# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step Qwen3 caption encoder built on the stock ``tt_transformers`` graph.

This is a thin subclass of
:class:`~models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm.Qwen3ForEmbedding`
(an existing tt-metal reference wrapper for Qwen3-Embedding models). The parent provides
all of the model construction:

- ``models.tt_transformers.tt.generator_vllm.initialize_vllm_text_transformer`` →
  :class:`~models.tt_transformers.tt.model.Transformer` (Embedding, Attention with fused
  QKV / paged SDPA / q_norm / k_norm, MLP, DistributedNorm(RMSNorm), HfRotarySetup, LMHead)
- :class:`~models.tt_transformers.tt.common.PagedAttentionConfig` paged KV cache
- :class:`~models.tt_transformers.tt.generator.Generator` driver

The **only** ACE-Step-specific surface here is the override of :meth:`forward` to return
**per-token** hidden states ``[B, 1, S, H]`` (post final RMSNorm, pre-LMHead). The parent's
``Generator.prefill_forward_text(return_hidden_states=True)`` returns the *pooled*
last-token vector ``[B, H]`` (typical embedding-model semantics) — that's not what
ACE-Step's DiT cross-attention needs. The override skips ``Generator`` and calls
``Transformer.ttnn_prefill_forward(get_last_token=-1)`` directly, then applies the final
``DistributedNorm`` to the whole sequence.

Drop-in replacement for the deleted ``TtQwen3EmbeddingEncoder`` — same constructor
signature ``(device, hf_model_dir, qwen_safetensors_path)`` and same
``forward(input_ids_np, attention_mask_np) -> ttnn.Tensor [B, 1, S, H]`` API, plus the
``embed_tokens(input_ids_np) -> ttnn.Tensor`` helper used by the lyric token-embedding path.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch

import ttnn
from models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm import Qwen3ForEmbedding
from models.tt_transformers.tt.common import Mode, num_blocks_in_seq


@contextmanager
def _hf_model_env(hf_model_dir: str):
    """Temporarily set ``HF_MODEL`` so ``ModelArgs._set_hf_params`` reads our checkpoint dir.

    The parent ``Qwen3ForEmbedding._initialize_model`` eventually calls
    ``initialize_vllm_text_transformer`` which constructs a ``ModelArgs`` driven off the
    ``HF_MODEL`` env var. Same trick the 5 Hz causal LM wrapper uses.
    """
    key = "HF_MODEL"
    old = os.environ.get(key)
    os.environ[key] = str(hf_model_dir)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class AceStepQwen3Encoder(Qwen3ForEmbedding):
    """Per-token Qwen3 caption encoder for ACE-Step DiT conditioning.

    Subclass of ``Qwen3ForEmbedding``; overrides only the ``forward`` pooling step.
    """

    def __init__(
        self,
        *,
        device,
        hf_model_dir: str,
        qwen_safetensors_path: Optional[str] = None,
        max_batch_size: int = 1,
        max_seq_len: int = 256,
        **kwargs: Any,
    ) -> None:
        # ``qwen_safetensors_path`` is accepted for drop-in compatibility with the deleted
        # ``TtQwen3EmbeddingEncoder`` constructor but is unused — the parent loads weights
        # via ``ModelArgs.load_state_dict`` which reads from ``hf_model_dir`` directly.
        del qwen_safetensors_path

        self._hf_model_dir = str(hf_model_dir)
        with _hf_model_env(self._hf_model_dir):
            super().__init__(
                device=device,
                max_batch_size=int(max_batch_size),
                max_seq_len=int(max_seq_len),
                model_name=self._hf_model_dir,  # AutoConfig accepts a local path
                **kwargs,
            )
            # Eagerly build the model + paged KV cache so the first forward() call doesn't
            # pay the model-construction cost (matches the deleted TtQwen3EmbeddingEncoder's
            # behaviour where everything was built in __init__).
            self._initialize_model(int(max_batch_size), int(max_seq_len))

    # ------------------------------------------------------------------
    # ACE-Step API surface (drop-in for the deleted TtQwen3EmbeddingEncoder)
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        input_ids,
        attention_mask=None,
    ):
        """Return per-token hidden states as a TTNN tensor ``[B, 1, S, H]``.

        Args:
            input_ids: ``np.ndarray`` (``uint32``) or ``torch.Tensor`` of shape ``[B, S]``.
            attention_mask: optional ``[B, S]`` mask. Unused by this wrapper: the parent's
                ``prepare_inputs_prefill`` path pads tokens to ``max_seq_len`` and lets the
                stock ``Attention.forward_prefill`` build the causal mask internally. The
                caller is expected to right-pad ``input_ids`` to a length ``<= max_seq_len``
                with ``pad_token_id`` (the encoder slices the output back to the input width).

        Returns:
            TTNN tensor of shape ``[B, 1, S, H]`` (TILE layout, ``bfloat16``).
        """
        del attention_mask  # see docstring

        ids_t = _to_torch_int64(input_ids)
        if ids_t.dim() != 2:
            raise ValueError(f"input_ids must be [B, S], got {tuple(ids_t.shape)}")
        b, s = int(ids_t.shape[0]), int(ids_t.shape[1])
        if b > self.max_batch_size:
            raise ValueError(f"batch size {b} > max_batch_size {self.max_batch_size}")
        if s > self.max_seq_len:
            raise ValueError(f"seq_len {s} > max_seq_len {self.max_seq_len}")

        with _hf_model_env(self._hf_model_dir):
            self._initialize_model(self.max_batch_size, self.max_seq_len)

        # Pad each user's ids to ``max_seq_len`` (trace reuse + ``Attention.forward_prefill``
        # ``seq_len % 128 == 0`` assert). Then run prefill per-user — TT executes embedding
        # models one user at a time anyway (see Qwen3ForEmbedding._initialize_model comment).
        if s < self.max_seq_len:
            pad = torch.zeros((b, self.max_seq_len - s), dtype=ids_t.dtype)
            ids_padded = torch.cat([ids_t, pad], dim=-1)
        else:
            ids_padded = ids_t

        tt_model = self.model[0]
        block_size = self.paged_attention_config.block_size
        n_blocks = num_blocks_in_seq(self.max_seq_len, block_size)

        per_user_hs: list[torch.Tensor] = []
        for user_idx in range(b):
            # Per-user page table with disjoint physical blocks (mirrors the parent's
            # forward(): "each batch item gets its own set of distinct sequential blocks").
            page_table = torch.arange(user_idx * n_blocks, (user_idx + 1) * n_blocks, dtype=torch.int32).reshape(
                1, n_blocks
            )

            ids_user = ids_padded[user_idx : user_idx + 1].contiguous()
            prep = tt_model.prepare_inputs_prefill(
                ids_user,
                page_table=page_table,
                batch_size=1,
                user_id=0,
            )
            prefill_input, rot_mats_global, rot_mats_local, page_table_tt, _ = prep

            # ``get_last_token=-1`` ⇒ Transformer.forward returns the raw post-block hidden
            # states (before final RMSNorm + LMHead). The LMHead slicing isn't applied, so
            # the output covers every token position.
            hidden = tt_model.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                page_table=page_table_tt,
                chunk_page_table=None,
                chunk_start_idx=None,
                get_last_token=-1,
                kv_cache=self._kv_cache[0],
                batch_size=1,
            )

            # Apply the final ``DistributedNorm`` over the whole sequence. With mode=PREFILL
            # and no sharded ``norm_config``, the inner ``RMSNorm`` is shape-agnostic on the
            # sequence axis (unlike the LMHead path which requires shard_height=32).
            hidden_normed = tt_model.norm(hidden, mode=Mode.PREFILL)
            ttnn.deallocate(hidden)

            # Read back; trim TILE padding on the sequence axis to the real ``s`` tokens.
            host = ttnn.to_torch(hidden_normed).float().contiguous()
            ttnn.deallocate(hidden_normed)
            host = host.reshape(1, 1, -1, host.shape[-1])
            if int(host.shape[2]) > s:
                host = host[:, :, :s, :]
            per_user_hs.append(host)

        out_torch = torch.cat(per_user_hs, dim=0)  # [B, 1, S, H]
        out_tt = ttnn.from_torch(
            out_torch.to(dtype=torch.bfloat16).contiguous(),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out_tt

    def embed_tokens(self, input_ids) -> Any:
        """Embedding lookup only — returns ``[B, S, H]`` device tensor.

        Drop-in for the deleted ``TtQwen3EmbeddingEncoder.embed_tokens`` (used by
        ``official_lm_preprocess._lyric_replacement`` for lyric tokens — those don't go
        through the transformer body, just the embedding table).
        """
        with _hf_model_env(self._hf_model_dir):
            self._initialize_model(self.max_batch_size, self.max_seq_len)

        ids_t = _to_torch_int64(input_ids)
        if ids_t.dim() != 2:
            raise ValueError(f"input_ids must be [B, S], got {tuple(ids_t.shape)}")

        tt_model = self.model[0]
        # Embedding.forward expects ``uint32`` ROW_MAJOR ids; the stock ``Embedding`` op
        # returns TILE ``[B, S, H]`` (the bridge in qwen_tt_transformers_lm.py uses the
        # exact same pattern).
        ids_tt = ttnn.from_torch(
            ids_t.to(torch.int64),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        h = tt_model.embd(ids_tt)
        try:
            ttnn.deallocate(ids_tt)
        except Exception:
            pass
        return h


def _to_torch_int64(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.asarray(x, dtype=np.int64))
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.int64, device="cpu")
    raise TypeError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")


__all__ = ["AceStepQwen3Encoder"]
