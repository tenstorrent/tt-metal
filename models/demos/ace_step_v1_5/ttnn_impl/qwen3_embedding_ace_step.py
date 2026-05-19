# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step Qwen3 caption encoder built on stock ``tt_transformers`` primitives.

Mirrors the canonical non-vLLM Qwen3-Embedding-8B reference at
``models/demos/wormhole/qwen3_embedding_8b/demo/demo.py`` (the standalone CLI demo),
not the vLLM-serving wrapper ``Qwen3ForEmbedding`` in the sibling ``generator_vllm.py``.

The whole encoder is built directly from
:func:`models.tt_transformers.tt.common.create_tt_model` →
:class:`~models.tt_transformers.tt.model.Transformer` with paged KV cache; no detour
through ``tt_transformers/tt/generator_vllm.py`` (so this module loads cleanly in
environments without vLLM installed). The full graph is the stock ``tt_transformers``
stack:

- :class:`~models.tt_transformers.tt.embedding.Embedding`
- :class:`~models.tt_transformers.tt.attention.Attention` (fused QKV, q_norm/k_norm, paged SDPA)
- :class:`~models.tt_transformers.tt.mlp.MLP`
- :class:`~models.tt_transformers.tt.distributed_norm.DistributedNorm` wrapping
  :class:`models.common.rmsnorm.RMSNorm` for input / post-attn / final norms
- :class:`~models.tt_transformers.tt.rope.HfRotarySetup` (HF cos/sin via ``get_rot_mats_hf``)
- Paged KV cache via :class:`~models.tt_transformers.tt.common.PagedAttentionConfig`
  (``paged_fill_cache`` for prefill)

ACE-Step's DiT cross-attention needs **per-token** hidden states ``[B, 1, S, H]``, not
the pooled last-token vector ``[B, H]`` that ``Generator.prefill_forward_text(
return_hidden_states=True)`` returns. The :meth:`forward` method therefore skips
``Generator`` and calls
:meth:`Transformer.ttnn_prefill_forward(get_last_token=-1) <models.tt_transformers.tt.model.Transformer.ttnn_prefill_forward>`
directly to get raw post-block hidden states, then applies the final ``DistributedNorm``
over the whole sequence (the LMHead is bypassed entirely — this is an encoder, not a
causal LM).

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
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, create_tt_model


@contextmanager
def _hf_model_env(hf_model_dir: str):
    """Temporarily set ``HF_MODEL`` so ``ModelArgs._set_hf_params`` reads our checkpoint dir.

    ``create_tt_model`` constructs a ``ModelArgs`` driven off the ``HF_MODEL`` env var;
    we set it transiently here so other ACE-Step components that read ``HF_MODEL`` (or
    the rest of the test session) are not affected by our model load.
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


class AceStepQwen3Encoder:
    """Qwen3 caption encoder for ACE-Step DiT conditioning.

    Mirrors the canonical ``models/demos/wormhole/qwen3_embedding_8b/demo/demo.py`` pattern:
    build the model via :func:`create_tt_model` with paged KV, then drive prefill through
    :meth:`Transformer.ttnn_prefill_forward` directly to expose per-token hidden states
    (the standard embedding-model pooled output is wrong for ACE-Step's DiT cross-attention).

    Public API is drop-in compatible with the deleted ``TtQwen3EmbeddingEncoder``.
    """

    def __init__(
        self,
        *,
        device: Any,
        hf_model_dir: str,
        qwen_safetensors_path: Optional[str] = None,
        max_batch_size: int = 1,
        max_seq_len: int = 256,
        page_block_size: int = 32,
        page_max_num_blocks: Optional[int] = None,
        dtype=None,
        use_hf_rope: bool = True,
    ) -> None:
        # ``qwen_safetensors_path`` is accepted for drop-in compatibility with the deleted
        # ``TtQwen3EmbeddingEncoder`` constructor but is unused — ``create_tt_model`` loads
        # weights via ``ModelArgs.load_state_dict`` which reads from ``hf_model_dir`` directly.
        del qwen_safetensors_path

        self.device = device
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self._hf_model_dir = str(hf_model_dir)

        # Paged KV config sized to hold ``max_batch_size`` users at ``max_seq_len`` tokens
        # each (same recipe as the demo.py reference for the embedding model). Each user
        # gets a disjoint band of blocks via the page_table in ``forward()``.
        block_size = int(page_block_size)
        blocks_per_seq = (self.max_seq_len + block_size - 1) // block_size
        min_blocks = max(1024, blocks_per_seq * self.max_batch_size)
        max_blocks = int(page_max_num_blocks) if page_max_num_blocks is not None else min_blocks
        if max_blocks < blocks_per_seq * self.max_batch_size:
            raise ValueError(
                f"page_max_num_blocks={max_blocks} too small: need at least "
                f"blocks_per_seq * max_batch_size = {blocks_per_seq * self.max_batch_size}"
            )
        self._paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_blocks)

        tt_dtype = dtype if dtype is not None else ttnn.bfloat8_b

        # Stage HF_MODEL so ModelArgs reads our local checkpoint dir, then build the model
        # exactly the way the canonical Qwen3-Embedding demo does.
        with _hf_model_env(self._hf_model_dir):
            self.model_args, self.tt_model, self.tt_kv_cache, self.state_dict = create_tt_model(
                device,
                instruct=False,  # embedding model, not chat — matches demo.py's instruct=False
                max_batch_size=self.max_batch_size,
                optimizations=None,  # default = accuracy
                max_seq_len=self.max_seq_len,
                paged_attention_config=self._paged_cfg,
                dtype=tt_dtype,
                use_hf_rope=bool(use_hf_rope),
            )

        # HF-compatible config view for callers that read encoder hidden_size etc.
        self.config = self.model_args.hf_config
        self._blocks_per_seq = blocks_per_seq

    # ------------------------------------------------------------------
    # ACE-Step API surface (drop-in for the deleted TtQwen3EmbeddingEncoder)
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask=None) -> Any:
        """Return per-token hidden states as a TTNN tensor ``[B, 1, S, H]``.

        Args:
            input_ids: ``np.ndarray`` (``uint32``) or ``torch.Tensor`` of shape ``[B, S]``.
            attention_mask: optional ``[B, S]`` mask. Unused by this wrapper: ``input_ids``
                is right-padded to ``max_seq_len`` and ``Attention.forward_prefill`` builds
                a strict causal mask internally. Callers are expected to right-pad with
                ``pad_token_id`` (the encoder slices the output back to the real width).

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

        # Pad each user's ids to ``max_seq_len`` (Attention.forward_prefill asserts
        # ``seq_len % 128 == 0``; also keeps the trace-key consistent across calls).
        if s < self.max_seq_len:
            pad = torch.zeros((b, self.max_seq_len - s), dtype=ids_t.dtype)
            ids_padded = torch.cat([ids_t, pad], dim=-1)
        else:
            ids_padded = ids_t

        per_user_hs: list[torch.Tensor] = []
        for user_idx in range(b):
            # Per-user page table with disjoint physical blocks (same pattern the
            # canonical demo.py uses: each batch item gets its own block band so the
            # paged KV cache doesn't collide between users).
            page_table = torch.arange(
                user_idx * self._blocks_per_seq,
                (user_idx + 1) * self._blocks_per_seq,
                dtype=torch.int32,
            ).reshape(1, self._blocks_per_seq)

            ids_user = ids_padded[user_idx : user_idx + 1].contiguous()
            prep = self.tt_model.prepare_inputs_prefill(
                ids_user,
                page_table=page_table,
                batch_size=1,
                user_id=0,
            )
            prefill_input, rot_mats_global, rot_mats_local, page_table_tt, _ = prep

            # ``get_last_token=-1`` ⇒ Transformer.forward returns raw post-block hidden
            # states (before final RMSNorm + LMHead). The LMHead is never invoked — this
            # is an encoder, so we skip it entirely.
            hidden = self.tt_model.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                page_table=page_table_tt,
                chunk_page_table=None,
                chunk_start_idx=None,
                get_last_token=-1,
                kv_cache=self.tt_kv_cache,
                batch_size=1,
            )

            # Apply the final ``DistributedNorm`` over the whole sequence. With
            # ``mode=PREFILL`` and no sharded ``norm_config``, the inner RMSNorm is
            # shape-agnostic on the sequence axis (unlike the LMHead path which
            # demands ``shard_height==32``).
            hidden_normed = self.tt_model.norm(hidden, mode=Mode.PREFILL)
            try:
                ttnn.deallocate(hidden)
            except Exception:
                pass

            host = ttnn.to_torch(hidden_normed).float().contiguous()
            try:
                ttnn.deallocate(hidden_normed)
            except Exception:
                pass
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
        ``official_lm_preprocess._lyric_replacement`` for lyric tokens that don't go
        through the transformer body — just the embedding table).
        """
        ids_t = _to_torch_int64(input_ids)
        if ids_t.dim() != 2:
            raise ValueError(f"input_ids must be [B, S], got {tuple(ids_t.shape)}")

        # ``Transformer.embd`` is the stock ``tt_transformers.tt.embedding.Embedding``
        # instance built during ``create_tt_model``; its forward expects ``uint32``
        # ROW_MAJOR ids and returns TILE ``[B, S, H]``.
        ids_tt = ttnn.from_torch(
            ids_t.to(torch.int64),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        h = self.tt_model.embd(ids_tt)
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
