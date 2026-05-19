# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full ``tt_transformers``-backed causal LM stack for ACE-Step 5 Hz checkpoints.

This module is the **only** TTNN body for the 5 Hz causal LM. It is a thin driver
around the stock ``tt_transformers`` model graph constructed by
:func:`models.tt_transformers.tt.common.create_tt_model` — every LM-handler component
maps directly to a ``models/tt_transformers`` (or ``models/common``) primitive:

================================  =====================================================
LM-handler component               Stock ``tt_transformers`` building block
================================  =====================================================
``model.embed_tokens``             :class:`models.tt_transformers.tt.embedding.Embedding`
                                   (auto-promoted to ``ScaledEmbedding`` when
                                   ``ModelArgs.embed_scale`` is set)
Prefill causal mask                Built inside
                                   :meth:`Attention.forward_prefill <models.tt_transformers.tt.attention.Attention.forward_prefill>`
                                   (no host mask tensor in this wrapper)
Position IDs / RoPE caches         :class:`models.tt_transformers.tt.rope.HfRotarySetup`
                                   (HF cos/sin via
                                   :func:`~models.tt_transformers.tt.rope.get_rot_mats_hf`)
RoPE rotate Q, K                   Inside ``Attention`` via
                                   :class:`HfRotarySetup` / ``ttnn.experimental.rotary_embedding_hf``
``input_layernorm`` /              :class:`~models.tt_transformers.tt.distributed_norm.DistributedNorm`
``post_attention_layernorm``       wrapping :class:`models.common.rmsnorm.RMSNorm`
``q_norm`` / ``k_norm``            Auto-wired inside ``Attention`` from state-dict
                                   keys ``*.q_norm.weight`` / ``*.k_norm.weight``
Q / K / V projection matmuls       Fused QKV inside ``Attention``
Q / K / V bias add                 ``Attention`` picks up ``*.{q,k,v}_proj.bias`` from
                                   state-dict (Qwen2 yes, Qwen3 no — both handled)
Reshape / permute to ``[B,H,S,D]`` ``ttnn.experimental.nlp_create_qkv_heads`` inside
                                   ``Attention``
KV cache growth                    Paged:
                                   ``ttnn.experimental.paged_fill_cache`` (prefill) /
                                   ``ttnn.experimental.paged_update_cache`` (decode),
                                   configured via
                                   :class:`~models.tt_transformers.tt.common.PagedAttentionConfig`
GQA repeat_interleave              Implicit in
                                   ``ttnn.transformer.paged_scaled_dot_product_attention_*``
Attention scores ``Q · Kᵀ``        Fused SDPA inside ``Attention``
Attention · V                      Fused SDPA inside ``Attention``
Output projection (``o_proj``)     Inside ``Attention`` (``self.wo``)
Residual add                       Inside
                                   :class:`~models.tt_transformers.tt.decoder.TransformerBlock`
                                   (raw op: ``ttnn.add``)
MLP (``gate_proj`` / ``up_proj``   :class:`models.tt_transformers.tt.mlp.MLP`
/ ``down_proj``)
Scaled softmax (fp32)              Fused SDPA (``numeric_stable=True`` baked in)
================================  =====================================================

The :class:`AceStepFiveHzExperimentalTtnnCausalLM` wrapper continues to receive
**torch** logits (one ``ttnn.to_torch`` at the model boundary), so all downstream
consumers (sampling, repetition-penalty, CFG combine) keep working unchanged.

**Caveats**

- This wrapper assumes the ACE-Step 5 Hz LM checkpoint is loadable via
  HuggingFace ``AutoModelForCausalLM`` from a standard HF directory
  (e.g. ``acestep-5Hz-lm-1.7B``).  ``ModelArgs._set_hf_params`` drives off
  ``AutoConfig.from_pretrained(HF_MODEL)``; for an out-of-the-box Qwen3 1.7B
  config this works without any registration.
- Currently fixed to ``batch_size=1`` (matches the bridge's contract).
- Paged KV is **mandatory** here: ``tt_transformers``' non-paged prefill is not
  covered by upstream tests for Qwen3 and we don't want to silently degrade.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

import torch

import ttnn
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    get_padded_prefill_len,
    num_blocks_in_seq,
)


@contextmanager
def _hf_model_env(hf_model_dir: str):
    """Temporarily set ``HF_MODEL`` so ``ModelArgs._set_hf_params`` reads our checkpoint dir."""
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


def _make_page_table(max_batch_size: int, paged_cfg: PagedAttentionConfig) -> torch.Tensor:
    """Identity-style per-user page table covering the full paged pool.

    Shape: ``[max_batch_size, max_num_blocks // max_batch_size]`` (the layout
    :class:`~models.tt_transformers.tt.generator.Generator` and
    :mod:`models.tt_transformers.demo.simple_text_demo` use).
    """
    perm = torch.randperm(paged_cfg.max_num_blocks)
    reverse = torch.argsort(perm)
    return reverse.reshape(max_batch_size, paged_cfg.max_num_blocks // max_batch_size)


class QwenModelTtTransformers:
    """Stock-``tt_transformers`` driver compatible with :class:`AceStepFiveHzExperimentalTtnnCausalLM`.

    Exposes a minimal surface (``forward(tokens, start_pos)``, ``reset_kv_cache()``,
    ``.config``) consumed by the experimental causal LM bridge.
    """

    def __init__(
        self,
        model_name: str,
        device: Any,
        *,
        max_seq_len: int = 16384,
        page_block_size: Optional[int] = None,
        page_max_num_blocks: Optional[int] = None,
        dtype=None,
        use_hf_rope: bool = True,
        validate_against_hf: bool = False,
    ) -> None:
        if validate_against_hf:
            raise RuntimeError("QwenModelTtTransformers does not support validate_against_hf=True.")

        self.device = device
        self.max_seq_len = int(max_seq_len)
        # Pick paged-attention params that comfortably cover ``max_seq_len`` for batch=1.
        # block_size=32 keeps the paged blocks aligned with the TILE size and matches the
        # simple_text_demo default; max_num_blocks is the smallest power-of-two block count
        # that holds ``max_seq_len`` per user (rounded up).
        block = int(page_block_size if page_block_size is not None else 32)
        min_blocks = max(1, (self.max_seq_len + block - 1) // block)
        # Round up to a multiple of max_batch_size (=1 here) — trivially satisfied.
        max_blocks = int(page_max_num_blocks if page_max_num_blocks is not None else max(min_blocks, 1024))
        if max_blocks * block < self.max_seq_len:
            raise ValueError(
                f"page_max_num_blocks={max_blocks} * page_block_size={block} = {max_blocks*block} "
                f"cannot hold max_seq_len={self.max_seq_len}."
            )
        self._paged_cfg = PagedAttentionConfig(block_size=block, max_num_blocks=max_blocks)

        # Stage the HF checkpoint into the ``HF_MODEL`` env var so ``ModelArgs._set_hf_params``
        # finds the right config / weights. Revert immediately after construction so other
        # ACE-Step components that read ``HF_MODEL`` aren't affected.
        with _hf_model_env(model_name):
            tt_dtype = dtype if dtype is not None else ttnn.bfloat8_b
            (
                self.model_args,
                self.tt_model,
                self.tt_kv_cache,
                self.state_dict,
            ) = create_tt_model(
                mesh_device=device,
                instruct=True,  # ACE-Step LM uses chat-style prompts
                max_batch_size=1,
                optimizations=None,  # default = accuracy
                max_seq_len=self.max_seq_len,
                paged_attention_config=self._paged_cfg,
                dtype=tt_dtype,
                use_hf_rope=bool(use_hf_rope),
            )

        # HF-compatible config view for the LM bridge (`AceStepFiveHzExperimentalTtnnCausalLM`
        # reads ``self.config.vocab_size``).
        self.config = self.model_args.hf_config
        # Stable random page table (same per-user mapping for the lifetime of this model;
        # we re-allocate on ``reset_kv_cache`` since paged-fill writes data into specific blocks
        # and we want fresh blocks for the next user).
        self._page_table_torch = _make_page_table(self.model_args.max_batch_size, self._paged_cfg)
        # Decode position cursor (per-user; we only support batch_size=1 here).
        self._cursor = 0
        # Stashed by ``_prefill`` so the experimental bridge can pull the real last-token
        # logits row out of the ``[1, 1, 32, padded_vocab]`` tile-aligned LMHead output.
        # ``None`` until at least one prefill has run.
        self._prefill_last_token_offset_in_tile: Optional[int] = None

    # ------------------------------------------------------------------
    # KV cache lifecycle
    # ------------------------------------------------------------------

    def reset_kv_cache(self) -> None:
        """Reset paged KV state by re-initializing per-layer paged buffers and reshuffling the page table.

        The simplest correct reset for a single-user paged setup is to drop the existing block
        contents (they will be overwritten on next prefill anyway, since paged_fill_cache writes
        to whichever blocks the page_table addresses) and zero the cursor. We also reshuffle the
        page table so a stale `current_pos` cannot accidentally point at valid old data.
        """
        self._cursor = 0
        self._page_table_torch = _make_page_table(self.model_args.max_batch_size, self._paged_cfg)
        # Re-init paged KV blocks in place. ``Attention.init_kv_cache`` allocates fresh ttnn
        # tensors; we then rebuild ``tt_kv_cache`` so the in-flight reference matches.
        for layer in self.tt_model.layers:
            attn = layer.attention
            if hasattr(attn, "init_kv_cache"):
                try:
                    attn.init_kv_cache(self.model_args, None)
                except Exception:
                    # init_kv_cache raises if paged_attention_config isn't on the Attention.
                    # That path means there's nothing to reset anyway.
                    pass
        self.tt_kv_cache = [layer.attention.layer_past for layer in self.tt_model.layers]

    # ------------------------------------------------------------------
    # Forward entry point
    # ------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> Any:
        """Return TTNN logits.

        - ``start_pos == 0`` and ``tokens.shape[1] > 1`` ⇒ prefill (returns
          ``[1, 1, 32, padded_vocab]`` — a single tile-row containing the real
          last-token logits at row ``self._prefill_last_token_offset_in_tile``;
          other rows belong to neighbouring real or pad tokens within the same tile).
        - Otherwise ⇒ decode (single new token; returns ``[1, 1, 32, padded_vocab]``
          where row 0 is the real user, rows 1–31 are batch-pad slots).

        The :class:`AceStepFiveHzExperimentalTtnnCausalLM` bridge reads
        ``self._prefill_last_token_offset_in_tile`` and composes / trims the TTNN
        tensor to a torch ``[1, seq_len, vocab_size]`` view downstream.
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [batch, seq], got shape {tuple(tokens.shape)}")
        if int(tokens.shape[0]) != 1:
            raise RuntimeError("QwenModelTtTransformers only supports batch_size=1 today.")
        S = int(tokens.shape[1])
        if int(start_pos) == 0 and S > 1:
            return self._prefill(tokens)
        if S != 1:
            raise RuntimeError(
                f"Decode path expects exactly 1 new token per step, got seq={S} at start_pos={start_pos}."
            )
        return self._decode(tokens, int(start_pos))

    # ------------------------------------------------------------------
    # Prefill (last-token logits via the stock Transformer.forward path)
    # ------------------------------------------------------------------

    def _prefill(self, tokens: torch.Tensor) -> Any:
        seq_len = int(tokens.shape[1])
        prefill_seq_len = get_padded_prefill_len(seq_len)
        # Right-pad the prompt with zeros so the prefill input is a TTNN-friendly length
        # (``Attention.forward_prefill`` asserts ``seq_len % 128 == 0``). The pad tokens still
        # feed the KV cache, but downstream sampling only reads the last real position.
        if prefill_seq_len != seq_len:
            pad = torch.zeros(1, prefill_seq_len - seq_len, dtype=tokens.dtype, device=tokens.device)
            padded_tokens = torch.cat([tokens, pad], dim=-1)
        else:
            padded_tokens = tokens

        # Match ``Generator._get_prefill_user_page_table``: pass only the page-table entries
        # that cover the padded prefill sequence (``paged_fill_cache`` writes exactly
        # ``ceil(prefill_seq_len/block_size)`` blocks). Passing the full pool would also write
        # outside those blocks and corrupt unrelated KV state on subsequent prefills.
        block_size = self._paged_cfg.block_size
        n_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
        page_table_for_prefill = self._page_table_torch[:, :n_blocks].contiguous()

        # ``prepare_inputs_prefill`` runs the Embedding lookup and slices the prefill RoPE caches.
        # Output: (tokens_embd, rot_mats_global, rot_mats_local, page_table_tt, chunk_page_table_tt).
        inputs = self.tt_model.prepare_inputs_prefill(
            padded_tokens,
            page_table=page_table_for_prefill,
            batch_size=1,
            user_id=0,
        )
        (
            prefill_input,
            rot_mats_global,
            rot_mats_local,
            page_table_tt,
            _chunk_page_table_tt,
        ) = inputs

        # The stock tt_transformers ``LMHead`` is *width-sharded* and requires exactly one
        # TILE-row of input (``shard_height == 32``). We therefore can NOT call
        # ``_apply_norm_and_lm_head`` over the full ``[1,1,prefill_seq_len,H]`` hidden state —
        # ``interleaved_to_sharded`` trips ``TT_FATAL: Shard height 32 must match physical
        # height 128 for width sharded`` for any prefill_seq_len > 32. Instead we ask
        # ``Transformer.forward`` to slice the hidden state to the 32-tile-row containing
        # the last real token *before* applying the final RMSNorm and LMHead. This is exactly
        # what ``Generator.prefill_forward_text`` does for every other ``tt_transformers``
        # model and matches the canonical sharded-LMHead contract.
        last_token_idx = seq_len - 1
        get_last_token = (last_token_idx // 32) * 32

        logits_tt = self.tt_model.ttnn_prefill_forward(
            prefill_input,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=0,
            page_table=page_table_tt,
            chunk_page_table=None,
            chunk_start_idx=None,
            get_last_token=get_last_token,
            kv_cache=self.tt_kv_cache,
            batch_size=1,
        )
        # Tell the bridge which row of the returned tile corresponds to the real last token
        # (other rows in [get_last_token, get_last_token+32) are unused / pad-token logits).
        self._prefill_last_token_offset_in_tile = int(last_token_idx % 32)
        # Cursor is the number of *real* tokens we've processed (excluding right-pad).
        self._cursor = seq_len
        return logits_tt

    # ------------------------------------------------------------------
    # Decode (single new token)
    # ------------------------------------------------------------------

    def _decode(self, tokens: torch.Tensor, start_pos: int) -> Any:
        cur = int(start_pos)
        # tt_transformers' decode path expects [B] tokens and a [B] current_pos vector.
        decode_tokens = tokens.view(1).to(torch.int32)
        current_pos = torch.tensor([cur], dtype=torch.int32)
        # ``prepare_decode_inputs_host`` pads tokens to 32 (one tile row), builds the rot_idxs and
        # current_pos tensors on the mesh in the right sharding, and (when paged) replicates the
        # page table to every device.
        host_inputs = self.tt_model.prepare_decode_inputs_host(
            decode_tokens,
            current_pos,
            page_table=self._page_table_torch,
        )
        from models.tt_transformers.tt.common import copy_host_to_device

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.device)
        tt_tokens, tt_current_pos, rope_idxs, tt_page_table = device_inputs

        out = self.tt_model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=rope_idxs,
            page_table=tt_page_table,
            kv_cache=self.tt_kv_cache,
            sampling_on_device=False,
        )
        # ``ttnn_decode_forward`` returns either (logits, None) when sampling is off-device, or
        # (tokens, log_probs) when on-device sampling is engaged. We disable on-device sampling
        # for parity with the legacy bridge, so we always get the (logits, None) tuple.
        if isinstance(out, tuple):
            logits_tt = out[0]
        else:
            logits_tt = out

        self._cursor = cur + 1
        return logits_tt


__all__ = ["QwenModelTtTransformers"]
