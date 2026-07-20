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

**Trace**

- With ``use_prefill_trace=True``, prefill uses ``_prefill_traced`` (``transform_and_embed`` + prefill in
  capture, matching ``Generator._capture_trace_prefill``). Traces are keyed by ``(padded_prefill_len,
  real_seq_len)`` because ``get_last_token`` is fixed at capture time.
- With ``use_decode_trace=True``, decode uses ``_decode_traced`` (per-token ``execute_trace``).
- ``use_trace=True`` enables both (legacy convenience).

**Memory (P1)**

- ``ACE_STEP_LM_PREFILL_L1``: prefill activations in L1 (default **on**; ``0`` opts out to DRAM).
  Attention/MLP use swept ``l1/dram/l1`` + ``l1/dram/ws`` pins; residual skip stays DRAM.
- ``ACE_STEP_LM_UNIFIED_DECODE_SHARD=1`` (default): reserved hook for decode shard unification
  (see :mod:`qwen_decode_shard`; currently no-op — matmul output grids differ from residual).
- ``ACE_STEP_LM_DECODE_QK_NORM_SHARDED=1`` (default): sharded Q/K head norms (see :mod:`qwen_decode_qk_norm`).
- ``ACE_STEP_LM_SDPA_GATHER_UNIFIED=1`` (default): reserved hook for post-SDPA gather layout
  (see :mod:`qwen_decode_sdpa_layout`; currently no-op — stock ``[32, 32]`` users grid).
- ``ACE_STEP_LM_NARROW_AUDIO_VOCAB=1`` (default): narrow ``LMHead`` column band in codes phase (see :mod:`ace_step_lm_head_narrow`).
- ``ACE_STEP_LM_LM_HEAD_SHARDED_NORM=1`` (default): sharded prefill final RMSNorm before ``LMHead`` (see :mod:`qwen_lm_head_sharded_norm`).
- ``ACE_STEP_LM_TP=1``: multi-device LM — pass a multi-chip mesh into this wrapper (demo:
  ``open_preprocess_device`` opens the full mesh). ``ModelArgs.num_devices`` and the
  ``all_gather`` after ``LMHead`` (below) activate automatically. Default remains 1×1 preprocess.

**Caveats**

- This wrapper assumes the ACE-Step 5 Hz LM checkpoint is loadable via
  HuggingFace ``AutoModelForCausalLM`` from a standard HF directory
  (e.g. ``acestep-5Hz-lm-1.7B``).  ``ModelArgs._set_hf_params`` drives off
  ``AutoConfig.from_pretrained(HF_MODEL)``; for an out-of-the-box Qwen3 1.7B
  config this works without any registration.
- Currently fixed to ``batch_size=1`` (matches the bridge's contract).
- Paged KV is **mandatory** here: ``tt_transformers``' non-paged prefill is not
  covered by upstream tests for Qwen3 and we don't want to silently degrade.

**PCC / accuracy**

Decoder precision/fidelity comes from
``models/experimental/ace_step_v1_5/model_params/<variant>/accuracy_decoder_config.json``
via :func:`~math_perf_env.ace_step_five_hz_lm_accuracy_optimizations` (BF16 + HiFi4).

Default path also uses swept prefill matmul program configs (HiFi4 + BF16, 1D 8×4 w8):
QKV 128×2048×4096 (l1/dram/l1), WO 128×2048×2048 (l1/dram/ws), MLP w1/w3 128×2048×6144 and
w2 128×6144×2048 (l1/dram/ws) via :func:`~math_perf_env.ace_step_lm_prefill_qkv_sweep_enabled`
and :func:`~math_perf_env.ace_step_lm_prefill_mlp_sweep_enabled` (both default on).
Set ``ACE_STEP_LM_PREFILL_QKV_SWEEP=0`` / ``ACE_STEP_LM_PREFILL_MLP_SWEEP=0`` for stock configs.
Explicit ``ACE_STEP_LM_BFLOAT8_WEIGHTS=1`` opts into HiFi2 + ``bfloat8_b`` weights without
the pinned program config (unless sweep is also on).
"""

from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch

import ttnn
from models.experimental.ace_step_v1_5.ttnn_impl.ace_step_lm_head_narrow import ace_step_patch_lm_head_narrow_forward
from models.experimental.ace_step_v1_5.ttnn_impl.lm_logits_debug import ace_step_debug_lm_logits_enabled
from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_five_hz_lm_accuracy_optimizations,
    ace_step_five_hz_lm_bfloat8_weights_enabled,
    ace_step_five_hz_lm_optimizations,
    ace_step_lm_decode_qk_norm_sharded_enabled,
    ace_step_lm_head_sharded_norm_enabled,
    ace_step_lm_narrow_audio_vocab_enabled,
    ace_step_lm_prefill_l1_enabled,
    ace_step_lm_prefill_mlp_sweep_enabled,
    ace_step_lm_prefill_qkv_sweep_enabled,
    ace_step_lm_sdpa_concat_width_enabled,
    ace_step_lm_unified_decode_shard_enabled,
)
from models.experimental.ace_step_v1_5.ttnn_impl.qwen_decode_qk_norm import ace_step_apply_qwen_decode_qk_norm
from models.experimental.ace_step_v1_5.ttnn_impl.qwen_decode_sdpa_layout import (
    ace_step_patch_model_args_sdpa_gather_unified,
)
from models.experimental.ace_step_v1_5.ttnn_impl.qwen_decode_shard import ace_step_patch_model_args_decode_unified_shard
from models.experimental.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm import ace_step_apply_lm_head_sharded_norm
from models.experimental.ace_step_v1_5.ttnn_impl.qwen_prefill_l1 import (
    ace_step_apply_qwen_prefill_l1,
    ace_step_patch_model_args_lm_prefill_mlp_ff1_3_matmul,
    ace_step_patch_model_args_lm_prefill_mlp_ff2_matmul,
    ace_step_patch_model_args_lm_prefill_qkv_matmul,
    ace_step_patch_model_args_lm_prefill_wo_matmul,
    ace_step_promote_attention_wqkv_to_dram_interleaved,
    ace_step_promote_mlp_prefill_dram_interleaved,
    ace_step_qwen_prefill_l1_op_context,
)
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    copy_host_to_device,
    create_tt_model,
    get_padded_prefill_len,
    num_blocks_in_seq,
)


@dataclass
class _DecodeTraceState:
    trace_id: Optional[int] = None
    device_inputs: Optional[tuple] = None
    logits_tt: Optional[Any] = None


@dataclass
class _PrefillTraceState:
    trace_id: Optional[int] = None
    device_inputs: Optional[tuple] = None
    rot_mats_global: Any = None
    rot_mats_local: Any = None
    logits_tt: Optional[Any] = None
    get_last_token: int = 0
    last_token_offset_in_tile: int = 0


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
        use_trace: bool = False,
        use_prefill_trace: Optional[bool] = None,
        use_decode_trace: Optional[bool] = None,
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
            _bf8_weights = ace_step_five_hz_lm_bfloat8_weights_enabled()
            if dtype is not None:
                tt_dtype = dtype
            elif _bf8_weights:
                tt_dtype = ttnn.bfloat8_b
            else:
                tt_dtype = ttnn.bfloat16
            lm_optimizations = (
                ace_step_five_hz_lm_optimizations if _bf8_weights else ace_step_five_hz_lm_accuracy_optimizations
            )
            (
                self.model_args,
                self.tt_model,
                self.tt_kv_cache,
                self.state_dict,
            ) = create_tt_model(
                mesh_device=device,
                instruct=True,  # ACE-Step LM uses chat-style prompts
                max_batch_size=1,
                optimizations=lm_optimizations,
                max_seq_len=self.max_seq_len,
                paged_attention_config=self._paged_cfg,
                dtype=tt_dtype,
                use_hf_rope=bool(use_hf_rope),
            )

        if ace_step_lm_prefill_l1_enabled():
            ace_step_apply_qwen_prefill_l1(self.tt_model, self.model_args)
        if ace_step_lm_prefill_qkv_sweep_enabled():
            ace_step_patch_model_args_lm_prefill_qkv_matmul(self.model_args, device)
            ace_step_patch_model_args_lm_prefill_wo_matmul(self.model_args, device)
            ace_step_promote_attention_wqkv_to_dram_interleaved(self.tt_model)
        if ace_step_lm_prefill_mlp_sweep_enabled():
            ace_step_patch_model_args_lm_prefill_mlp_ff1_3_matmul(self.model_args, device)
            ace_step_patch_model_args_lm_prefill_mlp_ff2_matmul(self.model_args, device)
            ace_step_promote_mlp_prefill_dram_interleaved(self.tt_model)
        if ace_step_lm_unified_decode_shard_enabled():
            ace_step_patch_model_args_decode_unified_shard(self.model_args)
        if ace_step_lm_sdpa_concat_width_enabled():
            ace_step_patch_model_args_sdpa_gather_unified(self.model_args)
        if ace_step_lm_decode_qk_norm_sharded_enabled():
            ace_step_apply_qwen_decode_qk_norm(self.tt_model, self.model_args)
        if ace_step_lm_head_sharded_norm_enabled():
            ace_step_apply_lm_head_sharded_norm(self.tt_model, self.model_args)
        if ace_step_lm_narrow_audio_vocab_enabled() and hasattr(self.tt_model, "lm_head"):
            ace_step_patch_lm_head_narrow_forward(self.tt_model.lm_head)

        self._narrow_vocab_indices: torch.Tensor | None = None

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
        _trace_api = hasattr(ttnn, "begin_trace_capture") and hasattr(ttnn, "execute_trace")
        if use_prefill_trace is None and use_decode_trace is None:
            _prefill = bool(use_trace)
            _decode = bool(use_trace)
        else:
            _prefill = bool(use_prefill_trace if use_prefill_trace is not None else use_trace)
            _decode = bool(use_decode_trace if use_decode_trace is not None else False)
        self._use_prefill_trace = _prefill and _trace_api
        self._use_decode_trace = _decode and _trace_api
        self._decode_trace = _DecodeTraceState()
        # Keyed by (padded_prefill_len, real_seq_len): ``get_last_token`` is baked into the
        # captured graph and must match ``seq_len``, not just the padded length bucket.
        self._prefill_traces: dict[tuple[int, int], _PrefillTraceState] = {}

    @contextlib.contextmanager
    def _prefill_l1_op_context(self):
        if ace_step_lm_prefill_l1_enabled():
            with ace_step_qwen_prefill_l1_op_context():
                yield
        else:
            yield

    # ------------------------------------------------------------------
    # KV cache lifecycle
    # ------------------------------------------------------------------

    def reset_kv_cache(self) -> None:
        """Reset paged KV state by reshuffling the page table and zeroing the cursor.

        The KV buffers are allocated once at construction (``Attention.init_kv_cache`` runs in
        the layer constructors). We deliberately do **not** re-allocate them here: ``paged_fill_cache``
        overwrites whichever blocks the page table addresses on the next prefill, and decode fills
        the remaining positions incrementally, so the previous contents are never read. Reshuffling
        the page table additionally prevents a stale ``current_pos`` from pointing at valid old data.

        Re-allocating per reset used to dominate device time: ``init_kv_cache`` builds the cache from
        ``torch.zeros`` (FP32) and converts to bf16 TILE on device, i.e. a Tilize (FP32=>FP32) plus
        Typecast (FP32=>BF16) over a multi-MB DRAM buffer, ×2 (K,V) ×every layer, on every prefill.
        """
        self.reset_kv_state_only()
        self.release_trace()

    def reset_kv_state_only(self) -> None:
        """Reshuffle page table and zero cursor without releasing captured traces."""
        self._cursor = 0
        self._page_table_torch = _make_page_table(self.model_args.max_batch_size, self._paged_cfg)
        self.tt_kv_cache = [layer.attention.layer_past for layer in self.tt_model.layers]

    def warmup_jit_compile(self, tokens: torch.Tensor) -> None:
        """Eager prefill + decode to populate the JIT cache (untimed; mirrors Devstral demos)."""
        was_prefill_trace = self._use_prefill_trace
        was_decode_trace = self._use_decode_trace
        self._use_prefill_trace = False
        self._use_decode_trace = False
        try:
            self.reset_kv_state_only()
            _ = self._prefill_eager(tokens)
            decode_pos = int(tokens.shape[1])
            last_tok = tokens[:, -1:]
            _ = self._decode_eager(last_tok.view(-1), decode_pos)
            ttnn.synchronize_device(self.device)
        finally:
            self._use_prefill_trace = was_prefill_trace
            self._use_decode_trace = was_decode_trace
            self.reset_kv_state_only()

    def warmup_trace_capture(self, tokens: torch.Tensor) -> None:
        """Capture prefill/decode traces after eager JIT warmup; keeps traces for replay."""
        if not self._use_prefill_trace and not self._use_decode_trace:
            return
        self.reset_kv_state_only()
        _ = self._prefill(tokens)
        decode_pos = int(tokens.shape[1])
        last_tok = tokens[:, -1:]
        _ = self._decode(last_tok.view(-1), decode_pos)
        ttnn.synchronize_device(self.device)
        self.reset_kv_state_only()

    def release_trace(self) -> None:
        if self._decode_trace.trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._decode_trace.trace_id)
            except Exception:
                pass
        self._decode_trace = _DecodeTraceState()
        for st in self._prefill_traces.values():
            if st.trace_id is not None:
                try:
                    ttnn.release_trace(self.device, st.trace_id)
                except Exception:
                    pass
        self._prefill_traces.clear()

    def set_narrow_audio_vocab_indices(self, indices: torch.Tensor | None) -> None:
        """Optional audio-code column band for ``LMHead`` (``ACE_STEP_LM_NARROW_AUDIO_VOCAB=1``)."""
        self._narrow_vocab_indices = indices
        lm_head = getattr(self.tt_model, "lm_head", None)
        if lm_head is None:
            return
        lm_head._ace_narrow_vocab_indices = indices  # type: ignore[attr-defined]

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
        # Trace capture includes ``transform_and_embed_prefill_inputs_device`` + prefill
        # (same graph as ``Generator._capture_trace_prefill``).
        if self._use_prefill_trace:
            return self._prefill_traced(tokens)
        return self._prefill_eager(tokens)

    def _prefill_eager(self, tokens: torch.Tensor) -> Any:
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
            _chunk_start_idx_tt,
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
        if ace_step_debug_lm_logits_enabled():
            self._last_debug_params = {
                "mode": "prefill",
                "seq_len": seq_len,
                "prefill_seq_len": prefill_seq_len,
                "last_token_idx": last_token_idx,
                "get_last_token": get_last_token,
                "offset_in_tile": int(last_token_idx % 32),
                "n_page_blocks": int(n_blocks),
            }
            print(
                f"[ace_step_lm_logits_debug] prefill.qwen_params "
                f"seq_len={seq_len} padded={prefill_seq_len} "
                f"last_token_idx={last_token_idx} get_last_token={get_last_token} "
                f"offset_in_tile={last_token_idx % 32} n_page_blocks={n_blocks}",
                flush=True,
            )

        with self._prefill_l1_op_context():
            logits_tt = self.tt_model.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                page_table=page_table_tt,
                chunk_page_table=_chunk_page_table_tt,
                chunk_start_idx=_chunk_start_idx_tt,
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

    def _prefill_traced(self, tokens: torch.Tensor) -> Any:
        seq_len = int(tokens.shape[1])
        prefill_seq_len = int(get_padded_prefill_len(seq_len))
        if prefill_seq_len != seq_len:
            pad = torch.zeros(1, prefill_seq_len - seq_len, dtype=tokens.dtype, device=tokens.device)
            padded_tokens = torch.cat([tokens, pad], dim=-1)
        else:
            padded_tokens = tokens

        block_size = self._paged_cfg.block_size
        n_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
        page_table_for_prefill = self._page_table_torch[:, :n_blocks].contiguous()

        last_token_idx = seq_len - 1
        get_last_token = (last_token_idx // 32) * 32
        last_off = int(last_token_idx % 32)

        trace_key = (prefill_seq_len, seq_len)
        st = self._prefill_traces.get(trace_key)
        if (
            st is None
            or st.trace_id is None
            or st.get_last_token != get_last_token
            or st.last_token_offset_in_tile != last_off
        ):
            host_inputs = self.tt_model.prepare_prefill_inputs_trace(
                padded_tokens,
                page_table=page_table_for_prefill,
            )
            tt_rot_global = host_inputs[1]
            tt_rot_local = host_inputs[2]
            host_tuple = (host_inputs[0], host_inputs[3], host_inputs[4], host_inputs[5])

            device_inputs = copy_host_to_device(host_tuple, mesh_device=self.device)
            (
                x_embd,
                tt_page_table,
                tt_chunk_page_table,
                tt_chunk_start_idx,
            ) = self.tt_model.transform_and_embed_prefill_inputs_device(*device_inputs)
            with self._prefill_l1_op_context():
                warm = self.tt_model.ttnn_prefill_forward(
                    x_embd,
                    rot_mats_global=tt_rot_global,
                    rot_mats_local=tt_rot_local,
                    page_table=tt_page_table,
                    chunk_page_table=tt_chunk_page_table,
                    chunk_start_idx=tt_chunk_start_idx,
                    kv_cache=self.tt_kv_cache,
                    get_last_token=get_last_token,
                )
            ttnn.synchronize_device(self.device)
            try:
                ttnn.deallocate(warm)
            except Exception:
                pass

            device_inputs = copy_host_to_device(host_tuple, mesh_device=self.device)
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            (
                x_embd,
                tt_page_table,
                tt_chunk_page_table,
                tt_chunk_start_idx,
            ) = self.tt_model.transform_and_embed_prefill_inputs_device(*device_inputs)
            with self._prefill_l1_op_context():
                _ = self.tt_model.ttnn_prefill_forward(
                    x_embd,
                    rot_mats_global=tt_rot_global,
                    rot_mats_local=tt_rot_local,
                    page_table=tt_page_table,
                    chunk_page_table=tt_chunk_page_table,
                    chunk_start_idx=tt_chunk_start_idx,
                    kv_cache=self.tt_kv_cache,
                    get_last_token=get_last_token,
                )
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)
            st = _PrefillTraceState(
                trace_id=trace_id,
                device_inputs=device_inputs,
                rot_mats_global=tt_rot_global,
                rot_mats_local=tt_rot_local,
                logits_tt=_,
                get_last_token=get_last_token,
                last_token_offset_in_tile=last_off,
            )
            self._prefill_traces[trace_key] = st

        host_inputs = self.tt_model.prepare_prefill_inputs_trace(
            padded_tokens,
            page_table=page_table_for_prefill,
        )
        host_tuple = (host_inputs[0], host_inputs[3], host_inputs[4], host_inputs[5])
        copy_host_to_device(host_tuple, st.device_inputs, mesh_device=self.device)
        ttnn.execute_trace(self.device, st.trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(self.device)
        logits_tt = st.logits_tt

        self._prefill_last_token_offset_in_tile = last_off
        self._cursor = seq_len
        return logits_tt

    # ------------------------------------------------------------------
    # Decode (single new token)
    # ------------------------------------------------------------------

    def _decode(self, tokens: torch.Tensor, start_pos: int) -> Any:
        if self._use_decode_trace:
            return self._decode_traced(tokens, start_pos)
        return self._decode_eager(tokens, start_pos)

    def _decode_eager(self, tokens: torch.Tensor, start_pos: int) -> Any:
        cur = int(start_pos)
        if ace_step_debug_lm_logits_enabled():
            self._last_debug_params = {
                "mode": "decode",
                "start_pos": cur,
                "token_id": int(tokens.view(-1)[0].item()),
            }
            print(
                f"[ace_step_lm_logits_debug] decode.qwen_params start_pos={cur} "
                f"token_id={int(tokens.view(-1)[0].item())}",
                flush=True,
            )
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
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.device)
        tt_tokens, tt_current_pos, rope_idxs, tt_page_table = device_inputs

        out = self.tt_model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=rope_idxs,
            page_table=tt_page_table,
            kv_cache=self.tt_kv_cache,
            on_device_logits=False,
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

    def _decode_traced(self, tokens: torch.Tensor, start_pos: int) -> Any:
        """Decode with trace replay (matches ``Generator._decode_forward_trace_text``).

        Every inference step copies fresh host inputs and runs ``execute_trace``, including
        the first step after capture setup. Returning logits from the in-capture forward
        without a replay left KV / output buffers out of sync with later steps.
        """
        cur = int(start_pos)
        decode_tokens = tokens.view(1).to(torch.int32)
        current_pos = torch.tensor([cur], dtype=torch.int32)
        host_inputs = self.tt_model.prepare_decode_inputs_host(
            decode_tokens,
            current_pos,
            page_table=self._page_table_torch,
        )

        if self._decode_trace.trace_id is None:
            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.device)
            warm = self.tt_model.ttnn_decode_forward(
                *device_inputs,
                kv_cache=self.tt_kv_cache,
                on_device_logits=False,
            )
            if isinstance(warm, tuple):
                warm_logits = warm[0]
            else:
                warm_logits = warm
            ttnn.synchronize_device(self.device)
            try:
                ttnn.deallocate(warm_logits)
            except Exception:
                pass

            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.device)
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            out = self.tt_model.ttnn_decode_forward(
                *device_inputs,
                kv_cache=self.tt_kv_cache,
                on_device_logits=False,
            )
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)
            if isinstance(out, tuple):
                logits_tt = out[0]
            else:
                logits_tt = out
            self._decode_trace = _DecodeTraceState(
                trace_id=trace_id,
                device_inputs=device_inputs,
                logits_tt=logits_tt,
            )

        copy_host_to_device(host_inputs, self._decode_trace.device_inputs, mesh_device=self.device)
        ttnn.execute_trace(self.device, self._decode_trace.trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(self.device)
        logits_tt = self._decode_trace.logits_tt

        self._cursor = cur + 1
        return logits_tt

    # ------------------------------------------------------------------
    # Post-transformer profiling (final norm + LMHead + logits postprocess)
    # ------------------------------------------------------------------

    def capture_prefill_hidden_tile(self, tokens: torch.Tensor) -> Any:
        """Run prefill transformer only; return ``[1,1,32,H]`` tile before final norm."""
        seq_len = int(tokens.shape[1])
        prefill_seq_len = get_padded_prefill_len(seq_len)
        if prefill_seq_len != seq_len:
            pad = torch.zeros(1, prefill_seq_len - seq_len, dtype=tokens.dtype, device=tokens.device)
            padded_tokens = torch.cat([tokens, pad], dim=-1)
        else:
            padded_tokens = tokens

        block_size = self._paged_cfg.block_size
        n_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
        page_table_for_prefill = self._page_table_torch[:, :n_blocks].contiguous()

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
            _chunk_start_idx_tt,
        ) = inputs

        with self._prefill_l1_op_context():
            hidden = self.tt_model.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                page_table=page_table_tt,
                chunk_page_table=_chunk_page_table_tt,
                chunk_start_idx=_chunk_start_idx_tt,
                get_last_token=-1,
                kv_cache=self.tt_kv_cache,
                batch_size=1,
            )

        last_token_idx = seq_len - 1
        get_last_token = (last_token_idx // 32) * 32
        return ttnn.slice(
            hidden,
            (0, 0, get_last_token, 0),
            (1, 1, get_last_token + 32, hidden.shape[-1]),
        )

    def forward_post_transformer_prefill(self, hidden_tile: Any) -> Any:
        """Final RMSNorm + sharded ``LMHead`` on a prefill last-token tile."""
        return self.tt_model._apply_norm_and_lm_head(hidden_tile)

    def capture_decode_hidden_tile(self, tokens: torch.Tensor, start_pos: int) -> Any:
        """Run decode transformer stack only; return hidden tile before final norm."""
        from models.tt_transformers.tt.common import Mode
        from models.tt_transformers.tt.model_config import TensorGroup

        tt_model = self.tt_model
        decode_tokens = tokens.view(1).to(torch.int32)
        current_pos = torch.tensor([int(start_pos)], dtype=torch.int32)
        host_inputs = tt_model.prepare_decode_inputs_host(
            decode_tokens,
            current_pos,
            page_table=self._page_table_torch,
        )
        tt_tokens, tt_current_pos, rope_idxs, tt_page_table = copy_host_to_device(host_inputs, mesh_device=self.device)

        rot_mats_global = tt_model.rope_setup.get_rot_mats(rope_idxs)
        rot_mats_local = (
            tt_model.rope_local_setup.get_rot_mats(rope_idxs) if hasattr(tt_model, "rope_local_setup") else None
        )

        x = tt_model._transform_decode_inputs_device(tt_tokens)
        if tt_model.prefetcher is not None:
            tt_model.prefetcher.run()

        for i, layer in enumerate(tt_model.layers):
            activation_dtype = tt_model.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )
            if not tt_model.args.is_galaxy:
                x = ttnn.to_memory_config(
                    x,
                    tt_model.args.get_residual_mem_config(Mode.DECODE, tt_model.prefetcher),
                    activation_dtype,
                )
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            x = layer(
                x,
                tt_current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                mode=Mode.DECODE,
                page_table=tt_page_table,
                chunk_page_table=None,
                chunk_start_idx=None,
                kv_cache=self.tt_kv_cache[i] if self.tt_kv_cache is not None else None,
                batch_size=1,
            )

        if tt_model.prefetcher is not None:
            tt_model.prefetcher.stop()
        return x

    def forward_post_transformer_decode(self, hidden_tile: Any) -> Any:
        """Decode-path final norm + ``LMHead`` (+ gather/untilize when multi-device)."""
        from models.tt_transformers.tt.common import Mode

        tt_model = self.tt_model
        x = hidden_tile
        x = tt_model.norm(
            x,
            mode=Mode.DECODE,
            norm_config=tt_model.args.get_norm_config("lm_head", Mode.DECODE, tt_model.prefetcher),
        )
        if tt_model.prefetcher is not None:
            x = ttnn.to_memory_config(x, tt_model.args.get_lm_head_input_mem_config(Mode.DECODE, tt_model.prefetcher))
        x = tt_model.lm_head(x)

        if tt_model.args.num_devices > 1:
            cluster_axis = 0 if tt_model.args.is_galaxy else None
            num_links = 2 if tt_model.args.is_galaxy else 1
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=tt_model.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=num_links,
                memory_config=x.memory_config() if tt_model.prefetcher is None else ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=cluster_axis,
                topology=tt_model.args.ccl_topology(),
                barrier_semaphore=tt_model.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
                subdevice_id=tt_model.prefetcher.worker_sub_device_id if tt_model.prefetcher is not None else None,
            )

        return ttnn.untilize(
            x,
            use_multicore=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=tt_model.prefetcher.all_worker_cores_range_set if tt_model.prefetcher is not None else None,
        )


__all__ = ["QwenModelTtTransformers"]
