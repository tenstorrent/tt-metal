# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
pplx-embed-v1-4B bidirectional attention subclass.

pplx-embed uses full bidirectional attention (every token attends to every
other token) rather than the causal mask used by standard Qwen3.  The only
change required at the TT graph level is flipping ``is_causal=True`` to
``is_causal=False`` in the ``ttnn.transformer.scaled_dot_product_attention``
call inside ``forward_prefill``.

The 4B backbone (Qwen3-4B: hidden=2560, 36 layers, 32 Q-heads / 8 KV-heads,
head_dim=128) is architecturally identical to the 0.6B variant apart from
those scale dimensions, so this subclass is shared verbatim with the 0.6B
model — only the SDPA causal flag, the optional bidirectional padding mask,
and the LoFi RoPE kernel-config plumbing live here.

Also incorporates the bs=1/ISL=512 Q-BFP8 skip optimisation from
Qwen3EmbeddingAttention: when ``TT_SKIP_KV_CACHE_FILL=1`` is set, K/V stay
native bf16, so the Q->BFP8 typecast is pure overhead and can be skipped.

Implementation uses a temporary wrapper around ``ttnn.transformer.scaled_dot_product_attention``
to inject ``is_causal=False`` without duplicating the entire ``forward_prefill``
method, keeping the subclass resilient to upstream changes.
"""

import functools
import os

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_concat_heads import nlp_concat_heads_headsplit
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_concat_heads import (
    supported as _concat_headsplit_supported,
)
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads import nlp_create_qkv_heads_headsplit
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads import supported as _qkv_headsplit_supported
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import (
    make_norm_constants,
    nlp_create_qkv_heads_norm_headsplit,
)
from models.tt_transformers.tt.attention import Attention

_OPTIMIZED_BATCH = 1
_OPTIMIZED_SEQ_LEN = 512

# Optional additive padding mask injected into the bidirectional prefill SDPA.
# Shape [b, nqh, S, S] (batch/head broadcastable). Real key columns are 0, padded
# key columns are a large negative value so padding is excluded from attention.
# A single shared device buffer is reused across all layers; the serving harness
# updates its contents per request before replaying the trace.
_PAD_ATTN_MASK = None


def set_pad_attn_mask(mask):
    """Set (or clear with ``None``) the shared bidirectional SDPA padding mask."""
    global _PAD_ATTN_MASK
    _PAD_ATTN_MASK = mask


def _interleaved_out(memory_config):
    """Head-split writers address the output by flat tile id, which only matches
    the interleaved layout. Sharded destinations must use the stock op."""
    return memory_config is None or not memory_config.is_sharded()


def _wrap_create_qkv_heads_headsplit(original_fn):
    """Route ``nlp_create_qkv_heads`` to the model-local head-split kernels.

    Falls back to the stock op whenever the fast path cannot express the call
    (sharded input, transposed K heads, indivisible head counts, …), so this is
    always safe to leave enabled.
    """

    @functools.wraps(original_fn)
    def wrapper(qkv_fused, *args, **kwargs):
        num_heads = kwargs.get("num_heads")
        num_kv_heads = kwargs.get("num_kv_heads", num_heads)
        transpose_k_heads = kwargs.get("transpose_k_heads", True)
        if (
            not args
            and num_heads is not None
            and _interleaved_out(kwargs.get("memory_config"))
            and _qkv_headsplit_supported(qkv_fused, num_heads, num_kv_heads, transpose_k_heads)
        ):
            return nlp_create_qkv_heads_headsplit(
                qkv_fused,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                memory_config=kwargs.get("memory_config"),
            )
        return original_fn(qkv_fused, *args, **kwargs)

    return wrapper


def _wrap_create_qkv_heads_norm(original_fn, consts):
    """Route ``nlp_create_qkv_heads`` to the head-split + Q/K RMSNorm fused op.

    ``consts`` = (gamma_q_tiles, gamma_k_tiles, scaler, eps) built once per layer. The
    caller makes ``q_norm``/``k_norm`` identities for the same forward so the norm is
    applied exactly once. Falls back to the stock op where the fast path cannot express
    the call (sharded input, transposed K heads, indivisible head counts).
    """
    gq, gk, sc, ep = consts

    @functools.wraps(original_fn)
    def wrapper(qkv_fused, *args, **kwargs):
        num_heads = kwargs.get("num_heads")
        num_kv_heads = kwargs.get("num_kv_heads", num_heads)
        transpose_k_heads = kwargs.get("transpose_k_heads", True)
        if (
            not args
            and num_heads is not None
            and _interleaved_out(kwargs.get("memory_config"))
            and _qkv_headsplit_supported(qkv_fused, num_heads, num_kv_heads, transpose_k_heads)
        ):
            return nlp_create_qkv_heads_norm_headsplit(
                qkv_fused,
                gq,
                gk,
                sc,
                ep,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                memory_config=kwargs.get("memory_config"),
            )
        return original_fn(qkv_fused, *args, **kwargs)

    return wrapper


def _wrap_concat_heads_headsplit(original_fn):
    """Route ``nlp_concat_heads`` to the model-local head-split kernels."""

    @functools.wraps(original_fn)
    def wrapper(context, *args, **kwargs):
        if not args and _interleaved_out(kwargs.get("memory_config")) and _concat_headsplit_supported(context):
            return nlp_concat_heads_headsplit(context, memory_config=kwargs.get("memory_config"))
        return original_fn(context, *args, **kwargs)

    return wrapper


def _wrap_sdpa_bidirectional(original_fn):
    """Return a wrapper that forces ``is_causal=False`` and injects the padding mask."""

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        kwargs["is_causal"] = False
        if _PAD_ATTN_MASK is not None and kwargs.get("attn_mask") is None:
            kwargs["attn_mask"] = _PAD_ATTN_MASK
        return original_fn(*args, **kwargs)

    return wrapper


class PplxBidirectionalAttention(Attention):
    """Drop-in replacement for ``tt_transformers.tt.attention.Attention``
    that uses bidirectional (non-causal) SDPA for pplx-embed models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fused head-split + Q/K RMSNorm (QWEN_FUSED_HEADS_NORM=1): one pass over the
        # fused QKV activation replaces nlp_create_qkv_heads + q_norm + k_norm. Constants
        # (row-replicated gamma tiles, 1/head_dim scaler, eps) are built once per layer.
        self._fused_norm_consts = None
        if os.getenv("QWEN_FUSED_HEADS_NORM", "0") == "1":
            names = (
                "mesh_device",
                "state_dict",
                "weight_cache_path",
                "layer_num",
                "dtype",
                "transformation_mats",
                "configuration",
            )
            bound = dict(zip(names, args))
            bound.update({k: v for k, v in kwargs.items() if k in names})
            configuration, state_dict, layer_num = bound["configuration"], bound["state_dict"], bound["layer_num"]
            layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
            qk, kk = f"{layer_name}.q_norm.weight", f"{layer_name}.k_norm.weight"
            if qk in state_dict and kk in state_dict and os.getenv("QWEN_SKIP_QK_NORM", "0") != "1":
                self._fused_norm_consts = make_norm_constants(
                    state_dict[qk], state_dict[kk], configuration.norm_eps, bound["mesh_device"]
                )
        # Ablation knob (DO NOT ENABLE): skipping the trained Q/K RMSNorm shaves
        # device time but collapses retrieval accuracy — the per-head Q/K norm is
        # load-bearing, not redundant. Kept gated/off as documentation of the
        # measured trade-off (see 0.6B notes: STS-B 0.848 -> 0.236 when skipped).
        if os.getenv("QWEN_SKIP_QK_NORM", "0") == "1":
            self.q_norm = lambda x, mode, norm_config: x
            self.k_norm = lambda x, mode, norm_config: x

    def _mllama_rope_prefill(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        # The rotary_embedding_llama op defaults to MathFidelity::HiFi4 (4 math
        # passes). RoPE is just a cos/sin rotation (operands in [-1, 1]), so a
        # lower fidelity is near-lossless while cutting the kernel's math passes.
        # QWEN_ROPE_FIDELITY: lofi | hifi2 | hifi4 (unset -> stock HiFi4).
        fidelity = os.getenv("QWEN_ROPE_FIDELITY", "").lower()
        if fidelity == "lofi":
            ckc = self.args.compute_kernel_config_lofi
        elif fidelity == "hifi2":
            ckc = self.args.compute_kernel_config_hifi2
        else:
            return super()._mllama_rope_prefill(q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
            compute_kernel_config=ckc,
        )
        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
            compute_kernel_config=ckc,
        )
        return q_heads_1QSD, k_heads_1KSD

    def _prepare_q_for_sdpa(self, q_heads_1QSD: ttnn.Tensor) -> ttnn.Tensor:
        if (
            self.max_batch_size == _OPTIMIZED_BATCH
            and self.max_seq_len == _OPTIMIZED_SEQ_LEN
            and getattr(self.args, "skip_kv_cache_fill", False)
        ):
            return q_heads_1QSD
        return super()._prepare_q_for_sdpa(q_heads_1QSD)

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        original_sdpa = ttnn.transformer.scaled_dot_product_attention
        ttnn.transformer.scaled_dot_product_attention = _wrap_sdpa_bidirectional(original_sdpa)

        # Head-split QKV/concat live in tt/custom_ops as generic_op kernels and are
        # injected the same way as the bidirectional SDPA above.
        original_create_heads = ttnn.experimental.nlp_create_qkv_heads
        original_concat_heads = ttnn.experimental.nlp_concat_heads
        _saved_norms = None
        if self._fused_norm_consts is not None:
            ttnn.experimental.nlp_create_qkv_heads = _wrap_create_qkv_heads_norm(
                original_create_heads, self._fused_norm_consts
            )
            _saved_norms = (self.q_norm, self.k_norm)
            self.q_norm = lambda x, mode, norm_config: x
            self.k_norm = lambda x, mode, norm_config: x
        elif os.getenv("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "0") == "1":
            ttnn.experimental.nlp_create_qkv_heads = _wrap_create_qkv_heads_headsplit(original_create_heads)
        if os.getenv("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "0") == "1":
            ttnn.experimental.nlp_concat_heads = _wrap_concat_heads_headsplit(original_concat_heads)
        try:
            return super().forward_prefill(
                x_11SH,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        finally:
            ttnn.transformer.scaled_dot_product_attention = original_sdpa
            ttnn.experimental.nlp_create_qkv_heads = original_create_heads
            ttnn.experimental.nlp_concat_heads = original_concat_heads
            if _saved_norms is not None:
                self.q_norm, self.k_norm = _saved_norms


PplxBidirectionalAttention.__name__ = "Attention"
