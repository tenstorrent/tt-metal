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


def _wrap_create_qkv_heads_norm(original_fn, consts, rot=None, q_dtype=None, kv_dtype=None, norm_eps=None):
    """Route ``nlp_create_qkv_heads`` to the head-split + Q/K RMSNorm fused op.

    ``consts`` = (gamma_q_tiles, gamma_k_tiles, scaler, eps) built once per layer. The
    caller makes ``q_norm``/``k_norm`` identities for the same forward so the norm is
    applied exactly once. ``rot`` = (cos, sin, T) folds RoPE in as well; ``q_dtype`` /
    ``kv_dtype`` make the op emit Q / K,V in SDPA's operand dtypes (no Typecast op).
    Falls back to the stock op where the fast path cannot express the call (sharded
    input, transposed K heads, indivisible head counts).
    """
    gq, gk, sc, ep = consts
    rot_kwargs = {} if rot is None else {"rot_cos": rot[0], "rot_sin": rot[1], "trans_mat": rot[2]}
    if q_dtype is not None:
        rot_kwargs["q_dtype"] = q_dtype
    if kv_dtype is not None:
        rot_kwargs["kv_dtype"] = kv_dtype
    # QWEN_FUSED_RESIDENT_CONSTS=1 (bs1 default): cos/sin, the rotation tile, scaler and eps come from a per-core L1
    # shard shared by all layers (CBs alias it), and gamma is read after the first unit; the op falls back when a
    # core's units span more than one seq tile.
    if rot is not None and norm_eps is not None and os.getenv("QWEN_FUSED_RESIDENT_CONSTS", "0") == "1":
        rot_kwargs.update(resident=True, norm_eps=norm_eps)

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
                **rot_kwargs,
            )
        return original_fn(qkv_fused, *args, **kwargs)

    return wrapper


def _wrap_matmul_out_bfp8(original_fn, weight):
    """Make the QKV projection (``b is weight``) emit bfp8 instead of the pinned bf16.

    Upstream pins the QKV output to bf16 only because the stock rotary op asserts bf16;
    the fused head-split op reads bfp8 directly, so the projection can write half the
    bytes and the fused op read half the bytes (QWEN_QKV_OUT_BFP8=1).
    """

    @functools.wraps(original_fn)
    def wrapper(a, b, *args, **kwargs):
        if b is weight:
            kwargs["dtype"] = ttnn.bfloat8_b
        return original_fn(a, b, *args, **kwargs)

    return wrapper


# Tensors the guarded ``ttnn.deallocate`` must skip once (id -> remaining skips): the fused RoPE pass hands
# back Q/K that the base forward believes it may free, and the concat-free SDPA output is handed straight
# to the output projection while the base forward still frees "the SDPA output" after concat_heads.
_SKIP_DEALLOC_ONCE: dict[int, int] = {}
# SDPA outputs already written in the [B, 1, S, H*d] layout (QWEN_SDPA_CONCAT_OUT=1): concat_heads is a no-op.
_SDPA_CONCAT_OUT_IDS: set[int] = set()


def _wrap_concat_heads_headsplit(original_fn):
    """Route ``nlp_concat_heads`` to the model-local head-split kernels."""

    @functools.wraps(original_fn)
    def wrapper(context, *args, **kwargs):
        if id(context) in _SDPA_CONCAT_OUT_IDS:
            # SDPA already produced [B, 1, S, H*d]; the base forward deallocates its "SDPA output" right after
            # this call, so protect the tensor once and hand it back as the concatenated result.
            _SDPA_CONCAT_OUT_IDS.discard(id(context))
            _SKIP_DEALLOC_ONCE[id(context)] = _SKIP_DEALLOC_ONCE.get(id(context), 0) + 1
            return context
        if not args and _interleaved_out(kwargs.get("memory_config")) and _concat_headsplit_supported(context):
            return nlp_concat_heads_headsplit(context, memory_config=kwargs.get("memory_config"))
        return original_fn(context, *args, **kwargs)

    return wrapper


def _wrap_reshape_concat_out(original_fn):
    """bs1: the base forward reshapes the [1, H, S, d] SDPA output to [1, H, -1, d] before concat_heads; a
    concat-free SDPA output is already [1, 1, S, H*d], so that reshape must hand the tensor back untouched
    (the concat wrapper then does the same). The batched [1, 1, B*S, -1] reshape (shape[1] == 1) still runs."""

    @functools.wraps(original_fn)
    def wrapper(tensor, shape=None, *args, **kwargs):
        if shape is not None and id(tensor) in _SDPA_CONCAT_OUT_IDS and len(shape) == 4 and int(shape[1]) > 1:
            return tensor
        return original_fn(tensor, shape, *args, **kwargs)

    return wrapper


def _wrap_sdpa_bidirectional(original_fn, concat_out=False):
    """Return a wrapper that forces ``is_causal=False`` and injects the padding mask.

    ``concat_out`` (QWEN_SDPA_CONCAT_OUT=1, batched prefill only): ask SDPA for its output in the
    ``[B, 1, S, H*d]`` layout (``output_heads_concat=True``), which is what ``nlp_concat_heads`` would
    produce, so that pass (142 MB per layer at bs32) is skipped. The concat wrapper recognises the tensor.
    """

    # QWEN_SDPA_CAUSAL=1 keeps the base forward's causal attention (Qwen3-Embedding-4B: same backbone as
    # pplx-embed, causal attention + last-token pooling); default is pplx-embed's bidirectional attention.
    causal = os.getenv("QWEN_SDPA_CAUSAL", "0") == "1"
    # bs1 as well (QWEN_SDPA_CONCAT_OUT_BS1=1): the q256 8x8 SDPA drains faster into [1, 1, S, H*d]
    # (standalone 57.3 -> 54.1 us) and the 4.6 us model-local concat op disappears.
    concat_out_bs1 = os.getenv("QWEN_SDPA_CONCAT_OUT_BS1", "1") == "1"
    # bs1 (QWEN_SDPA_GQA_PACK=1): SDPA's pack_gqa_heads schedules the 4 Q heads sharing a KV head as one head
    # of 4*S rows, so each KV head's K/V streams once down one 8-core chain instead of once per Q head.
    # Unmasked non-causal calls only (the serving pad mask takes the unpacked call).
    gqa_pack = os.getenv("QWEN_SDPA_GQA_PACK", "0") == "1"
    # Packed calls take their own q chunk and grid (QWEN_SDPA_GQA_PACK_Q_CHUNK, QWEN_SDPA_GQA_PACK_GRID=x,y): at bs1,
    # q192 on 11x8 gives each KV head 11 chunks of its 64 packed row tiles, one per core of one grid row (88 cores,
    # 6 row tiles each vs 8 on 8x8). Unpacked calls keep the model's config.
    pack_q_chunk = int(os.getenv("QWEN_SDPA_GQA_PACK_Q_CHUNK", "0"))
    pack_grid = tuple(int(x) for x in os.getenv("QWEN_SDPA_GQA_PACK_GRID", "0,0").split(","))
    pack_cfgs = {}

    def packed_program_config(pc):
        if pc is None or not pack_q_chunk:
            return pc
        key = (pc.k_chunk_size, pc.exp_approx_mode)
        if key not in pack_cfgs:
            grid = pc.compute_with_storage_grid_size
            pack_cfgs[key] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*pack_grid) if pack_grid[0] else grid,
                q_chunk_size=pack_q_chunk,
                k_chunk_size=pc.k_chunk_size,
                exp_approx_mode=pc.exp_approx_mode,
            )
        return pack_cfgs[key]

    # Batched (QWEN_SDPA_REUSE_KV=1): SDPA's reuse_kv keeps K/V in a core's CBs across its consecutive Q chunks of the
    # same (batch, KV head) instead of re-reading them per chunk (non-causal, unmasked, one K chunk). With K/V no
    # longer re-read per chunk, finer Q chunks balance the grid for free: those calls take QWEN_SDPA_REUSE_Q_CHUNK
    # (the grid stays). Masked / causal calls keep the model's config (a small q chunk without reuse is slower).
    reuse_kv = os.getenv("QWEN_SDPA_REUSE_KV", "0") == "1"
    reuse_q_chunk = int(os.getenv("QWEN_SDPA_REUSE_Q_CHUNK", "0"))
    reuse_cfgs = {}

    def reuse_program_config(pc):
        if pc is None or not reuse_q_chunk:
            return pc
        grid = pc.compute_with_storage_grid_size
        key = (int(grid.x), int(grid.y), pc.k_chunk_size, pc.exp_approx_mode)
        if key not in reuse_cfgs:
            reuse_cfgs[key] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=reuse_q_chunk,
                k_chunk_size=pc.k_chunk_size,
                exp_approx_mode=pc.exp_approx_mode,
            )
        return reuse_cfgs[key]

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        if not causal:
            kwargs["is_causal"] = False
            if _PAD_ATTN_MASK is not None and kwargs.get("attn_mask") is None:
                kwargs["attn_mask"] = _PAD_ATTN_MASK
        q = args[0] if args else kwargs.get("input_tensor_q")
        k = args[1] if len(args) > 1 else kwargs.get("input_tensor_k")
        if (
            gqa_pack
            and not causal
            and kwargs.get("attn_mask") is None
            and q is not None
            and int(q.shape[0]) == 1
            and int(q.shape[1]) > int(k.shape[1])
            and int(q.shape[1]) % int(k.shape[1]) == 0
            # the op needs a tile-aligned Q sequence to view the group's heads as one
            and int(q.shape[2]) % 32 == 0
        ):
            kwargs["pack_gqa_heads"] = True
            kwargs["program_config"] = packed_program_config(kwargs.get("program_config"))
        pc = kwargs.get("program_config")
        if (
            reuse_kv
            and not causal
            and kwargs.get("attn_mask") is None
            and q is not None
            and int(q.shape[0]) > 1
            and pc is not None
            and pc.k_chunk_size >= int(k.shape[2])  # one K chunk
        ):
            kwargs["reuse_kv"] = True
            kwargs["program_config"] = reuse_program_config(pc)
        if concat_out and q is not None and (int(q.shape[0]) > 1 or concat_out_bs1):
            kwargs["output_heads_concat"] = True
            out = original_fn(*args, **kwargs)
            _SDPA_CONCAT_OUT_IDS.add(id(out))
            return out
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
        self._fused_norm_eps = None
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
                self._fused_norm_eps = configuration.norm_eps
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

    def _q_bf16_shortcut(self) -> bool:
        """bs1/ISL512 embedding path: K/V stay bf16 and Q is handed to SDPA uncast."""
        return (
            self.max_batch_size == _OPTIMIZED_BATCH
            and self.max_seq_len == _OPTIMIZED_SEQ_LEN
            and getattr(self.args, "skip_kv_cache_fill", False)
        )

    def _prepare_q_for_sdpa(self, q_heads_1QSD: ttnn.Tensor) -> ttnn.Tensor:
        if q_heads_1QSD.dtype == (self.activation_dtype or ttnn.bfloat8_b):
            # Already in SDPA's Q dtype: the fused head-split op emitted it that way
            # (QWEN_FUSED_Q_BFP8), so the upstream Typecast is not needed.
            return q_heads_1QSD
        if self._q_bf16_shortcut():
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
        concat_out = os.getenv("QWEN_SDPA_CONCAT_OUT", "0") == "1"
        ttnn.transformer.scaled_dot_product_attention = _wrap_sdpa_bidirectional(original_sdpa, concat_out=concat_out)

        # Head-split QKV/concat live in tt/custom_ops as generic_op kernels and are
        # injected the same way as the bidirectional SDPA above.
        original_create_heads = ttnn.experimental.nlp_create_qkv_heads
        original_concat_heads = ttnn.experimental.nlp_concat_heads
        _saved_norms = None
        _saved_rope = None
        _saved_dealloc = None
        _saved_mm = None
        _saved_reshape = None
        if concat_out:
            _saved_reshape = ttnn.reshape
            ttnn.reshape = _wrap_reshape_concat_out(_saved_reshape)
        if self._fused_norm_consts is not None:
            # QWEN_FUSED_ROTARY=1 also folds RoPE into the same pass (rot_mats = [cos, sin] for
            # this chunk; the single 32x32 tile-local rotation is transformation_mats["prefill"]).
            rot = None
            if os.getenv("QWEN_FUSED_ROTARY", "0") == "1" and rot_mats is not None and self.transformation_mats:
                rot = (rot_mats[0], rot_mats[1], self.transformation_mats["prefill"])
            # Output dtypes: with RoPE fused, Q (and optionally K/V) can leave the op already
            # in SDPA's operand dtype, deleting the per-layer Typecast. QWEN_FUSED_Q_BFP8=1
            # matches what _prepare_q_for_sdpa would cast to (bf16 on the bs1 shortcut),
            # "force" emits bfp8 Q there too. QWEN_FUSED_KV_BFP8=1 emits K/V in bfp8 (only
            # meaningful when the KV-cache fill is skipped and SDPA reads K/V directly).
            q_dtype = kv_dtype = None
            if rot is not None:
                q_mode = os.getenv("QWEN_FUSED_Q_BFP8", "0")
                if q_mode == "force" or (q_mode == "1" and not self._q_bf16_shortcut()):
                    q_dtype = self.activation_dtype or ttnn.bfloat8_b
                if os.getenv("QWEN_FUSED_KV_BFP8", "0") == "1" and getattr(self.args, "skip_kv_cache_fill", False):
                    kv_dtype = ttnn.bfloat8_b
                if os.getenv("QWEN_QKV_OUT_BFP8", "0") == "1" and getattr(self.args, "skip_kv_cache_fill", False):
                    # QKV projection output in bfp8: the fused op is the only reader. Q/K/V
                    # are then already bfp8-quantised, so emit them in bfp8 as well.
                    q_dtype = kv_dtype = ttnn.bfloat8_b
                    _saved_mm = (ttnn.experimental.minimal_matmul, ttnn.linear)
                    ttnn.experimental.minimal_matmul = _wrap_matmul_out_bfp8(_saved_mm[0], self.wqkv)
                    ttnn.linear = _wrap_matmul_out_bfp8(_saved_mm[1], self.wqkv)
            ttnn.experimental.nlp_create_qkv_heads = _wrap_create_qkv_heads_norm(
                original_create_heads, self._fused_norm_consts, rot, q_dtype, kv_dtype, self._fused_norm_eps
            )
            _saved_norms = (self.q_norm, self.k_norm)
            self.q_norm = lambda x, mode, norm_config: x
            self.k_norm = lambda x, mode, norm_config: x
            if rot is not None:
                # RoPE already applied inside the fused op: hand Q/K straight through. The
                # upstream forward then deallocates the "pre-rotary" tensors, which are now
                # the tensors SDPA reads, so skip exactly one deallocation of each (later
                # frees still happen, nothing leaks).
                _saved_rope = self.rotary_embedding_prefill

                def _identity_rope(q, k, rm):
                    _SKIP_DEALLOC_ONCE[id(q)] = _SKIP_DEALLOC_ONCE.get(id(q), 0) + 1
                    _SKIP_DEALLOC_ONCE[id(k)] = _SKIP_DEALLOC_ONCE.get(id(k), 0) + 1
                    return q, k

                self.rotary_embedding_prefill = _identity_rope
        elif os.getenv("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "0") == "1":
            ttnn.experimental.nlp_create_qkv_heads = _wrap_create_qkv_heads_headsplit(original_create_heads)
        if os.getenv("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "0") == "1" or concat_out:
            ttnn.experimental.nlp_concat_heads = _wrap_concat_heads_headsplit(original_concat_heads)
        if self._fused_norm_consts is not None or concat_out:
            _saved_dealloc = ttnn.deallocate

            def _guarded_dealloc(t, *a, **kw):
                if _SKIP_DEALLOC_ONCE.get(id(t), 0) > 0:
                    _SKIP_DEALLOC_ONCE[id(t)] -= 1
                    if _SKIP_DEALLOC_ONCE[id(t)] == 0:
                        del _SKIP_DEALLOC_ONCE[id(t)]
                    return None
                return _saved_dealloc(t, *a, **kw)

            ttnn.deallocate = _guarded_dealloc
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
            if _saved_rope is not None:
                self.rotary_embedding_prefill = _saved_rope
            if _saved_dealloc is not None:
                ttnn.deallocate = _saved_dealloc
            if _saved_mm is not None:
                ttnn.experimental.minimal_matmul, ttnn.linear = _saved_mm
            if _saved_reshape is not None:
                ttnn.reshape = _saved_reshape


PplxBidirectionalAttention.__name__ = "Attention"
