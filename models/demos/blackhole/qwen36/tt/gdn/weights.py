# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading + precompute for the Qwen3.5-9B Gated DeltaNet layer.

Loads the projection / conv / norm weights for one `linear_attn` substate and
precomputes the derived device tensors (conv taps, fused matmuls, prefill-kernel
constants, cached chunk masks). Behavior-preserving extraction of the original
`Qwen36GatedDeltaNet.__init__` weight code — every dtype / layout / memory_config
and every env-var read is preserved verbatim.
"""
import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.config import GDNConfig
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import create_chunk_masks_seq

# Channel-chunk widths for the native ttnn.conv1d chunk-prefill path (conv1d_native.py), measured
# to fit L1 on P150 with conv1d_bench.py (T=2048, K=4, HEIGHT_SHARDED + Conv2dL1FullSliceConfig):
# FIR fallback is 3.13 ms; native 2x3072 (chunk_width=3072) is 1.63 ms. chunk_width=2048 (3x2048 on
# C=6144, or 4x2048 on C=8192) overflows L1 ("grow to 1684544 B beyond max L1 1572864 B"), and a
# single 1x6144 call clashes with L1 buffers instead. chunk_width=4096 (2x4096 on C=8192) also
# clashes with L1 buffers, so it is deliberately left off this list. Ordered widest-first so the
# smallest n_cc (fewest conv1d calls) wins when more than one width divides C.
_NATIVE_CONV_CHUNK_WIDTHS = (3072, 2560)


@dataclass
class GDNWeights:
    # Projection weights
    qkv_proj_weight: ttnn.Tensor
    q_proj_weight: object  # None — dead split (op runs fused QKV); kept as kwarg
    k_proj_weight: object
    v_proj_weight: object
    a_proj_weight: ttnn.Tensor
    b_proj_weight: ttnn.Tensor
    g_proj_weight: ttnn.Tensor
    o_proj_weight: ttnn.Tensor
    # Conv weights (stay on HOST, ROW_MAJOR) + biases
    q_conv_weight: ttnn.Tensor
    k_conv_weight: ttnn.Tensor
    v_conv_weight: ttnn.Tensor
    q_conv_bias: object
    k_conv_bias: object
    v_conv_bias: object
    # 1D params
    A_log: ttnn.Tensor
    dt_bias: ttnn.Tensor
    o_norm_weight: ttnn.Tensor
    A_neg: ttnn.Tensor
    # Precomputed conv taps / bias on device
    q_weight_taps: list
    k_weight_taps: list
    v_weight_taps: list
    q_bias_dev: object
    k_bias_dev: object
    v_bias_dev: object
    fused_conv_weight_taps: list
    fused_conv_bias_dev: object
    # Depthwise conv1d weight [C, 1, K], host ROW_MAJOR, split into equal channel chunks for the
    # native ttnn.conv1d chunk-prefill path (see gdn/conv1d_native.py). Same source + channel order
    # as fused_conv_weight_taps. dict {n_cc: [chunk tensors]} — conv1d_native.py picks n_cc per T.
    fused_conv_w1d_chunks: object
    # Fused projection weights
    ab_proj_weight: ttnn.Tensor
    mega_fused_weight: object
    mega_qkv_dim: object
    mega_a_dim: object
    mega_b_dim: object
    mega_g_dim: object
    # Chunk-parallel prefill kernel (gated_delta_attn_seq) — always on
    use_chunk_seq_prefill: bool
    chunk_seq_masks_long: object


def load_gdn_weights(mesh_device, config: GDNConfig, state_dict, tensor_cache_path=None) -> GDNWeights:
    """Load + precompute all device weights for one Gated DeltaNet layer.

    `state_dict` is the per-layer `linear_attn` substate (keys already stripped of
    the `layers.{n}.linear_attn.` prefix). `tensor_cache_path` points at the
    `layers.{n}` directory (or None) — cache file names re-add the `linear_attn.`
    prefix to match the original cache keys exactly.
    """
    num_heads = config.num_heads
    num_v_heads = config.num_v_heads
    head_k_dim = config.head_k_dim
    head_v_dim = config.head_v_dim
    conv_kernel_size = config.conv_kernel_size
    norm_eps = config.norm_eps

    def load_weight_2d(name):
        """Load 2D weight, transposed to [in, out] for ttnn.linear (on a tensor-cache miss only)."""
        return ttnn.as_tensor(
            state_dict[name],
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"linear_attn.{name}") if tensor_cache_path else None,
            preprocess=lambda t: t.T.contiguous(),
        )

    def load_conv_weight(name):
        """Load conv1d weight — stays on HOST (not device), ROW_MAJOR layout."""
        t = state_dict[name]
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    def load_1d(name):
        """Load 1D param — must use TILE_LAYOUT on device like all other tensors."""
        t = state_dict[name]
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"linear_attn.{name}") if tensor_cache_path else None,
        )

    # Fused QKV projection: one matmul instead of three
    qkv_key = "qkv_proj.weight"
    if qkv_key not in state_dict:
        raise ValueError(
            f"DeltaNet layer requires the combined qkv_proj weight "
            f"(key '{qkv_key}' missing; the split q/k/v_proj were removed in the weight refactor)."
        )
    qkv_proj_weight = ttnn.as_tensor(
        state_dict[qkv_key],
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=(tensor_cache_path / "linear_attn.qkv_proj.weight") if tensor_cache_path else None,
        preprocess=lambda t: t.T.contiguous(),  # [4096, 8192]; cache-miss only
    )
    # The split q/k/v_proj are dead (the op runs the fused QKV from the combined weight; it
    # reads the splits only in a fallback reached when qkv_proj_weight is None). Not created,
    # saving ~33MB/layer of device memory. The op still receives them as kwargs (None).
    q_proj_weight = None
    k_proj_weight = None
    v_proj_weight = None
    a_proj_weight = load_weight_2d("in_proj_a.weight")
    b_proj_weight = load_weight_2d("in_proj_b.weight")
    g_proj_weight = load_weight_2d("in_proj_z.weight")
    o_proj_weight = load_weight_2d("out_proj.weight")

    q_conv_weight = load_conv_weight("q_conv.weight")
    k_conv_weight = load_conv_weight("k_conv.weight")
    v_conv_weight = load_conv_weight("v_conv.weight")

    def load_conv_bias_or_none(name):
        if name in state_dict:
            t = state_dict[name]
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        return None

    q_conv_bias = load_conv_bias_or_none("q_conv.bias")
    k_conv_bias = load_conv_bias_or_none("k_conv.bias")
    v_conv_bias = load_conv_bias_or_none("v_conv.bias")

    A_log = load_1d("A_log")
    dt_bias = load_1d("dt_bias")
    # DeltaNet output norm uses STANDARD RMSNorm (raw weights ~0.88),
    # NOT zero-centered like the decoder/attention norms (raw weights ~0.03).
    o_norm_weight = load_1d("norm.weight")

    # Precompute A_neg = -exp(A_log) once (constant per layer, saves 2 ops per decode step)
    A_neg = ttnn.neg(ttnn.exp(A_log))

    # ---- precompute helpers (bodies verbatim; self.device -> mesh_device, self.<dim> -> config.<dim>) ----

    def _precompute_weight_taps(conv_weight):
        """Pre-slice conv weight [D, 1, K] into K device tensors [1, 1, D] for FIR decode."""

        weight_torch = ttnn.to_torch(conv_weight)
        D = weight_torch.shape[0]
        taps = []
        for k in range(conv_kernel_size):
            w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
            taps.append(ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device))
        return taps

    def _precompute_bias_dev(conv_bias):
        """Pre-convert conv bias to [1, 1, D] device tensor."""
        if conv_bias is None:
            return None

        bias_torch = ttnn.to_torch(conv_bias)
        D = bias_torch.numel()
        bias_reshaped = bias_torch.reshape(1, 1, D).contiguous()
        return ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def _precompute_fused_weight_taps():
        """Pre-concatenate Q+K+V conv weight taps into fused taps [1, 1, D_total]."""
        q_w = ttnn.to_torch(q_conv_weight)  # [D_q, 1, K]
        k_w = ttnn.to_torch(k_conv_weight)  # [D_k, 1, K]
        v_w = ttnn.to_torch(v_conv_weight)  # [D_v, 1, K]
        # Concatenate along D dimension: [D_total, 1, K]
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        D_total = fused_w.shape[0]
        taps = []
        for k_idx in range(conv_kernel_size):
            w_k = fused_w[:, 0, k_idx].reshape(1, 1, D_total).contiguous()
            taps.append(ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device))
        return taps

    def _precompute_fused_bias_dev():
        """Pre-concatenate Q+K+V conv biases into fused bias [1, 1, D_total]."""
        parts = []
        for bias in [q_conv_bias, k_conv_bias, v_conv_bias]:
            if bias is not None:
                parts.append(ttnn.to_torch(bias))
            else:
                return None  # If any is None, skip fused bias
        fused = torch.cat(parts, dim=0)
        D_total = fused.numel()
        fused_reshaped = fused.reshape(1, 1, D_total).contiguous()
        return ttnn.from_torch(fused_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def _precompute_fused_conv_w1d_chunks():
        """Depthwise conv1d weight(s) for the native ttnn.conv1d chunk-prefill path (conv1d_native.py).

        Same source + same channel order as _precompute_fused_weight_taps's fused_w: already
        [C, 1, K], i.e. torch.stack(taps, dim=-1).reshape(C, 1, K) as in load_gdn_weights_tp.

        Returns a dict {n_cc: [chunk tensors]}, one entry per channel-chunk-count this layer might
        use at runtime (conv1d_native.py picks between them per-call, by T — see its T3_MAX gate):
          - the default entry: smallest n_cc such that C is divisible by n_cc and C // n_cc is an
            allow-listed width in `_NATIVE_CONV_CHUNK_WIDTHS` (see its comment for the measured
            numbers, e.g. n_cc=2 / width 3072 on C=6144);
          - an n_cc=3 entry whenever C % 3 == 0 (width C // 3, e.g. 2048 on C=6144), IN ADDITION to
            the default — conv1d_native.py only ever selects it when that width also equals
            q_dim == k_dim == v_dim (so the 3 conv outputs are usable as Q/K/V directly, no
            concat) and it fits L1 for the current T. Host tensors, kept on CPU — no tensor cache
            needed (tiny: C*K bf16 values total, cheap to precompute even if unused at runtime).

        env QWEN36_GDN_CONV_CHUNKS=<n> overrides: returns only {n: [chunks]} for that n_cc (still
        must evenly divide C) — conv1d_native.py then always uses it regardless of T.
        Returns None if no allow-listed width fits and n_cc=3 doesn't divide C either — caller
        then keeps the FIR fallback path.
        """
        q_w = ttnn.to_torch(q_conv_weight)  # [D_q, 1, K]
        k_w = ttnn.to_torch(k_conv_weight)  # [D_k, 1, K]
        v_w = ttnn.to_torch(v_conv_weight)  # [D_v, 1, K]
        W1d = torch.cat([q_w, k_w, v_w], dim=0).contiguous()  # [C, 1, K]
        C = W1d.shape[0]

        def _split(n_cc):
            cw = C // n_cc
            return [
                ttnn.from_torch(
                    W1d[i * cw : (i + 1) * cw].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                for i in range(n_cc)
            ]

        override = os.environ.get("QWEN36_GDN_CONV_CHUNKS")
        if override is not None:
            n_cc = int(override)
            assert n_cc > 0 and C % n_cc == 0, f"QWEN36_GDN_CONV_CHUNKS={override!r} must evenly divide C={C}"
            return {n_cc: _split(n_cc)}

        chunks_by_ncc = {}
        candidates = [C // cw for cw in _NATIVE_CONV_CHUNK_WIDTHS if C % cw == 0]
        if candidates:
            default_n_cc = min(candidates)
            chunks_by_ncc[default_n_cc] = _split(default_n_cc)
        if C % 3 == 0 and 3 not in chunks_by_ncc:
            chunks_by_ncc[3] = _split(3)
        return chunks_by_ncc or None

    def _precompute_fused_ab_weight():
        """Pre-concatenate a_proj + b_proj weights into [4096, 64] for fused matmul."""
        a_w = ttnn.to_torch(a_proj_weight)  # [4096, 32]
        b_w = ttnn.to_torch(b_proj_weight)  # [4096, 32]
        fused = torch.cat([a_w, b_w], dim=1).contiguous()  # [4096, 64]
        return ttnn.from_torch(fused, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def _precompute_mega_fused_weight():
        """Fuse QKV + g + a + b projections into one [4096, D_total] weight.

        Saves 2 matmul kernel launches per decode step (QKV=1, ab=1, g=1 -> mega=1).
        Output split: [qkv_dim | g_dim | a_dim | b_dim] — qkv_dim and g_dim are both tile-width
        multiples, so ttnn_gated_deltanet.py can device-slice qkv and gate straight off mega_out
        (tile-aligned begin) and only pays a small untilize on the combined 1-tile-wide a|b slice,
        instead of untiling the whole [*, D_total] tensor (a/b used to sit mid-tensor at a
        non-tile-aligned boundary between qkv and g). Cached under a new name (below) so an
        old qkv|a|b|g-ordered cache entry is never loaded for this new qkv|g|a|b layout.
        """
        if qkv_proj_weight is None:
            return None
        qkv_w = ttnn.to_torch(qkv_proj_weight)  # [4096, 8192]
        a_w = ttnn.to_torch(a_proj_weight)  # [4096, 32]
        b_w = ttnn.to_torch(b_proj_weight)  # [4096, 32]
        g_w = ttnn.to_torch(g_proj_weight)  # [4096, 4096]
        fused = torch.cat([qkv_w, g_w, a_w, b_w], dim=1).contiguous()
        return ttnn.as_tensor(
            fused,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # New cache name (qgab, not the old qkv|a|b|g order) so a stale cache file from the
            # previous concat order is never loaded.
            cache_file_name=(tensor_cache_path / "linear_attn.mega_fused_weight_qgab") if tensor_cache_path else None,
        )

    # Precompute conv weight taps and bias on device to avoid CPU round-trips during decode
    q_weight_taps = _precompute_weight_taps(q_conv_weight)
    k_weight_taps = _precompute_weight_taps(k_conv_weight)
    v_weight_taps = _precompute_weight_taps(v_conv_weight)
    q_bias_dev = _precompute_bias_dev(q_conv_bias)
    k_bias_dev = _precompute_bias_dev(k_conv_bias)
    v_bias_dev = _precompute_bias_dev(v_conv_bias)

    # Precompute fused QKV conv weight taps [1, 1, D_total] for fused conv decode
    fused_conv_weight_taps = _precompute_fused_weight_taps()
    fused_conv_bias_dev = _precompute_fused_bias_dev()
    fused_conv_w1d_chunks = _precompute_fused_conv_w1d_chunks()

    # Fused a+b projection weight: [4096, 64] — saves 1 matmul per decode step
    ab_proj_weight = _precompute_fused_ab_weight()

    # Mega-fused weight: QKV + g + a + b in one [4096, 12352] matmul (see
    # _precompute_mega_fused_weight for why g moved before a/b).
    # Eliminates 2 separate matmuls (g_proj, ab_proj) per decode step
    mega_fused_weight = _precompute_mega_fused_weight()
    if mega_fused_weight is not None:
        # Dims below are widths, not offsets — order in the concatenated tensor is qkv|g|a|b
        # (see _precompute_mega_fused_weight); ttnn_gated_deltanet.py computes offsets from these.
        mega_qkv_dim = config.q_dim + config.k_dim + config.v_dim
        mega_a_dim = num_v_heads
        mega_b_dim = num_v_heads
        mega_g_dim = num_v_heads * head_v_dim
    else:
        mega_qkv_dim = None
        mega_a_dim = None
        mega_b_dim = None
        mega_g_dim = None

    # Chunk-parallel prefill via the C++ gated_delta_attn_seq kernel (float32) is
    # the default prefill path. The kernel hardcodes Ct=4 diagonal blocks, so it
    # ONLY supports chunk_size=128 (= long_prefill_chunk_size). Precompute its
    # float32 masks (incl. eye_32) once.
    use_chunk_seq_prefill = True
    chunk_seq_masks_long = create_chunk_masks_seq(config.long_prefill_chunk_size, mesh_device)

    return GDNWeights(
        qkv_proj_weight=qkv_proj_weight,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        a_proj_weight=a_proj_weight,
        b_proj_weight=b_proj_weight,
        g_proj_weight=g_proj_weight,
        o_proj_weight=o_proj_weight,
        q_conv_weight=q_conv_weight,
        k_conv_weight=k_conv_weight,
        v_conv_weight=v_conv_weight,
        q_conv_bias=q_conv_bias,
        k_conv_bias=k_conv_bias,
        v_conv_bias=v_conv_bias,
        A_log=A_log,
        dt_bias=dt_bias,
        o_norm_weight=o_norm_weight,
        A_neg=A_neg,
        q_weight_taps=q_weight_taps,
        k_weight_taps=k_weight_taps,
        v_weight_taps=v_weight_taps,
        q_bias_dev=q_bias_dev,
        k_bias_dev=k_bias_dev,
        v_bias_dev=v_bias_dev,
        fused_conv_weight_taps=fused_conv_weight_taps,
        fused_conv_bias_dev=fused_conv_bias_dev,
        fused_conv_w1d_chunks=fused_conv_w1d_chunks,
        ab_proj_weight=ab_proj_weight,
        mega_fused_weight=mega_fused_weight,
        mega_qkv_dim=mega_qkv_dim,
        mega_a_dim=mega_a_dim,
        mega_b_dim=mega_b_dim,
        mega_g_dim=mega_g_dim,
        use_chunk_seq_prefill=use_chunk_seq_prefill,
        chunk_seq_masks_long=chunk_seq_masks_long,
    )
