# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Weight loading + precompute for the Qwen3.5-9B Gated DeltaNet layer.

Loads the projection / conv / norm weights for one `linear_attn` substate and
precomputes the derived device tensors (conv taps, fused matmuls, prefill-kernel
constants, cached chunk masks). Behavior-preserving extraction of the original
`Qwen35GatedDeltaNet.__init__` weight code — every dtype / layout / memory_config
and every env-var read is preserved verbatim.
"""
import math
import os
from dataclasses import dataclass

import torch

# Install the experimental backend on sys.path BEFORE importing its ops.
import models.demos.blackhole.qwen3_5_9b.tt.gdn._experimental_path  # noqa: F401
import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.gdn.config import GDNConfig
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import create_chunk_masks
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import create_chunk_masks_seq


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
    # Fused projection weights
    ab_proj_weight: ttnn.Tensor
    mega_fused_weight: object
    mega_qkv_dim: object
    mega_a_dim: object
    mega_b_dim: object
    mega_g_dim: object
    # Cached chunk masks
    cached_masks: object
    cached_masks_long: object
    # Derived flags
    use_prefill_kernel: bool
    use_chunk_seq_prefill: bool
    use_decode_kernel: bool
    chunk_seq_masks_long: object
    # Prefill-kernel constants (None unless use_prefill_kernel)
    kernel_neg_exp_A: object = None
    kernel_dt_bias: object = None
    kernel_norm_w: object = None
    kernel_scale_tt: object = None
    kernel_rms_scale_tt: object = None
    kernel_rms_eps_tt: object = None


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
        """Load 2D weight, transpose to [in, out] for ttnn.linear."""
        t = state_dict[name].T.contiguous()
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"linear_attn.{name}") if tensor_cache_path else None,
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
    t = state_dict[qkv_key].T.contiguous()  # [4096, 8192]
    qkv_proj_weight = ttnn.as_tensor(
        t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=(tensor_cache_path / "linear_attn.qkv_proj.weight") if tensor_cache_path else None,
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

    def _precompute_fused_ab_weight():
        """Pre-concatenate a_proj + b_proj weights into [4096, 64] for fused matmul."""
        a_w = ttnn.to_torch(a_proj_weight)  # [4096, 32]
        b_w = ttnn.to_torch(b_proj_weight)  # [4096, 32]
        fused = torch.cat([a_w, b_w], dim=1).contiguous()  # [4096, 64]
        return ttnn.from_torch(fused, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def _precompute_mega_fused_weight():
        """Fuse QKV + a + b + g projections into one [4096, D_total] weight.

        Saves 2 matmul kernel launches per decode step (QKV=1, ab=1, g=1 -> mega=1).
        Output split: [qkv_dim | a_dim | b_dim | g_dim]
        """
        if qkv_proj_weight is None:
            return None
        qkv_w = ttnn.to_torch(qkv_proj_weight)  # [4096, 8192]
        a_w = ttnn.to_torch(a_proj_weight)  # [4096, 32]
        b_w = ttnn.to_torch(b_proj_weight)  # [4096, 32]
        g_w = ttnn.to_torch(g_proj_weight)  # [4096, 4096]
        fused = torch.cat([qkv_w, a_w, b_w, g_w], dim=1).contiguous()
        return ttnn.from_torch(fused, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)

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

    # Fused a+b projection weight: [4096, 64] — saves 1 matmul per decode step
    ab_proj_weight = _precompute_fused_ab_weight()

    # Mega-fused weight: QKV + a + b + g in one [4096, 12352] matmul
    # Eliminates 2 separate matmuls (g_proj, ab_proj) per decode step
    mega_fused_weight = _precompute_mega_fused_weight()
    if mega_fused_weight is not None:
        mega_qkv_dim = config.q_dim + config.k_dim + config.v_dim
        mega_a_dim = num_v_heads
        mega_b_dim = num_v_heads
        mega_g_dim = num_v_heads * head_v_dim
    else:
        mega_qkv_dim = None
        mega_a_dim = None
        mega_b_dim = None
        mega_g_dim = None

    # Pre-cache chunk masks for prefill (shared across all calls)
    cached_masks = create_chunk_masks(config.prefill_chunk_size, mesh_device)

    # The Neumann approximation overflows at chunk_size=256, but we now use
    # exact forward substitution (LAPACK triangular solve) which is numerically
    # stable at any chunk size. chunk_size=128 balances device compute (128x128
    # matmuls) vs sub-chunk count (8 per 1024-token outer chunk) vs LAPACK
    # solve cost ([256, 128, 128] = 16MB transfer).
    # NOTE: chunk_size=64 is ~39% faster per-layer but compounds PCC errors
    # across 32 layers (PCC drops from 0.999 single-layer to 0.87 full-model).
    cached_masks_long = create_chunk_masks(config.long_prefill_chunk_size, mesh_device)

    # Prefill kernel constants (precomputed once, reused across all prefill calls)
    use_prefill_kernel = not os.environ.get("GDN_DISABLE_PREFILL_KERNEL", "")
    kernel_consts = {}
    if use_prefill_kernel:
        Nv = num_v_heads
        Dk = head_k_dim
        Dv = head_v_dim

        # neg_exp_A: [1, 1, Nv] — already have A_neg but need correct shape
        neg_exp_A_torch = ttnn.to_torch(A_neg).float().reshape(1, 1, Nv)
        kernel_consts["kernel_neg_exp_A"] = ttnn.from_torch(
            neg_exp_A_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # dt_bias: [1, 1, Nv]
        dt_bias_torch = ttnn.to_torch(dt_bias).float().reshape(1, 1, Nv)
        kernel_consts["kernel_dt_bias"] = ttnn.from_torch(
            dt_bias_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # norm_w: [1, 1, Dv] — RMS norm weight (reader loads Vt tiles, not used by compute)
        norm_w_torch = ttnn.to_torch(o_norm_weight).float().reshape(1, 1, Dv)
        kernel_consts["kernel_norm_w"] = ttnn.from_torch(
            norm_w_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # scale_tt: Dk^(-0.5) as [1, 1, 1] scalar
        kernel_consts["kernel_scale_tt"] = ttnn.from_torch(
            torch.tensor([[[Dk**-0.5]]], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # rms_scale_tt: sqrt(Dv) as [1, 1, 1] — loaded by reader, not used by compute
        kernel_consts["kernel_rms_scale_tt"] = ttnn.from_torch(
            torch.tensor([[[math.sqrt(Dv)]]], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # rms_eps_tt: Dv * eps as [1, 1, 1] — loaded by reader, not used by compute
        kernel_consts["kernel_rms_eps_tt"] = ttnn.from_torch(
            torch.tensor([[[Dv * norm_eps]]], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Chunk-parallel prefill via the C++ gated_delta_attn_seq kernel (float32),
    # opt-in via QWEN9B_GDN_CHUNK_SEQ=1. Replaces the per-token prefill kernel.
    # The kernel hardcodes Ct=4 diagonal blocks, so it ONLY supports
    # chunk_size=128 (= long_prefill_chunk_size). Precompute its float32 masks
    # (incl. eye_32) once; other chunk sizes fall back to the per-token kernel.
    use_chunk_seq_prefill = bool(os.environ.get("QWEN9B_GDN_CHUNK_SEQ", ""))
    chunk_seq_masks_long = None
    if use_chunk_seq_prefill:
        chunk_seq_masks_long = create_chunk_masks_seq(config.long_prefill_chunk_size, mesh_device)

    # Decode recurrence kernel: opt-in fused on-device kernel for T=1 decode.
    use_decode_kernel = bool(os.environ.get("QWEN9B_GDN_DECODE_KERNEL", ""))

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
        ab_proj_weight=ab_proj_weight,
        mega_fused_weight=mega_fused_weight,
        mega_qkv_dim=mega_qkv_dim,
        mega_a_dim=mega_a_dim,
        mega_b_dim=mega_b_dim,
        mega_g_dim=mega_g_dim,
        cached_masks=cached_masks,
        cached_masks_long=cached_masks_long,
        use_prefill_kernel=use_prefill_kernel,
        use_chunk_seq_prefill=use_chunk_seq_prefill,
        use_decode_kernel=use_decode_kernel,
        chunk_seq_masks_long=chunk_seq_masks_long,
        **kernel_consts,
    )
