# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro TTNN convolution helpers (TT-prefixed params, ``tt_`` functions).

- :func:`tt_weight_norm_materialize` — effective Conv weight from ``(v, g)`` tensors
- :func:`tt_conv1d_nlc` — Conv1d on NLC activations ``[B, L, C]``
- :func:`tt_conv_transpose1d_nlc` — ConvTranspose1d via ``conv_transpose2d`` (``spatial_style`` width vs height)
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn


# --- Opt-in prepared-weight caching for metal-trace capture ---------------------------------
# ``ttnn.conv1d`` / ``ttnn.conv_transpose2d`` prepare (tilize + reshard) their weights on host and
# UPLOAD them to device on every invocation.  That host->device write is illegal inside
# ``ttnn.begin_trace_capture`` ("Writes are not supported during trace capture"), so the generator
# forward cannot be trace-captured while convs re-upload weights each call.
#
# When trace weight-prep is enabled, the FIRST call to each conv site asks the op to hand back the
# tensors it prepared (``return_weights_and_bias=True``) and caches them on device; every later call
# reuses those prepared tensors (weights already on device -> no upload), which is what makes the
# graph trace-capturable.  Reusing the op's own prepared weights is byte-identical to the raw path
# (verified PCC = 1.0 for both conv1d and conv_transpose2d), so this changes only *where* the weight
# prep happens, never the math.  Default OFF: every existing (non-traced) caller is untouched.
#
# The enable flag + cache are shared across Kokoro modules (see tt_trace_prep); re-exported here so
# existing ``from tt_conv import set_trace_weight_prep`` call sites keep working.
from .tt_trace_prep import (  # noqa: E402
    clear_trace_weight_prep_cache,
    prep_cache_get as _prep_cache_get,
    prep_cache_set as _prep_cache_set,
    set_trace_weight_prep,
    trace_weight_prep_enabled as _trace_weight_prep_enabled,
    traced_zeros as _traced_zeros,
)


def _prep_signature(x, out_dtype, extra) -> tuple:
    """Cache key discriminator: identical (shape, dtype, mem-config, out-dtype, site) -> same prep."""
    return (
        tuple(int(d) for d in x.shape),
        str(x.dtype),
        str(x.memory_config()),
        str(out_dtype),
    ) + tuple(extra)


def _weights_for_conv(params, sig):
    """Resolve (weight, bias, want_weights_and_bias) honouring the trace-prep cache.

    ``sig is None`` (prep disabled, or a code path that opts out) -> raw weights, no reuse.
    Cache hit -> reuse the on-device prepared tensors. Cache miss with prep enabled -> raw weights
    but ask the op to return its prepared tensors so the caller can store them under ``sig``.
    """
    if sig is None:
        return params.weight, params.bias, False
    cached = _prep_cache_get((id(params.weight), sig))
    if cached is not None:
        return cached[0], cached[1], False
    return params.weight, params.bias, True


def upload_conv1d_params_from_module(
    conv: nn.Conv1d,
    _device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> "TTConv1dParams":
    """Upload Conv1d weights on host ROW_MAJOR (``ttnn.conv1d`` prepares once per call)."""
    w = conv.weight.detach().cpu().unsqueeze(-1)
    w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = None
    if conv.bias is not None:
        b_tt = ttnn.from_torch(
            conv.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    return TTConv1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=int(conv.in_channels),
        out_channels=int(conv.out_channels),
        kernel_size=int(conv.kernel_size[0]),
        stride=int(conv.stride[0]),
        padding=int(conv.padding[0]),
        groups=int(conv.groups),
        dilation=int(conv.dilation[0]),
    )


def upload_conv_transpose_pool_params_from_module(
    module: nn.ConvTranspose1d,
    _device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> "TTConvTranspose1dParams":
    """Upload pool weights on host ROW_MAJOR (same as legacy Kokoro bring-up)."""
    w = module.weight.detach().cpu().unsqueeze(-1)
    w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = None
    if module.bias is not None:
        b_tt = ttnn.from_torch(
            module.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    return TTConvTranspose1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=int(module.in_channels),
        out_channels=int(module.out_channels),
        kernel_size=int(module.kernel_size[0]),
        stride=int(module.stride[0]),
        padding=int(module.padding[0]),
        output_padding=int(module.output_padding[0]),
        groups=int(module.groups),
        mirror_kernel=True,
        spatial_style="height",
    )


def tt_weight_norm_materialize(weight_v: torch.Tensor, weight_g: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Effective weight for legacy ``torch.nn.utils.weight_norm`` (``v``, ``g`` tensors).

    For Conv1d, ``v`` has shape ``[out_ch, in_ch/groups, k]``; ``g`` is broadcastable to ``v``.
    """
    v = weight_v
    g = weight_g
    while g.dim() < v.dim():
        g = g.unsqueeze(-1)
    v_norm = torch.linalg.vector_norm(v.reshape(v.shape[0], -1), ord=2, dim=1).clamp_min(eps)
    v_norm = v_norm.view(v.shape[0], *([1] * (v.dim() - 1)))
    return v * (g / v_norm)


def dram_height_slice_num_slices(
    sliced_dim: int,
    *,
    min_slices: int = 8,
    max_slices: int = 256,
    target_rows_per_slice: int = 512,
) -> int:
    """Pick ``Conv2dDRAMSliceHeight`` ``num_slices`` to stay under Blackhole L1 (~1.4 MiB/bank).

    Each DRAM slice still runs an L1 ``conv_transpose2d`` on its height partition.  Empirically
    ~512 input rows per slice fits on BH; Kokoro max length (``F`` ~ 127k) needs ~256 slices.
    """
    if sliced_dim <= target_rows_per_slice:
        return min_slices
    n = max(min_slices, (sliced_dim + target_rows_per_slice - 1) // target_rows_per_slice)
    p2 = 1
    while p2 < n:
        p2 <<= 1
    return min(p2, max_slices)


# BH L1: height-style conv_transpose2d circular buffers scale with rows/slice. 512 is fine for
# moderate lengths; long generator upsample (seq ~8k+) and iSTFT OLA need tighter slicing.
_DRAM_SLICE_TARGET_ROWS_DEFAULT = 512
_DRAM_SLICE_TARGET_ROWS_LONG = 128
_DRAM_SLICE_LONG_SEQ_THRESHOLD = 4096
# Decoder decode block depthwise pool (groups≈1090) at T_mel ~500–1000 overflows L1 unless sliced.
_DRAM_SLICE_HIGH_CHANNEL_THRESHOLD = 512
_DRAM_SLICE_HIGH_CHANNEL_MIN_SEQ = 256


def dram_height_slice_target_rows(
    sliced_dim: int,
    *,
    channels: int = 1,
    activation_dtype=None,
) -> int:
    """Pick per-slice row budget for ``Conv2dDRAMSliceHeight`` on Blackhole."""
    if sliced_dim >= _DRAM_SLICE_LONG_SEQ_THRESHOLD:
        base = _DRAM_SLICE_TARGET_ROWS_LONG
    else:
        base = _DRAM_SLICE_TARGET_ROWS_DEFAULT

    elem_bytes = 4 if activation_dtype == ttnn.float32 else 2
    # Keep per-slice activation footprint under ~256 KiB (empirical BH L1 CB headroom).
    budget_rows = max(32, (256 * 1024) // max(channels * elem_bytes, 1))
    if channels >= 1024:
        return min(base, 32, budget_rows)
    if channels >= _DRAM_SLICE_HIGH_CHANNEL_THRESHOLD:
        return min(base, 64, budget_rows)
    return min(base, budget_rows)


def dram_height_slice_config(
    sliced_dim: int,
    *,
    channels: int = 1,
    activation_dtype=None,
    target_rows_per_slice: int | None = None,
) -> ttnn.Conv2dSliceConfig:
    if target_rows_per_slice is None:
        target_rows_per_slice = dram_height_slice_target_rows(
            sliced_dim, channels=channels, activation_dtype=activation_dtype
        )
    return ttnn.Conv2dSliceConfig(
        slice_type=ttnn.Conv2dDRAMSliceHeight,
        num_slices=dram_height_slice_num_slices(sliced_dim, target_rows_per_slice=target_rows_per_slice),
    )


def _conv_transpose_use_dram_height_slice(
    *,
    spatial_style: str,
    seq: int,
    channels: int,
    activation_dtype,
) -> bool:
    """Whether height-style ``conv_transpose2d`` needs DRAM height slicing on BH."""
    if spatial_style != "height":
        return False
    if seq >= 1024:
        return True
    if channels >= _DRAM_SLICE_HIGH_CHANNEL_THRESHOLD and seq >= _DRAM_SLICE_HIGH_CHANNEL_MIN_SEQ:
        return True
    return activation_dtype == ttnn.float32 and channels >= 256 and seq >= 384


@dataclass(frozen=True)
class TTConv1dParams:
    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor]
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    groups: int = 1
    dilation: int = 1


def _chunked_tt_conv1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: "TTConv1dParams",
    device,
    compute_config=None,
    out_dtype=ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    chunk_out: int = 512,
) -> ttnn.Tensor:
    """Dilated conv1d on large L via overlapping sliding-window chunks (device-only).

    For dilated convolutions with L > 2048, the default sharding configuration exceeds
    L1 on BH.  Reducing act_block_h_override would corrupt boundary outputs because the
    halo rows (positions ±dilation*(k//2) outside the block) are not fetched for dilated
    kernels.  Instead, we slice the input into overlapping windows on-device so each chunk
    has L_chunk = chunk_out + (kernel_size-1)*dilation < 2048, avoiding the overflow.

    Each chunk's input includes its left and right halo explicitly; the conv is called
    with padding=0 so no virtual padding is applied inside ttnn.conv1d.
    """
    B = int(x_nlc.shape[0])
    L = int(x_nlc.shape[1])
    C_in = int(x_nlc.shape[2])

    out_L = (L + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) // params.stride + 1

    # chunk_params: no padding — we include halo in each extracted slice
    cp = TTConv1dParams(
        weight=params.weight,
        bias=params.bias,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=0,
        groups=params.groups,
        dilation=params.dilation,
    )

    output_chunks: list[ttnn.Tensor] = []
    out_s = 0
    while out_s < out_L:
        out_e = min(out_s + chunk_out, out_L)

        # Virtual input range needed for output positions [out_s, out_e)
        vin_s = out_s * params.stride - params.padding
        vin_e = (out_e - 1) * params.stride - params.padding + params.dilation * (params.kernel_size - 1)

        # Actual (physical) input range — clamp to [0, L)
        in_s = max(0, vin_s)
        in_e = min(L - 1, vin_e)
        left_pad = max(0, -vin_s)
        right_pad = max(0, vin_e - (L - 1))

        x_slice = ttnn.slice(x_nlc, [0, in_s, 0], [B, in_e + 1, C_in], [1, 1, 1], memory_config=memory_config)

        if left_pad > 0:
            z_l = _traced_zeros(
                [B, left_pad, C_in],
                dtype=x_nlc.dtype,
                device=device,
                memory_config=memory_config,
                key=(id(params.weight), "chunk_conv1d_zl", B, left_pad, C_in, str(x_nlc.dtype), str(memory_config)),
            )
            x_slice = ttnn.concat([z_l, x_slice], dim=1, memory_config=memory_config)
            ttnn.deallocate(z_l)
        if right_pad > 0:
            z_r = _traced_zeros(
                [B, right_pad, C_in],
                dtype=x_nlc.dtype,
                device=device,
                memory_config=memory_config,
                key=(id(params.weight), "chunk_conv1d_zr", B, right_pad, C_in, str(x_nlc.dtype), str(memory_config)),
            )
            x_slice = ttnn.concat([x_slice, z_r], dim=1, memory_config=memory_config)
            ttnn.deallocate(z_r)

        y_c = tt_conv1d_nlc(
            x_nlc=x_slice,
            params=cp,
            device=device,
            compute_config=compute_config,
            out_dtype=out_dtype,
            memory_config=memory_config,
        )
        ttnn.deallocate(x_slice)

        if y_c.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            y_c = ttnn.to_memory_config(y_c, memory_config)
        output_chunks.append(y_c)
        out_s = out_e

    out = ttnn.concat(output_chunks, dim=1, memory_config=memory_config)
    for c in output_chunks:
        ttnn.deallocate(c)
    return out


def tt_conv1d_nlc_cpu(
    *,
    x_nlc: ttnn.Tensor,
    params: TTConv1dParams,
    device,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    out_dtype=None,
) -> ttnn.Tensor:
    """Run Conv1d on CPU using uploaded TT weights (NLC in/out).

    Used for Kokoro ``F0_conv`` / ``N_conv`` where ``ttnn.conv1d`` diverges from PyTorch
    on stride-2 ``kernel_size=3`` paths at inference lengths (e.g. ``T_f0=162``).
    """
    x = ttnn.to_torch(x_nlc).float()
    while x.dim() > 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    w = ttnn.to_torch(params.weight).float().squeeze(-1)
    b = ttnn.to_torch(params.bias).float().reshape(-1) if params.bias is not None else None
    x_bct = x.transpose(1, 2).contiguous()
    with torch.no_grad():
        y_bct = F.conv1d(
            x_bct,
            w,
            b,
            stride=params.stride,
            padding=params.padding,
            dilation=params.dilation,
            groups=params.groups,
        )
    y_nlc = y_bct.transpose(1, 2).contiguous()
    dtype = out_dtype if out_dtype is not None else x_nlc.dtype
    return ttnn.from_torch(y_nlc, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _batched_tt_conv1d_nlc(
    *,
    x_flat: ttnn.Tensor,
    params: "TTConv1dParams",
    device,
    batch: int,
    seq: int,
    compute_config=None,
    out_dtype=ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    output_sharded: bool = False,
) -> ttnn.Tensor:
    """Single ``ttnn.conv1d`` over all ``batch`` rows (Blackhole-only; see ``batched_shape``).

    Returns the conv's native ``[1, 1, batch*out_len, C]`` rows **without** an extra reshape: each
    ReshapeView is a fixed ~5µs device dispatch, so the caller chains conv→LN→activation directly on
    this rank-4 shape (all rank-agnostic over the channel dim) and defers the one un-flatten to
    ``[batch, out_len, C]`` until right before the LSTM.
    """
    # bfloat8_b conv weights (this batched path is TextEncoder-ONLY — sole caller is
    # tt_text_encoder.py, verified — so the F0/prosody/generator convs are untouched). The L1 sweep
    # (perf/conv_sweep_l1_results.md) found this the single real lever: the conv re-reads the
    # [Cin*k, Cout] weight per output block, so halving the weight bytes cuts the conv 16.1->11.4µs
    # (-29%) at isolated PCC 0.99988 vs 0.99989 (bf16). The tolerant ASR TextEncoder absorbs it; the
    # F0-feeding convs (which reject precision drops) never take this path.
    conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat8_b)
    conv_config.config_tensors_in_dram = False
    conv_config.deallocate_activation = True
    # Config sweep (perf/test_conv_text_encoder_perf_sweep.py, production 512x512x5 @ B*T=96 shape):
    # the conv op was 31.6µs on the default auto-shard/no-double-buffer config. Block-sharded with both
    # double buffers on is the fastest PCC-passing config (1.0 unchanged) at 21.7µs (1.46x). Levers, in
    # order of impact: weights-double-buffer (~10µs — the conv re-reads the [2560,512] weight per output
    # block), act-double-buffer (~4µs), explicit BLOCK shard (beats auto/width). act_block_h_override=32
    # ties the default but 64 regresses hard (40-50µs); split_reader is noise. Double buffering costs
    # extra L1 — guarded to the larger convs that benefit and verified to fit the full kmodel's L1.
    conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    conv_config.enable_act_double_buffer = True
    conv_config.enable_weights_double_buffer = True
    conv_config.act_block_h_override = 32
    if params.out_channels >= 256 or params.kernel_size >= 7:
        conv_config.force_split_reader = True
    if compute_config is None:
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True
        )

    if x_flat.layout != ttnn.TILE_LAYOUT:
        x_flat = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT, memory_config=memory_config)

    # ttnn#47927 ("conv1d: enable DRAM slicing by default") dropped conv1d's historical L1_FULL
    # default: a missing slice_config now auto-routes by input location, so a DRAM input (the embedding
    # output that feeds block 0) gets width-sliced through DRAM and the conv returns its output
    # *interleaved in DRAM*. That breaks ``output_sharded`` — the LayerNorm can no longer read the
    # conv's block-sharded L1 output in place and falls back to a 3-core DRAM LN, with a per-block
    # ShardedToInterleaved/InterleavedToSharded/ReshapeView round-trip. When the caller wants the
    # sharded output, force L1_FULL to restore the block-sharded L1 result (the B*T≈96 TextEncoder
    # shape fits L1; long-sequence DRAM-slicing callers leave output_sharded=False and are untouched).
    slice_config = ttnn.Conv2dL1FullSliceConfig if output_sharded else None

    # Trace-prep weight caching (same mechanism as the standard per-item path above): with prep
    # enabled the first call reuses the op's own prepared (bf8, tiled, resharded) weights and caches
    # them on device; later calls (and the trace capture) reuse them so ttnn.conv1d does NO
    # host->device weight upload — which is what makes the TextEncoder conv trace-capturable. Byte-
    # identical to the raw path. Keyed by the batched shape so it never collides with the per-item sig.
    _sig = (
        _prep_signature(x_flat, out_dtype, extra=("conv1d_batched", batch, seq))
        if _trace_weight_prep_enabled()
        else None
    )
    _w, _b, _want_wb = _weights_for_conv(params, _sig)
    _res = ttnn.conv1d(
        input_tensor=x_flat,
        weight_tensor=_w,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        device=device,
        bias_tensor=_b,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=params.padding,
        dilation=params.dilation,
        batch_size=batch,
        input_length=seq,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=slice_config,
        groups=params.groups,
        dtype=out_dtype,
        return_weights_and_bias=_want_wb,
    )
    if _want_wb:
        y, _wb = _res
        _prep_cache_set((id(params.weight), _sig), (_wb[0], _wb[1]))
    else:
        y = _res
    # With ``output_sharded`` the conv's native block-sharded L1 output is returned as-is, so the
    # consumer (the TextEncoder channel-LayerNorm) reads it in place — no ShardedToInterleaved here AND
    # no InterleavedToSharded before the LayerNorm (the conv output is already a valid block-sharded LN
    # input; the LN derives its program config from this shard spec). Saves two reshards/CNN stage.
    if output_sharded and y.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return y
    # Otherwise move the (block-sharded) conv output to interleaved DRAM and return as-is; no reshape.
    if y.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        y = ttnn.to_memory_config(y, memory_config)
    return y


def tt_conv1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTConv1dParams,
    device,
    conv_config: Optional[ttnn.Conv1dConfig] = None,
    compute_config=None,
    out_dtype=ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    preserve_input_dtype: bool = False,
    batched_shape: Optional[tuple[int, int]] = None,
    output_sharded: bool = False,
) -> ttnn.Tensor:
    """
    Conv1d on activations ``[B, L, C]`` (NLC). Returns ``[B, out_L, out_C]`` (NLC).

    With ``preserve_input_dtype=True`` the output dtype is forced to match ``x_nlc.dtype`` (useful
    for fp32 paths where the default bf16 output would clamp precision below WH's accumulator).

    ``batched_shape=(B, L)`` (Blackhole only) takes a batch-flattened input (``[1, 1, B*L, C]`` /
    ``[1, B*L, C]``) and runs a single batched ``ttnn.conv1d`` (``batch_size=B``), returning the
    conv's native ``[1, 1, B*out_L, C]`` (no extra reshape — see ``_batched_tt_conv1d_nlc``). The
    per-item split below exists only to dodge a Wormhole-B0 ``B*L`` correctness bug; on Blackhole the
    batched op is correct (verified PCC≈1.0 for the TextEncoder shapes) and roughly halves
    conv-related dispatch (one conv/halo/shard set instead of B, no per-item concat). The caller keeps
    activations batch-flattened across the CNN stack and un-flattens once downstream (before the LSTM).
    """
    if preserve_input_dtype:
        out_dtype = x_nlc.dtype

    # Blackhole batched fast path: one flattened conv over all B rows (see ``batched_shape`` above).
    if batched_shape is not None and conv_config is None and device.arch() == ttnn.device.Arch.BLACKHOLE:
        Bb, Lb = batched_shape
        return _batched_tt_conv1d_nlc(
            x_flat=x_nlc,
            params=params,
            device=device,
            batch=Bb,
            seq=Lb,
            compute_config=compute_config,
            out_dtype=out_dtype,
            memory_config=memory_config,
            output_sharded=output_sharded,
        )
    if batched_shape is not None:
        # Non-Blackhole (or custom conv_config): un-flatten, run the standard (per-item) path, then
        # re-flatten to the batched [1, 1, B*out_L, C] output contract so callers are arch-agnostic.
        Bb, Lb = batched_shape
        C = int(x_nlc.shape[-1])
        x_nlc = ttnn.reshape(x_nlc, (Bb, Lb, C), memory_config=memory_config)
        y = tt_conv1d_nlc(
            x_nlc=x_nlc,
            params=params,
            device=device,
            compute_config=compute_config,
            out_dtype=out_dtype,
            memory_config=memory_config,
            preserve_input_dtype=preserve_input_dtype,
        )
        return ttnn.reshape(y, (1, 1, int(y.shape[0]) * int(y.shape[1]), int(y.shape[2])), memory_config=memory_config)

    B = int(x_nlc.shape[0])
    L = int(x_nlc.shape[1])
    C_in = int(x_nlc.shape[2])

    # Workaround: ttnn.conv1d produces incorrect results on Wormhole B0 when B*L
    # falls in [97..192] (or near 384). For B>1 we process each item separately.
    if B > 1:
        slices = []
        for b in range(B):
            x_b = ttnn.slice(x_nlc, [b, 0, 0], [b + 1, L, C_in], [1, 1, 1], memory_config=memory_config)
            if x_b.layout != ttnn.TILE_LAYOUT:
                x_b = ttnn.to_layout(x_b, ttnn.TILE_LAYOUT, memory_config=memory_config)
            y_b = tt_conv1d_nlc(
                x_nlc=x_b,
                params=params,
                device=device,
                compute_config=compute_config,
                out_dtype=out_dtype,
                memory_config=memory_config,
                preserve_input_dtype=False,
            )
            # conv output may be sharded; move to DRAM before concat
            if y_b.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                y_b = ttnn.to_memory_config(y_b, memory_config)
            slices.append(y_b)
        ttnn.deallocate(x_nlc)
        return ttnn.concat(slices, dim=0, memory_config=memory_config)

    # Workaround: B=1 with L in (96, 194) also hits the broken range; pad to 194.
    _BL_BROKEN_MAX = 96
    _BL_PAD_TARGET = 194
    orig_L = L
    if _BL_BROKEN_MAX < L < _BL_PAD_TARGET:
        pad = _BL_PAD_TARGET - L
        # ttnn.pad (one op, zero fill) instead of concat([x, zeros], dim=1): the dim=1 concat of
        # TILE tensors unpads both inputs (untilize+untilize+concat triple) along the tile-padded
        # length dim. pad appends the zero rows directly. Bit-identical (same 0.0 fill, x untouched).
        x_nlc = ttnn.pad(x_nlc, padding=[(0, 0), (0, pad), (0, 0)], value=0.0, memory_config=memory_config)
        L = _BL_PAD_TARGET

    # For large L with stride=1: use sliding-window chunked conv to avoid L1 CB overflow.
    # act_block_h_override cannot fix CB conflicts for DRAM-interleaved inputs at very
    # large L, and does not work at all for dilated kernels (boundary halo rows are not
    # fetched when act_block_h < dilation*(k-1), corrupting ~30% of outputs).
    # _chunked_tt_conv1d_nlc builds the explicit halo for every chunk and is correct for
    # all dilation values including 1.  Stride > 1 convolutions (e.g. noise_conv) are
    # excluded: chunking them requires chunk_out*stride input rows per chunk which can
    # again exceed 2048 causing unbounded recursion; act_block_h_override=32 covers them.
    if L > 2048 and params.stride == 1 and conv_config is None:
        return _chunked_tt_conv1d_nlc(
            x_nlc=x_nlc,
            params=params,
            device=device,
            compute_config=compute_config,
            out_dtype=out_dtype,
            memory_config=memory_config,
        )

    if conv_config is None:
        conv_config = ttnn.Conv1dConfig(weights_dtype=params.weight.dtype)
        conv_config.config_tensors_in_dram = True
        conv_config.deallocate_activation = True
        if params.out_channels >= 256 or params.kernel_size >= 7:
            conv_config.force_split_reader = True
        # Wide convs re-read the [Cin*k, Cout] weight per output block, so weights/act
        # double-buffering is the dominant wall-clock lever (proven 1.46x / ~10µs on the
        # identical-shape TextEncoder conv, see _batched_tt_conv1d_nlc). Explicit BLOCK sharding
        # pins the output to a 2D core grid so a wide-channel generator conv uses every output
        # tile as a core (ceil(H/32)*ceil(C/32)) instead of the auto-heuristic's inconsistent
        # 20-26 for the 320x128 body convs. Numerically inert — this is spatial distribution of
        # independent output tiles, not a reduction split, so the per-element MAC order is
        # unchanged (verified: resblocks 0.99990/0.99990, conv_post 0.99994 identical to the
        # pre-change baseline). Guarded to wide convs and the non-chunked (L<=2048) region so the
        # extra double-buffer L1 is bounded; act_block_h_override=32 caps the per-core CB (64
        # regresses hard on BH per the TextEncoder sweep). The out_channels band is capped at 256:
        # the double-buffered weight CB scales with Cin*k*Cout, and the Decoder's wide AdainResBlk1d
        # convs (out=1024, e.g. encode 514->1024) overflow BH L1 (>2.5 MB CBs vs 1.5 MB/bank) if
        # this recipe is applied — 128..256 covers every generator body/source conv while excluding
        # those.
        if L <= 2048 and 128 <= params.out_channels <= 256:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
            conv_config.enable_act_double_buffer = True
            conv_config.enable_weights_double_buffer = True
            conv_config.act_block_h_override = 32
        # For large L with dilation=1: capping act_block_h to 32 (one TILE row) reduces
        # the per-core CB footprint to avoid L1 overflow.  For dilation=1 this is correct
        # because the halo is only (kernel_size-1)//2 rows, well within a 32-row block.
        if L > 2048:
            conv_config.act_block_h_override = 32
    if compute_config is None:
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True
        )

    _sig = _prep_signature(x_nlc, out_dtype, extra=("conv1d", L)) if _trace_weight_prep_enabled() else None
    _w, _b, _want_wb = _weights_for_conv(params, _sig)
    _res = ttnn.conv1d(
        input_tensor=x_nlc,
        weight_tensor=_w,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        device=device,
        bias_tensor=_b,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=params.padding,
        dilation=params.dilation,
        batch_size=1,
        input_length=L,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=params.groups,
        dtype=out_dtype,
        return_output_dim=True,
        return_weights_and_bias=_want_wb,
    )
    if _want_wb:
        y, out_len, _wb = _res
        _prep_cache_set((id(params.weight), _sig), (_wb[0], _wb[1]))
    else:
        y, out_len = _res
    if len(y.shape) == 4 and y.shape[0] == 1:
        y = ttnn.squeeze(y, 0)
    if len(y.shape) == 4 and y.shape[1] == 1:
        y = ttnn.reshape(y, [y.shape[0], y.shape[2], y.shape[3]], memory_config=memory_config)
    if len(y.shape) == 3 and y.shape[1] != out_len:
        y = ttnn.reshape(y, (1, out_len, y.shape[-1]), memory_config=memory_config)

    if orig_L != L:
        orig_out_L = (orig_L + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) // params.stride + 1
        C_out = int(y.shape[-1])
        y = ttnn.slice(y, [0, 0, 0], [1, orig_out_L, C_out], [1, 1, 1], memory_config=memory_config)
    return y


def tt_conv1d_stride2_k3_1ch_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTConv1dParams,
    device,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """TT-native Conv1d(1→1, k=3, stride=2, padding=1) via reshape+permute+elementwise.

    Avoids ``ttnn.conv1d`` which produces ~10 Hz interior errors at T_f0~162 on BH due to
    the broken B*L range (96, 194) — see P1 in DECODE_STACK_NOTES.md.

    Algorithm: zero-pad x by 1 on each side → reshape [B, T+2, 1] to [B, (T+2)/2, 2]
    (C-order pairs → even/odd channels) → permute(0,2,1) to [B, 2, half] so the even and
    odd rows are memory-contiguous → slice three kernel-position views s0/s1/s2 → compute
    w0·s0 + w1·s1 + w2·s2 + b using elementwise ops (no MAC, no BF16 rounding).

    Only valid for in_channels=out_channels=1, kernel_size=3, stride=2, padding=1, dilation=1.
    """
    assert params.in_channels == 1 and params.out_channels == 1, "only 1-channel conv"
    assert params.kernel_size == 3 and params.stride == 2 and params.padding == 1
    assert getattr(params, "dilation", 1) == 1 and getattr(params, "groups", 1) == 1

    B = int(x_nlc.shape[0])
    T = int(x_nlc.shape[1])
    T_out = (T - 1) // 2 + 1  # == (T + 2*1 - 3) // 2 + 1

    dtype = x_nlc.dtype

    # Upload kernel weights [1,1,3,1] → [1,1,3] and bias to device (TILE, broadcast over B and T).
    # ``ttnn.to_device`` of the host weight is a host->device write that trace capture forbids, so
    # when trace weight prep is enabled we upload once and reuse the cached device tensors across
    # calls (``_own_weights`` then stays False so the compute below does not free them). Prep off =
    # the original upload-and-free-each-call behaviour, byte-identical.
    _own_weights = not _trace_weight_prep_enabled()
    _cache_key = (id(params.weight), "stride2_k3", str(memory_config))
    _cached = _prep_cache_get(_cache_key) if _trace_weight_prep_enabled() else None
    if _cached is not None:
        w_tt, b_tt = _cached
    else:
        w_dev = ttnn.to_device(ttnn.reshape(params.weight, [1, 1, 3]), device, memory_config=memory_config)
        w_tt = ttnn.to_layout(w_dev, ttnn.TILE_LAYOUT, memory_config=memory_config)
        ttnn.deallocate(w_dev)

        b_tt: Optional[ttnn.Tensor] = None
        if params.bias is not None:
            b_dev = ttnn.to_device(ttnn.reshape(params.bias, [1, 1, 1]), device, memory_config=memory_config)
            b_tt = ttnn.to_layout(b_dev, ttnn.TILE_LAYOUT, memory_config=memory_config)
            ttnn.deallocate(b_dev)

        if _trace_weight_prep_enabled():
            _prep_cache_set(_cache_key, (w_tt, b_tt))

    # Step 1: pad [B, T, 1] → [B, T+2, 1]. The zero pad column is created with ttnn.zeros(device=...),
    # a host->device write that trace capture forbids; under trace weight prep it is created once and
    # reused (the same all-zeros [B,1,1] serves left/right/even pads — byte-identical either way).
    if x_nlc.layout != ttnn.TILE_LAYOUT:
        x_nlc = ttnn.to_layout(x_nlc, ttnn.TILE_LAYOUT, memory_config=memory_config)
    _zkey = (id(params.weight), "stride2_zeros", B, str(dtype), str(memory_config))
    _zcached = _prep_cache_get(_zkey) if _trace_weight_prep_enabled() else None
    if _zcached is not None:
        z_pad = _zcached
    else:
        z_pad = ttnn.zeros([B, 1, 1], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
        if _trace_weight_prep_enabled():
            _prep_cache_set(_zkey, z_pad)
    x_padded = ttnn.concat([z_pad, x_nlc, z_pad], dim=1, memory_config=memory_config)

    # Ensure T+2 is even for the reshape into (even, odd) pairs
    T_padded = T + 2
    if T_padded % 2 != 0:
        x_padded_even = ttnn.concat([x_padded, z_pad], dim=1, memory_config=memory_config)
        ttnn.deallocate(x_padded)
        x_padded = x_padded_even
        T_padded += 1
    if _own_weights:
        ttnn.deallocate(z_pad)
    half = T_padded // 2

    # Step 2: reshape [B, T_padded, 1] → [B, half, 2] in ROW_MAJOR (C-order)
    # After reshape: element [b, j, 0] = x_padded[2j] (even), [b, j, 1] = x_padded[2j+1] (odd)
    x_rm = ttnn.to_layout(x_padded, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    ttnn.deallocate(x_padded)
    x_pairs = ttnn.reshape(x_rm, [B, half, 2], memory_config=memory_config)
    ttnn.deallocate(x_rm)

    # Step 3: permute [B, half, 2] → [B, 2, half] so each "row" is memory-contiguous
    # Row 0: even positions [x_padded[0], x_padded[2], ..., x_padded[2*(half-1)]]
    # Row 1: odd  positions [x_padded[1], x_padded[3], ..., x_padded[2*(half-1)+1]]
    x_pairs_tile = ttnn.to_layout(x_pairs, ttnn.TILE_LAYOUT, memory_config=memory_config)
    ttnn.deallocate(x_pairs)
    x_deint = ttnn.permute(x_pairs_tile, (0, 2, 1), memory_config=memory_config)
    ttnn.deallocate(x_pairs_tile)

    # Step 4: slice three contiguous kernel-position views from [B, 2, half] in ROW_MAJOR
    # s0: even positions 0, 2, ..., 2*(T_out-1)    → row 0, cols 0..T_out-1
    # s1: odd  positions 1, 3, ..., 2*(T_out-1)+1  → row 1, cols 0..T_out-1
    # s2: even positions 2, 4, ..., 2*T_out         → row 0, cols 1..T_out
    x_deint_rm = ttnn.to_layout(x_deint, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    ttnn.deallocate(x_deint)
    s0 = ttnn.slice(x_deint_rm, [0, 0, 0], [B, 1, T_out], [1, 1, 1], memory_config=memory_config)
    s1 = ttnn.slice(x_deint_rm, [0, 1, 0], [B, 2, T_out], [1, 1, 1], memory_config=memory_config)
    s2 = ttnn.slice(x_deint_rm, [0, 0, 1], [B, 1, T_out + 1], [1, 1, 1], memory_config=memory_config)
    ttnn.deallocate(x_deint_rm)

    # Step 5: convert each [B, 1, T_out] slice to TILE and reshape to NLC [B, T_out, 1]
    def _to_nlc(s: ttnn.Tensor) -> ttnn.Tensor:
        s = ttnn.to_layout(s, ttnn.TILE_LAYOUT, memory_config=memory_config)
        return ttnn.reshape(s, [B, T_out, 1], memory_config=memory_config)

    s0 = _to_nlc(s0)
    s1 = _to_nlc(s1)
    s2 = _to_nlc(s2)

    # Step 6: concat [B,T_out,1] × 3 → [B,T_out,3], broadcast-multiply by weights [1,1,3],
    # then reduce over channel dim to get [B,T_out,1] = w0·s0 + w1·s1 + w2·s2
    s_all = ttnn.concat([s0, s1, s2], dim=2, memory_config=memory_config)
    ttnn.deallocate(s0)
    ttnn.deallocate(s1)
    ttnn.deallocate(s2)
    weighted = ttnn.multiply(s_all, w_tt, memory_config=memory_config)
    ttnn.deallocate(s_all)
    if _own_weights:
        ttnn.deallocate(w_tt)
    y = ttnn.sum(weighted, dim=2, keepdim=True, memory_config=memory_config)
    ttnn.deallocate(weighted)
    if b_tt is not None:
        y_b = ttnn.add(y, b_tt, memory_config=memory_config)
        if _own_weights:
            ttnn.deallocate(b_tt)
        ttnn.deallocate(y)
        y = y_b
    return y


# Height-style conv_transpose2d L1 output limit.  Above this output length TTNN pre-allocates
# the full output in L1 even with dram_slice_config, overflowing BH (1.5 MiB/bank).
_L1_CONV_TRANSPOSE_HEIGHT_MAX_OUT = 8192
# Maximum input rows processed per OLA chunk.  For stride≤10, kernel≤20:
#   chunk_out = (256-1)*10 + 20 = 2 570 rows × 256 ch × 4 B / 126 banks ≈ 21 KB/bank ✓
_CONV_TRANSPOSE_HEIGHT_CHUNK = 256


@dataclass(frozen=True)
class TTConvTranspose1dParams:
    # TTNN conv_transpose2d weights (NHWC op): default ``[in_ch, out_ch/groups, 1, k]`` (``spatial_style="width"``);
    # Kokoro istftnet depthwise pool uses ``[in_ch, out_ch/groups, k, 1]`` (``spatial_style="height"``).
    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor]
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    output_padding: int
    groups: int = 1
    mirror_kernel: bool = True
    spatial_style: str = "width"  # "width" | "height"


def _chunked_tt_conv_transpose1d_height(
    x_nlc: ttnn.Tensor,
    params: "TTConvTranspose1dParams",
    device,
    chunk_size: int,
    memory_config: ttnn.MemoryConfig,
    out_dtype,
) -> ttnn.Tensor:
    """Overlap-add chunked height-style conv_transpose for sequences whose full output would OOM L1.

    Each chunk of ``chunk_size`` input rows is processed with ``padding=0`` on device.
    Chunk outputs are downloaded and accumulated into a pre-allocated float32 buffer on CPU.
    After all chunks, the global padding trim (``params.padding`` from each end) is applied
    and the bias (if any) is added once.  Result is uploaded to device in one call.

    Why padding=0 per chunk: applying the original ``params.padding`` to each chunk would
    trim boundary contributions that should only be trimmed at the *global* sequence edges.
    Internal chunks' inter-chunk overlap (K-S rows) is correctly accumulated via ``+=`` into
    the unpadded buffer; the global trim then reproduces the exact padded output.

    The overlap-add runs **entirely on device** (no host download / re-upload), so the op is
    metal-trace-capturable: each chunk output is zero-padded to full length at its global ``out_start``
    offset and added into a float32 accumulator. TILE layout can't front-pad on device, so the
    placement pad goes through ROW_MAJOR. Accumulation is float32 (matching the previous host
    accumulation) and cast to ``out_dtype`` once at the end; the global ``padding`` trim and bias are
    applied on device. Validated vs ``torch.conv_transpose1d`` at a chunking length: PCC 1.0 (bf16) /
    0.999995 (fp32).
    """
    B = int(x_nlc.shape[0])
    L = int(x_nlc.shape[1])
    C = int(x_nlc.shape[2])
    S = params.stride
    K = params.kernel_size
    P = params.padding
    OP = params.output_padding
    C_out = params.out_channels

    # Unpadded buffer length: sum of all per-input contributions without global trim.
    out_H_unpadded = (L - 1) * S + K
    # Final output length after global trim and output_padding.
    out_H = out_H_unpadded - 2 * P + OP

    # Per-chunk params: no padding (we trim globally), no bias (applied once at the end).
    chunk_params = _dc_replace(params, padding=0, output_padding=0, bias=None)

    acc: Optional[ttnn.Tensor] = None  # float32 device accumulator [B, out_H_unpadded, C_out]
    for chunk_start in range(0, L, chunk_size):
        chunk_end = min(chunk_start + chunk_size, L)

        # Slice input chunk on device (chunk_start multiples of chunk_size=256 are tile-aligned).
        x_chunk = ttnn.slice(x_nlc, [0, chunk_start, 0], [B, chunk_end, C], [1, 1, 1], memory_config=memory_config)

        # Run conv_transpose with padding=0 (seq ≤ chunk_size < 1024, no DRAM slice).
        y = tt_conv_transpose1d_nlc(
            x_nlc=x_chunk,
            params=chunk_params,
            device=device,
            memory_config=memory_config,
            out_dtype=out_dtype,
        )
        ttnn.deallocate(x_chunk)

        # Upcast to fp32 for the overlap-add (matches the reference fp32 accumulation).
        y_f = y if y.dtype == ttnn.float32 else ttnn.typecast(y, ttnn.float32, memory_config=memory_config)
        if y_f is not y:
            ttnn.deallocate(y)

        # Place this chunk at unpadded global coords [out_start : out_start + chunk_out] (zeros else).
        # TILE can't front-pad on device, so pad in ROW_MAJOR (which supports it), then back to TILE.
        out_start = chunk_start * S
        chunk_out = int(y_f.shape[1])
        right = out_H_unpadded - out_start - chunk_out
        if out_start == 0 and right == 0:
            y_placed = y_f
        else:
            y_rm = ttnn.to_layout(y_f, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
            ttnn.deallocate(y_f)
            y_pad_rm = ttnn.pad(
                y_rm, padding=[(0, 0), (out_start, right), (0, 0)], value=0.0, memory_config=memory_config
            )
            ttnn.deallocate(y_rm)
            y_placed = ttnn.to_layout(y_pad_rm, ttnn.TILE_LAYOUT, memory_config=memory_config)
            ttnn.deallocate(y_pad_rm)

        if acc is None:
            acc = y_placed
        else:
            new_acc = ttnn.add(acc, y_placed, memory_config=memory_config)
            ttnn.deallocate(acc)
            ttnn.deallocate(y_placed)
            acc = new_acc

    # Global trim: remove P from each end. When P==0 and out_H==out_H_unpadded the trim is a no-op —
    # a full-range ``ttnn.slice`` would *alias* acc and freeing acc would invalidate it — so return acc
    # directly in that (generator ups) case.
    if P == 0 and out_H == out_H_unpadded:
        trimmed = acc
    else:
        trimmed = ttnn.slice(acc, [0, P, 0], [B, P + out_H, C_out], [1, 1, 1], memory_config=memory_config)
        ttnn.deallocate(acc)

    # Apply bias once on device (excluded from chunk_params); broadcast [1,1,C_out] over rows.
    # ``params.bias`` is a host ROW_MAJOR tensor, so it must be uploaded — a host->device write. Under
    # trace prep the uploaded fp32 device bias is cached (keyed by the stable host-bias id) and reused.
    if params.bias is not None:
        _bkey = (id(params.bias), "ct_ola_bias", C_out, str(memory_config))
        _own_bias = not _trace_weight_prep_enabled()
        bias_dev = _prep_cache_get(_bkey) if _trace_weight_prep_enabled() else None
        if bias_dev is None:
            bias_dev = ttnn.to_device(params.bias, device, memory_config=memory_config)
            bias_dev = ttnn.to_layout(bias_dev, ttnn.TILE_LAYOUT, memory_config=memory_config)
            bias_dev = ttnn.reshape(bias_dev, [1, 1, C_out], memory_config=memory_config)
            if bias_dev.dtype != ttnn.float32:
                bias_dev = ttnn.typecast(bias_dev, ttnn.float32, memory_config=memory_config)
            if _trace_weight_prep_enabled():
                _prep_cache_set(_bkey, bias_dev)
        trimmed_b = ttnn.add(trimmed, bias_dev, memory_config=memory_config)
        ttnn.deallocate(trimmed)
        if _own_bias:
            ttnn.deallocate(bias_dev)
        trimmed = trimmed_b

    upload_dtype = out_dtype if out_dtype is not None else ttnn.bfloat16
    if trimmed.dtype != upload_dtype:
        out = ttnn.typecast(trimmed, upload_dtype, memory_config=memory_config)
        ttnn.deallocate(trimmed)
        return out
    return trimmed


def tt_conv_transpose1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTConvTranspose1dParams,
    device,
    conv_config: Optional[ttnn.Conv2dConfig] = None,
    compute_config=None,
    out_dtype=ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """1D transpose conv via ``conv_transpose2d`` (NHWC ``[N,H,W,C]`` staging inside the op).

    When the expected output length falls in TTNN's broken range (96, 194) the input is
    zero-padded on-device to push the output to 194 (just outside the range), then sliced
    back to the desired length.  Zero-padding is safe because each output position in
    [0, out_H) only receives contributions from input positions already present in the
    original (unpadded) tensor; the extra zero columns add nothing.
    """
    bsz = int(x_nlc.shape[0])
    seq = int(x_nlc.shape[1])

    # Broken-range check: ttnn.conv_transpose2d produces wrong results when the spatial output
    # dimension lands in (96, 194).  Pad input on-device to push output to 194, then slice.
    _TRANSPOSE_BROKEN_MAX = 96
    _TRANSPOSE_BROKEN_MIN = 194
    if params.spatial_style == "height":
        out_H = (seq - 1) * params.stride + params.kernel_size - 2 * params.padding + params.output_padding
        if out_H >= _L1_CONV_TRANSPOSE_HEIGHT_MAX_OUT:
            return _chunked_tt_conv_transpose1d_height(
                x_nlc=x_nlc,
                params=params,
                device=device,
                chunk_size=_CONV_TRANSPOSE_HEIGHT_CHUNK,
                memory_config=memory_config,
                out_dtype=out_dtype,
            )
        if _TRANSPOSE_BROKEN_MAX < out_H < _TRANSPOSE_BROKEN_MIN:
            numerator = _TRANSPOSE_BROKEN_MIN - params.kernel_size + 2 * params.padding - params.output_padding
            target_seq = (numerator + params.stride - 1) // params.stride + 1
            pad_len = target_seq - seq
            if x_nlc.layout != ttnn.TILE_LAYOUT:
                x_nlc = ttnn.to_layout(x_nlc, ttnn.TILE_LAYOUT, memory_config=memory_config)
            # ttnn.pad (one op) vs concat([x, zeros], dim=1) (untilize+untilize+concat). Bit-identical.
            x_padded = ttnn.pad(x_nlc, padding=[(0, 0), (0, pad_len), (0, 0)], value=0.0, memory_config=memory_config)
            y_full = tt_conv_transpose1d_nlc(
                x_nlc=x_padded,
                params=params,
                device=device,
                compute_config=compute_config,
                out_dtype=out_dtype,
                memory_config=memory_config,
            )
            C_out = int(y_full.shape[-1])
            y = ttnn.slice(y_full, [0, 0, 0], [bsz, out_H, C_out], [1, 1, 1], memory_config=memory_config)
            ttnn.deallocate(y_full)
            return y

    # Keep NLC activations in TILE; request TILE output so downstream AdaIN/concat stays TILE
    # (avoids explicit untilize→ROW_MAJOR→conv→tilize on the hot prosody upsample path).
    if x_nlc.layout != ttnn.TILE_LAYOUT:
        x_nlc = ttnn.to_layout(x_nlc, ttnn.TILE_LAYOUT, memory_config=memory_config)
    x = ttnn.reshape(x_nlc, (x_nlc.shape[0], 1, x_nlc.shape[1], x_nlc.shape[2]), memory_config=memory_config)
    # Large height-style sequences (generator upsample at max phoneme length) need DRAM
    # slicing.  With dram_slice_config, requesting output_layout=TILE_LAYOUT causes TTNN
    # to pre-allocate the full output as L1-sharded, overflowing BH L1 (1.5 MiB/bank).
    # Skip output_layout for that path; we convert to TILE in DRAM afterward.
    _use_dram_slice = _conv_transpose_use_dram_height_slice(
        spatial_style=params.spatial_style,
        seq=seq,
        channels=params.in_channels,
        activation_dtype=x_nlc.dtype,
    )
    if conv_config is None:
        conv_config = ttnn.Conv2dConfig(weights_dtype=params.weight.dtype)
        # Keep conv-transpose configuration/bias tensors in DRAM and free activations eagerly
        # to reduce pressure on L1_SMALL (the generator path can otherwise OOM on BH).
        conv_config.config_tensors_in_dram = True
        conv_config.deallocate_activation = True
        if not _use_dram_slice:
            conv_config.output_layout = ttnn.TILE_LAYOUT
        conv_config.enable_act_double_buffer = False
        if params.out_channels >= 256 or params.kernel_size >= 7:
            conv_config.force_split_reader = True
    if compute_config is None:
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True
        )

    if params.spatial_style == "height":
        # Matches ``ttnn_adain_resblk_encode._TtDepthwiseConvTransposePool`` (Kokoro istftnet pool).
        _height_slice_cfg = (
            dram_height_slice_config(
                seq,
                channels=params.in_channels,
                activation_dtype=x_nlc.dtype,
            )
            if _use_dram_slice
            else None
        )
        # Prepared-weight reuse on both the sliced and non-sliced height paths (the chunked generator
        # ups take the DRAM-sliced path at real sequence lengths). ``seq`` and the slice flag are in
        # the key so a given (seq, slice) reuses its prepared weight; the prepared format is a
        # function of those, so reuse across forwards/chunks is safe.
        _sig = (
            _prep_signature(x, out_dtype, extra=("ctH", seq, _height_slice_cfg is not None))
            if _trace_weight_prep_enabled()
            else None
        )
        _w, _b, _want_wb = _weights_for_conv(params, _sig)
        _res = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=_w,
            in_channels=params.in_channels,
            out_channels=params.out_channels,
            device=device,
            bias_tensor=_b,
            kernel_size=(params.kernel_size, 1),
            stride=(params.stride, 1),
            padding=(params.padding, 0),
            output_padding=(params.output_padding, 0),
            dilation=(1, 1),
            batch_size=bsz,
            input_height=seq,
            input_width=1,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=params.groups,
            dtype=out_dtype,
            dram_slice_config=_height_slice_cfg,
            mirror_kernel=params.mirror_kernel,
            return_output_dim=True,
            return_weights_and_bias=_want_wb,
        )
        if _want_wb:
            y, out_hw, _wb = _res
            _prep_cache_set((id(params.weight), _sig), (_wb[0], _wb[1]))
        else:
            y, out_hw = _res
        oh, ow = int(out_hw[0]), int(out_hw[1])
        flat = oh * ow
        y = ttnn.reshape(y, (y.shape[0], flat, y.shape[3]), memory_config=memory_config)
        # DRAM-sliced path omits output_layout=TILE in conv_config to avoid L1 OOM;
        # convert to TILE in DRAM here so downstream ops (add, leaky_relu) see TILE layout.
        if _use_dram_slice and y.layout != ttnn.TILE_LAYOUT:
            y = ttnn.to_layout(y, ttnn.TILE_LAYOUT, memory_config=memory_config)
        return y

    _sig = _prep_signature(x, out_dtype, extra=("ctW", seq)) if _trace_weight_prep_enabled() else None
    _w, _b, _want_wb = _weights_for_conv(params, _sig)
    _res = ttnn.conv_transpose2d(
        input_tensor=x,
        weight_tensor=_w,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        device=device,
        bias_tensor=_b,
        kernel_size=(1, params.kernel_size),
        stride=(1, params.stride),
        padding=(0, params.padding),
        output_padding=(0, params.output_padding),
        dilation=(1, 1),
        batch_size=bsz,
        input_height=1,
        input_width=seq,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=params.groups,
        dtype=out_dtype,
        mirror_kernel=params.mirror_kernel,
        return_output_dim=True,
        return_weights_and_bias=_want_wb,
    )
    if _want_wb:
        y, out_hw, _wb = _res
        _prep_cache_set((id(params.weight), _sig), (_wb[0], _wb[1]))
    else:
        y, out_hw = _res
    y = ttnn.reshape(y, (y.shape[0], out_hw[1], y.shape[3]), memory_config=memory_config)
    return y
