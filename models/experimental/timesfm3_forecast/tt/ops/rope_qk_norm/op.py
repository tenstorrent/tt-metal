# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R3-2: RoPE (``rotary_embedding_llama``'s pair rotation) and the QK RMSNorm over q and k in one pass, through
``ttnn.generic_op``. Replaces two RoPE calls and two ``rms_norm`` calls (four DRAM round trips of q and k).

One work unit is one tile row of one head (32 tokens x 96). Units run seq-tile-major, so a core reloads cos / sin
only when its seq tile changes. Kernels are model-local; their paths are absolute, from this file.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

TILE = 32
KERNELS = Path(__file__).resolve().parent / "kernels"
X_CB, COS_CB, SIN_CB, TRANS_CB, SCALER_CB, G_CB, V_CB, OUT_CB = 0, 1, 2, 3, 4, 5, 6, 16
ROT_CB, Y_CB, SQ_CB, RSTD_CB, T_CB = 24, 25, 26, 27, 28
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _cb(index, pages, dtype, cores):
    page = _TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _accessor(t) -> list[int]:
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _split(device, n: int):
    grid = device.compute_with_storage_grid_size()
    _, cores, group_1, _, per_1, per_2 = ttnn.split_work_to_cores(grid, n, True)
    n1, work, start = group_1.num_cores(), [], 0
    for i, core in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
        c = per_1 if i < n1 else per_2
        work.append((core, c, start))
        start += c
    return cores, work


def _check(name, t):
    if t.layout != ttnn.TILE_LAYOUT or t.is_sharded() or t.dtype != ttnn.bfloat16:
        raise ValueError(f"rope_qk_norm: {name} must be bf16 interleaved TILE, got {t.dtype} {t.memory_config()}")


def rope_qk_norm(q, k, cos, sin, trans, q_gain, k_gain, *, epsilon: float, math_fidelity=ttnn.MathFidelity.HiFi4):
    """(qn, kn) = RMSNorm(RoPE(q)) * q_gain, RMSNorm(RoPE(k)) * k_gain for bf16 interleaved tile tensors q, k of
    shape (R, H, S, W); cos, sin (1, 1, S, W); trans (1, 1, 32, 32); gains (1, W)."""
    for name, t in (("q", q), ("k", k)):
        _check(name, t)
    if tuple(q.padded_shape) != tuple(k.padded_shape):
        raise ValueError(f"rope_qk_norm: q and k differ, {q.shape} vs {k.shape}")
    r, h, s, width = q.padded_shape
    if q.shape[-1] != width:
        raise ValueError(f"rope_qk_norm: needs a tile-aligned head width, got {q.shape}")
    qn, kn, _ = _launch((q, k), r, h, s, width, cos, sin, trans, q_gain, k_gain, epsilon, math_fidelity, False)
    return qn, kn


def rope_qk_norm_qkv(
    qkv, rows: int, heads: int, cos, sin, trans, q_gain, k_gain, *, epsilon: float, math_fidelity=ttnn.MathFidelity.HiFi4
):
    """R3-5: (qn, kn, v) of shape (R, H, S, W) straight from the fused QKV projection ``qkv`` (R * S rows, 3 * H * W
    columns: the q heads, the k heads, the v heads), with RoPE and the QK norm as ``rope_qk_norm``; replaces the
    head split as well."""
    _check("qkv", qkv)
    n = 1
    for d in list(qkv.padded_shape)[:-1]:
        n *= d
    width = qkv.padded_shape[-1] // (3 * heads)
    if qkv.shape[-1] != qkv.padded_shape[-1] or width % TILE or n % rows or (n // rows) % TILE:
        raise ValueError(f"rope_qk_norm_qkv: shape {qkv.shape} does not split into {rows} rows x {heads} heads")
    return _launch((qkv, qkv), rows, heads, n // rows, width, cos, sin, trans, q_gain, k_gain, epsilon, math_fidelity, True)


def _launch(srcs, r, h, s, width, cos, sin, trans, q_gain, k_gain, epsilon, math_fidelity, from_qkv: bool):
    for name, t in (("cos", cos), ("sin", sin), ("trans", trans), ("q_gain", q_gain), ("k_gain", k_gain)):
        _check(name, t)
    if tuple(cos.padded_shape)[-2:] != (s, width) or tuple(sin.padded_shape) != tuple(cos.padded_shape):
        raise ValueError(f"rope_qk_norm: cos {cos.shape} / sin {sin.shape} do not match S = {s}, W = {width}")
    wt, st, rh = width // TILE, s // TILE, r * h
    if wt > 4:
        raise ValueError("rope_qk_norm: a row must fit DST (W <= 128)")
    device = srcs[0].device()
    mem = ttnn.DRAM_MEMORY_CONFIG
    shape = (r, h, s, width)
    outs = [ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem) for _ in range(2 + from_qkv)]
    groups = 3 if from_qkv else 2
    cores, work = _split(device, st * groups * rh)
    bf, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        _cb(X_CB, 2 * wt, bf, cores),
        _cb(COS_CB, 2 * wt, bf, cores),
        _cb(SIN_CB, 2 * wt, bf, cores),
        _cb(TRANS_CB, 1, bf, cores),
        _cb(SCALER_CB, 1, bf, cores),
        _cb(G_CB, 2 * wt, bf, cores),
        _cb(V_CB, 2 * wt, bf, cores),
        _cb(OUT_CB, 2 * wt, bf, cores),
        _cb(ROT_CB, wt, bf, cores),
        _cb(Y_CB, wt, f32, cores),
        _cb(SQ_CB, wt, f32, cores),
        _cb(RSTD_CB, 1, f32, cores),
        _cb(T_CB, wt, f32, cores),
    ]
    ins = (*srcs, cos, sin, trans, q_gain, k_gain)
    dsts = (outs[0], outs[1], outs[-1])  # without from_qkv the v slot repeats k_out (never written)
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "reader_rope_qk_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, st, rh, h, int(from_qkv), *(a for t in ins for a in _accessor(t))],
        runtime_args=[((c.x, c.y), [*(t.buffer_address() for t in ins), n, s0]) for c, n, s0 in work],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "writer_rope_qk_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, st, rh, int(from_qkv), *(a for t in dsts for a in _accessor(t))],
        runtime_args=[((c.x, c.y), [*(t.buffer_address() for t in dsts), n, s0]) for c, n, s0 in work],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "compute_rope_qk_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, rh, _f32_bits(1.0 / width), _f32_bits(epsilon), int(from_qkv)],
        runtime_args=[((c.x, c.y), [n, s0]) for c, n, s0 in work],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True),
    )
    io = [*ins[1:], *outs] if from_qkv else [*ins, *outs]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
    return outs[0], outs[1], outs[2] if from_qkv else None
