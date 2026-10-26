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
X_CB, COS_CB, SIN_CB, TRANS_CB, SCALER_CB, G_CB, OUT_CB = 0, 1, 2, 3, 4, 5, 16
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


def rope_qk_norm(q, k, cos, sin, trans, q_gain, k_gain, *, epsilon: float, math_fidelity=ttnn.MathFidelity.HiFi4):
    """(qn, kn) = RMSNorm(RoPE(q)) * q_gain, RMSNorm(RoPE(k)) * k_gain for bf16 interleaved tile tensors q, k of
    shape (R, H, S, W); cos, sin (1, 1, S, W); trans (1, 1, 32, 32); gains (1, W)."""
    tensors = (("q", q), ("k", k), ("cos", cos), ("sin", sin), ("trans", trans), ("q_gain", q_gain), ("k_gain", k_gain))
    for name, t in tensors:
        if t.layout != ttnn.TILE_LAYOUT or t.is_sharded() or t.dtype != ttnn.bfloat16:
            raise ValueError(f"rope_qk_norm: {name} must be bf16 interleaved TILE, got {t.dtype} {t.memory_config()}")
    if tuple(q.padded_shape) != tuple(k.padded_shape):
        raise ValueError(f"rope_qk_norm: q and k differ, {q.shape} vs {k.shape}")
    r, h, s, width = q.padded_shape
    if (
        q.shape[-1] != width
        or tuple(cos.padded_shape)[-2:] != (s, width)
        or tuple(sin.padded_shape) != tuple(cos.padded_shape)
    ):
        raise ValueError(f"rope_qk_norm: shapes {q.shape}, cos {cos.shape}, sin {sin.shape}")
    wt, st, rh = width // TILE, s // TILE, r * h
    if wt > 4:
        raise ValueError("rope_qk_norm: a row must fit DST (W <= 128)")
    device = q.device()
    mem = ttnn.DRAM_MEMORY_CONFIG
    qn = ttnn.allocate_tensor_on_device(q.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    kn = ttnn.allocate_tensor_on_device(k.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    cores, work = _split(device, st * 2 * rh)
    bf, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        _cb(X_CB, 2 * wt, bf, cores),
        _cb(COS_CB, 2 * wt, bf, cores),
        _cb(SIN_CB, 2 * wt, bf, cores),
        _cb(TRANS_CB, 1, bf, cores),
        _cb(SCALER_CB, 1, bf, cores),
        _cb(G_CB, 2 * wt, bf, cores),
        _cb(OUT_CB, 2 * wt, bf, cores),
        _cb(ROT_CB, wt, bf, cores),
        _cb(Y_CB, wt, f32, cores),
        _cb(SQ_CB, wt, f32, cores),
        _cb(RSTD_CB, 1, f32, cores),
        _cb(T_CB, wt, f32, cores),
    ]
    ins = (q, k, cos, sin, trans, q_gain, k_gain)
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "reader_rope_qk_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, st, rh, *(a for t in ins for a in _accessor(t))],
        runtime_args=[((c.x, c.y), [*(t.buffer_address() for t in ins), n, s0]) for c, n, s0 in work],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "writer_rope_qk_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, st, rh, *_accessor(qn), *_accessor(kn)],
        runtime_args=[((c.x, c.y), [qn.buffer_address(), kn.buffer_address(), n, s0]) for c, n, s0 in work],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "compute_rope_qk_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, rh, _f32_bits(1.0 / width), _f32_bits(epsilon)],
        runtime_args=[((c.x, c.y), [n, s0]) for c, n, s0 in work],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True),
    )
    ttnn.generic_op([*ins, qn, kn], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
    return qn, kn
