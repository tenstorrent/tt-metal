# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Residual add + RMSNorm in one ``ttnn.generic_op`` (ported from pplx_embed_4b/tt/custom_ops).

``sum = a + b`` (the next residual) and ``normed = rmsnorm(sum) * gamma`` come out of one
pass. Separately, ``ttnn.add`` reads a and b and writes the sum, and ``rms_norm`` reads the
sum again and writes the normalized tensor: 5 passes over [M, W]. Fused, a and b are read
once and both outputs written once: 4 passes, and one launch instead of two. Per tile-row:

    sum = a + b              -> sum output CB + a working copy (``norm_dtype``) for the norm
    x2  = sum * sum
    ms  = row-sum(x2) / W    (reduce with a 1/W scaler tile)
    inv = rsqrt(ms + eps)
    out = (sum * bcast_cols(inv)) * gamma

Rows are split across cores; each core streams whole tile-rows.
"""

import os

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops._common import (
    TILE_BYTES,
    core_ranges,
    split_work_to_cores,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNEL = os.path.join(_HERE, "kernels", "reader_add_rmsnorm.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_add_rmsnorm.cpp")
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_add_rmsnorm.cpp")
_BF16_TILE = TILE_BYTES[ttnn.bfloat16]
_CHUNK = 4  # DST tiles per session with fp32 accumulation (half-sync)
TILE = 32


def make_add_norm_constants(gamma: torch.Tensor, eps: float, device):
    """gamma ``[W]`` -> (gamma as row-replicated tiles ``[1, 1, 32, W]``, 1/W scaler tile, eps tile), bf16 on device."""
    width = int(gamma.numel())
    rep = gamma.to(torch.float32).reshape(1, 1, 1, width).expand(1, 1, TILE, width).contiguous()
    gamma_tiles = ttnn.from_torch(rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def const(val):
        t = torch.full((1, 1, TILE, TILE), float(val), dtype=torch.float32)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return gamma_tiles, const(1.0 / width), const(eps)


def fused_add_rmsnorm(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    consts,
    *,
    sum_dtype=None,
    out_dtype=None,
    norm_dtype=ttnn.bfloat16,
    memory_config=None,
    out_memory_config=None,
    inplace=False,
    max_buffers=2,
    fidelity=ttnn.MathFidelity.HiFi4,
    defines=(),
):
    """``a``, ``b``: ``[..., M, W]`` TILE interleaved. Returns ``(a + b, rmsnorm(a + b) * gamma)``.

    ``inplace``: the sum is written over ``a`` (each core reads a row before it writes that row, and
    no two cores share a row), so the residual stream needs one buffer, e.g. resident in L1.
    ``out_memory_config`` places the normed output (default: ``memory_config``)."""
    gamma_tiles, scaler_tile, eps_tile = consts
    memory_config = memory_config or a.memory_config()
    device = a.device()
    shape = list(a.padded_shape)
    Wt = shape[-1] // TILE
    assert Wt % _CHUNK == 0 and shape[-2] % TILE == 0 and list(b.padded_shape) == shape
    rows_t = 1
    for d in shape[:-1]:
        rows_t *= int(d)
    rows_t //= TILE
    sum_dtype = sum_dtype or a.dtype
    out_dtype = out_dtype or sum_dtype

    if inplace:
        assert sum_dtype == a.dtype, "in-place sum keeps a's dtype"
        sum_tensor = a
    else:
        sum_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), sum_dtype, ttnn.TILE_LAYOUT, device, memory_config
        )
    out_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, device, out_memory_config or memory_config
    )

    grid = device.compute_with_storage_grid_size()
    _, per_core = split_work_to_cores(rows_t, int(grid.x), int(grid.y))
    used_cores = core_ranges(per_core)
    # Double-buffer the streamed rows only when a core gets more than one; at bs1 each core
    # gets one row and the second buffer would only take L1 from interleaved L1 activations.
    nbuf = min(max_buffers, 2) if max(n for _, _, n in per_core) > 1 else 1

    def cb(index, tiles, dtype):
        tsize = TILE_BYTES[dtype]
        return ttnn.CBDescriptor(
            total_size=tiles * tsize,
            core_ranges=used_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tsize)],
        )

    cbs = [
        cb(0, Wt * nbuf, a.dtype),  # a rows
        cb(1, Wt * nbuf, b.dtype),  # b rows
        cb(2, Wt, ttnn.bfloat16),  # gamma tiles (resident)
        cb(3, 1, ttnn.bfloat16),  # 1/W scaler
        cb(4, 1, ttnn.bfloat16),  # eps
        cb(5, Wt, norm_dtype),  # working copy of the sum the norm reads
        cb(6, Wt, ttnn.bfloat16),  # sum^2
        cb(7, 1, ttnn.bfloat16),  # mean square
        cb(8, 1, ttnn.bfloat16),  # rsqrt
        cb(16, Wt * nbuf, sum_dtype),  # sum out
        cb(17, Wt * nbuf, out_dtype),  # normed out
    ]

    reader_ct = [Wt]
    for t in (a, b, gamma_tiles, scaler_tile, eps_tile):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [Wt, _CHUNK]
    writer_ct = [Wt]
    for t in (sum_tensor, out_tensor):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_rt, compute_rt, writer_rt = [], [], []
    cursor = 0
    for cx, cy, n_rows in per_core:
        core = (cx, cy)
        reader_rt.append(
            (
                core,
                [
                    a.buffer_address(),
                    b.buffer_address(),
                    gamma_tiles.buffer_address(),
                    scaler_tile.buffer_address(),
                    eps_tile.buffer_address(),
                    n_rows,
                    cursor,
                ],
            )
        )
        compute_rt.append((core, [n_rows]))
        writer_rt.append((core, [sum_tensor.buffer_address(), out_tensor.buffer_address(), n_rows, cursor]))
        cursor += n_rows

    def kernel(src, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=used_cores,
            compile_time_args=ct,
            runtime_args=rt,
            defines=list(defines),
            config=config,
        )

    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel(READER_KERNEL, reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
            kernel(
                COMPUTE_KERNEL,
                compute_ct,
                compute_rt,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=fidelity,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                    bfp8_pack_precise=True,
                ),
            ),
            kernel(WRITER_KERNEL, writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
        ],
        cbs=cbs,
    )
    io = [a, b, gamma_tiles, scaler_tile, eps_tile] + ([] if inplace else [sum_tensor]) + [out_tensor]
    ttnn.generic_op(io, program)
    return sum_tensor, out_tensor
