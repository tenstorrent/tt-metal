# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 decode attention prologue as one generic_op (Laguna; kernels/ap1_*.cpp).

Replaces the head split of the fused qkv matmul output, the per-head q / k RMSNorms, the HF rotate_half RoPE and
the layout moves between them (~16 ops). Core 0 builds q, core 1 builds k and copies v: each gathers row 0 of the
qkv output into a head-major [32 (heads), 128] block, then RMSNorm + RoPE run on the tile engine."""

import struct
from pathlib import Path

import torch
import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"


def _tile_bytes(dt):
    return {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}[dt]


def _cb(grid, i, fmt, page, pages):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
    )


def reduce_scaler(device, head_dim, mesh_mapper=None):
    """bf16 tile of 1 / head_dim (the row-reduce scaler: the reduced sum is the mean)."""
    return ttnn.from_torch(
        torch.full((1, 1, 32, 32), 1.0 / head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        mesh_mapper=mesh_mapper,
    )


def attn_prologue1(qkv, cos, sin, q_norm, k_norm, scaler, nq, nkv, rd, eps, k_mem, v_mem):
    """qkv: [.., 32, W] bf16 TILE (row 0 real; [q heads | k heads | v heads | ...], head_dim 128); cos / sin:
    [1, 1, 1, rd] bf16 TILE rows; q_norm / k_norm: [1, 1, 1, 128] TILE weight rows. Returns q [1, 1, nq, 128]
    (DRAM), k [1, 1, nkv, 128] in k_mem and v in v_mem (the KV-update height shards)."""
    device = qkv.device()
    hd = 128
    assert qkv.dtype == ttnn.bfloat16 and qkv.padded_shape[-1] >= (nq + 2 * nkv) * hd and nq <= 32 and nkv <= 32
    assert rd in (64, 128), rd
    q = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, nq, hd]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
                                       ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, nkv, hd]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, k_mem)
    v = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, nkv, hd]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, v_mem)
    rd_tiles = rd // 32
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    w_page, cs_page = _tile_bytes(q_norm.dtype), _tile_bytes(cos.dtype)
    kernels, cbs = [], []
    for role, (core, w, nh, off, out) in enumerate(
        (((0, 0), q_norm, nq, 0, q), ((1, 0), k_norm, nkv, nq * 4, k))
    ):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))})
        acc = []
        for t in (qkv, w, cos, sin, scaler):
            acc.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "ap1_reader.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[role, nh, off, (nq + nkv) * 4, rd_tiles, w_page, cs_page] + acc,
            common_runtime_args=[qkv.buffer_address(), cos.buffer_address(), sin.buffer_address(),
                                 scaler.buffer_address(), w.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        ))
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "ap1_writer.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[role] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(v).get_compile_time_args()),
            common_runtime_args=[out.buffer_address(), v.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        ))
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.fp32_dest_acc_en = True
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "ap1_compute.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[rd_tiles, eps_bits],
            config=cfg,
        ))
        cbs += [
            _cb(grid, 0, ttnn.bfloat16, 2048, 4),
            _cb(grid, 1, w.dtype, w_page, 4),
            _cb(grid, 2, cos.dtype, cs_page, rd_tiles),
            _cb(grid, 3, sin.dtype, cs_page, rd_tiles),
            _cb(grid, 4, ttnn.bfloat16, 2048, 1),
            _cb(grid, 5, ttnn.bfloat16, 2048, 4),
            _cb(grid, 16, ttnn.bfloat16, 2048, 4),
            _cb(grid, 24, ttnn.float32, 4096, 4),
            _cb(grid, 25, ttnn.float32, 4096, 1),
            _cb(grid, 26, ttnn.float32, 4096, 4),
            _cb(grid, 27, ttnn.bfloat16, 2048, 4),
        ]
    ttnn.generic_op(
        [qkv, q_norm, k_norm, cos, sin, scaler, q, k, v],
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
    )
    return q, k, v
