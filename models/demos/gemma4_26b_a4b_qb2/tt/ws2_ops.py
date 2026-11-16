# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 program builders for the WS2 decode path (perf/WS2.md).

``qkv_heads_norm_rope_b1`` runs the models/demos/gemma4 ``QkvHeadsNormRope`` kernels unchanged with one
difference in placement: at batch 1 the three (user, tensor) units run on three cores and the v shard is
written to its own core (1, 0) instead of sharing core (0, 0) with q and k. The k and v shards then sit on
different cores, which is what ``paged_fused_update_cache`` needs to write both caches in one program.
"""

import struct

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid, core_range_set, row_major_cores
from models.demos.gemma4.custom_ops.qkv_heads_norm_rope.op import KERNEL_DIR, TILE, QkvHeadsNormRope


def _one_core_heads_memory(core, head_dim):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(core, core)}), [TILE, head_dim], ttnn.ShardOrientation.ROW_MAJOR
        ),
    )


def qkv_heads_norm_rope_b1(x, hq, hkv, head_dim, gq, gk, cos, sin, eps, v_core=ttnn.CoreCoord(1, 0)):
    """Same contract as ``QkvHeadsNormRope.op`` for one user, with v sharded on ``v_core``."""
    cls = QkvHeadsNormRope
    D = int(head_dim)
    assert x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and not x.is_sharded()
    assert x.memory_config().buffer_type == ttnn.BufferType.L1
    shape = [int(v) for v in x.shape]
    assert shape[:3] == [1, 1, 1] and shape[3] == (hq + 2 * hkv) * D, shape
    assert D % (2 * TILE) == 0
    cols_t, half_t = D // TILE, D // TILE // 2
    device = x.device()
    grid = compute_grid(device)
    cores = row_major_cores(grid, 3)  # one tensor per core: q, k, v
    core_set = core_range_set(cores)
    home = ttnn.CoreCoord(0, 0)
    memories = (_one_core_heads_memory(home, D), _one_core_heads_memory(home, D), _one_core_heads_memory(v_core, D))
    outs = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, h, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory)
        for h, memory in zip((hq, hkv, hkv), memories)
    ]
    targets = [device.worker_core_from_logical_core(c) for c in (home, home, v_core)]
    tile_bytes, f32_tile = TILE * TILE * 2, TILE * TILE * 4

    def cb(idx, pages, page=tile_bytes, fmt=ttnn.bfloat16):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(cls.CB_X, cols_t),
        cb(cls.CB_OUT, cols_t),
        cb(cls.CB_G, cols_t),
        cb(cls.CB_COS, cols_t),
        cb(cls.CB_SIN, cols_t),
        cb(cls.CB_SCALER, 1 + 2 * cols_t),
        cb(cls.CB_SQ, cols_t, page=f32_tile, fmt=ttnn.float32),
        cb(cls.CB_INV, 1, page=f32_tile, fmt=ttnn.float32),
        cb(cls.CB_N, cols_t),
    ]
    reader_ct = [cls.CB_X, cls.CB_G, cls.CB_COS, cls.CB_SIN, cls.CB_SCALER, cols_t, hq, hkv]
    for t in (x, gq, gk, cos, sin):
        reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    writer_ct = [cls.CB_OUT, cols_t]
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    inv_d_bits = struct.unpack("<I", struct.pack("<f", 1.0 / D))[0]
    compute_ct = [
        cls.CB_X, cls.CB_OUT, cls.CB_G, cls.CB_COS, cls.CB_SIN, cls.CB_SCALER, cls.CB_SQ, cls.CB_INV, cols_t, half_t,
        eps_bits, inv_d_bits, cls.CB_N,
    ]  # fmt: skip
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    out_addrs = [o.buffer_address() for o in outs]
    for t, core in enumerate(cores):
        reader_rt[core.x][core.y] = [
            x.buffer_address(), gq.buffer_address(), gk.buffer_address(), cos.buffer_address(),
            sin.buffer_address(), 0, t, 1,
        ]  # fmt: skip
        writer_rt[core.x][core.y] = [targets[t].x, targets[t].y, t, 1] + out_addrs
        compute_rt[core.x][core.y] = [t, 1]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "reader_qkv_heads_norm_rope.cpp"),
            core_ranges=core_set,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer_qkv_heads_norm_rope.cpp"),
            core_ranges=core_set,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "compute_qkv_heads_norm_rope.cpp"),
            core_ranges=core_set,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
    ]
    ttnn.generic_op([x, gq, gk, cos, sin] + outs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return outs[0], outs[1], outs[2]
