# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE tail for T < 32 token rows per device: ONE generic_op replacing ``tilize_with_val_padding`` +
``deepseek_moe_fast_reduce_nc_fused``.

combine [K, T, H] RM bf16 (moe_compute output, slots whose expert is outside this device's mesh column hold stale data) ->
out [1, 1, T, H] TILE bf16 = sum_k mask(t,k) * score(t,k) * combine[k, t, :]   (mask = expert of slot (t,k) is in my column).
The reader builds the activation tiles straight from the RM rows (only rows 0..T-1 are read), the compute kernel is the same
MAC-with-column-broadcast as the fast_reduce kernel, so numerics match the unfused path."""

import os
from pathlib import Path

import ttnn

KDIR = str(Path(__file__).resolve().parent / "moe_tail_kernels")


def make_col_tensor(mesh):
    """uint32 [1,1,1,16] per device: element 0 = this device's mesh column."""
    import torch

    rows, cols = tuple(mesh.shape)
    t = torch.zeros(rows, cols, 1, 64, dtype=torch.int32)
    for c in range(cols):
        t[:, c, 0, 0] = c
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cb(core_set, idx, pages, page, dt):
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
    )


def moe_tail(combine, scores, indices, mapping, col, n_cores=80):
    sh = [int(v) for v in combine.shape]
    K, T, H = sh[0], sh[-2], sh[-1]
    assert T <= 32 and H % 32 == 0
    nt = H // 32
    while nt % n_cores:
        n_cores -= 1
    tpc = nt // n_cores
    mesh = combine.device()
    mesh_cols = int(tuple(mesh.shape)[1])
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:n_cores]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    rt = ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [i * tpc]
    A, S, TMP, O = 0, 1, 2, 16
    bf = ttnn.bfloat16
    map_page = int(mapping.shape[-1]) * 2
    tmp_bytes = K * T * tpc * 64 + 2 * T * 64 + 256 + map_page
    cbs = [
        _cb(core_set, A, K * tpc, 2048, bf),
        _cb(core_set, S, K, 2048, bf),
        _cb(core_set, TMP, 1, (tmp_bytes + 63) // 64 * 64, bf),
        _cb(core_set, O, tpc, 2048, bf),
    ]
    sc_page, idx_page, row_page = K * 2, K * 2, H * 2

    def kern(name, ct, cfg, common):
        return ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/{name}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt if name != "moe_tail_compute.cpp" else ttnn.RuntimeArgs(),
            common_runtime_args=common,
            config=cfg,
            defines=([("TAIL_DBG", "1")] if os.environ.get("DSV41_TAIL_DBG") and name == "moe_tail_reader.cpp" else []),
        )

    reader = kern(
        "moe_tail_reader.cpp",
        [A, S, TMP, T, K, tpc, mesh_cols, sc_page, idx_page, map_page, row_page]
        + _acc(combine)
        + _acc(scores)
        + _acc(indices)
        + _acc(mapping)
        + _acc(col),
        ttnn.ReaderConfigDescriptor(),
        [
            combine.buffer_address(),
            scores.buffer_address(),
            indices.buffer_address(),
            mapping.buffer_address(),
            col.buffer_address(),
        ],
    )
    writer = kern("moe_tail_writer.cpp", [O, tpc] + _acc(out), ttnn.WriterConfigDescriptor(), [out.buffer_address()])
    compute = kern(
        "moe_tail_compute.cpp",
        [tpc, K, A, S, O],
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        [],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x7A1 << 40) | (
        hash(
            (
                bool(os.environ.get("DSV41_TAIL_DBG")),
                T,
                K,
                H,
                n_cores,
                mesh_cols,
                tuple(_acc(combine)),
                tuple(_acc(scores)),
                tuple(_acc(indices)),
                tuple(_acc(mapping)),
                tuple(_acc(col)),
                tuple(_acc(out)),
            )
        )
        & ((1 << 40) - 1)
    )
    ttnn.generic_op([combine, scores, indices, mapping, col, out], prog)
    return out
