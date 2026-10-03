# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mHC expand (one generic_op): new_j = post_j * y + sum_i comb[i,j] x_i for x [T,1,n,D] fp32, y [T,1,1,D],
post [T,1,n,1], comb [T,1,n,n] (n = 4): one fp32 tile matmul per (token, 32-column tile)."""

import os

import ttnn

KDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mhc_kernels"
)  # absolute: the overlay package must use ITS kernels


def mhc_expand(y, x, post, comb, y2=None, n_cores=None):
    T, _, n, D = (int(v) for v in x.shape)
    assert n == 4 and T <= 32
    for t in (x, post, comb) + ((y2,) if y2 is not None else ()):
        assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT
    assert y.layout == ttnn.TILE_LAYOUT and y.dtype in (ttnn.float32, ttnn.bfloat16)
    y_row = int(y.shape[2]) == T  # [1,1,T,D] token rows (else [T,1,1,D])
    y_bf16 = y.dtype == ttnn.bfloat16
    has_y2 = y2 is not None
    assert not has_y2 or int(y2.shape[2]) == T and int(y2.shape[0]) == 1
    y2t = y2 if has_y2 else y
    nt = D // 32
    if n_cores is None:
        n_cores = 32 if T <= 4 else (40 if T <= 8 else 80)
    mesh = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([T, 1, n, D]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:n_cores]
    while nt % len(cores):
        cores.pop()
    gpc = nt // len(cores)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [k * gpc, (k + 1) * gpc]
        crt[cx][cy] = [gpc]
    CB_A, CB_B, CB_S, CB_O = 0, 1, 2, 3
    nb = gpc * T

    def cb(idx, pages, page, dt):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
        )

    cbs = [
        cb(CB_A, T, 4096, ttnn.float32),
        cb(CB_B, nb, 4096, ttnn.float32),
        cb(CB_S, max(2, (gpc * T * 128 + 4095) // 4096), 4096, ttnn.float32),
        cb(CB_O, nb, 4096, ttnn.float32),
    ]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/mhc_expand_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_A, CB_B, CB_S, T, nt, int(y_row), int(y_bf16), int(has_y2)]
        + acc(x)
        + acc(y)
        + acc(y2t)
        + acc(comb)
        + acc(post),
        runtime_args=rt,
        common_runtime_args=[
            x.buffer_address(),
            y.buffer_address(),
            comb.buffer_address(),
            post.buffer_address(),
            y2t.buffer_address(),
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/mhc_expand_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_O, T, nt] + acc(out),
        runtime_args=rt,
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/mhc_expand_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_A, CB_B, CB_O, T],
        runtime_args=crt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x5E2 << 40) | (
        hash(
            (
                T,
                n,
                D,
                len(cores),
                y_row,
                y_bf16,
                has_y2,
                tuple(acc(y2t)),
                tuple(acc(x)),
                tuple(acc(y)),
                tuple(acc(comb)),
                tuple(acc(post)),
                tuple(acc(out)),
            )
        )
        & ((1 << 40) - 1)
    )
    return ttnn.generic_op([x, y, comb, post, out], prog)
