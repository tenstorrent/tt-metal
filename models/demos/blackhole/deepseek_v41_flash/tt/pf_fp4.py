# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused fp4 (e2m1, per-32 e8m0 scale) quantise-dequantise of a bf16 TILE tensor [..., 128] in ONE generic_op (replaces the ~170 ttnn ops of ``pf_tune.fp4_fast``).
Same values (every step is an exact power-of-two scaling or an e2m1 grid value); DSV41_PFA_FP4=fused selects it (tt/pf_tune.py)."""

import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pf_kernels")
NB = 4  # tiles per block == the 4 column tiles of one 128-wide row


def fp4_fused(x, n_cores=None):
    assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
    shp = [int(s) for s in x.shape]
    assert shp[-1] % 128 == 0 and shp[-1] // 32 == NB
    n_tiles = 1
    for s in shp[:-2]:
        n_tiles *= s
    n_tiles *= (shp[-2] // 32) * (shp[-1] // 32)
    nblk = n_tiles // NB
    mesh = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shp), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = mesh.compute_with_storage_grid_size()
    ncores = n_cores or int(os.environ.get("DSV41_PFA_FP4_CORES", "110"))
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    ncores = min(ncores, len(allc), nblk)
    cores = allc[:ncores]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    base, rem = divmod(nblk, ncores)
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    b0 = 0
    for k, (cx, cy) in enumerate(cores):
        n = base + (1 if k < rem else 0)
        rt[cx][cy] = [b0, n]
        crt[cx][cy] = [b0, n]
        b0 += n

    def cb(idx, pages):
        return ttnn.CBDescriptor(
            total_size=pages * 2048,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=2048)],
        )

    CB_X, CB_ONE, CB_ABS, CB_INV, CB_SC, CB_Q, CB_OUT = range(7)
    cbs = [
        cb(CB_X, 2 * NB),
        cb(CB_ONE, 1),
        cb(CB_ABS, NB),
        cb(CB_INV, NB),
        cb(CB_SC, NB),
        cb(CB_Q, NB),
        cb(CB_OUT, 2 * NB),
    ]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/fp4_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_X, CB_ONE, NB] + acc(x),
        runtime_args=rt,
        common_runtime_args=[x.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/fp4_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_OUT, NB] + acc(out),
        runtime_args=rt,
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/fp4_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_X, CB_ONE, CB_ABS, CB_INV, CB_SC, CB_Q, CB_OUT, NB],
        runtime_args=crt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x6F4 << 40) | (hash((nblk, ncores, tuple(acc(x)), tuple(acc(out)))) & ((1 << 40) - 1))
    return ttnn.generic_op([x, out], prog)
