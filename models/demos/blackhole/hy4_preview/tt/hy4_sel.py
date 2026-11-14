# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact threshold selection of the decode indexer (task fuse-sparse-merge, B2; flag HY4_FUSE=idx): JIT kernels on the per-column scores [1,1,S,n] bf16 RM. See RESULT.md (design)."""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_sel_kernels")


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _grid_cores(md, nc):
    g = md.compute_with_storage_grid_size()
    cores = [(i % g.x, i // g.x) for i in range(nc)]
    rngs, left, r = [], nc, 0
    while left > 0:
        w = min(left, g.x)
        rngs.append(ttnn.CoreRange(ttnn.CoreCoord(0, r), ttnn.CoreCoord(w - 1, r)))
        left -= w
        r += 1
    return cores, ttnn.CoreRangeSet(rngs)


def hist1(scores, S, nc=64):
    """microbench / pass 1: u32 [S,256] histogram of the high byte of the sortable key (3-level NoC tree)."""
    md = scores.device()
    n = int(scores.shape[3])
    assert n % (nc * 2) == 0
    seg = n // nc
    cores, cs = _grid_cores(md, nc)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, S, 256]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    CD, CH, R1, R2, R3 = range(5)

    def cb(i, nbytes):
        return ttnn.CBDescriptor(total_size=nbytes, core_ranges=cs, format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=nbytes)])

    slot = S * 512
    cbs = [cb(CD, max(S * seg * 2, S * 1024)), cb(CH, slot), cb(R1, 7 * slot), cb(R2, 3 * slot), cb(R3, max(1, nc // 32 - 1) * slot)]
    rt = ttnn.RuntimeArgs()
    phys = []
    for (x, y) in cores:
        c = md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        phys += [c.x, c.y]
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i] + phys
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/sel_h1.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=[S, n * 2, seg, nc, CD, CH, R1, R2, R3, 0] + _acc(scores) + _acc(out),
        runtime_args=rt,
        common_runtime_args=[scores.buffer_address(), out.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=cs, initial_value=0) for i in range(3)], cbs=cbs)
    prog.custom_program_hash = (0x5E1 << 40) | (hash((S, n, nc, tuple(_acc(scores)), tuple(_acc(out)))) & ((1 << 40) - 1))
    ttnn.generic_op([scores, out], prog)
    return out
