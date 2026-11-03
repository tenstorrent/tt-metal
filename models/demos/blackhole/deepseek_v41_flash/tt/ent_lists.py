# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-program replacement of spec_paged._ent_matrix + _cs_ent_lists + the per-j slice / typecast / to_layout / reshape chain (about 45 ops, 230 us per owner / index layer
type and round on the critical chip): the key-slab update indices of the column-sharded key append, from the step positions only (overlap2)."""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlap_kernels")


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def ent_lists_fused(md, pos, col_u32, U, n, ratio, cols, n_head):
    """pos: int32 [U*n] row-major (one page); col_u32: the per-device column index page (indexer._cs_consts()['col_u32']). -> (vloc list, vhead list), n int32 [U] row-major tensors each."""
    assert U <= 16 and U * n <= 64
    outs = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([U]), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
        for _ in range(2 * n)
    ]
    cset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb = ttnn.CBDescriptor(
        total_size=4096,
        core_ranges=cset,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=4096)],
    )
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/ent_lists.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cset,
        compile_time_args=[0, U, n, ratio, cols, n_head] + _acc(pos) + _acc(col_u32) + _acc(outs[0]),
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[pos.buffer_address(), col_u32.buffer_address()] + [o.buffer_address() for o in outs],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (0x7E2 << 40) | (
        hash((U, n, ratio, cols, n_head, tuple(_acc(pos)), tuple(_acc(col_u32)), tuple(_acc(outs[0]))))
        & ((1 << 40) - 1)
    )
    ttnn.generic_op([pos, col_u32] + outs, prog)
    return outs[:n], outs[n:]
