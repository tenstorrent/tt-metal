# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# A REDUCE_SCALAR MAX whose running result is packed and copied back into DST before the next tile, as tt-mlir's D2M
# kernels do across reduction blocks: the copied tile must not clamp an all-negative maximum at 0.

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto data_args = TensorAccessorArgs<0>();
    const auto data = TensorAccessor(data_args, get_arg_val<uint32_t>(0));
    const auto scaler = TensorAccessor(
        TensorAccessorArgs<data_args.next_compile_time_args_offset()>(), get_arg_val<uint32_t>(1));
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t page = get_arg_val<uint32_t>(3);
    Noc noc;
    DataflowBuffer d_in(0);
    DataflowBuffer d_sc(1);
    d_sc.reserve_back(1);
    noc.async_read(scaler, d_sc, page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    d_sc.push_back(1);
    for (uint32_t i = 0; i < n; ++i) {
        d_in.reserve_back(1);
        noc.async_read(data, d_in, page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        d_in.push_back(1);
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto output = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0));
    const uint32_t page = get_arg_val<uint32_t>(1);
    Noc noc;
    DataflowBuffer d_out(16);
    d_out.wait_front(1);
    noc.async_write(d_out, output, page, {}, {.page_id = 0});
    noc.async_write_barrier();
    d_out.pop_front(1);
}
"""

COMPUTE = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t per_call = get_compile_time_arg_val(1);
    constexpr uint32_t in = 0, sc = 1, acc = 2, out = 16;
    DataflowBuffer d_in(in);
    DataflowBuffer d_sc(sc);
    DataflowBuffer d_acc(acc);
    compute_kernel_hw_startup(in, sc, out);
    d_sc.wait_front(1);
    for (uint32_t i = 0; i < n; i += per_call) {
        d_in.wait_front(per_call);
        if (i > 0) {
            d_acc.wait_front(1);
        }
        tile_regs_acquire();
        if (i > 0) {
            reconfig_data_format_srca(acc);
            copy_init(acc);
            copy_tile(acc, 0, 0);
            reconfig_data_format_srca(in);
        }
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(in, sc, acc);
        if constexpr (per_call == 1) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(in, sc, 0, 0, 0);
        } else {
            reduce_block<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(in, sc, 0, 0, 0, per_call, 0);
        }
        reduce_uninit();
        tile_regs_commit();
        d_in.pop_front(per_call);
        if (i > 0) {
            d_acc.pop_front(1);
        }
        const uint32_t ocb = (i + per_call == n) ? out : acc;
        DataflowBuffer d_o(ocb);
        d_o.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(ocb);
        pack_tile(0, ocb);
        tile_regs_release();
        d_o.push_back(1);
    }
    d_sc.pop_front(1);
}
"""


def reduce_max_with_reload(device, data, fp32_dest, per_call, acc_dtype):
    dtype, page = ttnn.bfloat16, 2048
    acc_page = 4096 if acc_dtype == ttnn.float32 else 2048
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    x = ttnn.from_torch(data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    s = ttnn.from_torch(torch.ones((1, 1, 32, 32)), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.from_torch(torch.zeros((1, 1, 32, 32)), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )
        for index, pages in ((0, 4), (1, 1), (16, 2))
    ] + [
        ttnn.CBDescriptor(
            total_size=2 * acc_page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2, data_format=acc_dtype, page_size=acc_page)],
        )
    ]
    reader_args = ttnn.RuntimeArgs()
    reader_args[0][0] = [x.buffer_address(), s.buffer_address(), data.shape[0], page]
    writer_args = ttnn.RuntimeArgs()
    writer_args[0][0] = [y.buffer_address(), page]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=READER,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args()
                + ttnn.TensorAccessorArgs(s).get_compile_time_args(),
                runtime_args=reader_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WRITER,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=ttnn.TensorAccessorArgs(y).get_compile_time_args(),
                runtime_args=writer_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=[data.shape[0], per_call],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_dest
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    ttnn.generic_op([x, s, y], program)
    return ttnn.to_torch(y)[0, 0, 0, 0].item()


@pytest.mark.skipif(not is_blackhole(), reason="the scratch row is cleared by the Blackhole reduce LLK")
@pytest.mark.parametrize("fp32_dest", [False, True])
@pytest.mark.parametrize("num_tiles, per_call", [(2, 1), (4, 1), (4, 2)], ids=["2x1", "4x1", "4_in_blocks_of_2"])
@pytest.mark.parametrize("acc_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16_acc", "fp32_acc"])
@pytest.mark.parametrize("sign", [-1, 1], ids=["all_negative", "mixed"])
def test_reduce_scalar_max_reload(device, num_tiles, per_call, fp32_dest, acc_dtype, sign):
    torch.manual_seed(0)
    if sign < 0:
        data = -(0.5 + torch.rand((num_tiles, 1, 32, 32)))
    else:
        data = torch.rand((num_tiles, 1, 32, 32)) * 2 - 1
    data = data.to(torch.bfloat16)
    got = reduce_max_with_reload(device, data, fp32_dest, per_call, acc_dtype)
    assert got == data.float().max().item()
