// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// lane_score_rows (data movement only, one core per output row): row `out_row` of the local block scores [1, 1, rows,
// blocks] bf16 ROW_MAJOR = lane `lane`'s window of the wide indexer's scores [1, 1, 32, lanes * cache_rows] (row
// out_row, columns lane * cache_rows .. + blocks): the lane chain's slices and concat as row copies (bytes at the
// 64-byte DRAM grain: cache_rows and blocks are multiples of 32 elements).  The plain lane body passes out_row = lane
// = u; the MTP lanes verify passes lane_of(out_row) for its lane-major rows.  CB 0: the row scratch (blocks * 2 bytes).
// Compile-time args: TensorAccessorArgs scores, out.  Runtime args: 0 scores, 1 out addresses, 2 out_row, 3 lane,
// 4 cache rows per lane, 5 blocks.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_row = get_arg_val<uint32_t>(2);
    const uint32_t lane = get_arg_val<uint32_t>(3);
    const uint32_t cache_rows = get_arg_val<uint32_t>(4);
    const uint32_t blocks = get_arg_val<uint32_t>(5);
    constexpr auto scores_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    const auto scores = TensorAccessor(scores_args, scores_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    cb_reserve_back(0, 1);
    const uint32_t l1 = get_write_ptr(0);
    noc_async_read(scores.get_noc_addr(out_row, lane * cache_rows * 2), l1, blocks * 2);
    noc_async_read_barrier();
    noc_async_write(l1, out.get_noc_addr(out_row, 0), blocks * 2);
    noc_async_write_barrier();
}
