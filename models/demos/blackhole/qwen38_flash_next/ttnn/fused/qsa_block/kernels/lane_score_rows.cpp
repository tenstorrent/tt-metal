// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// lane_score_rows (data movement only, one core per output row): row `out_row` of the local block scores [1, 1, rows,
// blocks] bf16 ROW_MAJOR = row out_row, columns lane * cache_rows .. + blocks of the wide indexer's scores
// [1, 1, 32, lanes * cache_rows]. Row copies at the 64-byte DRAM grain (cache_rows and blocks are multiples of 32
// elements).  CB 0 (cb_row): the row scratch (blocks * 2 bytes).
// Compile-time args: TensorAccessorArgs scores, out.  Runtime args: 0 scores, 1 out addresses, 2 out_row, 3 lane,
// 4 cache rows per lane, 5 blocks.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_row = get_arg_val<uint32_t>(2);
    const uint32_t lane = get_arg_val<uint32_t>(3);
    const uint32_t cache_rows = get_arg_val<uint32_t>(4);
    const uint32_t blocks = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_row = 0;
    constexpr auto scores_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    const auto scores = TensorAccessor(scores_args, scores_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const Noc noc;
    CircularBuffer(cb_row).reserve_back(1);
    // cb_row is scratch: reserved once and never pushed
    const CoreLocalMem<uint32_t> row_l1(CircularBuffer(cb_row).get_write_ptr());
    noc.async_read(scores, row_l1, blocks * 2, {.page_id = out_row, .offset_bytes = lane * cache_rows * 2}, {});
    noc.async_read_barrier();
    noc.async_write(row_l1, out, blocks * 2, {}, {.page_id = out_row, .offset_bytes = 0});
    noc.async_write_barrier();
}
