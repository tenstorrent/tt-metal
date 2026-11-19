// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Single-row append to the prefill stack's packed KV storage (data movement, one core per chip).
//
// For every user u: position p_u (from a device tensor, so a traced program follows the context) and one packed
// row (ROW_BYTES: FP8 latent | FP32 scales | BF16 RoPE). The row is written to this chip's storage only when the
// chip owns the position under the block-cyclic stripe of the cache:
//     chip(p) = (p / BLOCK) % CHIPS        row(p) = (p / (BLOCK * CHIPS)) * BLOCK + p % BLOCK
// at page (u * SLOT_STRIDE + layer) * R + row(p) (sm_common.hpp kv_row_*). Rows of other chips, and positions
// beyond the cache, are skipped. One NoC read and one NoC write per appended row; nothing else is touched.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "sm_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_row = get_named_compile_time_arg_val("cb_row");
    constexpr uint32_t cb_pos = get_named_compile_time_arg_val("cb_pos");
    constexpr uint32_t USERS = get_named_compile_time_arg_val("users");
    constexpr uint32_t ROW_BYTES = get_named_compile_time_arg_val("row_bytes");
    constexpr uint32_t STRIDE = get_named_compile_time_arg_val("row_stride");
    constexpr uint32_t ROWS_PER_SLOT = get_named_compile_time_arg_val("rows_per_slot");
    constexpr uint32_t SLOT_STRIDE = get_named_compile_time_arg_val("slot_stride");
    constexpr uint32_t BANKS = get_named_compile_time_arg_val("banks");
    constexpr uint32_t SHARD_ROWS = get_named_compile_time_arg_val("shard_rows");
    constexpr uint32_t BLOCK = get_named_compile_time_arg_val("stripe_block");
    constexpr uint32_t CHIPS = get_named_compile_time_arg_val("stripe_chips");
    static_assert(USERS <= sm::MAX_USERS);

    constexpr auto rows_args = TensorAccessorArgs<0>();
    constexpr auto pos_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();

    const uint32_t chip = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t rows_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t layer = get_common_arg_val<uint32_t>(3);

    const auto rows = TensorAccessor(rows_args, rows_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);

    cb_reserve_back(cb_pos, 1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_pos));
    noc_async_read(pos.get_noc_addr(0), reinterpret_cast<uint32_t>(p), USERS * sizeof(uint32_t));
    noc_async_read_barrier();

    cb_reserve_back(cb_row, USERS);
    const uint32_t scratch = get_write_ptr(cb_row);
    uint32_t page_of[sm::MAX_USERS];
    uint32_t mine = 0;
    for (uint32_t u = 0; u < USERS; ++u) {
        const uint32_t position = p[u];
        const uint32_t row = (position / (BLOCK * CHIPS)) * BLOCK + position % BLOCK;
        if ((position / BLOCK) % CHIPS != chip || row >= ROWS_PER_SLOT) {
            continue;
        }
        noc_async_read(rows.get_noc_addr(u), scratch + mine * STRIDE, ROW_BYTES);
        page_of[mine++] = (u * SLOT_STRIDE + layer) * ROWS_PER_SLOT + row;
    }
    noc_async_read_barrier();
    for (uint32_t i = 0; i < mine; ++i) {
        const uint64_t dst = get_noc_addr_from_bank_id<true>(
            sm::kv_row_bank<BANKS, SHARD_ROWS>(page_of[i]),
            kv_addr + sm::kv_row_offset<BANKS, SHARD_ROWS>(page_of[i], STRIDE));
        noc_async_write(scratch + i * STRIDE, dst, ROW_BYTES);
    }
    noc_async_write_barrier();
}
