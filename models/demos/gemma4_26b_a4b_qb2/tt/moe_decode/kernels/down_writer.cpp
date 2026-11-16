// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode down writer (RISCV_0, NOC_1): the column's 1x32 result tile into row 0 of output tile `col`
// ([1,1,1,H] bf16 TILE, 32x32 tiles): columns 0-15 to face 0 row 0, 16-31 to face 1 row 0. Rows 1-31 are not
// written (the batch-1 consumers read row 0 only).
// First it reads the second half of the routed experts' down tiles (NOC_1) into the reader's weight CB and pushes one
// cb_sync page.
// compile-time args: [cb_out, cb_w, cb_sync, cb_route, kts, k, cols, w_tile_bytes, route_page_bytes, banks]
//                    + TensorAccessorArgs(out, route)
// runtime args: [out_addr, col, w_addr, route_addr, bank, local column in the bank, cpb, flat_off (OUT_FLAT)]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t col = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t route_addr = get_arg_val<uint32_t>(3);
    const uint32_t bank = get_arg_val<uint32_t>(4);
    const uint32_t local = get_arg_val<uint32_t>(5);
    const uint32_t cpb = get_arg_val<uint32_t>(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t kts = get_compile_time_arg_val(4);
    constexpr uint32_t k = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t banks = get_compile_time_arg_val(9);
    constexpr auto o_args = TensorAccessorArgs<10>();
    constexpr auto r_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, face_bytes = 512, row_bytes = 32;
    const auto s_o = TensorAccessor(o_args, out_addr, tile_bytes);
    const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);

    const uint32_t scratch = get_write_ptr(cb_scr);
    noc_async_read(s_r.get_noc_addr(0), scratch, route_page_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
    const uint32_t chunk_bytes = kts * w_tile_bytes;
    uint32_t l1 = get_write_ptr(cb_w) + (k / 2) * chunk_bytes;  // the reader's reservation: the CB base
    for (uint32_t s = k / 2; s < k; ++s) {
        union {
            uint32_t u;
            float f;
        } id_bits;
        id_bits.u = static_cast<uint32_t>(row[k + s]) << 16;
        const uint32_t offset = (static_cast<uint32_t>(id_bits.f) * cpb + local) * chunk_bytes;
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_addr + offset), l1, chunk_bytes);
        l1 += chunk_bytes;
    }
    noc_async_read_barrier();
    cb_reserve_back(cb_sync, 1);
    cb_push_back(cb_sync, 1);

    cb_wait_front(cb_out, 1);
    const uint32_t src = get_read_ptr(cb_out);
#ifdef OUT_FLAT
    // flat output (fused_decode/boundary.py partial buffer): the column's 32 values at byte flat_off + 64 col
    noc_async_write(src, s_o.get_noc_addr(0) + get_arg_val<uint32_t>(7) + col * 2 * row_bytes, 2 * row_bytes);
#ifdef SPLIT_SLOTS
    // the second sum (the slots from SPLIT_SLOTS on) at flat_off2
    cb_wait_front(cb_out, 2);
    noc_async_write(
        src + 2 * row_bytes, s_o.get_noc_addr(0) + get_arg_val<uint32_t>(8) + col * 2 * row_bytes, 2 * row_bytes);
#endif
#else
    const uint64_t dst = s_o.get_noc_addr(col);
    noc_async_write(src, dst, row_bytes);
    noc_async_write(src + row_bytes, dst + face_bytes, row_bytes);
#endif
    noc_async_write_barrier();
#ifdef SPLIT_SLOTS
    cb_pop_front(cb_out, 2);
#else
    cb_pop_front(cb_out, 1);
#endif
}
