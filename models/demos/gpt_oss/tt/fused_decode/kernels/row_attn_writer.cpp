// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Row attention, writer (BRISC) (dflash/row_attention.py): tells compute the chunk count and which mask each chunk
// uses (mailbox: nk), then writes rows [8 g, 8 g + 8) (kv head g's query heads) of row r's two output tiles: in each
// 32x32 tile, face 0 and face 1 rows 8 g .. 8 g + 7 (256 bytes each) of the [1, M, heads, 64] output tensor.
//
// runtime args: [out_addr, pos_addr, r, g]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    const uint32_t r = get_arg_val<uint32_t>(2);
    const uint32_t g = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr2 = get_compile_time_arg_val(1);
    constexpr uint32_t window = get_compile_time_arg_val(2);
    constexpr uint32_t pos_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t max_chunks = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr auto pos_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;
    const auto s_out = TensorAccessor(out_args, out_addr, tile);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, pos_page_bytes);

    cb_reserve_back(cb_scr2, 1);
    const uint32_t scr = get_write_ptr(cb_scr2);
    noc_async_read(s_pos.get_noc_addr(0), scr, pos_page_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const int32_t p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[r];
    const int32_t lo_ = window ? p + 1 - static_cast<int32_t>(window) : 0;
    const uint32_t lo = lo_ > 0 ? static_cast<uint32_t>(lo_) : 0;
    uint32_t nk = (static_cast<uint32_t>(p) - (lo & ~31u)) / 32 + 1;
    nk = nk > max_chunks ? max_chunks : nk;
    // bit 16: chunk 0 is cut at the start; bit 17: the last chunk is cut at the end (else no mask is needed)
    const uint32_t info = nk | ((lo & 31u) ? 1u << 16 : 0u) | (((p & 31) != 31) ? 1u << 17 : 0u);
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, info);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, info);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, info);

    cb_wait_front(cb_out, 2);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t d = 0; d < 2; ++d) {
        const uint64_t dst = s_out.get_noc_addr(2 * r + d);
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t off = f * 512 + g * 8 * 32;
            noc_async_write(src + d * tile + off, dst + off, 256);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 2);
}
