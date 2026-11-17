// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode merge writer, HEAD-MAJOR output (flag HY4_ATTN_HM=1, S == 1 only): the merge tile of output column j (rows
// 0..7 = the 8 heads of user 0) is scattered to the head-major tensor out [1,8,1,512] TILE (padded [1,8,32,512],
// pre-zeroed, persistent): row h of CB tile j -> row 0 of out tile (h*16 + j0 + j), face 0 (cols 0-15) and face 1 (cols
// 16-31). DRAM writes are 64 B (2 rows: [row h | zeros]) so that the NoC write is 64 B aligned; row 1 of the out tile
// is padding anyway. The scratch lives in extra CB pages after the TPC output tiles.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t TPC = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2, 1>();
    constexpr uint32_t TILE_B = 2048;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const auto oa = TensorAccessor(o_args, o_addr, TILE_B);
    CircularBuffer o(CB_OUT);
    DeviceZoneScopedN("XW_ALL");
    o.wait_front(TPC);
    uint32_t p = o.get_read_ptr();
    uint32_t scr =
        p + TPC * TILE_B;  // extra pages (the CB has TPC + 8 pages; the first TPC tiles are pushed at the base)
    uint32_t n = 0;
    for (uint32_t j = 0; j < TPC; ++j) {
        for (uint32_t h = 0; h < 8; ++h) {
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + n * 64);
                volatile tt_l1_ptr uint32_t* s =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p + j * TILE_B + f * 512 + h * 32);
                for (uint32_t i = 0; i < 8; ++i) {
                    d[i] = s[i];
                    d[8 + i] = 0;
                }
                noc_async_write(scr + n * 64, oa.get_noc_addr(h * 16 + j0 + j, f * 512), 64);
                ++n;
            }
        }
    }
    noc_async_write_barrier();
    o.pop_front(TPC);
}
