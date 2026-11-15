// SPDX-License-Identifier: Apache-2.0
// tp32pf floor microbench, NCRISC: x tiles (resident), then per owned expert PASSES x 8 bank runs of 96 W01 tiles into
// a ring. CT: 0 cb_w, 1 cb_x, 2 MT, 3 PASSES, 4 ring blocks (of 96 tiles), 5 NE_total, 6 NC (cores), then
// TensorAccessorArgs(x). RT: 0 core idx, 1 phys y of dram vc (unused) Common: 0 w01 addr, 1 x addr, 2.. bank noc coords
// not needed (get_noc_addr_from_bank_id).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t MT = get_compile_time_arg_val(2);
    constexpr uint32_t PASSES = get_compile_time_arg_val(3);
    constexpr uint32_t NE = get_compile_time_arg_val(5);
    constexpr uint32_t NC = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr uint32_t TB = 576, BLK = 96 * TB, PA = 24 * TB;
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t w01 = get_common_arg_val<uint32_t>(0);
    const uint32_t xaddr = get_common_arg_val<uint32_t>(1);
    const auto xa = TensorAccessor(x_args, xaddr, 2048);
    // resident x: MT * 192 tiles
    cb_reserve_back(cb_x, MT * 192);
    uint32_t xl = get_write_ptr(cb_x);
    for (uint32_t i = 0; i < (me == 0 ? MT * 192 : 0); ++i) {
        noc_async_read(xa.get_noc_addr(i), xl + i * 2048, 2048);
        if ((i & 31) == 31) {
            noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, MT * 192);

    for (uint32_t e = me; e < NE; e += NC) {
        for (uint32_t p = 0; p < PASSES; ++p) {
            for (uint32_t b = 0; b < 8; ++b) {
                cb_reserve_back(cb_w, 96);
                uint32_t wr = get_write_ptr(cb_w);
                uint64_t base = get_noc_addr_from_bank_id<true>((b + me) & 7, w01 + e * BLK);
                for (uint32_t i = 0; i < 4; ++i) {
                    noc_async_read(base + i * PA, wr + i * PA, PA);
                }
                // keep up to 2 blocks in flight: barrier only the oldest via full barrier every block (simple)
                noc_async_read_barrier();
                cb_push_back(cb_w, 96);
            }
        }
    }
}
