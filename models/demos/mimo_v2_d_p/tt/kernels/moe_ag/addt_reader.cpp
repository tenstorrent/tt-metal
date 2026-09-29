// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Row add with a tiled output (out = a[a_off + r] + b[b_off + r], rows of H bf16 -> TILE), reader (NCRISC): per block
// (tile row tr, 1024-column chunk c) the 32 row segments of a and b (2 KB each, one "tile" page per row) into
// c_0 / c_1. Blocks j = me, me + P, ...
// CT: 0 ROW_BYTES, 1 NCH (chunks per row), 2 P
// Common RT: 0 a addr, 1 b addr, 2 a_off, 3 b_off, 4 blocks, 5 grid x
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_range.hpp"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0), NCH = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2);
    const InterleavedAddrGen<true> a = {.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> b = {.bank_base_address = get_common_arg_val<uint32_t>(1), .page_size = ROW_BYTES};
    const uint32_t a_off = get_common_arg_val<uint32_t>(2), b_off = get_common_arg_val<uint32_t>(3);
    const uint32_t blocks = get_common_arg_val<uint32_t>(4), me = core_index(get_common_arg_val<uint32_t>(5));
    for (uint32_t j = me; j < blocks; j += P) {
        const uint32_t tr = j / NCH, c = j % NCH;
        cb_reserve_back(tt::CBIndex::c_0, 32);
        cb_reserve_back(tt::CBIndex::c_1, 32);
        const uint32_t la = get_write_ptr(tt::CBIndex::c_0), lb = get_write_ptr(tt::CBIndex::c_1);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read(get_noc_addr(a_off + tr * 32 + r, a, c * 2048), la + r * 2048, 2048);
            noc_async_read(get_noc_addr(b_off + tr * 32 + r, b, c * 2048), lb + r * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(tt::CBIndex::c_0, 32);
        cb_push_back(tt::CBIndex::c_1, 32);
    }
}
