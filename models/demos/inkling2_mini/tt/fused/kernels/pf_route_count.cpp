// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill MoE routing, pass 1 (per core): expert histogram of this core's tokens. The top-k ids are a
// [T, 8] uint16 TILE tensor (one tile per 32 tokens; columns 0..7 sit in faces 0 and 2). Tokens >= n
// (the chunk's real row count, word 0 of the nreal buffer) are not routed.
// Runtime args: idx, nreal, hist, R0, R1 (this core's tile rows [R0, R1)), core index c.
// Output: hist row c = 128 uint32 counts (bytes [512 c, 512 c + 512) of the hist page).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// L1 scratch starts at 64 bytes (the DRAM alignment), so a DRAM transfer has both ends aligned alike.
static inline uint32_t al64(uint32_t a) { return (a + 63) & ~63u; }

void kernel_main() {
    constexpr uint32_t NE = get_named_compile_time_arg_val("ne");
    constexpr uint32_t K = get_named_compile_time_arg_val("k");
    constexpr bool idx_dram = get_named_compile_time_arg_val("idx_dram");
    constexpr uint32_t cb_idx = 0, cb_hist = 1, cb_n = 2;
    const uint32_t idx = get_arg_val<uint32_t>(0), nreal = get_arg_val<uint32_t>(1), hist = get_arg_val<uint32_t>(2);
    const uint32_t R0 = get_arg_val<uint32_t>(3), R1 = get_arg_val<uint32_t>(4), c = get_arg_val<uint32_t>(5);
    const InterleavedAddrGen<idx_dram> ia = {.bank_base_address = idx, .page_size = 2048};
    const InterleavedAddrGen<true> na = {.bank_base_address = nreal, .page_size = 64};
    const InterleavedAddrGen<true> ha = {.bank_base_address = hist, .page_size = 64};  // one page: only page 0

    const uint32_t nbuf = al64(get_write_ptr(cb_n));
    noc_async_read(na.get_noc_addr(0), nbuf, 64);
    const uint32_t ib = al64(get_write_ptr(cb_idx));
    for (uint32_t R = R0; R < R1; ++R) {  // faces 0 and 2 (columns 0..15) of each tile
        const uint32_t l = ib + (R - R0) * 1024;
        noc_async_read(ia.get_noc_addr(R), l, 512);
        noc_async_read(ia.get_noc_addr(R) + 1024, l + 512, 512);
    }
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(al64(get_write_ptr(cb_hist)));
    for (uint32_t e = 0; e < NE; ++e) {
        cnt[e] = 0;
    }
    noc_async_read_barrier();
    const uint32_t n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(nbuf)[0];
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib);
    const uint32_t tok1 = R1 * 32 < n ? R1 * 32 : n;
    for (uint32_t tok = R0 * 32; tok < tok1; ++tok) {
        volatile tt_l1_ptr uint16_t* row = ids + (tok - R0 * 32) * 16;
        for (uint32_t k = 0; k < K; ++k) {
            cnt[row[k]] += 1;
        }
    }
    noc_async_write(al64(get_write_ptr(cb_hist)), ha.get_noc_addr(0, c * NE * 4), NE * 4);
    noc_async_write_barrier();
}
