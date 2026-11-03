// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The global layers' per-chip decode positions under the KV pair split (tt/pair_split.py; host mirror
// reference/kv_pair_split.py positions_step), one core, at the start of every decode and verify step. Chip c of a
// pair (its parity, word 0 of a per-chip int32 page) keeps the global positions p with ((p >> 6) & 1) == c, the
// position p at local index 64 (p >> 7) + (p & 63). From the step's global positions (int32 [B], -1 an idle slot):
// - write [B]: where the chip writes the slot's K / V, its local index when it keeps p, else -1 (the cache update
//   skips it);
// - sdpa [B]: the slot's decode SDPA position on the chip, its last local index at a global position <= p,
//   64 (p >> 7) + clamp((p & 127) + 1 - 64 c, 0, 64) - 1 (-1: none);
// - S serial rows [B], row k holding the write positions of the slots j with j % S == k (a verify writes K / V one
//   column of its draft groups at a time, reference/spec_decode.py serial_positions).
// Every buffer is a one-page row-major interleaved DRAM tensor, read and written whole.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

FORCE_INLINE uint64_t dram_addr(uint32_t base, uint32_t page_size) {
    const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = page_size};
    return get_noc_addr(0, g, 0);
}

constexpr uint32_t align64(uint32_t bytes) { return (bytes + 63) & ~63u; }

}  // namespace

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);  // slots
    constexpr uint32_t S = get_compile_time_arg_val(1);  // serial rows
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t parity_bytes = get_compile_time_arg_val(3);  // the per-chip parity page
    constexpr uint32_t row_bytes = B * 4;
    constexpr uint32_t row_l1 = align64(row_bytes);
    static_assert(S <= B && parity_bytes >= 4);

    const uint32_t pos_addr = get_arg_val<uint32_t>(0);
    const uint32_t parity_addr = get_arg_val<uint32_t>(1);
    const uint32_t write_addr = get_arg_val<uint32_t>(2);
    const uint32_t sdpa_addr = get_arg_val<uint32_t>(3);  // the serial rows' addresses follow

    cb_reserve_back(cb_scratch, 1);
    const uint32_t pos_l1 = get_write_ptr(cb_scratch);
    const uint32_t parity_l1 = pos_l1 + row_l1;
    const uint32_t out_l1 = parity_l1 + align64(parity_bytes);  // write, sdpa, then the S serial rows
    noc_async_read(dram_addr(pos_addr, row_bytes), pos_l1, row_bytes);
    noc_async_read(dram_addr(parity_addr, parity_bytes), parity_l1, parity_bytes);
    noc_async_read_barrier();

    volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1);
    const int32_t c = reinterpret_cast<volatile tt_l1_ptr int32_t*>(parity_l1)[0];
    volatile tt_l1_ptr int32_t* write = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1);
    volatile tt_l1_ptr int32_t* sdpa = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + row_l1);
    for (uint32_t j = 0; j < B; ++j) {
        const int32_t p = pos[j];
        int32_t w = -1;
        int32_t s = -1;
        if (p >= 0) {
            const int32_t base = (p >> 7) << 6;
            if (((p >> 6) & 1) == c) {
                w = base + (p & 63);
            }
            int32_t seen = (p & 127) + 1 - (c << 6);
            seen = seen < 0 ? 0 : (seen > 64 ? 64 : seen);
            s = base + seen - 1;
        }
        write[j] = w;
        sdpa[j] = s;
    }
    for (uint32_t k = 0; k < S; ++k) {
        volatile tt_l1_ptr int32_t* row = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + (2 + k) * row_l1);
        for (uint32_t j = 0; j < B; ++j) {
            row[j] = j % S == k ? write[j] : -1;
        }
    }

    noc_async_write(out_l1, dram_addr(write_addr, row_bytes), row_bytes);
    noc_async_write(out_l1 + row_l1, dram_addr(sdpa_addr, row_bytes), row_bytes);
    for (uint32_t k = 0; k < S; ++k) {
        noc_async_write(out_l1 + (2 + k) * row_l1, dram_addr(get_arg_val<uint32_t>(4 + k), row_bytes), row_bytes);
    }
    noc_async_write_barrier();
}
