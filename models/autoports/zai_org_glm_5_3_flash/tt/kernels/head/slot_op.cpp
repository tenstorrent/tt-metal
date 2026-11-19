// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b serving backend (tt/stack_model.py SlotOps): per-user state operations between decode steps (request
// admission / release / vLLM slot remaps), one core, every layer in one launch. All tensors DRAM interleaved TILE.
// Runtime args: n_ops, then per op 8 words: kind, src address, src first page, dst address, dst first page, pages,
// page bytes, (src row << 16) | dst row.
//   kind 0: zero pages [dst first, + pages)                      (KDA S rows: user b's tiles are contiguous)
//   kind 1: copy pages src [first, + pages) -> dst [first, + pages)
//   kind 2: row in tiles: for each page p: dst tile p's row `dst row` <- src tile p's row `src row` (src address 0:
//           zeros); element bytes = page bytes / 1024 (KDA conv history: the users are rows of every tile)
// CB SLOT_CB: 2 x 4096 B of staging.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
FORCE_INLINE uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}
}  // namespace

void kernel_main() {
    const uint32_t n_ops = get_arg_val<uint32_t>(0);
    const uint32_t a = get_write_ptr(SLOT_CB), b = a + 4096;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < n_ops; ++o) {
        const uint32_t base = 1 + 8 * o;
        const uint32_t kind = get_arg_val<uint32_t>(base + 0);
        const uint32_t src = get_arg_val<uint32_t>(base + 1), src0 = get_arg_val<uint32_t>(base + 2);
        const uint32_t dst = get_arg_val<uint32_t>(base + 3), dst0 = get_arg_val<uint32_t>(base + 4);
        const uint32_t pages = get_arg_val<uint32_t>(base + 5), pb = get_arg_val<uint32_t>(base + 6);
        const uint32_t rows = get_arg_val<uint32_t>(base + 7);
        const InterleavedAddrGen<true> s{.bank_base_address = src, .page_size = pb};
        const InterleavedAddrGen<true> d{.bank_base_address = dst, .page_size = pb};
        if (kind == 0) {
            for (uint32_t off = 0; off < pb; off += MEM_ZEROS_SIZE) {
                noc_async_read(zeros, a + off, MEM_ZEROS_SIZE < pb - off ? MEM_ZEROS_SIZE : pb - off);
            }
            noc_async_read_barrier();
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_write(a, d.get_noc_addr(dst0 + p), pb);
            }
            noc_async_write_barrier();
        } else if (kind == 1) {
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read(s.get_noc_addr(src0 + p), a, pb);
                noc_async_read_barrier();
                noc_async_write(a, d.get_noc_addr(dst0 + p), pb);
                noc_async_write_barrier();
            }
        } else {
            const uint32_t sr = rows >> 16, dr = rows & 0xffffu, eb = pb / 1024;
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read(d.get_noc_addr(dst0 + p), a, pb);
                if (src != 0) {
                    noc_async_read(s.get_noc_addr(src0 + p), b, pb);
                }
                noc_async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    if (eb == 4) {
                        volatile tt_l1_ptr uint32_t* x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a);
                        volatile tt_l1_ptr uint32_t* y = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b);
                        x[tile_off(dr, c)] = src != 0 ? y[tile_off(sr, c)] : 0u;
                    } else {
                        volatile tt_l1_ptr uint16_t* x = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a);
                        volatile tt_l1_ptr uint16_t* y = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b);
                        x[tile_off(dr, c)] = src != 0 ? y[tile_off(sr, c)] : static_cast<uint16_t>(0);
                    }
                }
                noc_async_write(a, d.get_noc_addr(dst0 + p), pb);
                noc_async_write_barrier();
            }
        }
    }
}
