// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Greedy pick, stage 1 (tile-engine version), writer: from the unit's per-tile row maxima (cb_max, n tiles) each row
// takes the first tile holding its largest value, reads that tile's row (32 values) back from the logits and takes
// its first column with the largest value; writes per row one packed key (order-preserving bf16 value << 16 |
// 0xFFFF - column: the largest key is the largest value, and among equal values the smallest column) to page w of
// the partials [W, 32] uint32 -- the same partials the scalar stage 1 (argmax_part.cpp) writes.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t okey(uint32_t v) {  // order-preserving key of a bf16 bit pattern
    return v ^ (0x8000u | ((v >> 15) * 0x7FFFu));
}

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t per = get_compile_time_arg_val(3);
    constexpr uint32_t W = get_compile_time_arg_val(4);
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_max = 16, cb_buf = 2, T = 2048;
    const uint32_t w = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    if (w >= W) {
        return;
    }
    const uint32_t tr = w / S, j0 = (w % S) * per;
    const uint32_t j1 = j0 + per < Vt ? j0 + per : Vt;
    const uint32_t n = j1 > j0 ? j1 - j0 : 0;
    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0), T);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), 128);
    const uint32_t buf = get_write_ptr(cb_buf);  // 32 rows x 128 bytes, then the 32 result words
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + 32 * 128);
    if (n == 0) {
        for (uint32_t r = 0; r < 32; ++r) {
            res[r] = 0;
        }
    } else {
        cb_wait_front(cb_max, n);
        const volatile tt_l1_ptr uint16_t* mx = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_max));
        uint32_t bt[32];
        for (uint32_t r = 0; r < 32; ++r) {
            // row r's max: column 0 of row r -> face (r / 16) * 2, element (r % 16) * 16
            const uint32_t e = ((r / 16) * 2) * 256 + (r % 16) * 16;
            uint32_t best = 0, bj = 0;
            for (uint32_t t = 0; t < n; ++t) {
                const uint32_t k = okey(mx[t * (T / 2) + e]);
                if (k > best) {
                    best = k;
                    bj = t;
                }
            }
            bt[r] = bj;
            // row r of tile bj: 16 values in face (r / 16) * 2 at byte (r % 16) * 32, 16 in the next face; each half
            // read as the 64-byte aligned block around it (a DRAM source -- the draft's logits -- needs the alignment)
            const uint32_t off = ((r / 16) * 2) * 512 + (r % 16) * 32;
            const uint64_t src = in.get_noc_addr(tr * Vt + j0 + bj) + (off & ~63u);
            noc_async_read(src, buf + r * 128, 64);
            noc_async_read(src + 512, buf + r * 128 + 64, 64);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        cb_pop_front(cb_max, n);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t at = (((r / 16) * 2) * 512 + (r % 16) * 32) & 63u;  // the row's offset in its blocks
            const volatile tt_l1_ptr uint16_t* lo =
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(buf + r * 128 + at);
            const volatile tt_l1_ptr uint16_t* hi =
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(buf + r * 128 + 64 + at);
            uint32_t best = 0, bc = 0;
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t k = okey(c < 16 ? lo[c] : hi[c - 16]);
                if (k > best) {
                    best = k;
                    bc = c;
                }
            }
            const uint32_t col = (j0 + bt[r]) * 32 + bc;
            res[r] = (best << 16) | (0xFFFFu - col);
        }
    }
    noc_async_write(buf + 32 * 128, out.get_noc_addr(w), 128);
    noc_async_write_barrier();
}
