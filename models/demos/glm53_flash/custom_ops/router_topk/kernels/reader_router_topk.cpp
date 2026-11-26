// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Router top-k, token-row side (NCRISC; present only with X_MODE = 1): untilizes this core's token row of the
// MoE input x [1,1,T,H] bf16 TILE into the row-major [x | w] page the EP dispatch consumes, in parallel with the
// writer's score read + ranking on BRISC:
//   1. 2 x H/32 reads of 64 B: tile column ct, face half h, the face-row PAIR holding this row ((row & 15) & ~1):
//      DRAM offset (face + h) * 512 + (rf & ~1) * 32 inside tile page ct (64 B aligned), landing at the same
//      offset modulo 64 in the staging area (NoC alignment rule);
//   2. the row's 32 B half of every piece is copied into the contiguous 2 H-byte row of CB_XROW (bf16 column
//      c at byte 2 c), the K-wide tail (bytes [2 H, 2 H + 2 K)) is zeroed for the writer, which fills it with the
//      bf16 routing weights and sends the whole page;
//   3. CB_XROW is pushed (also for padded rows >= T, with no reads: the writer skips their page write).
// Ends with noc_async_full_barrier().

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t CB_XS = get_compile_time_arg_val(0);    // staging: 2 * HT pieces of 64 B
constexpr uint32_t CB_XROW = get_compile_time_arg_val(1);  // the row-major [x | w] page (2 * (H + K) bytes)
constexpr uint32_t HT = get_compile_time_arg_val(2);       // H / 32 tile columns of x
constexpr uint32_t K = get_compile_time_arg_val(3);        // routing weights per token (the page tail)
constexpr auto x_args = TensorAccessorArgs<4>();

constexpr uint32_t BF16_FACE = 512;
constexpr uint32_t PIECES = 2 * HT;
constexpr uint32_t ROW_BYTES = HT * 64;  // 2 B x 32 columns per tile column
}  // namespace

void kernel_main() {
    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t T = get_common_arg_val<uint32_t>(1);
    const bool valid = row < T;
    const auto s_x = TensorAccessor(x_args, x_addr);

    CircularBuffer cbs(CB_XS), cbrow(CB_XROW);
    cbs.reserve_back(1);
    cbrow.reserve_back(1);
    const uint32_t xs = cbs.get_write_ptr();
    const uint32_t xrow = cbrow.get_write_ptr();
    const uint32_t face = (row >> 4) * 2;
    const uint32_t rf = row & 15;
    const uint32_t pair_off = (rf & ~1u) * 32;  // the 64 B face-row pair holding this row
    const uint32_t half = (rf & 1) * 32;        // this row's 32 B inside the pair

    if (valid) {
        for (uint32_t ct = 0; ct < HT; ++ct) {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_read(s_x.get_noc_addr(ct, (face + h) * BF16_FACE + pair_off), xs + (ct * 2 + h) * 64, 64);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        // the row's halves -> the contiguous row (8 words per 32 B piece)
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xrow);
        const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(xs + half);
        for (uint32_t i = 0; i < PIECES; ++i) {
            const volatile tt_l1_ptr uint32_t* s = src + i * 16;  // 64 B piece stride = 16 words
            volatile tt_l1_ptr uint32_t* d = dst + i * 8;
            for (uint32_t q = 0; q < 8; ++q) {
                d[q] = s[q];
            }
        }
        volatile tt_l1_ptr uint32_t* tail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xrow + ROW_BYTES);
        for (uint32_t q = 0; q < (K * 2 + 3) / 4; ++q) {
            tail[q] = 0;
        }
    }
    cbs.push_back(1);
    cbrow.push_back(1);
    noc_async_full_barrier();
}
