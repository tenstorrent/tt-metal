// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode sampling head, step 1, writer (tt/device_sampler.py): the second half of the user slots' K largest logits of
// the core's range (users 2-3 of 4 slots, 4-7 of 8), from
// the tiles the reader loaded (sample_scan.cpp, which hands over their address), and each user's sum of
// exp((logit - the range's largest) / T) from the compute kernel's tile (sample_norm_compute.cpp): the 32 columns of
// row u (face 0 and face 1), added in Q30 fixed point.
// Output: those slots of this core's top-K partials block (byte offset slots / 2 * K * 8 of it; zero past `users`),
// and `slots` uint64 Q30 sums (zero past `users`) at byte offset core * slots * 8 of the sums partials page (4096
// bytes).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sample_fixed.hpp"
#include "sample_topk.hpp"

void kernel_main() {
    const uint32_t partials_addr = get_arg_val<uint32_t>(0);
    const uint32_t sums_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_begin = get_arg_val<uint32_t>(2);
    const uint32_t tile_end = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);

    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_ready = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);  // user slots of a core's block: 4, 8 or 16
    static_assert(users <= slots && (slots == 4 || slots == 8 || slots == 16));
    constexpr uint32_t half = slots / 2;  // the reader's slots
    constexpr uint32_t slot_bytes = K * 8;
    constexpr uint32_t block_bytes = slots * slot_bytes;
    constexpr uint32_t blocks_per_page = 4096 / block_bytes;

    cb_reserve_back(cb_out, 1);
    const uint32_t out_l1 = get_write_ptr(cb_out);  // the writer's slots, then the sums
    const uint32_t sums_l1 = out_l1 + half * slot_bytes;

    cb_wait_front(cb_ready, 1);
    const uint32_t x_l1 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ready));
    const uint32_t* x = reinterpret_cast<const uint32_t*>(x_l1);
    const uint32_t n = tile_end - tile_begin;
    uint32_t* out = reinterpret_cast<uint32_t*>(out_l1);
    for (uint32_t u = half; u < slots; ++u) {
        uint32_t* slot = out + (u - half) * 2 * K;
        if (u < users) {
            sample::range_topk<K>(x, n, tile_begin, u, slot);
        } else {
            for (uint32_t j = 0; j < 2 * K; ++j) {
                slot[j] = 0;
            }
        }
    }
    cb_pop_front(cb_ready, 1);
    const InterleavedAddrGen<false> part_gen = {.bank_base_address = partials_addr, .page_size = 4096};
    const uint32_t offset = (core % blocks_per_page) * block_bytes + half * slot_bytes;
    noc_async_write(out_l1, get_noc_addr(core / blocks_per_page, part_gen, offset), half * slot_bytes);

    uint64_t* sums = reinterpret_cast<uint64_t*>(sums_l1);
    cb_wait_front(cb_s, 1);
    const uint32_t* tile = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_s));
    for (uint32_t u = 0; u < slots; ++u) {
        uint64_t sum = 0;
        if (u < users) {
            for (uint32_t c = 0; c < 16; ++c) {
                sum += sample::fp32_to_q30(tile[u * 16 + c]);
                sum += sample::fp32_to_q30(tile[256 + u * 16 + c]);
            }
        }
        sums[u] = sum;
    }
    cb_pop_front(cb_s, 1);
    const InterleavedAddrGen<false> sums_gen = {.bank_base_address = sums_addr, .page_size = 4096};
    noc_async_write(sums_l1, get_noc_addr(0, sums_gen, core * slots * 8), slots * 8);
    noc_async_write_barrier();
}
