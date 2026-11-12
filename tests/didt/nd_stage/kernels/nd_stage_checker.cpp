// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Feeds the compute probe and checks its output run-to-run on the core itself. The inputs already sit in
// L1 (the CBs are backed by sharded tensors), so they are pushed once. Every output push is popped; on a
// checked iteration the last `check_last` pushes are compared word by word against iteration 0's copy in
// the ref region. Only a summary leaves the core, so the check costs no NoC traffic and no host time.
//
// Compile-time args: cb_in0, cb_in1, cb_out, elem_shift (0: 32-bit elements, 1: 16-bit elements)
// Runtime args: n_iters, pushes_per_iter, push_tiles, check_last, tile_bytes, check_every, in0_tiles,
//               in1_tiles, ref_addr, result_addr, core_id, inject_iter, inject_word, inject_xor
//
// inject_*: positive control. On iteration inject_iter the first checked push has inject_xor applied to
// word inject_word in L1 before the compare, as a fault in the pipeline would leave it. Iteration 0
// corrupts the reference copy instead. inject_iter = 0xFFFFFFFF disables it.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "nd_stage_result.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t elem_shift = get_compile_time_arg_val(3);

    const uint32_t n_iters = get_arg_val<uint32_t>(0);
    const uint32_t pushes_per_iter = get_arg_val<uint32_t>(1);
    const uint32_t push_tiles = get_arg_val<uint32_t>(2);
    const uint32_t check_last = get_arg_val<uint32_t>(3);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(4);
    const uint32_t check_every = get_arg_val<uint32_t>(5);
    const uint32_t in0_tiles = get_arg_val<uint32_t>(6);
    const uint32_t in1_tiles = get_arg_val<uint32_t>(7);
    const uint32_t ref_addr = get_arg_val<uint32_t>(8);
    const uint32_t result_addr = get_arg_val<uint32_t>(9);
    const uint32_t core_id = get_arg_val<uint32_t>(10);
    const uint32_t inject_iter = get_arg_val<uint32_t>(11);
    const uint32_t inject_word = get_arg_val<uint32_t>(12);
    const uint32_t inject_xor = get_arg_val<uint32_t>(13);

    nd_stage::Result res(result_addr, nd_stage::MAGIC_CHECKER, core_id);

    cb_reserve_back(cb_in0, in0_tiles);
    cb_push_back(cb_in0, in0_tiles);
    if (in1_tiles > 0) {
        cb_reserve_back(cb_in1, in1_tiles);
        cb_push_back(cb_in1, in1_tiles);
    }

    const uint32_t push_words = push_tiles * tile_bytes / sizeof(uint32_t);
    const uint32_t words_per_tile = tile_bytes / sizeof(uint32_t);
    const uint32_t first_checked = pushes_per_iter - check_last;

    for (uint32_t it = 0; it < n_iters; ++it) {
        const bool check_it = it == 0 || it + 1 == n_iters || it % check_every == 0;
        bool bad_it = false;
        for (uint32_t p = 0; p < pushes_per_iter; ++p) {
            cb_wait_front(cb_out, push_tiles);
            if (check_it && p >= first_checked) {
                // cb_wait_front does not invalidate the RISC's L1 data cache; without this a later check
                // could compare a stale cached line instead of what the packer wrote.
                invalidate_l1_cache();
                volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out));
                volatile tt_l1_ptr uint32_t* ref = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    ref_addr + (p - first_checked) * push_words * sizeof(uint32_t));
                if (it == inject_iter && p == first_checked) {
                    out[inject_word] ^= inject_xor;
                }
                if (it == 0) {
                    for (uint32_t w = 0; w < push_words; ++w) {
                        ref[w] = out[w];
                    }
                } else {
                    for (uint32_t w = 0; w < push_words; ++w) {
                        const uint32_t got = out[w];
                        const uint32_t want = ref[w];
                        if (got != want) {
                            bad_it = true;
                            res.record_mismatch<elem_shift>(
                                it, (p - first_checked) * push_words + w, got, want, words_per_tile);
                        }
                    }
                }
            }
            cb_pop_front(cb_out, push_tiles);
        }
        res.end_iteration(check_it && it > 0, bad_it, it);
    }
    res.finish();
}
