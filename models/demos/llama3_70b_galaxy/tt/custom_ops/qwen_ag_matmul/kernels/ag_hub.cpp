// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qwen_ag_matmul hub: the launch barrier and the block release of the fused all-gather + matmul.
//  1. Waits for the barrier incs of every peer (each peer's link-0 senders multicast one when they start), then
//     releases the local senders (go semaphores).
//  2. Walks each matmul column's block list in order and increments that column's in0 injector ready semaphore
//     once per block, as soon as the block is available: at once for a block of this device's own slot, after its
//     arrival word reached arrivals_expected (8: 4 links x 2 halves) for a remote one.
// The barrier and arrival words are global semaphores that remote devices increment; a peer may already be in the
// next call when it increments the barrier, so every word is decremented atomically, never set to 0.
//
// CT: 0 barrier_expected, 1 arrivals_expected, 2 num_cols, 3 go_fwd_sem_id, 4 go_bwd_sem_id, 5 ready_sem_id,
//     6 num_senders
// RT: 0 barrier_sem, then num_senders x (noc_x, noc_y), num_cols x (noc_x, noc_y) of the injectors, then per
//     column: num_blocks followed by num_blocks arrival word addresses (0 = own slot, available at once), then
//     num_words followed by the num_words distinct arrival word addresses (taken back down at the end)
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t MAX_COLS = 8;

void kernel_main() {
    constexpr uint32_t barrier_expected = get_compile_time_arg_val(0);
    constexpr uint32_t arrivals_expected = get_compile_time_arg_val(1);
    constexpr uint32_t num_cols = get_compile_time_arg_val(2);
    constexpr uint32_t go_fwd_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t go_bwd_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t num_senders = get_compile_time_arg_val(6);
    static_assert(num_cols <= MAX_COLS, "too many matmul columns");

    uint32_t a = 0;
    const uint32_t barrier_sem = get_arg_val<uint32_t>(a++);
    const uint32_t senders_idx = a;
    a += 2 * num_senders;
    const uint32_t inj_idx = a;
    a += 2 * num_cols;
    uint32_t list_idx[MAX_COLS];
    uint32_t list_len[MAX_COLS];
    uint32_t ptr[MAX_COLS];
    uint32_t remaining = 0;
    for (uint32_t c = 0; c < num_cols; c++) {
        list_len[c] = get_arg_val<uint32_t>(a++);
        list_idx[c] = a;
        a += list_len[c];
        ptr[c] = 0;
        remaining += list_len[c];
    }
    const uint32_t num_words = get_arg_val<uint32_t>(a++);
    const uint32_t words_idx = a;

    volatile tt_l1_ptr uint32_t* bs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem);
    noc_semaphore_wait_min(bs, barrier_expected);
    noc_semaphore_inc(get_noc_addr(barrier_sem), static_cast<uint32_t>(-static_cast<int32_t>(barrier_expected)));
    const uint32_t go_f = get_semaphore(go_fwd_sem_id);
    const uint32_t go_b = get_semaphore(go_bwd_sem_id);
    for (uint32_t i = 0; i < num_senders; ++i) {
        const uint32_t x = get_arg_val<uint32_t>(senders_idx + 2 * i);
        const uint32_t y = get_arg_val<uint32_t>(senders_idx + 2 * i + 1);
        noc_semaphore_inc(get_noc_addr(x, y, go_f), 1);
        noc_semaphore_inc(get_noc_addr(x, y, go_b), 1);
    }

    const uint32_t ready = get_semaphore(ready_sem_id);
    while (remaining > 0) {
        for (uint32_t c = 0; c < num_cols; c++) {
            while (ptr[c] < list_len[c]) {
                const uint32_t word = get_arg_val<uint32_t>(list_idx[c] + ptr[c]);
                if (word != 0) {
                    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word);
                    invalidate_l1_cache();
                    if (*w < arrivals_expected) {
                        break;
                    }
                }
#ifndef PROBE_NORELAY
                noc_semaphore_inc(
                    get_noc_addr(
                        get_arg_val<uint32_t>(inj_idx + 2 * c), get_arg_val<uint32_t>(inj_idx + 2 * c + 1), ready),
                    1);
#endif
                ptr[c]++;
                remaining--;
            }
        }
    }
    // every remote block's word reached arrivals_expected; take them back down (atomically, see above)
    for (uint32_t i = 0; i < num_words; i++) {
        const uint32_t word = get_arg_val<uint32_t>(words_idx + i);
        noc_semaphore_inc(get_noc_addr(word), static_cast<uint32_t>(-static_cast<int32_t>(arrivals_expected)));
    }
    noc_async_atomic_barrier();
}
