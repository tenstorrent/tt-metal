// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 greedy terminal, per-device merge (BRISC) (fused_decode/greedy.py: GreedyB1).
//
// Waits for the `workers` (key, local id) results of greedy_local.cpp in its slots (16 B each), keeps the largest key
// (ties: the lowest id) and writes (key, class, global id = offset + local id) as elements 0..2 of row 0 of the
// device's UINT32 tile `out` (sharded on this core).
//
// runtime args: [out_addr, offset]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t offset = get_arg_val<uint32_t>(1);
    constexpr uint32_t workers = get_compile_time_arg_val(0);
    constexpr uint32_t sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(2);
    const uint32_t slots_addr = get_write_ptr(cb_slots);

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, workers);
    noc_semaphore_set(sem, 0);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots_addr);
    // max key, then the lowest id among the entries in its softcap class (greedy_local.cpp)
    uint32_t best_key = s[0], low = s[1];
    for (uint32_t i = 1; i < workers; ++i) {
        if (s[4 * i] > best_key) {
            best_key = s[4 * i];
            low = s[4 * i + 1];
        }
    }
    uint32_t best_id = 0xFFFFFFFF;
    for (uint32_t i = 0; i < workers; ++i) {
        if (s[4 * i + 1] == low && s[4 * i + 2] < best_id) {
            best_id = s[4 * i + 2];
        }
    }
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
    o[0] = best_key;
    o[1] = low;
    o[2] = best_id + offset;
}
