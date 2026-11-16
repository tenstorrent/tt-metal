// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 LM head + greedy, writer (BRISC) (fused_decode/lmhead.py: LmHeadGreedyB1): the argmax of this core's logits
// (the 1x32 BF16 tiles of its vocab columns, kept in L1), with the final-logit softcap's ties resolved as
// greedy_local.cpp does: M = the largest BF16 order key, L = classes[M], id = the lowest id with key >= L. Posts
// (M, L, local id) to slot `slot` of the merge core (greedy_merge.cpp) and increments its semaphore.
//
// runtime args: [first_id, cols, merge_x, merge_y, slot, classes_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t bf16_key(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }

void kernel_main() {
    const uint32_t first_id = get_arg_val<uint32_t>(0);  // the local vocab id of this core's first column, value 0
    const uint32_t cols = get_arg_val<uint32_t>(1);
    const uint32_t merge_x = get_arg_val<uint32_t>(2);
    const uint32_t merge_y = get_arg_val<uint32_t>(3);
    const uint32_t slot = get_arg_val<uint32_t>(4);
    const uint32_t classes_addr = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(2);
    constexpr uint32_t sem_id = get_compile_time_arg_val(3);
    constexpr auto c_args = TensorAccessorArgs<4>();

    // every column's 32 values in L1 (cb_out holds them all), the max on the fly
    uint32_t best_key = 0, best_id = 0xFFFFFFFF;
    const uint32_t base = get_read_ptr(cb_out);
    for (uint32_t j = 0; j < cols; ++j) {
        cb_wait_front(cb_out, j + 1);
        volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + 64 * j);
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t key = bf16_key(v[c]);
            if (key > best_key) {
                best_key = key;
                best_id = 32 * j + c;
            }
        }
    }
    const uint32_t scr = (get_write_ptr(cb_scr) + 63) & ~63u;
    const auto s_c = TensorAccessor(c_args, classes_addr, 65536 * 4);
    noc_async_read(s_c.get_noc_addr(0, (4 * best_key) & ~63u), scr, 64);
    noc_async_read_barrier();
    const uint32_t low = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr)[best_key & 15];
    uint32_t low_id = best_id;
    for (uint32_t i = 0; i < 32 * cols && i < low_id; ++i) {
        const uint32_t key = bf16_key(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base)[i]);
        if (key >= low) {
            low_id = i;
        }
    }
    cb_pop_front(cb_out, cols);
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 64);
    r[0] = best_key;
    r[1] = low;
    r[2] = first_id + low_id;
    r[3] = 0;
    noc_async_write(scr + 64, get_noc_addr(merge_x, merge_y, get_write_ptr(cb_slots) + 16 * slot), 16);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(merge_x, merge_y, get_semaphore(sem_id)), 1);
    noc_async_atomic_barrier();
}
