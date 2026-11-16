// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 greedy terminal, local argmax worker (NCRISC) (fused_decode/greedy.py: GreedyB1).
//
// Scans row 0 of logits tiles [j0, j1) of this device's vocab shard ([1, 1, 32, vd] BF16 TILE, DRAM): the 64 B
// holding rows 0-1 of face 0 and of face 1 of each tile are read (64 B aligned for DRAM), row 0 = values
// 32 j .. 32 j + 15 and 32 j + 16 .. 32 j + 31. Keeps the largest value (BF16 bits mapped to an order-preserving
// unsigned key) M. Ties after the final-logit softcap: `classes` (greedy.py) maps every key to the first key of its run
// of equal softcapped BF16 values, L = classes[M]; the result is the lowest id with key >= L (every key in [L, M] ties
// with M after the softcap). Writes (M, L, id) to slot `slot` of the merge core, then increments its semaphore.
//
// runtime args: [logits_addr, j0, j1, merge_noc_x, merge_noc_y, slot, classes_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t bf16_key(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const uint32_t j1 = get_arg_val<uint32_t>(2);
    const uint32_t merge_x = get_arg_val<uint32_t>(3);
    const uint32_t merge_y = get_arg_val<uint32_t>(4);
    const uint32_t slot = get_arg_val<uint32_t>(5);
    const uint32_t classes_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(2);  // same address on every core of the program
    constexpr auto l_args = TensorAccessorArgs<3>();
    constexpr auto c_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    const uint32_t slots_addr = get_write_ptr(cb_slots);
    const auto s_l = TensorAccessor(l_args, logits_addr, 2048);

    const uint32_t scr = (get_write_ptr(cb_scr) + 63) & ~63u;  // 64 B aligned (DRAM reads): 128 B per tile, then 64 B
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_read(s_l.get_noc_addr(j, 0), scr + 128 * (j - j0), 64);
        noc_async_read(s_l.get_noc_addr(j, 512), scr + 128 * (j - j0) + 64, 64);
    }
    noc_async_read_barrier();
    uint32_t best_key = 0, best_id = 0xFFFFFFFF;
    for (uint32_t j = j0; j < j1; ++j) {
        volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 128 * (j - j0));
        for (uint32_t c = 0; c < 16; ++c) {
            const uint32_t k = bf16_key(v[c]);
            if (k > best_key) {
                best_key = k;
                best_id = 32 * j + c;
            }
        }
        for (uint32_t c = 0; c < 16; ++c) {
            const uint32_t k = bf16_key(v[32 + c]);
            if (k > best_key) {
                best_key = k;
                best_id = 32 * j + 16 + c;
            }
        }
    }
    const uint32_t res = scr + 128 * (j1 - j0);  // 64 B aligned: the class lookup block, then the result
    const auto s_c = TensorAccessor(c_args, classes_addr, 65536 * 4);
    noc_async_read(s_c.get_noc_addr(0, (4 * best_key) & ~63u), res, 64);
    noc_async_read_barrier();
    const uint32_t low = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res)[(best_key & 15)];
    uint32_t low_id = best_id;
    for (uint32_t j = j0; j < j1 && 32 * j < low_id; ++j) {
        volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 128 * (j - j0));
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t id = 32 * j + c;
            if (id < low_id && bf16_key(v[c < 16 ? c : 16 + c]) >= low) {
                low_id = id;
            }
        }
    }
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res);
    r[0] = best_key;
    r[1] = low;
    r[2] = low_id;
    r[3] = 0;
    noc_async_write(res, get_noc_addr(merge_x, merge_y, slots_addr + 16 * slot), 16);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(merge_x, merge_y, get_semaphore(sem_id)), 1);
    noc_async_atomic_barrier();
}
