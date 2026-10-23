// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DFlash drafter q / k prep, writer (BRISC) (dflash/qk_prep.py): this core's normed + rotated Q and K tiles -> its
// shards of the head-sharded Q / K outputs (same core, same L1 address on every core).
//
// runtime args: [q_out, k_out (, kv write core noc x, y with PREP_NOTIFY)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_out = get_arg_val<uint32_t>(0);
    const uint32_t k_out = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_qo = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ko = get_compile_time_arg_val(1);
    constexpr uint32_t tile = 2048;
    cb_wait_front(cb_qo, 2);
    noc_async_write(get_read_ptr(cb_qo), get_noc_addr(q_out), 2 * tile);
#ifdef PREP_JOINT
    // the joint tile's k rows NQ .. NQ + NKV - 1 -> rows 0 .. NKV - 1 of the K shard (faces 0 / 1, 32 bytes each)
    for (uint32_t d = 0; d < 2; ++d) {
        for (uint32_t f = 0; f < 2; ++f) {
            for (uint32_t h = 0; h < NKV; ++h) {
                noc_async_write(
                    get_read_ptr(cb_qo) + d * tile + f * 512 + (NQ + h) * 32,
                    get_noc_addr(k_out + d * tile + f * 512 + h * 32),
                    32);
            }
        }
    }
#else
    cb_wait_front(cb_ko, 2);
    noc_async_write(get_read_ptr(cb_ko), get_noc_addr(k_out), 2 * tile);
#endif
    noc_async_write_barrier();
#ifdef PREP_NOTIFY
    // K shard written: tell the fused KV write (semaphore PREP_SEM_BASE on its core)
    noc_semaphore_inc(
        get_noc_addr(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), get_semaphore(PREP_SEM_BASE)), 1);
    noc_async_atomic_barrier();  // the increment lands before the kernel ends
#endif
    cb_pop_front(cb_qo, 2);
#ifndef PREP_JOINT
    cb_pop_front(cb_ko, 2);
#endif
}
