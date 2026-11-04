// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// All-reduce reducer core, NCRISC: reads this chip's own chunk of the owned slice into slot c of the receive CB (a
// persistent L1 tensor), waits for the other 7 chips' chunks (global semaphore incremented by the fabric packets) and
// releases the compute kernel.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t CB_RECV = get_compile_time_arg_val(0);
    constexpr uint32_t Q = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t NS = get_compile_time_arg_val(3);  // sources (8)
    constexpr uint32_t IN_L1 = get_compile_time_arg_val(4);
    constexpr uint32_t CB_OWN =
        get_compile_time_arg_val(5);  // this chip's own chunk (not part of the receive CB: 7 remote slots)
    constexpr uint32_t NR = NS - 1;   // remote sources (7)
    constexpr auto in_args = TensorAccessorArgs<6>();
    constexpr uint32_t PIECE = 256;
    constexpr uint32_t TILEB = 2 * PIECE;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gsem = get_common_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(1);
    const auto in_acc = TensorAccessor(in_args, in_addr, 2048);
    auto src_addr = [&](uint32_t t, uint32_t off) -> uint64_t {
        if constexpr (IN_L1) {
            const uint32_t cx = get_arg_val<uint32_t>(2 + 2 * c), cy = get_arg_val<uint32_t>(3 + 2 * c);
            return get_noc_addr(cx, cy, in_addr + t * 2048 + off);
        } else {
            return in_acc.get_noc_addr(c * NT + t, off);
        }
    };

    cb_reserve_back(
        CB_RECV, NR * Q);  // the persistent receive tensor: one slot per REMOTE chip (slot = c' if c' < c else c' - 1)
    cb_reserve_back(CB_OWN, Q);  // a plain circular buffer of this program for the own chunk
    const uint32_t base = get_write_ptr(CB_OWN);
    for (uint32_t i = 0; i < Q; ++i) {
        noc_async_read(src_addr(p * Q + i, 0), base + i * TILEB, PIECE);
        noc_async_read(src_addr(p * Q + i, 512), base + i * TILEB + PIECE, PIECE);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem);
    {
        DeviceZoneScopedN("R_WAIT");
        noc_semaphore_wait_min(sem, NS - 1);
    }
    noc_semaphore_set(sem, 0);
    cb_push_back(CB_OWN, Q);
    cb_push_back(CB_RECV, NR * Q);
}
