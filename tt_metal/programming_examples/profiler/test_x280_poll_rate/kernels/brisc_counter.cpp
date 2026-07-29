// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// BRISC free-running counter. Runs on the data-movement RISC (RISCV_0 / BRISC)
// of one Tensix worker core and increments a u32 in its own L1, forever. The
// X280 polls this word over the NOC; a steadily increasing value lets the host
// confirm the X280 is reading live data.
//
// The `fence` after each store is REQUIRED: without it the in-loop stores sit
// in the core's store buffer and never drain to NoC-visible L1 (a returning
// kernel drains via tt-metal's BRISC epilogue, but this loop never returns).
//
// Launched non-blocking (EnqueueMeshWorkload blocking=false); never returns.
// The board reset at the start of the next run tears it down.
//
// COUNTER_ADDR is supplied by the host launcher as a compile define.

#include <cstdint>

#ifndef COUNTER_ADDR
#define COUNTER_ADDR 0x80000
#endif

void kernel_main() {
    volatile uint32_t* counter = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(COUNTER_ADDR));
    uint32_t v = 0;
    for (;;) {
        v++;
        *counter = v;
        __asm__ volatile("fence ow, ow" ::: "memory");  // drain the store to L1
    }
}
