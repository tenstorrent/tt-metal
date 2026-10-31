// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Bisect step 7: separate "the DRAM read is broken" from "the CB handoff is broken".
//
// Known good: compute fills Dst with a constant, packs, BRISC writes -> 1024/1024.
// Known bad:  NCRISC reads a tile from DRAM, compute copies it, BRISC writes -> zeros.
//
// That leaves two very different faults, and one run separates them. NCRISC reads
// the tile into the CB and then *overwrites the same L1 words with 5.0* before
// pushing. Now:
//
//   output all 5.0   -> NCRISC runs, its L1 write works, the CB handoff works and
//                       compute's copy works. Only the DRAM read failed.
//   output all 0     -> the handoff or the compute copy is broken, and the read is
//                       not even implicated.
//   output real data -> the read worked and the overwrite did not; then the earlier
//                       zeros came from somewhere else entirely.

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t cb_a = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> a_gen{a_addr, tile_bytes};
    cb_reserve_back(cb_a, 1);
    uint32_t dst = get_write_ptr(cb_a);
    noc_async_read_page(0, a_gen, dst);
    noc_async_read_barrier();
    // Overwrite what was just read with 5.0 (bf16 bit pattern 0x40a0).
    volatile uint16_t* q = (volatile uint16_t*)dst;
    for (uint32_t i = 0; i < tile_bytes / 2; ++i) {
        q[i] = 0x40a0u;
    }
    cb_push_back(cb_a, 1);
#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t cb_o = 16;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t o_bytes = 2048;

    const InterleavedAddrGenFast<true> o_gen{out_addr, o_bytes};
    cb_wait_front(cb_o, 1);
    noc_async_write_page(0, o_gen, get_read_ptr(cb_o));
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_o = 16;

    compute_kernel_hw_startup(cb_a, cb_a, cb_o);
    // copy_tile needs its op-specific init; pack_tile needs no init but the
    // unpack/math datacopy path does, and without it the copy yields zeros.
    copy_init(cb_a);
    cb_wait_front(cb_a, 1);
    tile_regs_acquire();
    copy_tile(cb_a, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_o, 1);
    pack_tile(0, cb_o);
    cb_push_back(cb_o, 1);
    tile_regs_release();
#endif
}
