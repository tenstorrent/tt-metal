// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR step 3b (oneshot_ar.py OneShotARP, MK_PS_DELTA_NS > 0): synthetic progressive producer standing in
// for the v12 down stage, whose owners complete their chunks one after another over the MLP tail. Per iteration: wait
// GO, then after rank x DELTA cycles release this owner's chunk: READY += 1 on the core that consumes it first (the
// link sender that reduce-scatters it, or this core's own receiver when this device owns the chunk).
// runtime args: [rank, n_targets, (target xy, READY word address or 0 = the shared READY) ...]
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include <cstdint>

// MK_PS_PROGW (debug, the hang hunt): per-RISC progress word at MK_PS_PROGW + 16 x slot = (it << 8) | step
#ifdef MK_PS_PROGW
#define PW(slot, it, step) (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>((MK_PS_PROGW) + 16 * (slot)) = ((it) << 8) | (step))
#else
#define PW(slot, it, step)
#endif
// GSEM (spec v0.9: no program semaphores in the engine): every *_sem_id compile arg is the uniform L1 address of a
// global semaphore instead of a program-semaphore id.
#ifndef MK_PS_SEM
#ifdef MK_PS_GSEM
#define MK_PS_SEM(id) (id)
#else
#define MK_PS_SEM(id) get_semaphore(id)
#endif
#endif

void kernel_main() {
    constexpr uint32_t ITERS = get_compile_time_arg_val(0);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t DELTA = get_compile_time_arg_val(3);
    const uint32_t rank = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(go_sem_id));
    for (uint32_t it = 0; it < ITERS; ++it) {
        PW(3, it, 0x10);
        noc_semaphore_wait_min(go, it + 1);
        PW(3, it, 0x11);
        if (rank > 0) {
            riscv_wait(rank * DELTA);
        }
        DeviceZoneScopedN("PS-P-RELEASE");
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t xy = get_arg_val<uint32_t>(2 + 2 * i);
            const uint32_t ra = get_arg_val<uint32_t>(3 + 2 * i);  // READY word: per-position (READYP) or the shared one
            noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, ra ? ra : MK_PS_SEM(ready_sem_id)), 1);
        }
        noc_async_atomic_barrier();
        PW(3, it, 0x12);
    }
#ifdef MK_PS_GSEM
    *go = 0;
#endif
}
