// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// REAL kernel_profiler marker generator for the data-movement RISCs (BRISC = RISCV_0, NCRISC = RISCV_1).
// Unlike producer_common.h (which hand-writes the ring), this emits genuine kernel_profiler.hpp zones via
// DeviceZoneScopedN -- the exact path a real workload takes -- so the X280 drain + host decode are exercised
// against the production 2-word marker + PROG/TIMER sticky format.
//
// Host controls (all via -D except the prog id, which is a runtime arg so it needn't recompile):
//   MARKER_COUNT : number of DeviceZoneScopedN zones this RISC emits (each = 1 START + 1 END marker)
//   WORK_SIZE    : nop iterations INSIDE each zone -> the zone duration (the production "rate" knob)
//   arg0         : the per-program runtime host-id; BRISC pushes it once as a STICKY_PROG (host forward-fills)
#include <cstdint>
// get_arg_val + the DeviceZone* macros come in via the data-movement kernel framework (kernel_includes.hpp
// -> brisck.cc/ncrisck.cc), same as the other dm example kernels -- no explicit dataflow include needed.

#ifndef MARKER_COUNT
#define MARKER_COUNT 4096u
#endif
#ifndef WORK_SIZE
#define WORK_SIZE 64u
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    // Runtime program id, pushed by the host (SetRuntimeArgs). Emitted once, at kernel start, as the
    // STICKY_PROG packet -- the host forward-fills it onto every following marker of this launch.
    uint32_t prog_id = get_arg_val<uint32_t>(0);
    DeviceZoneSetCounter(prog_id);
#endif
    for (uint32_t i = 0; i < (uint32_t)MARKER_COUNT; i++) {
        DeviceZoneScopedN("REALPROF-DM");
        for (volatile uint32_t j = 0; j < (uint32_t)WORK_SIZE; j++) {
            asm volatile("nop");
        }
    }
}
