// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of runtime_args_kernel.cpp. Reads positional
// vararg/common-vararg slots (bound via num_runtime_varargs /
// num_common_runtime_varargs in the KernelSpec schema) and writes them into
// the L1 result region anchored at RESULTS_ADDR via CoreLocalMem. The legacy
// raw-l1-ptr variant remains in runtime_args_kernel.cpp for callers still on
// the legacy host API.

#include <stdint.h>
#include "internal/risc_attribs.h"
#include "api/core_local_mem.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "dev_mem_map.h"
// TODO FIXME: this build system is ridiculously stupid
#ifdef COMPILE_FOR_TRISC
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "defines_generated.h"
#endif

void kernel_main() {
    // RESULTS_ADDR is the cached L1 address shared with the host (which reads/writes over the NOC and
    // must use the cached address). DM cores must reach that same physical memory through the uncached
    // alias, so add MEM_L1_UNCACHED_BASE here.
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    CoreLocalMem<uint32_t> results(RESULTS_ADDR + MEM_L1_UNCACHED_BASE);
#else
    CoreLocalMem<uint32_t> results(RESULTS_ADDR);
#endif
    constexpr uint32_t kCommonRTASeparation = 1024;
    uint64_t hartid = 0;
#ifdef COMPILE_FOR_DM
    // Quasar DM only: get the DM processor's hartid (DM2..DM7 on Quasar). Used
    // to index into a MAX_DMS-wide L1 region so each DM writes to its own slot.
    // TODO: Replace with get_thread_idx() kernel API when available.
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    // Quasar DM only: write the actual L1 base addresses at the end of CRTA
    // payload from all DMs so the host can verify per-DM CRTA placement.
    results[kCommonRTASeparation + MAX_DMS * NUM_RUNTIME_ARGS + hartid] = static_cast<uint32_t>(get_common_arg_addr(0));
#endif
    for (uint32_t i = 0; i < NUM_RUNTIME_ARGS; i++) {
#ifdef COMMON_RUNTIME_ARGS
        results[i + kCommonRTASeparation + hartid * NUM_RUNTIME_ARGS] = get_common_vararg(i);
#endif
        results[i] = get_vararg(i);
    }

#ifdef COORDS_ADDR
#ifdef DATA_MOVEMENT
    CoreLocalMem<uint32_t> coords(COORDS_ADDR);
    coords[0] = my_x[noc_index];
    coords[1] = my_y[noc_index];
    coords[2] = get_absolute_logical_x();
    coords[3] = get_absolute_logical_y();
    coords[4] = get_relative_logical_x();
    coords[5] = get_relative_logical_y();
#endif
#endif
}
