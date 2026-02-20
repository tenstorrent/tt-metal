// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "dev_mem_map.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC)
#include "ckernel.h"
#endif

namespace internal_ {

// Internal API - not for direct use in kernels
// Returns the hardware thread index for the current processor
//
// This index is used to access per-processor data structures shared between
// host and device (e.g. mailbox, RTAs, watcher debug)
// The ordering must match the host side expectation
//
// Quasar (tt-2xx) Tensix:
//   Index 0-7  : DM0-DM7 (data movement processors)
//   Index 8-11 : NEO0 Cluster (TRISC0-TRISC3)
//   Index 12-15: NEO1 Cluster (TRISC0-TRISC3)
//   Index 16-19: NEO2 Cluster (TRISC0-TRISC3)
//   Index 20-23: NEO3 Cluster (TRISC0-TRISC3)
//
// Blackhole/Wormhole (tt-1xx) Tensix:
//   Index 0: BRISC  (DM0)
//   Index 1: NCRISC (DM1)
//   Index 2: TRISC0
//   Index 3: TRISC1
//   Index 4: TRISC2
//
// Ethernet cores for all archs use PROCESSOR_INDEX
// ETH Wormhole: Index 0
// ETH Blackhole/Quasar: Index 0 to 1
inline __attribute__((always_inline)) uint32_t get_hw_thread_idx() {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC)
    uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    return NUM_DM_CORES + NUM_TRISC_CORES * neo_id + trisc_id;
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return static_cast<uint32_t>(hartid);
#else
    return PROCESSOR_INDEX;
#endif
}

}  // namespace internal_
