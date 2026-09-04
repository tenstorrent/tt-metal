// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "dev_mem_map.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC)
#include "ckernel.h"
#endif

#if defined(ARCH_QUASAR) && (defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_DM))
extern thread_local uint32_t hw_thread_idx;
#endif

namespace internal_ {

#if defined(ARCH_QUASAR) && (defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_DM))
// Defined in api/debug/assert.h, after ASSERT is available there. Deliberately not included
// directly from this file: assert.h includes this header (for internal_::get_hw_thread_idx), so an
// #include of assert.h here would re-enter this file before ASSERT is defined on whichever side
// loses the include-order race. Forward-declaring instead breaks the cycle.
void check_hw_thread_idx(uint32_t cached, uint32_t raw);

inline __attribute__((always_inline)) uint32_t read_hw_thread_idx() {
#if defined(COMPILE_FOR_TRISC)
    uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    return NUM_DM_CORES + NUM_TRISC_CORES * neo_id + trisc_id;
#else
    uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return static_cast<uint32_t>(hartid);
#endif
}

// Fills ::hw_thread_idx for the calling hardware thread. Must run once per thread before anything calls
// get_hw_thread_idx().
inline __attribute__((always_inline)) void init_hw_thread_idx() { hw_thread_idx = read_hw_thread_idx(); }
#endif

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
// On Quasar this reads the hardware index directly so debug/mailbox indexing is not vulnerable to
// cached local-memory corruption. When watcher ASSERT is already available in the include context, it
// also checks that the cached TLS copy still matches the raw hardware value.
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
#if defined(ARCH_QUASAR) && (defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_DM))
    check_hw_thread_idx(hw_thread_idx, read_hw_thread_idx());
    return hw_thread_idx;
#else
    return PROCESSOR_INDEX;
#endif
}

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC)
inline __attribute__((always_inline)) uint32_t get_trisc_id() { return ckernel::csr_read<ckernel::CSR::TRISC_ID>(); }
inline __attribute__((always_inline)) uint32_t get_neo_id() { return ckernel::csr_read<ckernel::CSR::NEO_ID>(); }
#endif

}  // namespace internal_
