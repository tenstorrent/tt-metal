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

// Returns the hardware thread index for the current processor.
// - Quasar DM   : 0-7
// - Quasar TRISC: 8-23
// - BH/WH       : 0-4
inline __attribute__((always_inline)) uint32_t get_hw_thread_idx() {
#if defined(ARCH_QUASAR)
#if defined(COMPILE_FOR_TRISC)
    uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    return NUM_DM_CORES + NUM_TRISC_CORES * neo_id + trisc_id;
#elif defined(COMPILE_FOR_DM)
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return hartid;
#endif
#else
    return PROCESSOR_INDEX;
#endif
}

}  // namespace internal_
