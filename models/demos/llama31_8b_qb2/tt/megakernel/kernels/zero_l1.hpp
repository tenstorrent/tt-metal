// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/dataflow_api.h"

// Clear a power-of-two, at least512B L1 region. Seed from the runtime's512B
// zero page, then double the initialized extent with disjoint local NoC reads.
// Every barrier protects the next copy's source; padding remains defined zero.
template <uint32_t Bytes>
inline void zero_l1(uint32_t destination) {
    static_assert(Bytes >= MEM_ZEROS_SIZE && (Bytes & (Bytes - 1)) == 0);
    noc_async_read(get_noc_addr(MEM_ZEROS_BASE), destination, MEM_ZEROS_SIZE);
    noc_async_read_barrier();
    for (uint32_t initialized = MEM_ZEROS_SIZE; initialized < Bytes; initialized *= 2) {
        noc_async_read(get_noc_addr(destination), destination + initialized, initialized);
        noc_async_read_barrier();
    }
}
