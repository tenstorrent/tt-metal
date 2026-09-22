// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/dataflow_api.h"

// Blackhole DRAM reads require identical low six address bits at both ends.
// Reserve at least128B of scratch for a scalar, or size+128B for a row. Keeping
// the actual read small avoids accessing padding outside a short page table.
inline uint32_t aligned_read_destination(uint32_t scratch, uint64_t source) {
    return ((scratch + 63u) & ~63u) + (static_cast<uint32_t>(source) & 63u);
}
inline uint32_t read_scalar_u32(uint64_t source, uint32_t scratch) {
    const uint32_t destination = aligned_read_destination(scratch, source);
    noc_async_read(source, destination, 4);
    noc_async_read_barrier();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination);
}
