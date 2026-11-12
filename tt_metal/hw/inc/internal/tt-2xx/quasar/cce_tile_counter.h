// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// LLK tile counters of a Keraunos CCE (a Quasar Ethernet core): 16 per CCE, each a 16-bit occupancy to which a
// register write adds atomically (Grendel CCE spec v1.0 §3.4.2, Table 7). The RESET, POSTED and BUFFER_CAPACITY offsets
// are the Quasar overlay's (llk_intf_api.hpp, interface 0), which equal the CCE's (chippy
// registers/ker/cce/cce_reg.h); its read-back and error-status offsets differ, so nothing here reads them.

#pragma once

#include <cstdint>

#ifndef KERAUNOS_CCE
#error "cce_tile_counter.h is for a Keraunos CCE (Quasar Ethernet core) only"
#endif

#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"

namespace internal_::cce_tile_counter {

constexpr uint32_t kNumCounters = 16;

// This CCE's address of counter @p id's POSTED register (@p id < kNumCounters). A 32-bit load returns the occupancy,
// adds to POSTED minus adds to ACKED mod 2^16; a store adds the value stored.
constexpr uint32_t posted_addr(uint32_t id) {
    return TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
           id * TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE +
           TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__POSTED_REG_OFFSET;
}

// Sets this CCE's counter @p id (< kNumCounters) to capacity @p capacity and occupancy 0, with its error flag clear.
// Capacity is written first because a capacity write may raise the error flag and RESET clears it (CM-077). An add
// that lands before the RESET is lost, so the counter's adders must not start before this returns; it fences, so the
// RESET is ordered before this hart's later loads and stores. Never blocks.
inline void reset(uint32_t id, uint32_t capacity) {
    overlay::llk_intf_set_capacity(overlay::LLK_INTERFACE_0, id, capacity);
    overlay::llk_intf_reset(overlay::LLK_INTERFACE_0, id);
    asm volatile("fence iorw, iorw" ::: "memory");
}

// Adds @p n to the counter whose POSTED register is at @p posted: posted_addr() on this CCE, or another tile's CCE
// register by system address, which reaches it over the SMN (Keraunos-E100 Table 30). One store, atomic at the
// counter, so any number of harts may add to one counter concurrently. Returns once the store issues; nothing tells
// the caller when it lands. Never blocks.
inline void add(uint64_t posted, uint32_t n) {
    *reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(posted)) = n;
}

}  // namespace internal_::cce_tile_counter
