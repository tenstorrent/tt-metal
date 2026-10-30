// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Helpers for code running on a CCE hart: the cycle counter, bounded waits, runtime args, and pointers into this
// CCE's TL1 and a peer CCE's TL1.

#pragma once

#include <stdint.h>

#include "internal/risc_attribs.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DRISC)

#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"

// Forward declared from dataflow_api.h
template <typename T>
T get_arg_val(int arg_idx);

namespace experimental::cce_hart {

FORCE_INLINE uint64_t read_cycle_counter() {
    uint64_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

// Counts a budget from construction. A budget of 0 never expires.
class WaitDeadline {
public:
    explicit WaitDeadline(uint64_t budget_cycles) :
        budget_cycles_(budget_cycles), start_cycles_(budget_cycles == 0 ? 0 : read_cycle_counter()) {}

    bool expired() const { return budget_cycles_ != 0 && read_cycle_counter() - start_cycles_ >= budget_cycles_; }

private:
    uint64_t budget_cycles_;
    uint64_t start_cycles_;
};

// Spins until `ready()` or `deadline` expires; returns whether it became ready.
template <typename Ready>
FORCE_INLINE bool wait_until(const WaitDeadline& deadline, Ready&& ready) {
    bool is_ready = ready();
    while (!is_ready && !deadline.expired()) {
        is_ready = ready();
    }
    return is_ready;
}

inline uint64_t arg_u64(uint32_t lo_index, uint32_t hi_index) {
    return (static_cast<uint64_t>(get_arg_val<uint32_t>(hi_index)) << 32) | get_arg_val<uint32_t>(lo_index);
}

// Words of this CCE's TL1 that another agent (the host or another hart) writes or reads.
template <typename T = uint32_t>
FORCE_INLINE volatile T* own_tl1_words(uint64_t tl1_offset) {
    return reinterpret_cast<volatile T*>(tl1_offset + static_cast<uint64_t>(MEM_L1_UNCACHED_BASE));
}

FORCE_INLINE volatile uint32_t* peer_tl1_word(uint32_t cce_index, uint32_t tl1_offset) {
    return reinterpret_cast<volatile uint32_t*>(cce_sram_address(cce_index, tl1_offset));
}

}  // namespace experimental::cce_hart

#endif
