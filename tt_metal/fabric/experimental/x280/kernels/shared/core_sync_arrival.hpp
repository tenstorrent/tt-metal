// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/shared/core_sync_arrival.hpp -- Arrival-direction vocabulary shared by
// every core_sync transport. One definition, because a translation unit may name
// more than one transport.
#ifndef X280_CORE_SYNC_ARRIVAL_HPP
#define X280_CORE_SYNC_ARRIVAL_HPP

#include <stdint.h>

#include "core_sync_layout.h"

namespace core_sync {

/* Which arrival primitive a core can originate. A property of the CALLING core,
 * not of the director it signals: the worker's own hardware decides.
 *
 * Counter: one shared counter on the director, incremented by a NoC atomic. The
 * increment carries no identity, so the director learns how many arrived and not
 * which. Word: a private word per participant, bearing that worker's ready
 * token, for a core that cannot issue a NoC atomic. */
enum class ArrivalKind : uint32_t { Counter = 0u, Word = 1u };

/* One tier-2 arrival: where its word lives in the director's memory, and which
 * worker index derives the token that word must hold. addr is a director-local
 * byte address of an 8-byte slot; worker_index is the argument to
 * core_sync_ready_token. */
struct WordArrival {
    uint32_t addr = 0u;
    uint32_t worker_index = 0u;
};

/* Validates a borrowed arrival array before any element is polled. A nonzero
 * count requires an array; every address names an aligned core-sync slot; and
 * addresses and worker indices are unique. */
static inline bool word_arrivals_valid(const WordArrival* arrivals, uint32_t count) {
    if (count > 0u && arrivals == nullptr) {
        return false;
    }
    for (uint32_t i = 0u; i < count; ++i) {
        if (arrivals[i].addr == 0u || (arrivals[i].addr % CORE_SYNC_SLOT_BYTES) != 0u) {
            return false;
        }
        for (uint32_t j = 0u; j < i; ++j) {
            if (arrivals[i].addr == arrivals[j].addr || arrivals[i].worker_index == arrivals[j].worker_index) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace core_sync

#endif /* X280_CORE_SYNC_ARRIVAL_HPP */
