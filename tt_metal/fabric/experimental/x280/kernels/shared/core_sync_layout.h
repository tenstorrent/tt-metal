// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/shared/core_sync_layout.h -- Slot layout and tokens for core_sync_barrier.
//
// Each slot is 8 bytes, taken to be one LIM ECC granule, so no two writers share
// a granule. The 8-byte granule width is assumed (docs/measurement.md
// "Unconfirmed hardware assumptions", item 9): the SiFive X280 MC Manual
// 21G3.04.00 §14.2.6 gives L3 ECC as SECDED with no width. The documentation
// does not say whether a sub-granule write becomes a read-modify-write, so a
// shared granule could lose a neighbour's update. LIM is uncached
// scratchpad, so cache-line false sharing is not the concern here.
#ifndef X280_CORE_SYNC_LAYOUT_H
#define X280_CORE_SYNC_LAYOUT_H

#include <stdint.h>

#include "ubench_units.h"

#define CORE_SYNC_SLOT_BYTES UBENCH_LIM_GRANULE_BYTES

#define CORE_SYNC_READY_TAG 0x594E4301u   /* "SYNC" | 1 */
#define CORE_SYNC_RELEASE_TAG 0x594E4302u /* "SYNC" | 2 */

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct __attribute__((aligned(8))) {
    volatile uint32_t token;
    uint32_t pad;
} core_sync_slot_t;

#if defined(__cplusplus)
}
static_assert(sizeof(core_sync_slot_t) == CORE_SYNC_SLOT_BYTES, "core_sync_slot_t must be CORE_SYNC_SLOT_BYTES");
#elif !defined(__ASSEMBLER__)
_Static_assert(sizeof(core_sync_slot_t) == CORE_SYNC_SLOT_BYTES, "core_sync_slot_t must be CORE_SYNC_SLOT_BYTES");
#endif

#if defined(__cplusplus)
constexpr uint32_t core_sync_ready_token(uint32_t run_nonce, uint32_t worker_idx) {
    return run_nonce ^ CORE_SYNC_READY_TAG ^ (worker_idx << 8);
}

constexpr uint32_t core_sync_release_token(uint32_t run_nonce) { return run_nonce ^ CORE_SYNC_RELEASE_TAG; }

constexpr uint32_t core_sync_slot_addr(uint32_t base_addr, uint32_t slot_idx) {
    return base_addr + slot_idx * CORE_SYNC_SLOT_BYTES;
}
#else
static inline uint32_t core_sync_ready_token(uint32_t run_nonce, uint32_t worker_idx) {
    return run_nonce ^ CORE_SYNC_READY_TAG ^ (worker_idx << 8);
}

static inline uint32_t core_sync_release_token(uint32_t run_nonce) { return run_nonce ^ CORE_SYNC_RELEASE_TAG; }

static inline uint32_t core_sync_slot_addr(uint32_t base_addr, uint32_t slot_idx) {
    return base_addr + slot_idx * CORE_SYNC_SLOT_BYTES;
}
#endif

#endif /* X280_CORE_SYNC_LAYOUT_H */
