/*
 * SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * kernels/shared/x280_counter_window.h
 *
 * The X280's credit and pointer publication: one counter word stored into a
 * peer tile's L1 through a write-only, posted, static-VC TLB window. Packet
 * injection's credit and STOP words and eth_credit's W2 and R1 use it. X280
 * device builds only.
 */
#ifndef X280_KERNELS_SHARED_X280_COUNTER_WINDOW_H
#define X280_KERNELS_SHARED_X280_COUNTER_WINDOW_H

#include <stdint.h>

#include "noc.h"

/* Request VC of every counter window. */
#define X280_COUNTER_WINDOW_VC 2u

/**
 * Opens 2 MiB window `slot` onto L1 offset 0 of the tile at `at` as a counter
 * window: ordering mode 2 (Posted Writes), static VC X280_COUNTER_WINDOW_VC,
 * unicast. Returns as noc_open_window_2m; the caller keeps its policy for an
 * unverified slot and clears the slot itself.
 *
 * Write-only: nothing may load through the window. A read and a write on one
 * static VC is unsafe at some destinations, and a static VC orders no read
 * behind a write.
 */
static inline noc_window_status_t x280_counter_window_open(uint32_t slot, x280_noc_coord_t at, volatile void** window) {
    return noc_open_window_2m(slot, at, 0, NOC_ORDER_POSTED, X280_COUNTER_WINDOW_VC, window);
}

/**
 * Stores `value` at `dst`, a word inside a counter window, after every
 * earlier load and store of this hart (fence iorw). Returns once the store is
 * accepted, not delivered, and cannot fail; only the peer observes delivery.
 *
 * Successive stores to one address through one counter window land in program
 * order (assumed, docs/measurement.md item 21). Nothing orders them
 * against stores to other addresses or through other windows, so a counter
 * that covers data needs that data complete first. A misaimed window loses the
 * write or hangs the hart.
 */
static inline void x280_counter_store(volatile uint32_t* dst, uint32_t value) {
    x280_fence_iorw();
    *dst = value;
}

#endif /* X280_KERNELS_SHARED_X280_COUNTER_WINDOW_H */
