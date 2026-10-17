// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/shared/core_sync_x280_roster.h -- The X280 core_sync transport's
// release and arrival targets, in the layout host-written argument blocks
// carry, so firmware executes them in place. Coordinates are virtual
// (translated), the space the L2CPU outbound TLB window takes.
#ifndef X280_CORE_SYNC_X280_ROSTER_H
#define X280_CORE_SYNC_X280_ROSTER_H

#include <stdint.h>

#include "../../include/x280_noc_coord.h"

/** One release destination outside every group. addr is a byte address in the
 * destination's own space; same_tile 1 selects a direct LIM store over a
 * windowed write, is only ever set for a hart of the sender's tile, and leaves
 * noc unread. */
typedef struct {
    x280_noc_coord_t noc;
    uint32_t addr;
    uint32_t same_tile;
} core_sync_x280_endpoint_t;

/** A rectangle of Tensix tiles reached by one broadcast, and the one address
 * the token lands on on every destination. num_dests is the logical receiver
 * count the host selected, nonzero and at most the rectangle's area; an L2CPU
 * is never one of them. Opted-out tiles inside the rectangle (translated
 * columns 8-9, harvested columns) receive nothing and are not counted. */
typedef struct {
    x280_noc_rect_t rect;
    uint32_t addr;
    uint32_t num_dests;
} core_sync_x280_group_t;

/** Where a tier-2 worker writes its arrival: the base of the director's ready
 * table and this worker's slot index in it, which is also the salt of the
 * token written there. The slot address is derived, never carried. */
typedef struct {
    x280_noc_coord_t noc;
    uint32_t ready_base;
    uint32_t worker_index;
    uint32_t same_tile;
} core_sync_x280_arrival_target_t;

#if defined(__cplusplus)
inline bool operator==(const core_sync_x280_endpoint_t& a, const core_sync_x280_endpoint_t& b) {
    return a.noc.x == b.noc.x && a.noc.y == b.noc.y && a.addr == b.addr && a.same_tile == b.same_tile;
}
inline bool operator!=(const core_sync_x280_endpoint_t& a, const core_sync_x280_endpoint_t& b) { return !(a == b); }
#endif

#endif /* X280_CORE_SYNC_X280_ROSTER_H */
