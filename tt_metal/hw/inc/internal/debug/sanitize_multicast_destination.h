// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Watcher checks of a multicast against the nodes of its destination: a destination's address is reserved only on its
// own nodes, so the multicast rectangle must stay within them. Part of the NoC sanitizer, which includes this header.
#pragma once

#include "internal/debug/node_range.h"
#include "internal/debug/sanitize.h"

// The nodes a multicast destination's address is reserved on, which its traits report through multicast_node_set:
// num_nodes packed coordinates (x in bits 8-15, y in bits 0-7) and num_ranges rectangles. A destination with neither is
// not checked. Defined without Watcher too, so the
// traits needn't depend on it.
struct DebugMulticastNodeSet {
    const uint16_t* packed_xy_coords = nullptr;
    uint32_t num_nodes = 0;
    const DebugNodeRange* ranges = nullptr;
    uint32_t num_ranges = 0;
};

#if defined(DEBUG_SANITIZE_NOC_ENABLED)

struct DebugNodeCoord {
    uint32_t x;
    uint32_t y;

    bool operator==(DebugNodeCoord other) const { return x == other.x && y == other.y; }
};
// A multicast rectangle as the NoC walks it.
struct DebugMulticastRect {
    DebugNodeCoord min;
    DebugNodeCoord max;
    // Corners given against the NoC's direction wrap around the grid edge, which the checks report rather than model. A
    // corner past the NoC coordinate range is reported too, before the scans over the rectangle could loop on it.
    bool is_malformed;

    // A rectangle whose corners are already ordered.
    DebugMulticastRect(DebugNodeCoord lower, DebugNodeCoord upper) : min{lower}, max{upper}, is_malformed{false} {}
    // NoC 0 runs from the start corner up to the end corner, NoC 1 down.
    DebugMulticastRect(uint8_t noc_id, DebugNodeCoord start, DebugNodeCoord end) :
        min{noc_id == 0 ? start : end},
        max{noc_id == 0 ? end : start},
        is_malformed{min.x > max.x || min.y > max.y || max.x > NOC_NODE_ID_MASK || max.y > NOC_NODE_ID_MASK} {}
};
inline bool debug_rect_contains(const DebugMulticastRect& rect, DebugNodeCoord node) {
    return node.x >= rect.min.x && node.x <= rect.max.x && node.y >= rect.min.y && node.y <= rect.max.y;
}
// Under ATT the coordinates are in the map's frame, so a worker is what the map resolves, as in the generic check.
inline bool debug_is_worker(uint8_t noc_id, DebugNodeCoord node) {
#if defined(NOC_ATT_ENABLED)
    return noc_att::resolve(ACTIVE_ATT_MAP, noc_att::Address::worker(node.x, node.y, 0)).valid;
#else
    bool unused_is_virtual_coord = false;
    return get_core_type(noc_id, node.x, node.y, unused_is_virtual_coord) == AddressableCoreType::TENSIX;
#endif
}
// Worker nodes in a rectangle whose first corner is a worker. Counted rather than taken from the area: worker
// coordinates have gaps (Blackhole's non-worker columns, harvested rows or columns). Non-worker nodes come in whole
// rows and columns, so the count is the worker columns along the first row times the worker rows along the first
// column.
inline uint32_t debug_count_workers(uint8_t noc_id, const DebugMulticastRect& rect) {
    uint32_t num_worker_cols = 0;
    for (uint32_t x = rect.min.x; x <= rect.max.x; ++x) {
        num_worker_cols += debug_is_worker(noc_id, {x, rect.min.y});
    }
    uint32_t num_worker_rows = 0;
    for (uint32_t y = rect.min.y; y <= rect.max.y; ++y) {
        num_worker_rows += debug_is_worker(noc_id, {rect.min.x, y});
    }
    return num_worker_cols * num_worker_rows;
}
// The sender's coordinates, in the frame of the rectangle and the destination's nodes.
inline DebugNodeCoord debug_sender_coord(uint8_t noc_id) {
#if defined(NOC_ATT_ENABLED)
    return {
        my_x[noc_id] - ACTIVE_ATT_MAP.node_id_offset_x,
        my_y[noc_id] - ACTIVE_ATT_MAP.node_id_offset_y,
    };
#else
    return {my_x[noc_id], my_y[noc_id]};
#endif
}
// Reports the multicast unless num_inside, the destination's worker nodes inside the rectangle, covers every worker
// node of it. A plain multicast skips the sender, so a sender inside the rectangle need not hold the destination; a
// loopback multicast (MCAST_INCL_SRC) writes it, so it must. Only a worker sender is among the counted nodes.
inline void debug_sanitize_multicast_rect_covered(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t len,
    const DebugMulticastRect& rect,
    uint32_t num_inside,
    DebugNodeCoord sender,
    bool is_sender_node,
    bool includes_sender) {
    bool covered = false;
    if (!rect.is_malformed) {
        num_inside +=
            !includes_sender && !is_sender_node && debug_rect_contains(rect, sender) && debug_is_worker(noc_id, sender);
        covered = num_inside == debug_count_workers(noc_id, rect);
    }
    if (!covered) {
        debug_sanitize_post_addr_and_hang(
            noc_id,
            noc_addr,
            l1_addr,
            len,
            DEBUG_SANITIZE_NOC_MULTICAST,
            DEBUG_SANITIZE_NOC_WRITE,
            DEBUG_SANITIZE_NOC_TARGET,
            DebugSanitizeNocMulticastOutsideNodeSet);
    }
}

// The destination's worker nodes inside the rectangle, and whether the sender is one of the destination's nodes.
struct DebugNodesInside {
    uint32_t num_inside = 0;
    bool has_sender = false;
};
// Over the destination's packed coordinates.
inline DebugNodesInside debug_coords_inside(
    const DebugMulticastRect& rect, DebugMulticastNodeSet nodes, DebugNodeCoord sender) {
    DebugNodesInside inside;
    for (uint32_t i = 0; i < nodes.num_nodes; ++i) {
        const DebugNodeCoord node{
            static_cast<uint32_t>((nodes.packed_xy_coords[i] >> 8) & 0xFF),
            static_cast<uint32_t>(nodes.packed_xy_coords[i] & 0xFF),
        };
        inside.num_inside += debug_rect_contains(rect, node);
        inside.has_sender |= node == sender;
    }
    return inside;
}
// Over the destination's rectangles, counting the workers of each one's overlap with the rectangle.
inline DebugNodesInside debug_ranges_inside(
    uint8_t noc_id, const DebugMulticastRect& rect, DebugMulticastNodeSet nodes, DebugNodeCoord sender) {
    DebugNodesInside inside;
    for (uint32_t i = 0; i < nodes.num_ranges; ++i) {
        const DebugNodeRange& node_range = nodes.ranges[i];
        const DebugMulticastRect range{
            DebugNodeCoord{node_range.start.x, node_range.start.y},
            DebugNodeCoord{node_range.end.x, node_range.end.y},
        };
        inside.has_sender |= debug_rect_contains(range, sender);
        // Both rectangles have worker corners, so their overlap does too.
        const DebugMulticastRect overlap{
            DebugNodeCoord{
                rect.min.x > range.min.x ? rect.min.x : range.min.x,
                rect.min.y > range.min.y ? rect.min.y : range.min.y,
            },
            DebugNodeCoord{
                rect.max.x < range.max.x ? rect.max.x : range.max.x,
                rect.max.y < range.max.y ? rect.max.y : range.max.y,
            },
        };
        if (overlap.min.x <= overlap.max.x && overlap.min.y <= overlap.max.y) {
            inside.num_inside += debug_count_workers(noc_id, overlap);
        }
    }
    return inside;
}

// Every worker node of the rectangle must be one of the destination's nodes.
//
// Takes the nodes as a pointer and count, not a template over the destination, so Watcher builds carry one copy.
void __attribute__((noinline)) debug_sanitize_noc_multicast_rect_in_node_set(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t len,
    DebugNodeCoord start,
    DebugNodeCoord end,
    DebugMulticastNodeSet nodes,
    bool includes_sender) {
    if (nodes.packed_xy_coords == nullptr && nodes.ranges == nullptr) {
        return;
    }
    const DebugMulticastRect rect{noc_id, start, end};
    if (!rect.is_malformed && !debug_is_worker(noc_id, rect.min)) {
        return;  // left to the generic multicast check
    }
    const DebugNodeCoord sender = debug_sender_coord(noc_id);
    // Assumes the nodes are distinct, or a duplicate inside the rectangle can hide a missing one. A tensor's
    // BufferDistributionSpec built from a CoreRangeSet satisfies this, as do a CoreRangeSet's rectangles; a
    // BufferDistributionSpec built from a core list is not checked.
    const DebugNodesInside coords = debug_coords_inside(rect, nodes, sender);
    // A malformed rectangle is reported without scanning it, as its corners may be out of range.
    const DebugNodesInside ranges =
        rect.is_malformed ? DebugNodesInside{} : debug_ranges_inside(noc_id, rect, nodes, sender);
    debug_sanitize_multicast_rect_covered(
        noc_id,
        noc_addr,
        l1_addr,
        len,
        rect,
        coords.num_inside + ranges.num_inside,
        sender,
        coords.has_sender || ranges.has_sender,
        includes_sender);
}
#define DEBUG_SANITIZE_NOC_MULTICAST_RECT_IN_NODE_SET(                                  \
    noc_id, noc_a, worker_a, l, x_start, y_start, x_end, y_end, nodes, includes_sender) \
    debug_sanitize_noc_multicast_rect_in_node_set(                                      \
        noc_id,                                                                         \
        noc_a,                                                                          \
        worker_a,                                                                       \
        l,                                                                              \
        DebugNodeCoord{x_start, y_start},                                               \
        DebugNodeCoord{x_end, y_end},                                                   \
        nodes,                                                                          \
        includes_sender)

#else  // !DEBUG_SANITIZE_NOC_ENABLED

#define DEBUG_SANITIZE_NOC_MULTICAST_RECT_IN_NODE_SET( \
    noc_id, noc_a, worker_a, l, x_start, y_start, x_end, y_end, nodes, includes_sender)

#endif  // DEBUG_SANITIZE_NOC_ENABLED
