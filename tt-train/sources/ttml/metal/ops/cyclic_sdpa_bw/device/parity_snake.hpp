// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The logical parity snake and its serpentine embedding into a core grid.
//
// Reference: main.tex, "Parity-snake placement" and "Example: C = 16 cores on
// a 4x4 grid". The Python reference is tt_flash_attn/device/topology.py.
//
// Two layers are kept apart, as they are there:
//
//   * the parity snake is a property of the schedule alone. For C cores it
//     lists the even cores in increasing order and then the odd cores in
//     decreasing order -- 2, 4, 6, 8, 7, 5, 3, 1 for C = 8. Consecutive
//     consumers of a row packet are equal or adjacent on it (folded
//     stride -2 locality), which is what makes the relay nearest-neighbour;
//   * an embedding maps snake positions to grid coordinates. The serpentine
//     (boustrophedon) embedding of a full w x h rectangle makes every snake
//     edge a single hop. Any other placement stays correct but can turn snake
//     edges into multi-hop routes, which silently changes the traffic the
//     collision-free lemma describes.
//
// Coordinates are (x, y) with x the column, relative to the rectangle's own
// origin; the program factory offsets them into the region it was given.
//
// Same constraints as cyclic_schedule.hpp: host and device, constexpr,
// integer-only, no std::, preconditions documented rather than checked.

#pragma once

#include <cstdint>

#include "cyclic_schedule.hpp"

namespace ttml::metal::ops::cyclic_sdpa_bw {

//: A position in the core rectangle, relative to its origin.
struct GridCoord {
    uint32_t x;
    uint32_t y;
};

//: A core's neighbours along the snake; kNoCore at either end.
struct SnakeNeighbors {
    uint32_t prev;
    uint32_t next;
};

//: The core at snake position k. Precondition: k < C.
//
// Positions 0 .. floor(C/2) - 1 hold the even cores ascending; the rest hold
// the odd cores descending from the largest odd core <= C.
constexpr uint32_t snake_core_at(uint32_t C, uint32_t k) {
    const uint32_t evens = C / 2u;
    if (k < evens) {
        return 2u * (k + 1u);
    }
    const uint32_t largest_odd = (C % 2u != 0u) ? C : C - 1u;
    return largest_odd - 2u * (k - evens);
}

//: The snake position of core c, the inverse of snake_core_at.
//: Precondition: 1 <= c <= C.
constexpr uint32_t snake_index_of(uint32_t C, uint32_t c) {
    if (c % 2u == 0u) {
        return c / 2u - 1u;
    }
    const uint32_t largest_odd = (C % 2u != 0u) ? C : C - 1u;
    return C / 2u + (largest_odd - c) / 2u;
}

//: The one or two snake neighbours of core c. Precondition: 1 <= c <= C.
constexpr SnakeNeighbors snake_neighbors(uint32_t C, uint32_t c) {
    const uint32_t k = snake_index_of(C, c);
    return {
        k > 0u ? snake_core_at(C, k - 1u) : kNoCore,
        k + 1u < C ? snake_core_at(C, k + 1u) : kNoCore,
    };
}

//: True if a and b are equal or adjacent on the snake.
//
// This is the property the relay depends on: between consecutive active
// timesteps a row packet either stays put or moves one step along the snake.
constexpr bool snake_adjacent(uint32_t C, uint32_t a, uint32_t b) {
    const uint32_t ka = snake_index_of(C, a);
    const uint32_t kb = snake_index_of(C, b);
    return (ka > kb ? ka - kb : kb - ka) <= 1u;
}

// The k-th cell of a boustrophedon path through a width x height rectangle:
// row 0 left to right, row 1 right to left, and so on. Consecutive cells are
// always a unit hop apart, which is what makes this a Hamiltonian path and
// the snake nearest-neighbour.
//
// Precondition: k < width * height.
constexpr GridCoord serpentine_coord(uint32_t width, uint32_t k) {
    const uint32_t y = k / width;
    const uint32_t along = k % width;
    return {(y % 2u == 0u) ? along : width - 1u - along, y};
}

//: Where core c sits when the snake is laid along the serpentine path.
//: Precondition: width * height == C and 1 <= c <= C.
constexpr GridCoord placement_of(uint32_t C, uint32_t width, uint32_t c) {
    return serpentine_coord(width, snake_index_of(C, c));
}

//: Manhattan distance between two cells.
constexpr uint32_t hops(GridCoord a, GridCoord b) {
    const uint32_t dx = a.x > b.x ? a.x - b.x : b.x - a.x;
    const uint32_t dy = a.y > b.y ? a.y - b.y : b.y - a.y;
    return dx + dy;
}

// True if every snake edge is a single physical hop under this embedding.
// The program factory should refuse a region where this fails rather than
// quietly accept multi-hop relay traffic.
constexpr bool placement_is_nearest_neighbor(uint32_t C, uint32_t width, uint32_t height) {
    if (width * height != C) {
        return false;
    }
    for (uint32_t k = 0u; k + 1u < C; ++k) {
        if (hops(serpentine_coord(width, k), serpentine_coord(width, k + 1u)) != 1u) {
            return false;
        }
    }
    return true;
}

}  // namespace ttml::metal::ops::cyclic_sdpa_bw
