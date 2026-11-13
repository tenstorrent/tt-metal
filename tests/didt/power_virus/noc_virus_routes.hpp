// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOC power-virus routes for Blackhole, adapted to the live Tensix columns.
// Host-only: the test computes each source core's destination and passes it to
// the NCRISC kernel.

#pragma once

#include <cstdint>

namespace noc_virus {

constexpr std::uint32_t NOC_SIZE_X = 17;

// NOC0 coordinates. X/Y transfers use NOC 0, x/y use NOC 1.
// Per Tensix row and NOC there's one X transfer to the adjacent Tensix column
// upstream, so it travels almost the whole row ring. Likewise one Y transfer per
// Tensix column and NOC. Each tile sources at most one transfer.
struct Work {
    std::uint8_t src_x;
    std::uint8_t src_y;
    char direction;
    std::uint8_t dest;
};

constexpr Work noc_map[] = {
    {2, 2, 'X', 1},    {5, 2, 'y', 3},    {10, 2, 'y', 3},  {15, 2, 'x', 16},  {3, 3, 'X', 2},    {6, 3, 'Y', 2},
    {11, 3, 'y', 4},   {16, 3, 'x', 1},   {1, 4, 'x', 2},   {4, 4, 'X', 3},    {6, 4, 'y', 5},    {7, 4, 'Y', 3},
    {12, 4, 'y', 5},   {2, 5, 'x', 3},    {5, 5, 'X', 4},   {7, 5, 'y', 6},    {10, 5, 'Y', 4},   {13, 5, 'y', 6},
    {1, 6, 'Y', 5},    {3, 6, 'x', 4},    {6, 6, 'X', 5},   {11, 6, 'Y', 5},   {14, 6, 'y', 7},   {1, 7, 'y', 8},
    {2, 7, 'Y', 6},    {4, 7, 'x', 5},    {7, 7, 'X', 6},   {12, 7, 'Y', 6},   {15, 7, 'y', 8},   {2, 8, 'y', 9},
    {3, 8, 'Y', 7},    {5, 8, 'x', 6},    {11, 8, 'X', 10}, {13, 8, 'Y', 7},   {16, 8, 'y', 9},   {3, 9, 'y', 10},
    {4, 9, 'Y', 8},    {6, 9, 'x', 7},    {12, 9, 'X', 11}, {14, 9, 'Y', 8},   {4, 10, 'y', 11},  {5, 10, 'Y', 9},
    {10, 10, 'x', 11}, {13, 10, 'X', 12}, {15, 10, 'Y', 9}, {11, 11, 'x', 12}, {14, 11, 'X', 13}, {16, 11, 'Y', 10}};

// Live column sets are bitmaps indexed by NOC0 x.
struct Route {
    std::uint32_t noc;
    std::uint32_t src_x;
    std::uint32_t src_y;
    std::uint32_t dst_x;
    std::uint32_t dst_y;
};

namespace detail {

inline bool is_live(std::uint32_t live, std::uint32_t x) { return (live >> x) & 1; }

inline std::uint32_t prev_live(std::uint32_t live, std::uint32_t x) {
    do {
        x = (x + NOC_SIZE_X - 1) % NOC_SIZE_X;
    } while (!is_live(live, x));
    return x;
}

inline std::uint32_t next_live(std::uint32_t live, std::uint32_t x) {
    do {
        x = (x + 1) % NOC_SIZE_X;
    } while (!is_live(live, x));
    return x;
}

}  // namespace detail

// Route sourced by NOC0 tile (x, y), given the live Tensix columns.
// Y transfers in dead columns are dropped. X transfers whose source is dead move
// to the nearest free live tile upstream in the same row. X destinations are
// always the adjacent live column upstream, so the ring traversal stays long.
// With no harvesting this is exactly noc_map.
inline bool find_route(std::uint32_t live, std::uint32_t x, std::uint32_t y, Route* route) {
    if (live == 0) {
        return false;
    }

    // Sources moved off dead columns so far. Only the first num_moved are valid,
    // so nothing needs zeroing (keeps memset out of NCRISC code).
    constexpr std::uint32_t MAX_MOVED = 8;
    std::uint8_t moved_x[MAX_MOVED];
    std::uint8_t moved_y[MAX_MOVED];
    std::uint32_t num_moved = 0;

    auto is_taken = [&](std::uint32_t tx, std::uint32_t ty) {
        for (const auto& work : noc_map) {
            if (work.src_x == tx && work.src_y == ty && detail::is_live(live, work.src_x)) {
                return true;
            }
        }
        for (std::uint32_t i = 0; i < num_moved; i++) {
            if (moved_x[i] == tx && moved_y[i] == ty) {
                return true;
            }
        }
        return false;
    };

    for (const auto& work : noc_map) {
        const bool along_x = (work.direction == 'X' || work.direction == 'x');
        const std::uint32_t noc = (work.direction == 'x' || work.direction == 'y');
        std::uint32_t src_x = work.src_x;

        if (!along_x) {
            if (!detail::is_live(live, src_x)) {
                continue;
            }
        } else if (!detail::is_live(live, src_x)) {
            std::uint32_t tries = NOC_SIZE_X;
            do {
                src_x = detail::prev_live(live, src_x);
            } while (is_taken(src_x, work.src_y) && --tries > 0);
            if (tries == 0 || num_moved == MAX_MOVED) {
                continue;
            }
            moved_x[num_moved] = src_x;
            moved_y[num_moved] = work.src_y;
            num_moved++;
        }

        if (src_x != x || work.src_y != y) {
            continue;
        }

        route->noc = noc;
        route->src_x = src_x;
        route->src_y = work.src_y;
        if (along_x) {
            route->dst_x = (noc == 0) ? detail::prev_live(live, src_x) : detail::next_live(live, src_x);
            route->dst_y = work.src_y;
        } else {
            route->dst_x = src_x;
            route->dst_y = work.dest;
        }
        return true;
    }

    return false;
}

}  // namespace noc_virus
