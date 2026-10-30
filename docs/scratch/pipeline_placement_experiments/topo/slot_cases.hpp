// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Stage-to-slot instances with stages bigger than one slice, shared by topo_solver_bench.cpp and
// resolver_same_cases.cpp.
//
// The mesh is carved first: each mesh (R x C host slices) is tiled into slots, rectangles of
// slices, uniform (2x2, 2x4) or mixed. A stage is then ONE node whose size (in slices) must match
// the slot's, and the pipeline is a ring or chain of such stages. This is the level the design's
// Tier C hands to the resolver after C1, and the only level at which a node-to-node engine solves
// multi-slice stages exactly: the carve is given, the engines decide stage -> slot and the
// crossing. Slots touch when any of their slices are adjacent or joined by a cable.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct SlotRect {
    int mesh, r0, c0, h, w;  // in slices
};
struct SlotCable {
    int mesh_a, row_a, col_a, mesh_b, row_b, col_b;  // slice coordinates
};
struct SlotCase {
    std::string name;
    std::vector<std::pair<int, int>> meshes;  // (R, C) in slices
    std::vector<SlotRect> slots;
    std::vector<SlotCable> cables;
    std::vector<std::pair<int, int>> stage_shapes;  // (h, w) in slices, ring order
    bool ring = true;
};

inline void tile_uniform(SlotCase& c, int mesh, int R, int C, int h, int w) {
    for (int r = 0; r + h <= R; r += h) {
        for (int col = 0; col + w <= C; col += w) {
            c.slots.push_back({mesh, r, col, h, w});
        }
    }
}

// 4x8 mesh: two 2x4 slots on top, four 2x2 slots below ("mixed4"-like).
inline void tile_mixed_4x8(SlotCase& c, int mesh) {
    c.slots.push_back({mesh, 0, 0, 2, 4});
    c.slots.push_back({mesh, 0, 4, 2, 4});
    for (int col = 0; col < 8; col += 2) {
        c.slots.push_back({mesh, 2, col, 2, 2});
    }
}

inline std::vector<SlotCase> slot_cases() {
    std::vector<SlotCase> out;
    const std::pair<int, int> B{2, 2}, W{2, 4};
    auto east_west_cables = [](SlotCase& c, int a, int b, int k) {
        // k cables from column 7 of mesh a to column 0 of mesh b, rows 0..k-1 (slice coordinates)
        for (int r = 0; r < k; ++r) {
            c.cables.push_back({a, r, 7, b, r, 0});
        }
    };
    {
        SlotCase c{"8x8 mesh tiled 2x2: ring of 16 x 2x2", {{8, 8}}};
        tile_uniform(c, 0, 8, 8, 2, 2);
        c.stage_shapes.assign(16, B);
        out.push_back(c);
    }
    {
        SlotCase c{"8x8 mesh tiled 2x2: ring of 15 x 2x2 (1 spare slot)", {{8, 8}}};
        tile_uniform(c, 0, 8, 8, 2, 2);
        c.stage_shapes.assign(15, B);
        out.push_back(c);
    }
    {
        SlotCase c{"8x8 mesh tiled 2x4: ring of 8 x 2x4", {{8, 8}}};
        tile_uniform(c, 0, 8, 8, 2, 4);
        c.stage_shapes.assign(8, W);
        out.push_back(c);
    }
    {
        SlotCase c{"16x16 mesh tiled 2x2: ring of 64 x 2x2", {{16, 16}}};
        tile_uniform(c, 0, 16, 16, 2, 2);
        c.stage_shapes.assign(64, B);
        out.push_back(c);
    }
    {
        // heterogeneous, feasible order: 2x4, 2x2, 2x2, 2x2, 2x2, 2x4 walks the bottom row and back
        SlotCase c{"4x8 mixed tiling: ring 2x4,2x2,2x2,2x2,2x2,2x4 (feasible)", {{4, 8}}};
        tile_mixed_4x8(c, 0);
        c.stage_shapes = {W, B, B, B, B, W};
        out.push_back(c);
    }
    {
        // heterogeneous, infeasible order: the two 2x4 slots are adjacent to each other and to two
        // 2x2 slots each, so 2x4,2x2,2x2,2x4,2x2,2x2 has no ring
        SlotCase c{"4x8 mixed tiling: ring 2x4,2x2,2x2,2x4,2x2,2x2 (none)", {{4, 8}}};
        tile_mixed_4x8(c, 0);
        c.stage_shapes = {W, B, B, W, B, B};
        out.push_back(c);
    }
    for (int n : {2, 4}) {
        SlotCase c{std::to_string(n) + " meshes of 4x8 tiled 2x2, 4 cables/pair: ring of " + std::to_string(8 * n) + " x 2x2", {}};
        for (int m = 0; m < n; ++m) {
            c.meshes.push_back({4, 8});
            tile_uniform(c, m, 4, 8, 2, 2);
        }
        for (int m = 0; m < n; ++m) {
            east_west_cables(c, m, (m + 1) % n, 4);
        }
        c.stage_shapes.assign(8 * n, B);
        out.push_back(c);
    }
    {
        SlotCase c{"2 meshes of 4x8 mixed tiling, 4 cables/pair: ring of 12 heterogeneous", {{4, 8}, {4, 8}}};
        tile_mixed_4x8(c, 0);
        tile_mixed_4x8(c, 1);
        east_west_cables(c, 0, 1, 4);
        east_west_cables(c, 1, 0, 4);
        c.stage_shapes = {W, B, B, B, B, W, W, B, B, B, B, W};
        out.push_back(c);
    }
    {
        SlotCase c{"4 meshes of 4x8 tiled 2x2, 4 cables/pair: chain of 32 x 2x2", {}};
        for (int m = 0; m < 4; ++m) {
            c.meshes.push_back({4, 8});
            tile_uniform(c, m, 4, 8, 2, 2);
        }
        for (int m = 0; m + 1 < 4; ++m) {
            east_west_cables(c, m, m + 1, 4);
        }
        c.stage_shapes.assign(32, B);
        c.ring = false;
        out.push_back(c);
    }
    return out;
}

// The same level with no slot smaller than two slices: meshes tiled into 1x2 or 2x1 dominoes, or
// 2x2 on the top half and 1x2 on the bottom half, on one mesh and across 2 or 4 meshes.
inline std::vector<SlotCase> slot_cases_min2() {
    std::vector<SlotCase> out;
    const std::pair<int, int> H{1, 2}, V{2, 1}, B{2, 2};
    auto east_west_cables = [](SlotCase& c, int a, int b, int k) {
        for (int r = 0; r < k; ++r) {
            c.cables.push_back({a, r, 7, b, r, 0});
        }
    };
    auto tile_half = [](SlotCase& c, int mesh, int R, int C) {  // top half 2x2, bottom half 1x2
        for (int r = 0; r + 2 <= R / 2; r += 2) {
            for (int col = 0; col + 2 <= C; col += 2) {
                c.slots.push_back({mesh, r, col, 2, 2});
            }
        }
        for (int r = R / 2; r < R; ++r) {
            for (int col = 0; col + 2 <= C; col += 2) {
                c.slots.push_back({mesh, r, col, 1, 2});
            }
        }
    };
    {
        SlotCase c{"no-1x1 4x8 tiled 1x2: ring of 16 x 1x2", {{4, 8}}};
        tile_uniform(c, 0, 4, 8, 1, 2);
        c.stage_shapes.assign(16, H);
        out.push_back(c);
    }
    {
        SlotCase c{"no-1x1 4x8 tiled 2x1: ring of 16 x 2x1", {{4, 8}}};
        tile_uniform(c, 0, 4, 8, 2, 1);
        c.stage_shapes.assign(16, V);
        out.push_back(c);
    }
    {
        SlotCase c{"no-1x1 8x8 tiled 1x2: ring of 32 x 1x2", {{8, 8}}};
        tile_uniform(c, 0, 8, 8, 1, 2);
        c.stage_shapes.assign(32, H);
        out.push_back(c);
    }
    {
        SlotCase c{"no-1x1 8x8 tiled 1x2: ring of 31 x 1x2 (1 spare slot)", {{8, 8}}};
        tile_uniform(c, 0, 8, 8, 1, 2);
        c.stage_shapes.assign(31, H);
        out.push_back(c);
    }
    {
        SlotCase c{"no-1x1 8x8 half 2x2 / half 1x2: ring of 8 x 2x2 then 16 x 1x2", {{8, 8}}};
        tile_half(c, 0, 8, 8);
        c.stage_shapes.assign(8, B);
        c.stage_shapes.insert(c.stage_shapes.end(), 16, H);
        out.push_back(c);
    }
    {
        SlotCase c{
            "no-1x1 8x8 half 2x2 / half 1x2: ring alternating 2x2,1x2 (none: 2x2 slots touch only 2 of 16 dominoes "
            "each)",
            {{8, 8}}};
        tile_half(c, 0, 8, 8);
        for (int i = 0; i < 8; ++i) {
            c.stage_shapes.push_back(B);
            c.stage_shapes.push_back(H);
            c.stage_shapes.push_back(H);
        }
        out.push_back(c);
    }
    {
        SlotCase c{"no-1x1 16x16 tiled 1x2: ring of 128 x 1x2", {{16, 16}}};
        tile_uniform(c, 0, 16, 16, 1, 2);
        c.stage_shapes.assign(128, H);
        out.push_back(c);
    }
    for (int n : {2, 4}) {
        for (int k : {1, 4}) {
            SlotCase c{
                "no-1x1 " + std::to_string(n) + " meshes of 4x8 tiled 1x2, " + std::to_string(k) +
                    " cables/pair: ring of " + std::to_string(16 * n) + " x 1x2",
                {}};
            for (int m = 0; m < n; ++m) {
                c.meshes.push_back({4, 8});
                tile_uniform(c, m, 4, 8, 1, 2);
            }
            for (int m = 0; m < n; ++m) {
                east_west_cables(c, m, (m + 1) % n, k);
            }
            c.stage_shapes.assign(16 * n, H);
            out.push_back(c);
        }
    }
    {
        SlotCase c{"no-1x1 4 meshes of 4x8 tiled 2x1, 4 cables/pair: chain of 64 x 2x1", {}};
        for (int m = 0; m < 4; ++m) {
            c.meshes.push_back({4, 8});
            tile_uniform(c, m, 4, 8, 2, 1);
        }
        for (int m = 0; m + 1 < 4; ++m) {
            east_west_cables(c, m, m + 1, 4);
        }
        c.stage_shapes.assign(64, V);
        c.ring = false;
        out.push_back(c);
    }
    {
        SlotCase c{
            "no-1x1 2 meshes of 4x8 half 2x2 / half 1x2, 4 cables/pair: ring of 2 x (4 x 2x2, 8 x 1x2)",
            {{4, 8}, {4, 8}}};
        tile_half(c, 0, 4, 8);
        tile_half(c, 1, 4, 8);
        east_west_cables(c, 0, 1, 4);
        east_west_cables(c, 1, 0, 4);
        for (int m = 0; m < 2; ++m) {
            c.stage_shapes.insert(c.stage_shapes.end(), 4, B);
            c.stage_shapes.insert(c.stage_shapes.end(), 8, H);
        }
        out.push_back(c);
    }
    return out;
}
