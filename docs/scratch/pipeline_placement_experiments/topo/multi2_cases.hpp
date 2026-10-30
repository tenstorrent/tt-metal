// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multi-mesh instances of 32-rank meshes (4x8 host slices each) shared by topo_solver_bench.cpp and resolver_same_cases.cpp.
//
// Every mesh is a 4x8 grid of host slices, 32 ranks (the largest host grid a repo MGD declares). Meshes
// sit in a ring (mesh i cabled to i+1, last to first) or a line (chains). Cables run east-west:
// from column 7 of mesh i to column 0 of mesh i+1, k per adjacent pair, rows chosen by a seeded LCG
// so each seed is one "seating". The pipeline is a ring or a chain of single-slice stages that
// visits `used_per_mesh` slices of every mesh; with 32 it fills the fabric.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Multi2Cable {
    int mesh_a, row_a, col_a, mesh_b, row_b, col_b;
};

struct Multi2Case {
    std::string name;
    int n_meshes;
    int R = 4, C = 8;
    std::vector<Multi2Cable> cables;
    int used_per_mesh;  // stages per mesh; 32 fills a 4x8 mesh
    bool ring;          // false: chain (no closing edge), meshes in a line
};

inline Multi2Case multi2_make(int n, int k, uint32_t seed, int used, bool ring) {
    auto lcg = [](uint32_t& s) {
        s = s * 1664525u + 1013904223u;
        return s >> 8;
    };
    Multi2Case c;
    c.n_meshes = n;
    c.used_per_mesh = used;
    c.ring = ring;
    uint32_t s = seed * 7919u + n * 31u + k;
    const int pairs = ring ? n : n - 1;
    for (int i = 0; i < pairs; ++i) {
        const int j = (i + 1) % n;
        // k distinct rows on the east edge of i and k on the west edge of j
        std::vector<int> ra, rb;
        while ((int)ra.size() < k) {
            int r = lcg(s) % 4;
            bool dup = false;
            for (int x : ra) {
                dup |= x == r;
            }
            if (!dup) {
                ra.push_back(r);
            }
        }
        while ((int)rb.size() < k) {
            int r = lcg(s) % 4;
            bool dup = false;
            for (int x : rb) {
                dup |= x == r;
            }
            if (!dup) {
                rb.push_back(r);
            }
        }
        for (int t = 0; t < k; ++t) {
            c.cables.push_back({i, ra[t], 7, j, rb[t], 0});
        }
    }
    c.name = std::to_string(n) + " 32-rank meshes " + (ring ? "ring" : "line") + ", " + std::to_string(k) +
             " cable/pair, " + std::to_string(used) + "/32 per mesh, seed " + std::to_string(seed);
    return c;
}

inline std::vector<Multi2Case> multi2_cases() {
    std::vector<Multi2Case> out;
    // rings of meshes, every slice used: the blitz decode shape
    for (int n : {2, 4, 8}) {
        for (int k : {1, 4}) {
            for (uint32_t seed : {0u, 1u}) {
                out.push_back(multi2_make(n, k, seed, 32, true));
            }
        }
    }
    // chains (prefill): no loopback, meshes in a line
    for (int n : {2, 4}) {
        for (int k : {1, 4}) {
            out.push_back(multi2_make(n, k, 0, 32, false));
        }
    }
    // spare slices: rings using 24 of 32 slices per mesh
    for (int n : {2, 4}) {
        out.push_back(multi2_make(n, 2, 0, 24, true));
    }
    return out;
}

// The wider sweep: three seeds, k = 1, 2, 4 cables per pair, rings and lines over 2, 4 and 8 meshes,
// and rings with 24 or 28 of 32 slices used. Seeds 10.. so no instance repeats multi2_cases().
// For k = 1 at full occupancy the colour test decides every case exactly (REPORT.md, "Which cases
// are solvable"); the other cases are ground-truthed by whichever engine returns a verified carve.
inline std::vector<Multi2Case> multi3_cases() {
    std::vector<Multi2Case> out;
    for (int n : {2, 4, 8}) {
        for (int k : {1, 2, 4}) {
            for (uint32_t seed : {10u, 11u, 12u}) {
                out.push_back(multi2_make(n, k, seed, 32, true));
            }
        }
    }
    for (int n : {2, 4, 8}) {
        for (int k : {1, 2, 4}) {
            for (uint32_t seed : {10u, 11u}) {
                out.push_back(multi2_make(n, k, seed, 32, false));
            }
        }
    }
    for (int n : {2, 4, 8}) {
        for (int k : {1, 2}) {
            for (int used : {24, 28}) {
                out.push_back(multi2_make(n, k, 10, used, true));
            }
        }
    }
    return out;
}
