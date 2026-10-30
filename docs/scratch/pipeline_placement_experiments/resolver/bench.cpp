// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tt::tt_fabric {
struct MeshId {
    uint32_t v;
    uint32_t operator*() const { return v; }
};
struct FabricNodeId {
    MeshId mesh_id;
    uint32_t chip_id;
};
using EdgeInputTuple = std::tuple<std::string, std::string, bool>;
struct ResolvedEdge {
    std::string src, dst;
    bool is_loopback = false;
    uint32_t exit_row = 0, exit_col = 0, entry_row = 0, entry_col = 0;
    std::optional<uint32_t> exit_core_slot, entry_core_slot;
};
struct GraphLayoutResult {
    std::vector<std::string> stage_order;
    std::map<std::string, size_t> node_to_submesh;
    std::vector<ResolvedEdge> resolved_edges;
    uint32_t h2d_entry_row = 0, h2d_entry_col = 0;
    std::optional<uint32_t> h2d_core_slot;
    uint32_t d2h_exit_row = 0, d2h_exit_col = 0;
    std::optional<uint32_t> d2h_core_slot;
};
namespace detail {
struct InternalChip {
    FabricNodeId fid;
    uint32_t row, col;
};
struct LinkPair {
    uint32_t exit_row, exit_col, entry_row, entry_col;
};
using ConnectionKey = std::pair<size_t, size_t>;
using DirectLinks = std::map<ConnectionKey, std::vector<LinkPair>>;
#include "search_body.inc"
}  // namespace detail
}  // namespace tt::tt_fabric

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::detail;

// Submesh = 4x2 chips. A grid of gr x gc submeshes per mesh.
struct Fabric {
    std::vector<std::vector<InternalChip>> chips;
    DirectLinks links;
};

void link_h(DirectLinks& l, size_t a, size_t b) {  // a left of b
    for (uint32_t r = 0; r < 4; ++r) {
        l[{a, b}].push_back({r, 1, r, 0});
        l[{b, a}].push_back({r, 0, r, 1});
    }
}
void link_v(DirectLinks& l, size_t a, size_t b) {  // a above b
    for (uint32_t c = 0; c < 2; ++c) {
        l[{a, b}].push_back({3, c, 0, c});
        l[{b, a}].push_back({0, c, 3, c});
    }
}

size_t add_grid(Fabric& f, uint32_t mesh, size_t gr, size_t gc) {
    const size_t base = f.chips.size();
    for (size_t i = 0; i < gr * gc; ++i) {
        std::vector<InternalChip> sub;
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 2; ++c) {
                sub.push_back({FabricNodeId{MeshId{mesh}, uint32_t(i * 8 + r * 2 + c)}, r, c});
            }
        }
        f.chips.push_back(sub);
    }
    for (size_t r = 0; r < gr; ++r) {
        for (size_t c = 0; c < gc; ++c) {
            if (c + 1 < gc) {
                link_h(f.links, base + r * gc + c, base + r * gc + c + 1);
            }
            if (r + 1 < gr) {
                link_v(f.links, base + r * gc + c, base + (r + 1) * gc + c);
            }
        }
    }
    return base;
}

void cable(DirectLinks& l, size_t a, size_t b) {
    l[{a, b}].push_back({0, 0, 0, 0});
    l[{b, a}].push_back({0, 0, 0, 0});
}

void run(const char* name, const Fabric& f, size_t stages) {
    std::vector<std::string> order;
    std::vector<EdgeInputTuple> edges;
    for (size_t i = 0; i < stages; ++i) {
        order.push_back("s" + std::to_string(i));
    }
    for (size_t i = 0; i + 1 < stages; ++i) {
        edges.emplace_back(order[i], order[i + 1], false);
    }
    edges.emplace_back(order.back(), order.front(), true);
    std::map<std::string, uint32_t> counts;
    for (auto& s : order) {
        counts[s] = 8;
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    try {
        resolve_pipeline_placement(order, edges, f.links, counts, f.chips, {});
    } catch (const std::exception&) {
        ok = false;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf(
        "%-44s stages=%3zu submeshes=%3zu %-10s %.6fs\n",
        name,
        stages,
        f.chips.size(),
        ok ? "feasible" : "INFEASIBLE",
        s);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    auto want = [&](const char* k) { return only.empty() || only == k; };
    if (want("feas")) {
        for (auto [r, c] : std::vector<std::pair<size_t, size_t>>{{2, 2}, {2, 4}, {4, 4}, {4, 8}, {8, 8}}) {
            Fabric f;
            add_grid(f, 0, r, c);
            run(("one mesh grid " + std::to_string(r) + "x" + std::to_string(c) + ", full ring").c_str(), f, r * c);
        }
        // Two 4x4 meshes, cables at same-row corners on opposite colours: feasible.
        Fabric f;
        size_t a = add_grid(f, 0, 4, 4), b = add_grid(f, 1, 4, 4);
        cable(f.links, a + 0, b + 0);
        cable(f.links, a + 1, b + 1);
        run("two 4x4 meshes, cables at (0,0),(0,1)", f, 32);
    }
    if (want("odd")) {
        // Odd bipartite grid: no Hamiltonian cycle, ring over all submeshes is infeasible.
        for (auto [r, c] : std::vector<std::pair<size_t, size_t>>{{3, 3}, {3, 5}, {5, 5}, {3, 7}, {5, 7}}) {
            Fabric f;
            add_grid(f, 0, r, c);
            run(("odd grid " + std::to_string(r) + "x" + std::to_string(c) + ", full ring").c_str(), f, r * c);
        }
    }
    if (want("cables")) {
        // Two gxg meshes, both cables land on same-colour slices: each mesh would need a Hamiltonian
        // path between same-colour cells, impossible in a bipartite grid.
        for (size_t g : {2, 3, 4}) {
            Fabric f;
            size_t a = add_grid(f, 0, g, g), b = add_grid(f, 1, g, g);
            cable(f.links, a + 0, b + 0);
            cable(f.links, a + g + 1, b + g + 1);
            run(("two " + std::to_string(g) + "x" + std::to_string(g) + " meshes, cables same colour").c_str(),
                f,
                2 * g * g);
        }
    }
    if (want("big")) {
        Fabric f;
        size_t a = add_grid(f, 0, 4, 8), b = add_grid(f, 1, 4, 8);
        cable(f.links, a + 0, b + 0);
        cable(f.links, a + 9, b + 9);
        run("two 4x8 meshes, cables same colour", f, 64);
    }
}
