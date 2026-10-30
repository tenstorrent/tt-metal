// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The same carve instances as topo_solver_bench.cpp, through the real resolver search of
// tt_metal/fabric/pipeline_builder.cpp (search_body.inc is copied out of the checkout by
// run_experiments.sh extract_resolver).
//
// Every host slice is one 4x2-chip submesh. Horizontally adjacent slices share 4 links (one per
// row), vertically adjacent ones 2 (one per column), a cable is one link at chip (0,0). Every slice
// of every stage is one resolver node of 8 chips ("s<k>_<i>_<j>"); a stage's own grid edges pin a
// 2x2 to a unit square and a 2xk to a straight ladder; the pipeline edge k -> k+1 runs between the
// same fixed port sub-nodes as in topo_solver_bench.cpp (bottom-right of k to top-left of k+1), with
// the closing edge marked as the loopback. Single-slice stages are the exact carve; multi-slice
// stages are the fixed-port variant. The resolver additionally chooses one link and core slots per
// edge, which the topology solver does not model.
//
// Every returned layout is re-checked independently (verify_layout); a failure prints BAD.
// Each solve runs in a forked child killed after TOPO_TIMEOUT seconds (default 60).
// Usage: resolver_same_cases [quick|multi|all]
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

#include "carve_planner_port.hpp"
#include "multi2_cases.hpp"
#include "slot_cases.hpp"

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::detail;
using Shape = std::pair<int, int>;

struct Mesh {
    int R, C;
};
struct Case {
    std::string name;
    std::vector<Mesh> meshes;
    std::vector<std::pair<size_t, size_t>> cables;  // submesh indices
    std::vector<Shape> shapes;
    bool fixed_ports;
    bool ring = true;  // false: chain, no closing edge
};

struct Fabric {
    std::vector<std::vector<InternalChip>> chips;
    DirectLinks links;
    std::vector<size_t> mesh_base;
};

size_t sid(const Fabric& f, int mesh, int r, int c, int C) { return f.mesh_base[mesh] + size_t(r) * C + c; }

void link_h(DirectLinks& l, size_t a, size_t b) {
    for (uint32_t r = 0; r < 4; ++r) {
        l[{a, b}].push_back({r, 1, r, 0});
        l[{b, a}].push_back({r, 0, r, 1});
    }
}
void link_v(DirectLinks& l, size_t a, size_t b) {
    for (uint32_t c = 0; c < 2; ++c) {
        l[{a, b}].push_back({3, c, 0, c});
        l[{b, a}].push_back({0, c, 3, c});
    }
}

Fabric build_fabric(const Case& cs) {
    Fabric f;
    for (size_t m = 0; m < cs.meshes.size(); ++m) {
        const auto [R, C] = cs.meshes[m];
        f.mesh_base.push_back(f.chips.size());
        for (int i = 0; i < R * C; ++i) {
            std::vector<InternalChip> sub;
            for (uint32_t r = 0; r < 4; ++r) {
                for (uint32_t c = 0; c < 2; ++c) {
                    sub.push_back({FabricNodeId{MeshId{uint32_t(m)}, uint32_t(i * 8 + r * 2 + c)}, r, c});
                }
            }
            f.chips.push_back(sub);
        }
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (c + 1 < C) {
                    link_h(f.links, sid(f, m, r, c, C), sid(f, m, r, c + 1, C));
                }
                if (r + 1 < R) {
                    link_v(f.links, sid(f, m, r, c, C), sid(f, m, r + 1, c, C));
                }
            }
        }
    }
    for (auto [a, b] : cs.cables) {
        f.links[{a, b}].push_back({0, 0, 0, 0});
        f.links[{b, a}].push_back({0, 0, 0, 0});
    }
    return f;
}

// Independent check of a returned layout: every stage on its own submesh, every edge's chosen link
// present in the link table for the two submeshes it joins, host I/O chips inside stage 0's submesh.
std::string verify_layout(
    const std::vector<std::string>& order,
    const std::vector<EdgeInputTuple>& edges,
    const DirectLinks& links,
    const std::vector<std::vector<InternalChip>>& chips,
    const GraphLayoutResult& res,
    const std::set<std::pair<size_t, size_t>>& cable_pairs) {
    if (res.node_to_submesh.size() != order.size()) {
        return "stage count";
    }
    std::set<size_t> used;
    for (const auto& [stage, sub] : res.node_to_submesh) {
        if (sub >= chips.size() || !used.insert(sub).second) {
            return "submesh reused or out of range";
        }
    }
    if (res.resolved_edges.size() != edges.size()) {
        return "edge count";
    }
    for (const auto& e : res.resolved_edges) {
        const size_t a = res.node_to_submesh.at(e.src), b = res.node_to_submesh.at(e.dst);
        const auto it = links.find({a, b});
        if (it == links.end()) {
            return "edge " + e.src + "->" + e.dst + " between unlinked submeshes";
        }
        bool found = false;
        for (const auto& l : it->second) {
            found |= l.exit_row == e.exit_row && l.exit_col == e.exit_col && l.entry_row == e.entry_row &&
                     l.entry_col == e.entry_col;
        }
        if (!found) {
            return "edge " + e.src + "->" + e.dst + " on a link not in the table";
        }
        // A crossing between meshes must be one of the declared cables (the logical topology), never
        // an arbitrary physical link: the table only holds declared cables, and this makes it explicit.
        if (*chips[a].front().fid.mesh_id != *chips[b].front().fid.mesh_id && !cable_pairs.contains({a, b}) &&
            !cable_pairs.contains({b, a})) {
            return "edge " + e.src + "->" + e.dst + " crosses meshes on an undeclared link";
        }
    }
    auto in_stage0 = [&](uint32_t r, uint32_t c) {
        for (const auto& ch : chips[res.node_to_submesh.at(order.front())]) {
            if (ch.row == r && ch.col == c) {
                return true;
            }
        }
        return false;
    };
    if (!in_stage0(res.h2d_entry_row, res.h2d_entry_col) || !in_stage0(res.d2h_exit_row, res.d2h_exit_col)) {
        return "host I/O chip outside stage 0";
    }
    return "";
}

std::string nname(int k, int i, int j) { return "s" + std::to_string(k) + "_" + std::to_string(i) + "_" + std::to_string(j); }

std::pair<std::string, double> timed(const Case& cs, int cap_s) {
    int fd[2];
    if (pipe(fd) != 0) {
        return {"err", 0};
    }
    const pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        alarm(cap_s);
        const Fabric f = build_fabric(cs);
        const int n = cs.shapes.size();
        std::vector<std::string> order;  // row-major inside a stage, stages in ring order: topological
        std::vector<EdgeInputTuple> edges;
        std::map<std::string, uint32_t> counts;
        for (int k = 0; k < n; ++k) {
            const auto [h, w] = cs.shapes[k];
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    order.push_back(nname(k, i, j));
                    counts[nname(k, i, j)] = 8;
                    if (j + 1 < w) {
                        edges.emplace_back(nname(k, i, j), nname(k, i, j + 1), false);
                    }
                    if (i + 1 < h) {
                        edges.emplace_back(nname(k, i, j), nname(k, i + 1, j), false);
                    }
                }
            }
        }
        for (int k = 0; k + (cs.ring ? 0 : 1) < n; ++k) {
            const auto [h, w] = cs.shapes[k];
            edges.emplace_back(nname(k, h - 1, w - 1), nname((k + 1) % n, 0, 0), k == n - 1);
        }
        const auto t0 = std::chrono::steady_clock::now();
        int ok = 1;
        try {
            const auto res = resolve_pipeline_placement(order, edges, f.links, counts, f.chips, {});
            const std::set<std::pair<size_t, size_t>> cable_pairs(cs.cables.begin(), cs.cables.end());
            if (!verify_layout(order, edges, f.links, f.chips, res, cable_pairs).empty()) {
                ok = 2;
            }
        } catch (const std::exception&) {
            ok = 0;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[64];
        std::snprintf(buf, sizeof buf, "%d %.6f", ok, s);
        (void)!write(fd[1], buf, std::strlen(buf) + 1);
        _exit(0);
    }
    close(fd[1]);
    char buf[64] = {0};
    const ssize_t got = read(fd[0], buf, sizeof buf - 1);
    close(fd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (got <= 0) {
        return {"cap", double(cap_s)};
    }
    int ok = 0;
    double s = 0;
    std::sscanf(buf, "%d %lf", &ok, &s);
    return {ok == 2 ? "BAD" : ok ? "yes" : "no ", s};
}

// The slot-level solve: chips with slot-local coordinates, links where slices of two slots are adjacent or
// a cable joins them, one node per stage with its chip count, then the resolver; returns
// (1 yes, 0 no, 2 BAD) and the solve seconds.
std::pair<int, double> solve_slots(const SlotCase& sc) {
    {
        std::vector<std::vector<InternalChip>> chips(sc.slots.size());
        std::map<std::tuple<int, int, int>, size_t> slot_of;
        for (size_t i = 0; i < sc.slots.size(); ++i) {
            const auto& t = sc.slots[i];
            for (int sr = t.r0; sr < t.r0 + t.h; ++sr) {
                for (int scn = t.c0; scn < t.c0 + t.w; ++scn) {
                    slot_of[{t.mesh, sr, scn}] = i;
                    for (uint32_t rr = 0; rr < 4; ++rr) {
                        for (uint32_t cc = 0; cc < 2; ++cc) {
                            const uint32_t C = sc.meshes[t.mesh].second;
                            chips[i].push_back(
                                {FabricNodeId{
                                     MeshId{uint32_t(t.mesh)}, uint32_t(((sr * 4 + rr) * C * 2) + scn * 2 + cc)},
                                 uint32_t((sr - t.r0) * 4 + rr),
                                 uint32_t((scn - t.c0) * 2 + cc)});
                        }
                    }
                }
            }
        }
        DirectLinks links;
        auto local = [&](size_t slot, int sr, int scn, uint32_t rr, uint32_t cc) {
            const auto& t = sc.slots[slot];
            return std::pair<uint32_t, uint32_t>{uint32_t((sr - t.r0) * 4 + rr), uint32_t((scn - t.c0) * 2 + cc)};
        };
        for (const auto& [key, a] : slot_of) {
            const auto [m, r, c] = key;
            auto it = slot_of.find({m, r, c + 1});
            if (it != slot_of.end() && it->second != a) {
                for (uint32_t rr = 0; rr < 4; ++rr) {
                    auto [er, ec] = local(a, r, c, rr, 1);
                    auto [nr, nc] = local(it->second, r, c + 1, rr, 0);
                    links[{a, it->second}].push_back({er, ec, nr, nc});
                    links[{it->second, a}].push_back({nr, nc, er, ec});
                }
            }
            it = slot_of.find({m, r + 1, c});
            if (it != slot_of.end() && it->second != a) {
                for (uint32_t cc = 0; cc < 2; ++cc) {
                    auto [er, ec] = local(a, r, c, 3, cc);
                    auto [nr, nc] = local(it->second, r + 1, c, 0, cc);
                    links[{a, it->second}].push_back({er, ec, nr, nc});
                    links[{it->second, a}].push_back({nr, nc, er, ec});
                }
            }
        }
        std::set<std::pair<size_t, size_t>> cable_pairs;
        for (const auto& cb : sc.cables) {
            const auto ia = slot_of.find({cb.mesh_a, cb.row_a, cb.col_a}),
                       ib = slot_of.find({cb.mesh_b, cb.row_b, cb.col_b});
            if (ia == slot_of.end() || ib == slot_of.end()) {
                continue;  // a cable on a slice no slot covers (spare slice after the carve)
            }
            const size_t a = ia->second, b = ib->second;
            cable_pairs.insert({a, b});
            auto [ar, ac] = local(a, cb.row_a, cb.col_a, 0, 0);
            auto [br, bc] = local(b, cb.row_b, cb.col_b, 0, 0);
            links[{a, b}].push_back({ar, ac, br, bc});
            links[{b, a}].push_back({br, bc, ar, ac});
        }
        const int n = sc.stage_shapes.size();
        std::vector<std::string> order;
        std::vector<EdgeInputTuple> edges;
        std::map<std::string, uint32_t> counts;
        for (int k = 0; k < n; ++k) {
            order.push_back("s" + std::to_string(k));
            counts[order.back()] = sc.stage_shapes[k].first * sc.stage_shapes[k].second * 8;
        }
        for (int k = 0; k + (sc.ring ? 0 : 1) < n; ++k) {
            edges.emplace_back(order[k], order[(k + 1) % n], k == n - 1);
        }
        const auto t0 = std::chrono::steady_clock::now();
        int ok = 1;
        try {
            const auto res = resolve_pipeline_placement(order, edges, links, counts, chips, {});
            if (!verify_layout(order, edges, links, chips, res, cable_pairs).empty()) {
                ok = 2;
            }
        } catch (const std::exception&) {
            ok = 0;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return {ok, s};
    }
}

// Pre-carved slots: every slot is one submesh of h*w slices (4x2 chips each); a stage is one node
// of h*w*8 chips. Links join slots wherever their slices are adjacent (4 per shared slice row,
// 2 per shared slice column) and at cables (one chip pair).
std::pair<std::string, double> timed_slots(const SlotCase& sc, int cap_s) {
    int fd[2];
    if (pipe(fd) != 0) {
        return {"err", 0};
    }
    const pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        alarm(cap_s);
        const auto [ok, s] = solve_slots(sc);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%d %.6f", ok, s);
        (void)!write(fd[1], buf, std::strlen(buf) + 1);
        _exit(0);
    }
    close(fd[1]);
    char buf[64] = {0};
    const ssize_t got = read(fd[0], buf, sizeof buf - 1);
    close(fd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (got <= 0) {
        return {"cap", double(cap_s)};
    }
    int ok = 0;
    double s = 0;
    std::sscanf(buf, "%d %lf", &ok, &s);
    return {ok == 2 ? "BAD" : ok ? "yes" : "no ", s};
}

Case grid_case(std::string name, int R, int C, std::vector<Shape> shapes) {
    bool multi = false;
    for (auto [h, w] : shapes) {
        multi |= h * w > 1;
    }
    return Case{std::move(name), {{R, C}}, {}, std::move(shapes), multi};
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "quick";
    const int cap_s = std::getenv("TOPO_TIMEOUT") ? std::atoi(std::getenv("TOPO_TIMEOUT")) : 60;
    const Shape S1{1, 1};
    (void)S1;
    std::vector<Case> cases;
    auto ring1 = [](int n) { return std::vector<Shape>(n, Shape{1, 1}); };
// Multi-slice stages are NOT in this list. A node-to-node mapping engine has to draw the pipeline edge
// k -> k+1 between two fixed sub-nodes, and with the natural choice (bottom-right of k to top-left of
// k+1) every step moves right or down on the grid, so a ring can never close: the instance is
// infeasible by construction, not a carve result. "Some slice of k next to some slice of k+1" has no
// node-to-node encoding, so the exact comparison is the single-slice rings, on one mesh and across
// cables, which is also what both engines exist for.
    if (mode == "quick" || mode == "all") {
        cases.push_back(grid_case("4x8 ring of 32 x 1x1", 4, 8, ring1(32)));
        cases.push_back(grid_case("8x8 ring of 64 x 1x1", 8, 8, ring1(64)));
        cases.push_back(grid_case("8x16 ring of 128 x 1x1", 8, 16, ring1(128)));
        cases.push_back(grid_case("16x16 ring of 256 x 1x1", 16, 16, ring1(256)));
        cases.push_back(grid_case("5x5 ring of 25 x 1x1 (odd: none)", 5, 5, ring1(25)));
        cases.push_back(grid_case("7x7 ring of 49 x 1x1 (odd: none)", 7, 7, ring1(49)));
        cases.push_back(grid_case("8x8 ring of 60 x 1x1 (4 spare)", 8, 8, ring1(60)));
    }
    if (mode == "multi" || mode == "all") {
        for (int g : {4, 8}) {
            const size_t n1 = size_t(g) * g;
            Case ok{"two " + std::to_string(g) + "x" + std::to_string(g) + " meshes, cables (0,0),(0,1)", {{g, g}, {g, g}}, {{0, n1}, {1, n1 + 1}}, ring1(2 * g * g), false};
            cases.push_back(ok);
            Case bad{"two " + std::to_string(g) + "x" + std::to_string(g) + " meshes, cables same colour", {{g, g}, {g, g}}, {{0, n1}, {size_t(g) + 1, n1 + g + 1}}, ring1(2 * g * g), false};
            cases.push_back(bad);
        }
        cases.push_back(Case{"two 4x8 meshes, cables same colour", {{4, 8}, {4, 8}}, {{0, 32}, {9, 41}}, ring1(64), false});
    }
    if (mode == "multi2" || mode == "multi3") {
        for (const auto& m : mode == "multi2" ? multi2_cases() : multi3_cases()) {
            Case c{m.name, {}, {}, ring1(m.n_meshes * m.used_per_mesh), false, m.ring};
            for (int i = 0; i < m.n_meshes; ++i) {
                c.meshes.push_back({m.R, m.C});
            }
            const size_t per = size_t(m.R) * m.C;
            for (const auto& cb : m.cables) {
                c.cables.push_back({cb.mesh_a * per + size_t(cb.row_a) * m.C + cb.col_a, cb.mesh_b * per + size_t(cb.row_b) * m.C + cb.col_b});
            }
            cases.push_back(c);
        }
    }
    // The blaze carve planner (C++ port, carve_planner_port.hpp) followed by the resolver on its slots: the
    // design's C1 (planned carve) then C2, on the case definitions carve_footprint_bench.cpp exports with
    // CARVE_CASE_OUT (CARVE_CASES=<file>). Rings only; the planner's slots (plus blaze's one-slice filler
    // slots for unused slices) go to the resolver, and PLANNER_PLAN_OUT=<file> records them for the
    // topology solver's slot-level run. PLANNER_ALLOW_WIDE_STAGE0=1 lifts the planner's stage-0 rule.
    if (mode == "planner") {
        const char* path = std::getenv("CARVE_CASES");
        if (path == nullptr) {
            std::fprintf(stderr, "planner mode needs CARVE_CASES=<file>\n");
            return 1;
        }
        const bool allow_wide = std::getenv("PLANNER_ALLOW_WIDE_STAGE0") != nullptr;
        const char* plan_out = std::getenv("PLANNER_PLAN_OUT");
        std::ifstream in(path);
        std::string line;
        std::printf(
            "engine: blaze carve planner (C++ port) then pipeline_builder.cpp resolver on its slots; cap %d s per "
            "case%s\n",
            cap_s,
            allow_wide ? "; stage-0 rule off" : "");
        auto split = [](const std::string& str) {
            std::vector<std::vector<int>> out;
            size_t p = 0;
            while (p < str.size()) {
                const size_t e = str.find(';', p);
                const std::string part = str.substr(p, e == std::string::npos ? std::string::npos : e - p);
                std::vector<int> nums;
                size_t q = 0;
                while (q < part.size()) {
                    int v = 0, used = 0;
                    if (std::sscanf(part.c_str() + q, " %d%n", &v, &used) != 1) {
                        break;
                    }
                    nums.push_back(v);
                    q += used;
                }
                if (!nums.empty()) {
                    out.push_back(nums);
                }
                p = e == std::string::npos ? str.size() : e + 1;
            }
            return out;
        };
        while (std::getline(in, line)) {
            std::vector<std::string> f;
            size_t start = 0;
            for (size_t t = line.find('\t');; t = line.find('\t', start)) {
                f.push_back(line.substr(start, t == std::string::npos ? std::string::npos : t - start));
                if (t == std::string::npos) {
                    break;
                }
                start = t + 1;
            }
            if (f.size() != 5) {
                continue;
            }
            SlotCase sc;
            sc.name = f[0];
            for (const auto& m : split(f[1])) {
                sc.meshes.push_back({m[0], m[1]});
            }
            for (const auto& c : split(f[2])) {
                sc.cables.push_back({c[0], c[1], c[2], c[3], c[4], c[5]});
            }
            for (const auto& sh : split(f[3])) {
                sc.stage_shapes.push_back({sh[0], sh[1]});
            }
            sc.ring = f[4] == "ring";
            if (!sc.ring) {
                std::printf(
                    "%-70s %3zu stages | planner n/a (%s: the ring planner only)\n",
                    sc.name.c_str(),
                    sc.stage_shapes.size(),
                    f[4].c_str());
                continue;
            }
            int fd[2];
            if (pipe(fd) != 0) {
                return 1;
            }
            const pid_t pid = fork();
            if (pid == 0) {
                close(fd[0]);
                alarm(cap_s);
                // slices and adjacency in slice units
                std::vector<carve_planner::HostSlice> slices;
                std::vector<int> base;
                for (size_t m = 0; m < sc.meshes.size(); ++m) {
                    base.push_back(int(slices.size()));
                    for (int r = 0; r < sc.meshes[m].first; ++r) {
                        for (int c = 0; c < sc.meshes[m].second; ++c) {
                            slices.push_back({int(m), r, c, 1, 1});
                        }
                    }
                }
                auto sid = [&](int m, int r, int c) { return base[m] + r * sc.meshes[m].second + c; };
                std::vector<std::pair<int, int>> adjacency;
                for (size_t m = 0; m < sc.meshes.size(); ++m) {
                    for (int r = 0; r < sc.meshes[m].first; ++r) {
                        for (int c = 0; c < sc.meshes[m].second; ++c) {
                            if (c + 1 < sc.meshes[m].second) {
                                adjacency.push_back({sid(m, r, c), sid(m, r, c + 1)});
                            }
                            if (r + 1 < sc.meshes[m].first) {
                                adjacency.push_back({sid(m, r, c), sid(m, r + 1, c)});
                            }
                        }
                    }
                }
                for (const auto& cb : sc.cables) {
                    adjacency.push_back({sid(cb.mesh_a, cb.row_a, cb.col_a), sid(cb.mesh_b, cb.row_b, cb.col_b)});
                }
                const auto t0 = std::chrono::steady_clock::now();
                const auto plan = carve_planner::plan_ring_carve(slices, adjacency, sc.stage_shapes, allow_wide);
                const double plan_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                char buf[512];
                if (!plan.ok) {
                    std::snprintf(
                        buf,
                        sizeof buf,
                        "P0 %.6f %ld %zu %s",
                        plan_s,
                        plan.expansions,
                        plan.cycles,
                        plan.phase.c_str());
                    (void)!write(fd[1], buf, std::strlen(buf) + 1);
                    _exit(0);
                }
                // independent check of the plan: own shape, disjoint, consecutive stages touching
                std::set<std::tuple<int, int, int>> covered;
                bool good = plan.slots.size() == sc.stage_shapes.size();
                for (size_t k = 0; good && k < plan.slots.size(); ++k) {
                    const auto& r = plan.slots[k];
                    good &= r.rows == sc.stage_shapes[k].first && r.cols == sc.stage_shapes[k].second;
                    for (int rr = r.row0; rr < r.row0 + r.rows; ++rr) {
                        for (int cc = r.col0; cc < r.col0 + r.cols; ++cc) {
                            good &= covered.insert({r.mesh_id, rr, cc}).second;
                        }
                    }
                }
                std::set<std::pair<int, int>> adj_set;
                for (auto [a, b] : adjacency) {
                    adj_set.insert({a, b});
                    adj_set.insert({b, a});
                }
                for (size_t k = 0; good && k < plan.slots.size(); ++k) {
                    const auto& a = plan.slots[k];
                    const auto& b = plan.slots[(k + 1) % plan.slots.size()];
                    bool touch = false;
                    for (int ra = a.row0; ra < a.row0 + a.rows; ++ra) {
                        for (int ca = a.col0; ca < a.col0 + a.cols; ++ca) {
                            for (int rb = b.row0; rb < b.row0 + b.rows; ++rb) {
                                for (int cb2 = b.col0; cb2 < b.col0 + b.cols; ++cb2) {
                                    touch |= adj_set.count({sid(a.mesh_id, ra, ca), sid(b.mesh_id, rb, cb2)}) > 0;
                                }
                            }
                        }
                    }
                    good &= touch;
                }
                if (!good) {
                    std::snprintf(buf, sizeof buf, "P2 %.6f %ld %zu BAD-plan", plan_s, plan.expansions, plan.cycles);
                    (void)!write(fd[1], buf, std::strlen(buf) + 1);
                    _exit(0);
                }
                // slots: the plan's rectangles in stage order, then blaze's one-slice filler per unused slice
                for (const auto& r : plan.slots) {
                    sc.slots.push_back({r.mesh_id, r.row0, r.col0, r.rows, r.cols});
                }
                for (size_t m = 0; m < sc.meshes.size(); ++m) {
                    for (int r = 0; r < sc.meshes[m].first; ++r) {
                        for (int c = 0; c < sc.meshes[m].second; ++c) {
                            if (!covered.count({int(m), r, c})) {
                                sc.slots.push_back({int(m), r, c, 1, 1});
                            }
                        }
                    }
                }
                if (plan_out != nullptr) {
                    if (FILE* pf = std::fopen(plan_out, "a")) {
                        std::fprintf(pf, "%s\t%s\t%s\t", sc.name.c_str(), f[1].c_str(), f[2].c_str());
                        for (size_t i = 0; i < sc.slots.size(); ++i) {
                            const auto& r = sc.slots[i];
                            std::fprintf(pf, "%s%d %d %d %d %d", i ? ";" : "", r.mesh, r.r0, r.c0, r.h, r.w);
                        }
                        std::fprintf(pf, "\tring\t%s\n", f[3].c_str());
                        std::fclose(pf);
                    }
                }
                const auto [ok, rs] = solve_slots(sc);
                std::snprintf(buf, sizeof buf, "P1 %.6f %ld %zu %d %.6f", plan_s, plan.expansions, plan.cycles, ok, rs);
                (void)!write(fd[1], buf, std::strlen(buf) + 1);
                _exit(0);
            }
            close(fd[1]);
            char buf[512] = {0};
            const ssize_t got = read(fd[0], buf, sizeof buf - 1);
            close(fd[0]);
            int status = 0;
            waitpid(pid, &status, 0);
            if (got <= 0) {
                std::printf(
                    "%-70s %3zu stages | planner cap %8.3fs | resolver -          | total cap\n",
                    sc.name.c_str(),
                    sc.stage_shapes.size(),
                    double(cap_s));
                std::fflush(stdout);
                continue;
            }
            char tag[4] = {0};
            double plan_s = 0;
            long exp = 0;
            size_t cyc = 0;
            int consumed = 0;
            std::sscanf(buf, "%3s %lf %ld %zu %n", tag, &plan_s, &exp, &cyc, &consumed);
            if (std::string(tag) == "P1") {
                int ok = 0;
                double rs = 0;
                std::sscanf(buf + consumed, "%d %lf", &ok, &rs);
                std::printf(
                    "%-70s %3zu stages | planner yes %8.3fs (%ld exp, %zu cyc) | resolver %s %8.3fs | total %8.3fs\n",
                    sc.name.c_str(),
                    sc.stage_shapes.size(),
                    plan_s,
                    exp,
                    cyc,
                    ok == 2 ? "BAD"
                    : ok    ? "yes"
                            : "no ",
                    rs,
                    plan_s + rs);
            } else {
                std::printf(
                    "%-70s %3zu stages | planner no  %8.3fs (%ld exp, %zu cyc) %s | resolver -          | total "
                    "%8.3fs\n",
                    sc.name.c_str(),
                    sc.stage_shapes.size(),
                    plan_s,
                    exp,
                    cyc,
                    buf + consumed,
                    plan_s);
            }
            std::fflush(stdout);
        }
        return 0;
    }
    // The slots a carve produced (CARVE_PLAN_OUT of carve_footprint_bench.cpp): one slot per stage in stage
    // order, so the resolver's job is the design's C2 after C1: links and core slots, with the stage-to-slot
    // assignment already pinned by chip counts. CARVE_PLAN=<file>.
    if (mode == "plan") {
        const char* path = std::getenv("CARVE_PLAN");
        if (path == nullptr) {
            std::fprintf(stderr, "plan mode needs CARVE_PLAN=<file>\n");
            return 1;
        }
        std::ifstream in(path);
        std::string line;
        std::printf(
            "engine: pipeline_builder.cpp resolver on the slots of each footprint carve; cap %d s per solve\n", cap_s);
        while (std::getline(in, line)) {
            std::vector<std::string> f;
            size_t start = 0;
            for (size_t t = line.find('\t');; t = line.find('\t', start)) {
                f.push_back(line.substr(start, t == std::string::npos ? std::string::npos : t - start));
                if (t == std::string::npos) {
                    break;
                }
                start = t + 1;
            }
            if (f.size() != 5 && f.size() != 6) {
                continue;
            }
            SlotCase sc;
            sc.name = f[0];
            sc.ring = f[4] == "ring";
            auto split = [](const std::string& str) {
                std::vector<std::vector<int>> out;
                size_t p = 0;
                while (p < str.size()) {
                    const size_t e = str.find(';', p);
                    const std::string part = str.substr(p, e == std::string::npos ? std::string::npos : e - p);
                    std::vector<int> nums;
                    size_t q = 0;
                    while (q < part.size()) {
                        int v = 0, used = 0;
                        if (std::sscanf(part.c_str() + q, " %d%n", &v, &used) != 1) {
                            break;
                        }
                        nums.push_back(v);
                        q += used;
                    }
                    if (!nums.empty()) {
                        out.push_back(nums);
                    }
                    p = e == std::string::npos ? str.size() : e + 1;
                }
                return out;
            };
            for (const auto& m : split(f[1])) {
                sc.meshes.push_back({m[0], m[1]});
            }
            for (const auto& c : split(f[2])) {
                sc.cables.push_back({c[0], c[1], c[2], c[3], c[4], c[5]});
            }
            for (const auto& r : split(f[3])) {
                sc.slots.push_back({r[0], r[1], r[2], r[3], r[4]});
                if (f.size() == 5) {
                    sc.stage_shapes.push_back({r[3], r[4]});
                }
            }
            if (f.size() == 6) {
                for (const auto& sh : split(f[5])) {
                    sc.stage_shapes.push_back({sh[0], sh[1]});
                }
            }
            auto [ok, s] = timed_slots(sc, cap_s);
            std::printf(
                "%-70s %3zu stages | resolver-after-carve %s %8.3fs\n",
                sc.name.c_str(),
                sc.stage_shapes.size(),
                ok.c_str(),
                s);
            std::fflush(stdout);
        }
        return 0;
    }
    if (mode == "slots" || mode == "slots2") {
        std::printf("engine: pipeline_builder.cpp resolver, pre-carved slots; cap %d s per solve\n", cap_s);
        for (const auto& sc : mode == "slots" ? slot_cases() : slot_cases_min2()) {
            auto [ok, s] = timed_slots(sc, cap_s);
            std::printf("%-70s %3zu stages/%3zu slots | resolver %s %8.3fs\n", sc.name.c_str(), sc.stage_shapes.size(), sc.slots.size(), ok.c_str(), s);
            std::fflush(stdout);
        }
        return 0;
    }
    std::printf("engine: pipeline_builder.cpp resolver (resolve_pipeline_placement); cap %d s per solve\n", cap_s);
    // TOPO_SKIP=n resumes a run after its first n cases.
    const size_t skip = std::getenv("TOPO_SKIP") ? std::atoi(std::getenv("TOPO_SKIP")) : 0;
    for (size_t ci = skip; ci < cases.size(); ++ci) {
        const auto& cs = cases[ci];
        int area = 0, slices = 0;
        for (auto [h, w] : cs.shapes) {
            area += h * w;
        }
        for (auto [R, C] : cs.meshes) {
            slices += R * C;
        }
        auto [ok, s] = timed(cs, cap_s);
        std::printf("%-42s %3d/%-3d %s | resolver %s %8.3fs\n", cs.name.c_str(), area, slices, cs.fixed_ports ? "ports" : "exact", ok.c_str(), s);
        std::fflush(stdout);
    }
}
