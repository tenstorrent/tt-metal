// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The carve as a footprint mapping through tt-metal's own topology solver (solve_topology_mapping),
// with no solver change: the same pattern the mesh seating uses for seats.
//
// Global graph: one node per POSITION, a place one stage shape fits inside one mesh (in host
// slices). Two positions are adjacent when they do not overlap and some slice of one borders some
// slice of the other, by grid adjacency inside a mesh or by a cable between meshes.
// Target graph: the pipeline, one node per stage; ring, chain or fork edges as given.
// Constraints: each stage may only take positions of its own shape (add_required_constraint), and
// no slice is used by two chosen positions (add_resource_constraint, position -> covered slices).
// A mapping is then exactly a carve: one rectangle per stage, disjoint, every pipeline edge touching.
//
// Unlike topo_solver_bench.cpp (one node per slice of every stage, fixed ports), this encoding is
// exact for multi-slice stages. Every returned carve is re-checked here; on cases with at most
// BRUTE_MAX_STAGES stages the yes/no answer is also checked against a brute-force search.
//
// Each solve runs in a forked child killed after TOPO_TIMEOUT seconds (default 20).
// Usage: carve_footprint_bench [quick|repo|multi|all|multi2|file]
//   multi2: the single-slice rings and chains over 2 to 8 meshes of 32 ranks (multi2_cases.hpp), the same
//           instances as topo_solver_bench multi2 and resolver_same_cases multi2.
//   file:   cases from CARVE_CASES (one per line: name<TAB>R<TAB>C<TAB>h w;h w;...), a ring over one mesh;
//           carve_bench_encodings.py QUICK exported this way gives the same instances as its tables.
//   CARVE_ENGINES=sat,dfs limits the engines run (auto always picks SAT here, so it is redundant).
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>

#include "multi2_cases.hpp"

using namespace tt::tt_fabric;
using Stage = uint32_t;
using Pos = uint32_t;
using Shape = std::pair<int, int>;  // (rows, cols) in host slices; orientation counts

constexpr size_t BRUTE_MAX_STAGES = 8;

struct Cable {
    int mesh_a, row_a, col_a, mesh_b, row_b, col_b;
};

struct Case {
    std::string name;
    std::vector<std::pair<int, int>> meshes;  // slice grid per mesh
    std::vector<Cable> cables;
    std::vector<Shape> shapes;  // per stage
    std::vector<std::pair<Stage, Stage>> edges;
};

struct Rect {
    int mesh, r0, c0, h, w;
};

struct Instance {
    std::vector<Rect> rects;
    std::vector<std::vector<uint32_t>> covers;  // position -> slice ids
    std::vector<std::vector<uint32_t>> nbrs;    // slice -> neighbouring slice ids
    std::vector<std::vector<Pos>> of_stage;     // stage -> positions of its shape
    AdjacencyGraph<Pos>::AdjacencyMap global_adj;
    AdjacencyGraph<Stage>::AdjacencyMap target_adj;
};

std::vector<std::pair<Stage, Stage>> ring_edges(size_t n, bool close = true) {
    std::vector<std::pair<Stage, Stage>> e;
    for (size_t k = 0; k + 1 < n; ++k) {
        e.push_back({Stage(k), Stage(k + 1)});
    }
    if (close && n > 2) {
        e.push_back({Stage(n - 1), Stage(0)});
    }
    return e;
}

Case grid_ring(std::string name, int R, int C, std::vector<Shape> shapes) {
    const size_t n = shapes.size();
    return Case{std::move(name), {{R, C}}, {}, std::move(shapes), ring_edges(n)};
}

Instance build(const Case& cs) {
    Instance in;
    std::vector<uint32_t> base;
    uint32_t n_slices = 0;
    for (auto [R, C] : cs.meshes) {
        base.push_back(n_slices);
        n_slices += R * C;
    }
    auto sid = [&](int m, int r, int c) { return base[m] + uint32_t(r * cs.meshes[m].second + c); };
    in.nbrs.assign(n_slices, {});
    for (size_t m = 0; m < cs.meshes.size(); ++m) {
        const auto [R, C] = cs.meshes[m];
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (c + 1 < C) {
                    in.nbrs[sid(m, r, c)].push_back(sid(m, r, c + 1));
                    in.nbrs[sid(m, r, c + 1)].push_back(sid(m, r, c));
                }
                if (r + 1 < R) {
                    in.nbrs[sid(m, r, c)].push_back(sid(m, r + 1, c));
                    in.nbrs[sid(m, r + 1, c)].push_back(sid(m, r, c));
                }
            }
        }
    }
    for (const auto& cb : cs.cables) {
        const uint32_t a = sid(cb.mesh_a, cb.row_a, cb.col_a);
        const uint32_t b = sid(cb.mesh_b, cb.row_b, cb.col_b);
        in.nbrs[a].push_back(b);
        in.nbrs[b].push_back(a);
    }

    // Positions per distinct shape; stages of the same shape share them.
    std::map<Shape, std::vector<Pos>> by_shape;
    for (const Shape& s : cs.shapes) {
        if (by_shape.contains(s)) {
            continue;
        }
        auto& list = by_shape[s];
        for (size_t m = 0; m < cs.meshes.size(); ++m) {
            const auto [R, C] = cs.meshes[m];
            for (int r0 = 0; r0 + s.first <= R; ++r0) {
                for (int c0 = 0; c0 + s.second <= C; ++c0) {
                    std::vector<uint32_t> cov;
                    for (int r = r0; r < r0 + s.first; ++r) {
                        for (int c = c0; c < c0 + s.second; ++c) {
                            cov.push_back(sid(m, r, c));
                        }
                    }
                    list.push_back(Pos(in.rects.size()));
                    in.rects.push_back({int(m), r0, c0, s.first, s.second});
                    in.covers.push_back(std::move(cov));
                }
            }
        }
    }
    for (const Shape& s : cs.shapes) {
        in.of_stage.push_back(by_shape[s]);
    }

    // Global edges: disjoint positions with a slice of one next to a slice of the other.
    const size_t P = in.rects.size();
    std::vector<std::vector<char>> inside(P, std::vector<char>(n_slices, 0));
    std::vector<std::vector<char>> halo(P, std::vector<char>(n_slices, 0));
    for (size_t p = 0; p < P; ++p) {
        for (uint32_t s : in.covers[p]) {
            inside[p][s] = 1;
        }
        for (uint32_t s : in.covers[p]) {
            for (uint32_t t : in.nbrs[s]) {
                if (!inside[p][t]) {
                    halo[p][t] = 1;
                }
            }
        }
    }
    for (Pos p = 0; p < P; ++p) {
        in.global_adj[p];
    }
    for (Pos p = 0; p < P; ++p) {
        for (Pos q = p + 1; q < P; ++q) {
            bool overlap = false, touch = false;
            for (uint32_t s : in.covers[q]) {
                overlap |= inside[p][s] != 0;
                touch |= halo[p][s] != 0;
            }
            if (!overlap && touch) {
                in.global_adj[p].push_back(q);
                in.global_adj[q].push_back(p);
            }
        }
    }

    for (Stage k = 0; k < cs.shapes.size(); ++k) {
        in.target_adj[k];
    }
    for (auto [a, b] : cs.edges) {
        in.target_adj[a].push_back(b);
        in.target_adj[b].push_back(a);
    }
    return in;
}

bool positions_touch(const Instance& in, Pos p, Pos q) {
    for (uint32_t s : in.covers[p]) {
        for (uint32_t t : in.nbrs[s]) {
            if (std::find(in.covers[q].begin(), in.covers[q].end(), t) != in.covers[q].end()) {
                return true;
            }
        }
    }
    return false;
}

// Independent check of a returned carve: own shape, disjoint, every edge touching.
std::string verify(const Case& cs, const Instance& in, const std::map<Stage, Pos>& plan) {
    if (plan.size() != cs.shapes.size()) {
        return "wrong stage count";
    }
    std::set<uint32_t> used;
    for (const auto& [k, p] : plan) {
        if (std::find(in.of_stage[k].begin(), in.of_stage[k].end(), p) == in.of_stage[k].end()) {
            return "stage " + std::to_string(k) + " on a position of another shape";
        }
        for (uint32_t s : in.covers[p]) {
            if (!used.insert(s).second) {
                return "slice used twice";
            }
        }
    }
    for (auto [a, b] : cs.edges) {
        if (!positions_touch(in, plan.at(a), plan.at(b))) {
            return "edge " + std::to_string(a) + "-" + std::to_string(b) + " does not touch";
        }
    }
    return "";
}

// Brute force over positions in stage order: an independent yes/no for small cases.
bool brute_force(const Case& cs, const Instance& in) {
    const size_t n = cs.shapes.size();
    std::vector<Pos> chosen(n);
    std::vector<char> used(in.nbrs.size(), 0);
    std::function<bool(size_t)> go = [&](size_t k) {
        if (k == n) {
            return true;
        }
        for (Pos p : in.of_stage[k]) {
            bool ok = true;
            for (uint32_t s : in.covers[p]) {
                ok &= !used[s];
            }
            for (auto [a, b] : cs.edges) {
                const size_t other = a == k ? b : (b == k ? a : n);
                if (ok && other < k) {
                    ok = positions_touch(in, p, chosen[other]);
                }
            }
            if (!ok) {
                continue;
            }
            chosen[k] = p;
            for (uint32_t s : in.covers[p]) {
                used[s] = 1;
            }
            if (go(k + 1)) {
                return true;
            }
            for (uint32_t s : in.covers[p]) {
                used[s] = 0;
            }
        }
        return false;
    };
    return go(0);
}

struct Outcome {
    std::string verdict;  // yes / no / cap / BAD (returned carve failed verification)
    double seconds = 0;
    bool used_sat = false;
    std::string plan;
};

Outcome timed(const Case& cs, TopologyMappingSolverEngine engine, int cap_s) {
    int fd[2];
    if (pipe(fd) != 0) {
        return {"err"};
    }
    const pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        alarm(cap_s);
        const Instance in = build(cs);
        MappingConstraints<Stage, Pos> constraints;
        for (Stage k = 0; k < cs.shapes.size(); ++k) {
            const std::set<Pos> allowed(in.of_stage[k].begin(), in.of_stage[k].end());
            constraints.add_required_constraint(k, allowed);
        }
        std::map<Pos, std::vector<uint32_t>> covers;
        for (Pos p = 0; p < in.rects.size(); ++p) {
            covers[p] = in.covers[p];
        }
        constraints.add_resource_constraint<uint32_t>(covers);
        const auto t0 = std::chrono::steady_clock::now();
        auto res = solve_topology_mapping(
            AdjacencyGraph<Stage>(in.target_adj),
            AdjacencyGraph<Pos>(in.global_adj),
            constraints,
            ConnectionValidationMode::STRICT,
            /*quiet_mode=*/true,
            engine);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::string verdict = res.success ? "yes" : "no";
        std::string plan;
        if (res.success) {
            const std::string bad = verify(cs, in, res.target_to_global);
            if (!bad.empty()) {
                verdict = "BAD";
                plan = bad;
            } else {
                for (const auto& [k, p] : res.target_to_global) {
                    const Rect& r = in.rects[p];
                    char b[64];
                    std::snprintf(
                        b, sizeof b, "%s%dx%d@m%d(%d,%d)", plan.empty() ? "" : " ", r.h, r.w, r.mesh, r.r0, r.c0);
                    plan += b;
                }
            }
        }
        char buf[1024];
        std::snprintf(buf, sizeof buf, "%s %.6f %d %s", verdict.c_str(), s, res.stats.used_sat ? 1 : 0, plan.c_str());
        (void)!write(fd[1], buf, std::strlen(buf) + 1);
        _exit(0);
    }
    close(fd[1]);
    char buf[1024] = {0};
    const ssize_t got = read(fd[0], buf, sizeof buf - 1);
    close(fd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (got <= 0) {
        return {"cap", double(cap_s)};
    }
    Outcome o;
    char verdict[8] = {0};
    int used_sat = 0, consumed = 0;
    std::sscanf(buf, "%7s %lf %d %n", verdict, &o.seconds, &used_sat, &consumed);
    o.verdict = verdict;
    o.used_sat = used_sat != 0;
    o.plan = buf + consumed;
    return o;
}

void run(const Case& cs, int cap_s, bool show_plan) {
    const Instance in = build(cs);
    int area = 0, slices = 0;
    for (auto [h, w] : cs.shapes) {
        area += h * w;
    }
    for (auto [R, C] : cs.meshes) {
        slices += R * C;
    }
    std::string brute = "-";
    if (cs.shapes.size() <= BRUTE_MAX_STAGES) {
        brute = brute_force(cs, in) ? "yes" : "no";
    }
    const char* only = std::getenv("CARVE_ENGINES");
    auto wanted = [&](const char* e) { return only == nullptr || std::string(only).find(e) != std::string::npos; };
    const Outcome skipped{"-"};
    const Outcome a = wanted("auto") ? timed(cs, TopologyMappingSolverEngine::Auto, cap_s) : skipped;
    const Outcome s = wanted("sat") ? timed(cs, TopologyMappingSolverEngine::Sat, cap_s) : skipped;
    const Outcome d = wanted("dfs") ? timed(cs, TopologyMappingSolverEngine::Dfs, cap_s) : skipped;
    bool mismatch = false;
    for (const Outcome* o : {&a, &s, &d}) {
        mismatch |= o->verdict == "BAD";
        mismatch |= brute != "-" && (o->verdict == "yes" || o->verdict == "no") && o->verdict != brute;
    }
    std::printf(
        "%-48s %3d/%-3d %4zu pos | brute %-3s | auto(%s) %-3s %8.3fs | sat %-3s %8.3fs | dfs %-3s %8.3fs%s\n",
        cs.name.c_str(),
        area,
        slices,
        in.rects.size(),
        brute.c_str(),
        a.verdict == "cap" || a.verdict == "-" ? "?" : (a.used_sat ? "sat" : "dfs"),
        a.verdict.c_str(),
        a.seconds,
        s.verdict.c_str(),
        s.seconds,
        d.verdict.c_str(),
        d.seconds,
        mismatch ? "  <-- MISMATCH" : "");
    if (show_plan && !a.plan.empty()) {
        std::printf("    auto plan: %s\n", a.plan.c_str());
    }
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "quick";
    const int cap_s = std::getenv("TOPO_TIMEOUT") ? std::atoi(std::getenv("TOPO_TIMEOUT")) : 20;
    const bool show_plan = std::getenv("CARVE_SHOW_PLAN") != nullptr;
    const Shape S1{1, 1}, H{1, 2}, V{2, 1}, B{2, 2}, W{2, 4};
    auto rep = [](Shape s, int n) { return std::vector<Shape>(n, s); };
    auto cat = [](std::initializer_list<std::vector<Shape>> parts) {
        std::vector<Shape> v;
        for (const auto& p : parts) {
            v.insert(v.end(), p.begin(), p.end());
        }
        return v;
    };
    std::vector<Case> cases;

    if (mode == "quick" || mode == "all") {
        // The rings blaze's planner rejects (carve_probe.py), and the ones it finds.
        cases.push_back(grid_ring("4x4: 1x1, 3x 2x2 (blaze: no-cycle-13)", 4, 4, cat({{S1}, rep(B, 3)})));
        cases.push_back(grid_ring("4x4: 1x1, 3x 2x2, 1x1", 4, 4, cat({{S1}, rep(B, 3), {S1}})));
        cases.push_back(grid_ring("4x4: 1x1, 2x 2x2, 1x1", 4, 4, cat({{S1}, rep(B, 2), {S1}})));
        cases.push_back(grid_ring("4x6: 1x1, 4x 2x2, 1x1 (blaze: no-alignment)", 4, 6, cat({{S1}, rep(B, 4), {S1}})));
        cases.push_back(grid_ring("4x8: 1x1, 6x 2x2, 1x1 (blaze: no-alignment)", 4, 8, cat({{S1}, rep(B, 6), {S1}})));
        cases.push_back(grid_ring("4x8: 1x1, 2x4, 2x4, 2x2, 1x1", 4, 8, {S1, W, W, B, S1}));
        // Infeasible: counting and area.
        cases.push_back(grid_ring("5x5: 1x1, 5x 2x2 (only 4 fit)", 5, 5, cat({{S1}, rep(B, 5)})));
        cases.push_back(grid_ring("6x6: 1x1, 9x 2x2 (area 37 > 36)", 6, 6, cat({{S1}, rep(B, 9)})));
        // A fork: router r, two branches of two 2x2 stages, both returning to r.
        Case fork{"6x6 fork: r 1x1, r-a1-a2-r, r-b1-b2-r (2x2 branches)", {{6, 6}}, {}, {S1, B, B, B, B}, {}};
        fork.edges = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {3, 4}, {4, 0}};
        cases.push_back(fork);
    }
    if (mode == "multi" || mode == "all") {
        // mixed4 from the design doc: two meshes, each a row of four 4x2-chip slices; ring
        // s0 1 slice, s1 2 slices, s2 2 slices, s3 1 slice.
        const std::vector<Shape> mixed4 = {S1, H, H, S1};
        Case pr{
            "mixed4 (P,R): cables A0-B0, A2-B2",
            {{1, 4}, {1, 4}},
            {{0, 0, 0, 1, 0, 0}, {0, 0, 2, 1, 0, 2}},
            mixed4,
            ring_edges(4)};
        cases.push_back(pr);
        Case pq{
            "mixed4 (P,Q): cables A0-B0, A3-B3",
            {{1, 4}, {1, 4}},
            {{0, 0, 0, 1, 0, 0}, {0, 0, 3, 1, 0, 3}},
            mixed4,
            ring_edges(4)};
        cases.push_back(pq);
        Case caseb{
            "mixed4 Case B: crossings only on A0-B0", {{1, 4}, {1, 4}}, {{0, 0, 0, 1, 0, 0}}, mixed4, ring_edges(4)};
        cases.push_back(caseb);
        // Two 4x4 meshes, ring of 2x2 stages across two cables.
        Case two{
            "two 4x4: 1x1, 6x 2x2, 1x1, cables (0,0),(0,1)",
            {{4, 4}, {4, 4}},
            {{0, 0, 0, 1, 0, 0}, {0, 0, 1, 1, 0, 1}},
            cat({{S1}, rep(B, 6), {S1}}),
            ring_edges(8)};
        cases.push_back(two);
        Case two1{
            "two 4x4 ring of 32 x 1x1, cables (0,0),(0,1)",
            {{4, 4}, {4, 4}},
            {{0, 0, 0, 1, 0, 0}, {0, 0, 1, 1, 0, 1}},
            rep(S1, 32),
            ring_edges(32)};
        cases.push_back(two1);
    }
    if (mode == "repo" || mode == "all") {
        // Repo slice grids (carve_bench_repo.py).
        cases.push_back(grid_ring("4x8 ring of 32 x 1x1 (fills)", 4, 8, rep(S1, 32)));
        cases.push_back(grid_ring("2x16 ring of 32 x 1x1 (fills)", 2, 16, rep(S1, 32)));
        cases.push_back(grid_ring("5x5 ring of 25 x 1x1 (odd: none)", 5, 5, rep(S1, 25)));
        cases.push_back(grid_ring("4x8: 1x1, 1x2 x6, 2x2 x4, 1x1", 4, 8, cat({{S1}, rep(H, 6), rep(B, 4), {S1}})));
        cases.push_back(grid_ring("8x4: 1x1, 2x1 x6, 2x2 x4, 1x1", 8, 4, cat({{S1}, rep(V, 6), rep(B, 4), {S1}})));
        cases.push_back(grid_ring("2x16: 1x1, 1x2 x15, 1x1", 2, 16, cat({{S1}, rep(H, 15), {S1}})));
        cases.push_back(grid_ring("2x16: 1x1, 2x2 x7, 1x1 (fills 30/32)", 2, 16, cat({{S1}, rep(B, 7), {S1}})));
        cases.push_back(grid_ring("4x8: 2x4 x4 (fills, ring of 4)", 4, 8, rep(W, 4)));
        cases.push_back(grid_ring("8x8: 1x1, 14x 2x2, 1x1", 8, 8, cat({{S1}, rep(B, 14), {S1}})));
    }

    if (mode == "multi2") {
        for (const auto& m : multi2_cases()) {
            Case c{
                m.name,
                {},
                {},
                rep(S1, m.n_meshes * m.used_per_mesh),
                ring_edges(size_t(m.n_meshes) * m.used_per_mesh, m.ring)};
            for (int i = 0; i < m.n_meshes; ++i) {
                c.meshes.push_back({m.R, m.C});
            }
            for (const auto& cb : m.cables) {
                c.cables.push_back({cb.mesh_a, cb.row_a, cb.col_a, cb.mesh_b, cb.row_b, cb.col_b});
            }
            cases.push_back(c);
        }
    }
    if (mode == "file") {
        const char* path = std::getenv("CARVE_CASES");
        if (path == nullptr) {
            std::fprintf(stderr, "file mode needs CARVE_CASES=<path>\n");
            return 1;
        }
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            std::vector<std::string> f;
            size_t start = 0;
            for (size_t t = line.find('\t');; t = line.find('\t', start)) {
                f.push_back(line.substr(start, t == std::string::npos ? std::string::npos : t - start));
                if (t == std::string::npos) {
                    break;
                }
                start = t + 1;
            }
            if (f.size() != 4) {
                std::fprintf(stderr, "bad case line: %s\n", line.c_str());
                return 1;
            }
            std::vector<Shape> shapes;
            size_t pos = 0;
            while (pos < f[3].size()) {
                int h = 0, w = 0;
                if (std::sscanf(f[3].c_str() + pos, "%d %d", &h, &w) != 2) {
                    std::fprintf(stderr, "bad shapes: %s\n", f[3].c_str());
                    return 1;
                }
                shapes.push_back({h, w});
                const size_t semi = f[3].find(';', pos);
                pos = semi == std::string::npos ? f[3].size() : semi + 1;
            }
            cases.push_back(grid_ring(f[0], std::atoi(f[1].c_str()), std::atoi(f[2].c_str()), std::move(shapes)));
        }
    }

    std::printf(
        "carve as footprint mapping (solve_topology_mapping): engines auto, sat, dfs; cap %d s per solve; "
        "brute force on <= %zu stages\n",
        cap_s,
        BRUTE_MAX_STAGES);
    for (const auto& cs : cases) {
        run(cs, cap_s, show_plan);
    }
}
