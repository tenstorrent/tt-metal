// The carve as a graph-to-graph mapping through tt-metal's own topology solver
// (solve_topology_mapping, SAT and DFS engines), against the same cases as carve_bench_encodings.py.
//
// Global graph: one node per host slice, grid adjacency inside a mesh, plus optional cable edges
// between meshes. Target graph: one node per slice of every stage. Inside a stage the sub-nodes carry
// the stage's own grid edges, which pins a 2x2 to a unit square and a 2xk to a straight ladder (the
// grid graph has no other 4-cycles or ladders). Pipeline edge k -> k+1 is one target edge between a
// FIXED port sub-node of each stage: the solver maps nodes to nodes, so "some slice of k next to
// some slice of k+1" has no direct encoding. For single-slice stages this is exactly the carve; for
// multi-slice stages it is stricter (a "no" may be a port choice, not a carve failure), and 1xk/kx1
// stages are not used because a path of k sub-nodes could bend.
//
// Each solve runs in a forked child killed after TOPO_TIMEOUT seconds (default 60).
// Usage: topo_solver_bench [quick|multi|all]
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>

#include "multi2_cases.hpp"
#include "slot_cases.hpp"

using namespace tt::tt_fabric;
using Target = uint32_t;
using Global = uint64_t;
using Shape = std::pair<int, int>;

struct Mesh {
    int R, C;
};

struct Case {
    std::string name;
    std::vector<Mesh> meshes;
    std::vector<std::pair<Global, Global>> cables;  // slice ids, bidirectional
    std::vector<Shape> shapes;                      // in ring order; ring closes last -> first
    bool fixed_ports;                               // true when some stage has more than one slice
    bool ring = true;                               // false: chain, no closing edge
};

Global gid(int mesh, int r, int c, int C) { return Global(mesh) * 4096 + Global(r) * C + c; }
Target tid(int k, int i, int j) { return Target(k) * 256 + Target(i) * 16 + j; }

AdjacencyGraph<Global> build_global(const Case& cs) {
    AdjacencyGraph<Global>::AdjacencyMap adj;
    for (size_t m = 0; m < cs.meshes.size(); ++m) {
        const auto [R, C] = cs.meshes[m];
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                auto& v = adj[gid(m, r, c, C)];
                if (c + 1 < C) {
                    v.push_back(gid(m, r, c + 1, C));
                }
                if (c > 0) {
                    v.push_back(gid(m, r, c - 1, C));
                }
                if (r + 1 < R) {
                    v.push_back(gid(m, r + 1, c, C));
                }
                if (r > 0) {
                    v.push_back(gid(m, r - 1, c, C));
                }
            }
        }
    }
    for (auto [a, b] : cs.cables) {
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    return AdjacencyGraph<Global>(adj);
}

AdjacencyGraph<Target> build_target(const Case& cs) {
    AdjacencyGraph<Target>::AdjacencyMap adj;
    const int n = cs.shapes.size();
    auto link = [&](Target a, Target b) {
        adj[a].push_back(b);
        adj[b].push_back(a);
    };
    for (int k = 0; k < n; ++k) {
        const auto [h, w] = cs.shapes[k];
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                adj[tid(k, i, j)];
                if (j + 1 < w) {
                    link(tid(k, i, j), tid(k, i, j + 1));
                }
                if (i + 1 < h) {
                    link(tid(k, i, j), tid(k, i + 1, j));
                }
            }
        }
    }
    for (int k = 0; k + (cs.ring ? 0 : 1) < n; ++k) {
        const int k2 = (k + 1) % n;
        const auto [h, w] = cs.shapes[k];
        // exit port: bottom-right sub-node of k; entry port: top-left sub-node of k+1
        link(tid(k, h - 1, w - 1), tid(k2, 0, 0));
    }
    return AdjacencyGraph<Target>(adj);
}

// Runs one solve in a child; returns "yes"/"no"/"cap" and seconds.
std::pair<std::string, double> timed(const Case& cs, TopologyMappingSolverEngine engine, int cap_s) {
    int fd[2];
    if (pipe(fd) != 0) {
        return {"err", 0};
    }
    const pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        alarm(cap_s);
        const auto target = build_target(cs);
        const auto global = build_global(cs);
        MappingConstraints<Target, Global> constraints;
        const auto t0 = std::chrono::steady_clock::now();
        auto res = solve_topology_mapping(
            target, global, constraints, ConnectionValidationMode::STRICT, /*quiet_mode=*/true, engine);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[64];
        std::snprintf(buf, sizeof buf, "%d %.6f %zu %zu", res.success ? 1 : 0, s, res.stats.dfs_calls, res.stats.sat_solve_calls);
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
    size_t dfs = 0, sat = 0;
    std::sscanf(buf, "%d %lf %zu %zu", &ok, &s, &dfs, &sat);
    return {ok ? "yes" : "no ", s};
}

void run(const Case& cs, int cap_s) {
    int area = 0, slices = 0;
    for (auto [h, w] : cs.shapes) {
        area += h * w;
    }
    for (auto [R, C] : cs.meshes) {
        slices += R * C;
    }
    auto [sat_ok, sat_s] = timed(cs, TopologyMappingSolverEngine::Sat, cap_s);
    auto [dfs_ok, dfs_s] = timed(cs, TopologyMappingSolverEngine::Dfs, cap_s);
    std::printf(
        "%-42s %3d/%-3d %s | topo-sat %s %8.3fs | topo-dfs %s %8.3fs\n",
        cs.name.c_str(),
        area,
        slices,
        cs.fixed_ports ? "ports" : "exact",
        sat_ok.c_str(),
        sat_s,
        dfs_ok.c_str(),
        dfs_s);
    std::fflush(stdout);
}


// Pre-carved slots: one global node per slot (edges where slices are adjacent across slots or a
// cable joins them), one target node per stage; a required trait constraint matches stage shape to
// slot shape (h*16 + w, so orientation counts).
std::pair<std::string, double> timed_slots(const SlotCase& sc, TopologyMappingSolverEngine engine, int cap_s) {
    int fd[2];
    if (pipe(fd) != 0) {
        return {"err", 0};
    }
    const pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        alarm(cap_s);
        std::map<std::tuple<int, int, int>, Global> slot_of;
        std::map<Global, int> slot_trait;
        for (size_t i = 0; i < sc.slots.size(); ++i) {
            const auto& t = sc.slots[i];
            slot_trait[Global(i)] = t.h * 16 + t.w;
            for (int sr = t.r0; sr < t.r0 + t.h; ++sr) {
                for (int scn = t.c0; scn < t.c0 + t.w; ++scn) {
                    slot_of[{t.mesh, sr, scn}] = Global(i);
                }
            }
        }
        AdjacencyGraph<Global>::AdjacencyMap gadj;
        for (size_t i = 0; i < sc.slots.size(); ++i) {
            gadj[Global(i)];
        }
        auto join = [&](Global a, Global b) {
            if (a == b) {
                return;
            }
            auto& va = gadj[a];
            if (std::find(va.begin(), va.end(), b) == va.end()) {
                va.push_back(b);
                gadj[b].push_back(a);
            }
        };
        for (const auto& [key, a] : slot_of) {
            const auto [m, r, c] = key;
            auto it = slot_of.find({m, r, c + 1});
            if (it != slot_of.end()) {
                join(a, it->second);
            }
            it = slot_of.find({m, r + 1, c});
            if (it != slot_of.end()) {
                join(a, it->second);
            }
        }
        for (const auto& cb : sc.cables) {
            join(slot_of.at({cb.mesh_a, cb.row_a, cb.col_a}), slot_of.at({cb.mesh_b, cb.row_b, cb.col_b}));
        }
        const int n = sc.stage_shapes.size();
        AdjacencyGraph<Target>::AdjacencyMap tadj;
        std::map<Target, int> stage_trait;
        for (int k = 0; k < n; ++k) {
            tadj[Target(k)];
            stage_trait[Target(k)] = sc.stage_shapes[k].first * 16 + sc.stage_shapes[k].second;
        }
        for (int k = 0; k + (sc.ring ? 0 : 1) < n; ++k) {
            tadj[Target(k)].push_back(Target((k + 1) % n));
            tadj[Target((k + 1) % n)].push_back(Target(k));
        }
        const AdjacencyGraph<Target> target(tadj);
        const AdjacencyGraph<Global> global(gadj);
        MappingConstraints<Target, Global> constraints;
        constraints.add_required_trait_constraint<int>(stage_trait, slot_trait);
        const auto t0 = std::chrono::steady_clock::now();
        auto res = solve_topology_mapping(target, global, constraints, ConnectionValidationMode::STRICT, /*quiet_mode=*/true, engine);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[64];
        std::snprintf(buf, sizeof buf, "%d %.6f", res.success ? 1 : 0, s);
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
    return {ok ? "yes" : "no ", s};
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
        // exact: single-slice rings (the Hamiltonian-cycle carves)
        cases.push_back(grid_case("4x8 ring of 32 x 1x1", 4, 8, ring1(32)));
        cases.push_back(grid_case("8x8 ring of 64 x 1x1", 8, 8, ring1(64)));
        cases.push_back(grid_case("8x16 ring of 128 x 1x1", 8, 16, ring1(128)));
        cases.push_back(grid_case("16x16 ring of 256 x 1x1", 16, 16, ring1(256)));
        cases.push_back(grid_case("5x5 ring of 25 x 1x1 (odd: none)", 5, 5, ring1(25)));
        cases.push_back(grid_case("7x7 ring of 49 x 1x1 (odd: none)", 7, 7, ring1(49)));
        cases.push_back(grid_case("8x8 ring of 60 x 1x1 (4 spare)", 8, 8, ring1(60)));
        // fixed ports: multi-slice stages
    }
    if (mode == "multi" || mode == "all") {
        // two meshes joined by two cables, ring of single-slice stages over every slice: exact.
        for (int g : {4, 8}) {
            Case ok{"two " + std::to_string(g) + "x" + std::to_string(g) + " meshes, cables (0,0),(0,1)", {{g, g}, {g, g}}, {}, ring1(2 * g * g), false};
            ok.cables = {{gid(0, 0, 0, g), gid(1, 0, 0, g)}, {gid(0, 0, 1, g), gid(1, 0, 1, g)}};
            cases.push_back(ok);
            Case bad{"two " + std::to_string(g) + "x" + std::to_string(g) + " meshes, cables same colour", {{g, g}, {g, g}}, {}, ring1(2 * g * g), false};
            bad.cables = {{gid(0, 0, 0, g), gid(1, 0, 0, g)}, {gid(0, 1, 1, g), gid(1, 1, 1, g)}};
            cases.push_back(bad);
        }
        Case big{"two 4x8 meshes, cables same colour", {{4, 8}, {4, 8}}, {}, ring1(64), false};
        big.cables = {{gid(0, 0, 0, 8), gid(1, 0, 0, 8)}, {gid(0, 1, 1, 8), gid(1, 1, 1, 8)}};
        cases.push_back(big);
    }
    if (mode == "multi2") {
        for (const auto& m : multi2_cases()) {
            Case c{m.name, {}, {}, ring1(m.n_meshes * m.used_per_mesh), false, m.ring};
            for (int i = 0; i < m.n_meshes; ++i) {
                c.meshes.push_back({m.R, m.C});
            }
            for (const auto& cb : m.cables) {
                c.cables.push_back({gid(cb.mesh_a, cb.row_a, cb.col_a, m.C), gid(cb.mesh_b, cb.row_b, cb.col_b, m.C)});
            }
            cases.push_back(c);
        }
    }
    if (mode == "slots") {
        std::printf("engines: topo-sat, topo-dfs (solve_topology_mapping), pre-carved slots; cap %d s per solve\n", cap_s);
        for (const auto& sc : slot_cases()) {
            auto [sat_ok, sat_s] = timed_slots(sc, TopologyMappingSolverEngine::Sat, cap_s);
            auto [dfs_ok, dfs_s] = timed_slots(sc, TopologyMappingSolverEngine::Dfs, cap_s);
            std::printf("%-70s %3zu stages/%3zu slots | topo-sat %s %8.3fs | topo-dfs %s %8.3fs\n", sc.name.c_str(), sc.stage_shapes.size(), sc.slots.size(), sat_ok.c_str(), sat_s, dfs_ok.c_str(), dfs_s);
            std::fflush(stdout);
        }
        return 0;
    }
    std::printf("engines: topo-sat, topo-dfs (solve_topology_mapping); cap %d s per solve\n", cap_s);
    // TOPO_SKIP=n resumes a run after its first n cases.
    const size_t skip = std::getenv("TOPO_SKIP") ? std::atoi(std::getenv("TOPO_SKIP")) : 0;
    for (size_t ci = skip; ci < cases.size(); ++ci) {
        const auto& cs = cases[ci];
        run(cs, cap_s);
    }
}
