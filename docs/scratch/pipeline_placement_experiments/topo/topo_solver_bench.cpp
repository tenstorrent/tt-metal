// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
#include <fstream>
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

// Independent check of a returned mapping: every target node mapped, no global node used twice, every
// target edge on a global edge, and (slots) every stage on a slot of its own shape.
std::string verify_mapping(
    const AdjacencyGraph<Target>& target,
    const AdjacencyGraph<Global>& global,
    const std::map<Target, Global>& m,
    const std::map<Target, int>* stage_trait,
    const std::map<Global, int>* slot_trait) {
    if (m.size() != target.get_nodes().size()) {
        return "node count";
    }
    std::set<Global> used;
    for (const auto& [t, g] : m) {
        if (!used.insert(g).second) {
            return "global node used twice";
        }
    }
    for (const auto& t : target.get_nodes()) {
        const Global g = m.at(t);
        for (const auto& u : target.get_neighbors(t)) {
            const auto& nb = global.get_neighbors(g);
            if (std::find(nb.begin(), nb.end(), m.at(u)) == nb.end()) {
                return "target edge not on a global edge";
            }
        }
        if (stage_trait != nullptr && stage_trait->at(t) != slot_trait->at(g)) {
            return "stage on a slot of another shape";
        }
    }
    return "";
}

// Runs one solve in a child; returns "yes"/"no"/"BAD"/"cap", seconds, and whether the SAT backend ran
// (for the Auto column). BAD: the solver said yes but the returned mapping fails verify_mapping.
// TOPO_ENGINES=auto,sat,dfs selects the columns (default sat,dfs); a column not run prints "-".
struct Timed {
    std::string ok;
    double s = 0;
    bool used_sat = false;
};

// TOPO_CONFLICT_CAP=n runs the solve through TopologyMappingEnumerationSession with that per-solve SAT
// conflict budget (solve_topology_mapping's default of 300k reports budget exhaustion as "no"); with 2e9 a
// "no" inside the wall-clock cap is a real UNSAT. The DFS engine stops after 10 million calls and also
// reports "no"; that case is printed as "lim" (undecided), never as "no".
constexpr size_t kDfsCallLimit = 10000000;
MappingResult<Target, Global> solve_with_cap(
    const AdjacencyGraph<Target>& target,
    const AdjacencyGraph<Global>& global,
    const MappingConstraints<Target, Global>& constraints,
    TopologyMappingSolverEngine engine) {
    // The DFS engine has no conflict budget, and only solve_topology_mapping reports stats.dfs_calls (the
    // session does not), which the "lim" verdict needs; so DFS always goes the direct way.
    if (const char* cap_env = std::getenv("TOPO_CONFLICT_CAP");
        cap_env != nullptr && engine != TopologyMappingSolverEngine::Dfs) {
        TopologyMappingEnumerationSession<Target, Global> session(
            target,
            global,
            constraints,
            ConnectionValidationMode::STRICT,
            /*quiet_mode=*/true,
            engine,
            /*unique_shapes=*/false,
            std::atoi(cap_env));
        return session.next();
    }
    return solve_topology_mapping(
        target, global, constraints, ConnectionValidationMode::STRICT, /*quiet_mode=*/true, engine);
}
int verdict_code(const MappingResult<Target, Global>& res, const std::string& bad) {
    if (res.success) {
        return bad.empty() ? 1 : 2;
    }
    // stats.dfs_calls is not always carried through; the limit also leaves its message
    return res.stats.dfs_calls >= kDfsCallLimit || res.error_message.find("exceeded call limit") != std::string::npos
               ? 3
               : 0;
}
const char* verdict_str(int ok) { return ok == 3 ? "lim" : ok == 2 ? "BAD" : ok == 1 ? "yes" : "no "; }

bool engine_wanted(const char* name) {
    const char* e = std::getenv("TOPO_ENGINES");
    return std::string(e ? e : "sat,dfs").find(name) != std::string::npos;
}

Timed timed(const Case& cs, TopologyMappingSolverEngine engine, int cap_s) {
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
        auto res = solve_with_cap(target, global, constraints, engine);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[64];
        const int ok = verdict_code(
            res, res.success ? verify_mapping(target, global, res.target_to_global, nullptr, nullptr) : "");
        std::snprintf(buf, sizeof buf, "%d %.6f %zu %zu", ok, s, res.stats.dfs_calls, res.stats.sat_solve_calls);
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
        return {"cap", double(cap_s), false};
    }
    int ok = 0;
    double s = 0;
    size_t dfs = 0, sat = 0;
    std::sscanf(buf, "%d %lf %zu %zu", &ok, &s, &dfs, &sat);
    return {verdict_str(ok), s, sat > 0};
}

void run(const Case& cs, int cap_s) {
    int area = 0, slices = 0;
    for (auto [h, w] : cs.shapes) {
        area += h * w;
    }
    for (auto [R, C] : cs.meshes) {
        slices += R * C;
    }
    const Timed none{"-  ", 0, false};
    const Timed a = engine_wanted("auto") ? timed(cs, TopologyMappingSolverEngine::Auto, cap_s) : none;
    const Timed sat = engine_wanted("sat") ? timed(cs, TopologyMappingSolverEngine::Sat, cap_s) : none;
    const Timed dfs = engine_wanted("dfs") ? timed(cs, TopologyMappingSolverEngine::Dfs, cap_s) : none;
    std::printf(
        "%-42s %3d/%-3d %s | topo-auto(%s) %s %8.3fs | topo-sat %s %8.3fs | topo-dfs %s %8.3fs\n",
        cs.name.c_str(),
        area,
        slices,
        cs.fixed_ports ? "ports" : "exact",
        a.ok == "-  " || a.ok == "cap" ? "?" : (a.used_sat ? "sat" : "dfs"),
        a.ok.c_str(),
        a.s,
        sat.ok.c_str(),
        sat.s,
        dfs.ok.c_str(),
        dfs.s);
    std::fflush(stdout);
}


// Pre-carved slots: one global node per slot (edges where slices are adjacent across slots or a
// cable joins them), one target node per stage; a required trait constraint matches stage shape to
// slot shape (h*16 + w, so orientation counts).
Timed timed_slots(const SlotCase& sc, TopologyMappingSolverEngine engine, int cap_s) {
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
            const auto ia = slot_of.find({cb.mesh_a, cb.row_a, cb.col_a}),
                       ib = slot_of.find({cb.mesh_b, cb.row_b, cb.col_b});
            if (ia == slot_of.end() || ib == slot_of.end()) {
                continue;  // a cable on a slice no slot covers (spare slice after the carve)
            }
            join(ia->second, ib->second);
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
        auto res = solve_with_cap(target, global, constraints, engine);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[64];
        const int ok = verdict_code(
            res, res.success ? verify_mapping(target, global, res.target_to_global, &stage_trait, &slot_trait) : "");
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
        return {"cap", double(cap_s), false};
    }
    int ok = 0;
    double s = 0;
    std::sscanf(buf, "%d %lf", &ok, &s);
    return {verdict_str(ok), s, false};
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
    if (mode == "multi2" || mode == "multi3") {
        for (const auto& m : mode == "multi2" ? multi2_cases() : multi3_cases()) {
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
    // Slots from a plan file (CARVE_PLAN=<file>; written by carve_footprint_bench.cpp CARVE_PLAN_OUT or by
    // resolver_same_cases.cpp PLANNER_PLAN_OUT): the topology solver at slot level after a carve.
    if (mode == "plan") {
        const char* path = std::getenv("CARVE_PLAN");
        if (path == nullptr) {
            std::fprintf(stderr, "plan mode needs CARVE_PLAN=<file>\n");
            return 1;
        }
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
        std::ifstream in(path);
        std::string line;
        std::printf(
            "engines: topo-sat, topo-dfs (solve_topology_mapping) on the slots of each carve; cap %d s per solve\n",
            cap_s);
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
            const Timed none{"-  ", 0, false};
            const Timed a = engine_wanted("auto") ? timed_slots(sc, TopologyMappingSolverEngine::Auto, cap_s) : none;
            const Timed sat = engine_wanted("sat") ? timed_slots(sc, TopologyMappingSolverEngine::Sat, cap_s) : none;
            const Timed dfs = engine_wanted("dfs") ? timed_slots(sc, TopologyMappingSolverEngine::Dfs, cap_s) : none;
            std::printf(
                "%-70s %3zu stages/%3zu slots | topo-auto %s %8.3fs | topo-sat %s %8.3fs | topo-dfs %s %8.3fs\n",
                sc.name.c_str(),
                sc.stage_shapes.size(),
                sc.slots.size(),
                a.ok.c_str(),
                a.s,
                sat.ok.c_str(),
                sat.s,
                dfs.ok.c_str(),
                dfs.s);
            std::fflush(stdout);
        }
        return 0;
    }
    if (mode == "slots" || mode == "slots2") {
        std::printf("engines: topo-sat, topo-dfs (solve_topology_mapping), pre-carved slots; cap %d s per solve\n", cap_s);
        for (const auto& sc : mode == "slots" ? slot_cases() : slot_cases_min2()) {
            auto [sat_ok, sat_s, sat_u] = timed_slots(sc, TopologyMappingSolverEngine::Sat, cap_s);
            auto [dfs_ok, dfs_s, dfs_u] = timed_slots(sc, TopologyMappingSolverEngine::Dfs, cap_s);
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
