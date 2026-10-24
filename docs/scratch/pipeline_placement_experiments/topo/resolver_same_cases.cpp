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
        bool ok = true;
        try {
            resolve_pipeline_placement(order, edges, f.links, counts, f.chips, {});
        } catch (const std::exception&) {
            ok = false;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[64];
        std::snprintf(buf, sizeof buf, "%d %.6f", ok ? 1 : 0, s);
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
                            chips[i].push_back({FabricNodeId{MeshId{uint32_t(t.mesh)}, uint32_t(((sr * 4 + rr) * C * 2) + scn * 2 + cc)},
                                                uint32_t((sr - t.r0) * 4 + rr), uint32_t((scn - t.c0) * 2 + cc)});
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
        for (const auto& cb : sc.cables) {
            const size_t a = slot_of.at({cb.mesh_a, cb.row_a, cb.col_a}), b = slot_of.at({cb.mesh_b, cb.row_b, cb.col_b});
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
        bool ok = true;
        try {
            resolve_pipeline_placement(order, edges, links, counts, chips, {});
        } catch (const std::exception&) {
            ok = false;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[64];
        std::snprintf(buf, sizeof buf, "%d %.6f", ok ? 1 : 0, s);
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
    if (mode == "multi2") {
        for (const auto& m : multi2_cases()) {
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
    if (mode == "slots") {
        std::printf("engine: pipeline_builder.cpp resolver, pre-carved slots; cap %d s per solve\n", cap_s);
        for (const auto& sc : slot_cases()) {
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
