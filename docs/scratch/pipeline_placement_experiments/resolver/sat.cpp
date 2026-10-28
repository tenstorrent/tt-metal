// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cadical.hpp"

// Submesh-level ring placement: stage i on submesh m, consecutive stages on adjacent submeshes.
struct Graph {
    size_t n = 0;
    std::vector<std::set<size_t>> adj;
};

size_t add_grid(Graph& g, size_t r, size_t c) {
    size_t base = g.n;
    g.n += r * c;
    g.adj.resize(g.n);
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            size_t a = base + i * c + j;
            if (j + 1 < c) {
                g.adj[a].insert(a + 1), g.adj[a + 1].insert(a);
            }
            if (i + 1 < r) {
                g.adj[a].insert(a + c), g.adj[a + c].insert(a);
            }
        }
    }
    return base;
}
void cable(Graph& g, size_t a, size_t b) { g.adj[a].insert(b), g.adj[b].insert(a); }

void run(const char* name, const Graph& g, size_t stages) {
    auto t0 = std::chrono::steady_clock::now();
    CaDiCaL::Solver s;
    auto x = [&](size_t i, size_t m) { return int(i * g.n + m + 1); };
    for (size_t i = 0; i < stages; ++i) {
        for (size_t m = 0; m < g.n; ++m) {
            s.add(x(i, m));
        }
        s.add(0);
        for (size_t m = 0; m < g.n; ++m) {
            for (size_t k = m + 1; k < g.n; ++k) {
                s.add(-x(i, m)), s.add(-x(i, k)), s.add(0);
            }
        }
    }
    for (size_t m = 0; m < g.n; ++m) {
        for (size_t i = 0; i < stages; ++i) {
            for (size_t k = i + 1; k < stages; ++k) {
                s.add(-x(i, m)), s.add(-x(k, m)), s.add(0);
            }
        }
    }
    for (size_t i = 0; i < stages; ++i) {
        size_t nx = (i + 1) % stages;
        for (size_t m = 0; m < g.n; ++m) {
            s.add(-x(i, m));
            for (size_t to : g.adj[m]) {
                s.add(x(nx, to));
            }
            s.add(0);
        }
    }
    int r = s.solve();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf(
        "%-44s stages=%3zu submeshes=%3zu %-10s %.6fs\n", name, stages, g.n, r == 10 ? "feasible" : "INFEASIBLE", sec);
    std::fflush(stdout);
}

// argv[1] == "big" adds the two-4x8-mesh case, which ran over 230 s.
int main(int argc, char** argv) {
    const bool big = argc > 1 && std::string(argv[1]) == "big";
    for (auto [r, c] : std::vector<std::pair<size_t, size_t>>{{4, 4}, {4, 8}, {8, 8}}) {
        Graph g;
        add_grid(g, r, c);
        run(("one mesh grid " + std::to_string(r) + "x" + std::to_string(c) + ", full ring").c_str(), g, r * c);
    }
    for (auto [r, c] : std::vector<std::pair<size_t, size_t>>{{3, 3}, {3, 5}, {5, 5}, {3, 7}, {5, 7}}) {
        Graph g;
        add_grid(g, r, c);
        run(("odd grid " + std::to_string(r) + "x" + std::to_string(c) + ", full ring").c_str(), g, r * c);
    }
    for (size_t k : {2, 4}) {
        Graph g;
        size_t a = add_grid(g, k, k), b = add_grid(g, k, k);
        cable(g, a, b), cable(g, a + k + 1, b + k + 1);
        run(("two " + std::to_string(k) + "x" + std::to_string(k) + " meshes, cables same colour").c_str(),
            g,
            2 * k * k);
    }
    if (!big) {
        return 0;
    }
    Graph g;
    size_t a = add_grid(g, 4, 8), b = add_grid(g, 4, 8);
    cable(g, a, b), cable(g, a + 9, b + 9);
    run("two 4x8 meshes, cables same colour", g, 64);
}
