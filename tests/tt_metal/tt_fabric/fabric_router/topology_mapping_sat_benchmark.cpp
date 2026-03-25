// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SAT-only benchmark: induced subgraph isomorphism via CryptoMiniSat (parallel links → simple edges).
// Degrees / histograms count distinct neighbors only (raw parallel links merged to one edge).
// Physical: always the full 64-node pipeline from fill_pipeline64_adj. Logical: simple n-cycle C_n on
// n stages (TT_METAL_SAT_N, default 64). Encoding: injective x[stage][device] (n×64 vars); each stage
// picks one device; no two stages share a device; adj on stages matches adj on chosen devices.
// Dependency: apt install libcryptominisat5-dev
// Run: ./topology_mapping_sat_benchmark
// Optional: TT_METAL_SAT_DUMP_CNF=/path/to/dimacs.cnf
// Optional: TT_METAL_SAT_N logical stages (2..64); physical stays 64 devices.

#include <cryptominisat5/cryptominisat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using AdjMap = std::map<uint32_t, std::vector<uint64_t>>;
// 0/1 undirected adjacency (any parallel hardware link counts as a single edge).
using AdjMatrix = std::vector<std::vector<char>>;

#include "fill_pipeline64_adj.inc"

using CMSat::l_False;
using CMSat::l_True;
using CMSat::lbool;
using CMSat::Lit;
using CMSat::SATSolver;

int count_parallel_to(const AdjMap& raw, int u, int v) {
    auto it = raw.find(static_cast<uint32_t>(u));
    if (it == raw.end()) {
        return 0;
    }
    int c = 0;
    for (uint64_t x64 : it->second) {
        if (static_cast<int>(x64) == v) {
            ++c;
        }
    }
    return c;
}

void build_global_adj_simple(const AdjMap& raw, int n, AdjMatrix& adj) {
    adj.assign(static_cast<size_t>(n), std::vector<char>(static_cast<size_t>(n), 0));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const char has = (std::max(count_parallel_to(raw, i, j), count_parallel_to(raw, j, i)) > 0) ? 1 : 0;
            adj[static_cast<size_t>(i)][static_cast<size_t>(j)] = has;
            adj[static_cast<size_t>(j)][static_cast<size_t>(i)] = has;
        }
    }
}

// Fill any still-missing device rows by listing v once per occurrence of u in v's adjacency list.
void complete_adj_missing_devices(AdjMap& m, int n) {
    for (int u = 0; u < n; ++u) {
        if (m.find(static_cast<uint32_t>(u)) != m.end()) {
            continue;
        }
        std::vector<uint64_t> back;
        for (int v = 0; v < n; ++v) {
            if (v == u) {
                continue;
            }
            auto it = m.find(static_cast<uint32_t>(v));
            if (it == m.end()) {
                continue;
            }
            for (uint64_t x64 : it->second) {
                if (static_cast<int>(x64) == u) {
                    back.push_back(static_cast<uint64_t>(v));
                }
            }
        }
        m[static_cast<uint32_t>(u)] = std::move(back);
    }
}

void build_ring_adj_simple(int n, AdjMatrix& adj) {
    adj.assign(static_cast<size_t>(n), std::vector<char>(static_cast<size_t>(n), 0));
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        adj[static_cast<size_t>(i)][static_cast<size_t>(j)] = 1;
        adj[static_cast<size_t>(j)][static_cast<size_t>(i)] = 1;
    }
}

int count_undirected_edges(const AdjMatrix& adj, int n) {
    int c = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            c += adj[static_cast<size_t>(i)][static_cast<size_t>(j)] != 0 ? 1 : 0;
        }
    }
    return c;
}

// Number of distinct neighbors (each parallel link in raw data collapsed to one edge first).
int simple_degree(const AdjMatrix& adj, int n, int v) {
    int d = 0;
    for (int j = 0; j < n; ++j) {
        if (j != v && adj[static_cast<size_t>(v)][static_cast<size_t>(j)] != 0) {
            ++d;
        }
    }
    return d;
}

void print_degree_histogram(const char* title, const AdjMatrix& adj, int n) {
    std::map<int, int> hist;
    for (int v = 0; v < n; ++v) {
        ++hist[simple_degree(adj, n, v)];
    }
    std::cout << title << " (unique neighbors per vertex):\n";
    for (const auto& [deg, cnt] : hist) {
        std::cout << "  degree " << deg << " -> " << cnt << " vertices\n";
    }
}

size_t count_embed_clauses(int n_log, int n_host, const AdjMatrix& adj_t, const AdjMatrix& adj_g) {
    const size_t pair_host = static_cast<size_t>(n_host) * (static_cast<size_t>(n_host - 1)) / 2;
    const size_t pair_log = static_cast<size_t>(n_log) * (static_cast<size_t>(n_log - 1)) / 2;
    size_t c = static_cast<size_t>(n_log) + static_cast<size_t>(n_log) * pair_host;
    c += static_cast<size_t>(n_host) * pair_log;
    for (int ti = 0; ti < n_log; ++ti) {
        for (int tj = ti + 1; tj < n_log; ++tj) {
            const char at = adj_t[static_cast<size_t>(ti)][static_cast<size_t>(tj)];
            for (int kk = 0; kk < n_host; ++kk) {
                for (int ll = 0; ll < n_host; ++ll) {
                    if (at != adj_g[static_cast<size_t>(kk)][static_cast<size_t>(ll)]) {
                        ++c;
                    }
                }
            }
        }
    }
    return c;
}

// Variable index: stage ti in [0,n_log), device g in [0,n_host) => ti * n_host + g
static Lit vpos_emb(int n_host, int ti, int g) { return Lit(static_cast<uint32_t>(ti * n_host + g), false); }

static Lit vneg_emb(int n_host, int ti, int g) { return Lit(static_cast<uint32_t>(ti * n_host + g), true); }

void add_induced_subgraph_iso_clauses(
    SATSolver& s, int n_log, int n_host, const AdjMatrix& adj_t, const AdjMatrix& adj_g) {
    const size_t nvar = static_cast<size_t>(n_log) * static_cast<size_t>(n_host);
    s.new_vars(nvar);

    for (int ti = 0; ti < n_log; ++ti) {
        std::vector<Lit> row;
        row.reserve(static_cast<size_t>(n_host));
        for (int g = 0; g < n_host; ++g) {
            row.push_back(vpos_emb(n_host, ti, g));
        }
        s.add_clause(row);
    }
    for (int ti = 0; ti < n_log; ++ti) {
        for (int k = 0; k < n_host; ++k) {
            for (int l = k + 1; l < n_host; ++l) {
                s.add_clause({vneg_emb(n_host, ti, k), vneg_emb(n_host, ti, l)});
            }
        }
    }
    for (int g = 0; g < n_host; ++g) {
        for (int ti = 0; ti < n_log; ++ti) {
            for (int tj = ti + 1; tj < n_log; ++tj) {
                s.add_clause({vneg_emb(n_host, ti, g), vneg_emb(n_host, tj, g)});
            }
        }
    }
    for (int ti = 0; ti < n_log; ++ti) {
        for (int tj = ti + 1; tj < n_log; ++tj) {
            const char at = adj_t[static_cast<size_t>(ti)][static_cast<size_t>(tj)];
            for (int kk = 0; kk < n_host; ++kk) {
                for (int ll = 0; ll < n_host; ++ll) {
                    if (at != adj_g[static_cast<size_t>(kk)][static_cast<size_t>(ll)]) {
                        s.add_clause({vneg_emb(n_host, ti, kk), vneg_emb(n_host, tj, ll)});
                    }
                }
            }
        }
    }
}

void write_embed_cnf_optional(
    const std::string& path, int n_log, int n_host, const AdjMatrix& adj_t, const AdjMatrix& adj_g) {
    auto lit_for = [n_host](int ti, int k, bool neg) {
        int v = ti * n_host + k + 1;
        return neg ? -v : v;
    };
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open " + path);
    }
    const int nvar = n_log * n_host;
    out << "p cnf " << nvar << " " << count_embed_clauses(n_log, n_host, adj_t, adj_g) << "\n";
    for (int ti = 0; ti < n_log; ++ti) {
        for (int k = 0; k < n_host; ++k) {
            out << lit_for(ti, k, false) << (k + 1 < n_host ? ' ' : '\n');
        }
    }
    for (int ti = 0; ti < n_log; ++ti) {
        for (int k = 0; k < n_host; ++k) {
            for (int l = k + 1; l < n_host; ++l) {
                out << lit_for(ti, k, true) << " " << lit_for(ti, l, true) << " 0\n";
            }
        }
    }
    for (int col = 0; col < n_host; ++col) {
        for (int ti = 0; ti < n_log; ++ti) {
            for (int tj = ti + 1; tj < n_log; ++tj) {
                out << lit_for(ti, col, true) << " " << lit_for(tj, col, true) << " 0\n";
            }
        }
    }
    for (int ti = 0; ti < n_log; ++ti) {
        for (int tj = ti + 1; tj < n_log; ++tj) {
            const char at = adj_t[static_cast<size_t>(ti)][static_cast<size_t>(tj)];
            for (int kk = 0; kk < n_host; ++kk) {
                for (int ll = 0; ll < n_host; ++ll) {
                    if (at != adj_g[static_cast<size_t>(kk)][static_cast<size_t>(ll)]) {
                        out << lit_for(ti, kk, true) << " " << lit_for(tj, ll, true) << " 0\n";
                    }
                }
            }
        }
    }
}

bool model_to_assignment(
    const std::vector<lbool>& model, int num_vars, std::vector<signed char>& assignment, std::string& err) {
    if (static_cast<int>(model.size()) < num_vars) {
        err = "model shorter than n_log*n_host vars";
        return false;
    }
    assignment.assign(static_cast<size_t>(num_vars), static_cast<signed char>(-1));
    for (int i = 0; i < num_vars; ++i) {
        if (model[static_cast<size_t>(i)] == l_True) {
            assignment[static_cast<size_t>(i)] = 1;
        } else if (model[static_cast<size_t>(i)] == l_False) {
            assignment[static_cast<size_t>(i)] = 0;
        } else {
            err = "variable " + std::to_string(i) + " undefined in model";
            return false;
        }
    }
    err.clear();
    return true;
}

bool mapping_from_witness_embed(
    int n_log,
    int n_host,
    const std::vector<signed char>& assignment,
    std::vector<int>& stage_to_device,
    std::string& err) {
    stage_to_device.assign(static_cast<size_t>(n_log), -1);
    for (int t = 0; t < n_log; ++t) {
        int chosen = -1;
        int ntrue = 0;
        for (int g = 0; g < n_host; ++g) {
            const int idx = t * n_host + g;
            if (assignment[static_cast<size_t>(idx)] == 1) {
                ++ntrue;
                chosen = g;
            }
        }
        if (ntrue != 1) {
            err = "stage row " + std::to_string(t) + " has " + std::to_string(ntrue) +
                  " positive x[stage,*] (expected 1)";
            return false;
        }
        stage_to_device[static_cast<size_t>(t)] = chosen;
    }
    std::vector<int> use_count(static_cast<size_t>(n_host), 0);
    for (int g : stage_to_device) {
        if (g >= 0 && g < n_host) {
            use_count[static_cast<size_t>(g)]++;
        }
    }
    for (int g = 0; g < n_host; ++g) {
        if (use_count[static_cast<size_t>(g)] > 1) {
            err = "device " + std::to_string(g) + " used " + std::to_string(use_count[static_cast<size_t>(g)]) +
                  " times (expected injective embedding)";
            return false;
        }
    }
    err.clear();
    return true;
}

void print_stage_to_device_mapping(int n_log, const std::vector<int>& stage_to_device) {
    std::cout << "\n--- injective map (logical stage -> pipeline device id, 0-based) ---\n";
    for (int t = 0; t < n_log; ++t) {
        std::cout << "  stage " << t << " -> device " << stage_to_device[static_cast<size_t>(t)] << "\n";
    }
    std::cout << "--- compact (devices in stage order 0.." << (n_log - 1) << ") ---\n  [";
    for (int t = 0; t < n_log; ++t) {
        if (t > 0) {
            std::cout << ", ";
        }
        std::cout << stage_to_device[static_cast<size_t>(t)];
    }
    std::cout << "]\n---\n";
}

}  // namespace

int main() {
    constexpr int n_host = 64;
    int n_log = n_host;
    if (const char* ns = std::getenv("TT_METAL_SAT_N"); ns != nullptr && ns[0] != '\0') {
        n_log = std::atoi(ns);
    }
    if (n_log < 2 || n_log > n_host) {
        std::cerr << "TT_METAL_SAT_N must be between 2 and " << n_host << " (got " << n_log << ")\n";
        return 1;
    }

    AdjMap mesh_raw;
    fill_pipeline64_adj(mesh_raw);
    complete_adj_missing_devices(mesh_raw, n_host);

    AdjMatrix adj_g;
    build_global_adj_simple(mesh_raw, n_host, adj_g);

    AdjMatrix adj_t;
    build_ring_adj_simple(n_log, adj_t);

    const int edges_logical = count_undirected_edges(adj_t, n_log);
    const int edges_physical = count_undirected_edges(adj_g, n_host);
    const int num_vars = n_log * n_host;

    std::cout << "=== topology_mapping_sat_benchmark (CryptoMiniSat C++ API) ===\n";
    std::cout << "physical = full " << n_host << "-device pipeline (simple adjacency)\n";
    std::cout << "logical = C_" << n_log << " on stages [0, " << n_log << ") (induced subgraph embed into " << n_host
              << " devices)\n";
    std::cout << "undirected edges: logical " << edges_logical << ", physical(full) " << edges_physical << "\n";
    print_degree_histogram("--- logical (C_n simple graph)", adj_t, n_log);
    print_degree_histogram("--- physical (64-device pipeline, simple graph)", adj_g, n_host);
    std::cout << "CNF vars " << num_vars << " clauses " << count_embed_clauses(n_log, n_host, adj_t, adj_g) << "\n";

    const char* dump = std::getenv("TT_METAL_SAT_DUMP_CNF");
    if (dump != nullptr && dump[0] != '\0') {
        try {
            write_embed_cnf_optional(dump, n_log, n_host, adj_t, adj_g);
            std::cout << "Wrote DIMACS to " << dump << " (TT_METAL_SAT_DUMP_CNF)\n";
        } catch (const std::exception& ex) {
            std::cerr << "DIMACS dump failed: " << ex.what() << "\n";
        }
    }

    try {
        SATSolver solver;
        solver.set_verbosity(0);
        auto t_c0 = std::chrono::steady_clock::now();
        add_induced_subgraph_iso_clauses(solver, n_log, n_host, adj_t, adj_g);
        auto t_c1 = std::chrono::steady_clock::now();
        lbool ret = solver.solve();
        auto t_c2 = std::chrono::steady_clock::now();

        auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_c1 - t_c0).count();
        auto solve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_c2 - t_c1).count();

        std::cout << "CMS: clause build " << build_ms << " ms, solve " << solve_ms << " ms\n";
        if (ret == l_True) {
            std::cout << "Result: SAT\n";
            std::vector<signed char> witness;
            std::string err;
            if (model_to_assignment(solver.get_model(), num_vars, witness, err)) {
                std::vector<int> embedding;
                std::string merr;
                if (mapping_from_witness_embed(n_log, n_host, witness, embedding, merr)) {
                    print_stage_to_device_mapping(n_log, embedding);
                } else {
                    std::cout << "Invalid embedding: " << merr << "\n";
                }
            } else {
                std::cout << "Model extraction: " << err << "\n";
            }
        } else if (ret == l_False) {
            std::cout << "Result: UNSAT\n";
        } else {
            std::cout << "Result: UNKNOWN (timeout / limit / interrupted)\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 2;
    }

    return 0;
}
