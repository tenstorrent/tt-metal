// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SAT-only benchmark: multigraph isomorphism (logical <-> physical 48-node mesh) via CryptoMiniSat.
// Physical multiplicities from hardware lists are 0/4/8 per unordered pair; we halve to 0/2/4 so the
// logical "fabric" uses only even parallel-link counts (2 or 4 nonzero). Logical = same multigraph with
// vertices relabeled by a fixed permutation (always SAT; solver recovers the inverse permutation).
// Encoding: bijection x[t][g]; for all t_i < t_j and k,l, forbid (t_i->k & t_j->l) when
// mult_logical(t_i,t_j) != mult_physical(k,l).
// Dependency: apt install libcryptominisat5-dev
// Run: ./topology_mapping_sat_benchmark
// Optional: TT_METAL_SAT_DUMP_CNF=/path/to/file.cnf to also write DIMACS for debugging.

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
using MultMatrix = std::vector<std::vector<int>>;
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

// Undirected multiplicity: symmetricize listings (each physical link may appear on both endpoints).
void build_global_mult_symmetric(const AdjMap& raw, int n, MultMatrix& mult) {
    mult.assign(static_cast<size_t>(n), std::vector<int>(static_cast<size_t>(n), 0));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int m = std::max(count_parallel_to(raw, i, j), count_parallel_to(raw, j, i));
            mult[static_cast<size_t>(i)][static_cast<size_t>(j)] = m;
            mult[static_cast<size_t>(j)][static_cast<size_t>(i)] = m;
        }
    }
}

// Raw mesh lists encode 4 or 8 hardware links per neighbor pair; shrink to 2 or 4 for logical multiplicities.
void scale_global_mult_by_half(int n, MultMatrix& mult) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int& m = mult[static_cast<size_t>(i)][static_cast<size_t>(j)];
            m /= 2;
        }
    }
}

// mult_out[i][j] = mult_in[perm[i]][perm[j]]; perm must be a permutation of 0..n-1.
void relabel_multigraph(const MultMatrix& mult_in, int n, const std::vector<int>& perm, MultMatrix& mult_out) {
    mult_out.assign(static_cast<size_t>(n), std::vector<int>(static_cast<size_t>(n), 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mult_out[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                mult_in[static_cast<size_t>(perm[static_cast<size_t>(i)])]
                       [static_cast<size_t>(perm[static_cast<size_t>(j)])];
        }
    }
}

// Deterministic nontrivial permutation: i -> (a*i + b) mod n with gcd(a,n)==1.
std::vector<int> make_muladd_perm(int n, int a, int b) {
    std::vector<int> perm(static_cast<size_t>(n), -1);
    std::vector<char> hit(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        const int v = (a * i + b) % n;
        if (v < 0 || hit[static_cast<size_t>(v)]) {
            throw std::runtime_error("make_muladd_perm: not a permutation");
        }
        hit[static_cast<size_t>(v)] = 1;
        perm[static_cast<size_t>(i)] = v;
    }
    return perm;
}

size_t count_clauses(int n, const MultMatrix& mult_t, const MultMatrix& mult_g) {
    const size_t pair_amo = static_cast<size_t>(n) * (static_cast<size_t>(n - 1)) / 2;
    size_t c = static_cast<size_t>(n) + static_cast<size_t>(n) * pair_amo * 2;
    for (int ti = 0; ti < n; ++ti) {
        for (int tj = ti + 1; tj < n; ++tj) {
            const int mt = mult_t[static_cast<size_t>(ti)][static_cast<size_t>(tj)];
            for (int kk = 0; kk < n; ++kk) {
                for (int ll = 0; ll < n; ++ll) {
                    if (mt != mult_g[static_cast<size_t>(kk)][static_cast<size_t>(ll)]) {
                        ++c;
                    }
                }
            }
        }
    }
    return c;
}

static Lit vpos(int n, int ti, int g) { return Lit(static_cast<uint32_t>(ti * n + g), false); }

static Lit vneg(int n, int ti, int g) { return Lit(static_cast<uint32_t>(ti * n + g), true); }

void add_multigraph_iso_clauses(SATSolver& s, int n, const MultMatrix& mult_t, const MultMatrix& mult_g) {
    const size_t nvar = static_cast<size_t>(n) * static_cast<size_t>(n);
    s.new_vars(nvar);

    for (int ti = 0; ti < n; ++ti) {
        std::vector<Lit> row;
        row.reserve(static_cast<size_t>(n));
        for (int g = 0; g < n; ++g) {
            row.push_back(vpos(n, ti, g));
        }
        s.add_clause(row);
    }
    for (int ti = 0; ti < n; ++ti) {
        for (int k = 0; k < n; ++k) {
            for (int l = k + 1; l < n; ++l) {
                s.add_clause({vneg(n, ti, k), vneg(n, ti, l)});
            }
        }
    }
    for (int g = 0; g < n; ++g) {
        for (int ti = 0; ti < n; ++ti) {
            for (int tj = ti + 1; tj < n; ++tj) {
                s.add_clause({vneg(n, ti, g), vneg(n, tj, g)});
            }
        }
    }
    for (int ti = 0; ti < n; ++ti) {
        for (int tj = ti + 1; tj < n; ++tj) {
            const int mt = mult_t[static_cast<size_t>(ti)][static_cast<size_t>(tj)];
            for (int kk = 0; kk < n; ++kk) {
                for (int ll = 0; ll < n; ++ll) {
                    if (mt != mult_g[static_cast<size_t>(kk)][static_cast<size_t>(ll)]) {
                        s.add_clause({vneg(n, ti, kk), vneg(n, tj, ll)});
                    }
                }
            }
        }
    }
}

void write_multigraph_iso_cnf_optional(
    const std::string& path, int n, const MultMatrix& mult_t, const MultMatrix& mult_g) {
    auto lit_for = [n](int ti, int k, bool neg) {
        int v = ti * n + k + 1;
        return neg ? -v : v;
    };
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open " + path);
    }
    out << "p cnf " << (n * n) << " " << count_clauses(n, mult_t, mult_g) << "\n";
    for (int ti = 0; ti < n; ++ti) {
        for (int k = 0; k < n; ++k) {
            out << lit_for(ti, k, false) << (k + 1 < n ? ' ' : '\n');
        }
    }
    for (int ti = 0; ti < n; ++ti) {
        for (int k = 0; k < n; ++k) {
            for (int l = k + 1; l < n; ++l) {
                out << lit_for(ti, k, true) << " " << lit_for(ti, l, true) << " 0\n";
            }
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int ti = 0; ti < n; ++ti) {
            for (int tj = ti + 1; tj < n; ++tj) {
                out << lit_for(ti, k, true) << " " << lit_for(tj, k, true) << " 0\n";
            }
        }
    }
    for (int ti = 0; ti < n; ++ti) {
        for (int tj = ti + 1; tj < n; ++tj) {
            const int mt = mult_t[static_cast<size_t>(ti)][static_cast<size_t>(tj)];
            for (int kk = 0; kk < n; ++kk) {
                for (int ll = 0; ll < n; ++ll) {
                    if (mt != mult_g[static_cast<size_t>(kk)][static_cast<size_t>(ll)]) {
                        out << lit_for(ti, kk, true) << " " << lit_for(tj, ll, true) << " 0\n";
                    }
                }
            }
        }
    }
}

void fill_mesh48(AdjMap& global_adj_map) {
    global_adj_map[0] = {29, 29, 22, 22, 29, 29, 27, 27, 13, 13, 27, 27,
                         29, 29, 27, 27, 22, 22, 27, 27, 29, 29, 13, 13};
    global_adj_map[1] = {32, 32, 22, 22, 32, 32, 13, 13, 28, 28, 32, 32,
                         28, 28, 32, 32, 22, 22, 28, 28, 13, 13, 28, 28};
    global_adj_map[2] = {30, 30, 23, 23, 30, 30, 20, 20, 23, 23, 34, 34,
                         34, 34, 20, 20, 34, 34, 34, 34, 30, 30, 30, 30};
    global_adj_map[3] = {34, 34, 14, 14, 34, 34, 12, 12, 34, 34, 34, 34,
                         14, 14, 30, 30, 30, 30, 30, 30, 12, 12, 30, 30};
    global_adj_map[4] = {21, 21, 24, 24, 22, 22, 24, 24, 25, 25, 25, 25, 18, 18,
                         25, 25, 25, 25, 21, 21, 24, 24, 22, 22, 18, 18, 24, 24};
    global_adj_map[5] = {32, 32, 32, 32, 19, 19, 32, 32, 28, 28, 17, 17,
                         17, 17, 32, 32, 28, 28, 28, 28, 19, 19, 28, 28};
    global_adj_map[6] = {25, 25, 24, 24, 16, 16, 25, 25, 24, 24, 24, 24,
                         15, 15, 16, 16, 25, 25, 15, 15, 25, 25, 24, 24};
    global_adj_map[7] = {26, 26, 21, 21, 33, 33, 26, 26, 18, 18, 33, 33,
                         26, 26, 21, 21, 18, 18, 33, 33, 26, 26, 33, 33};
    global_adj_map[8] = {31, 31, 23, 23, 35, 35, 35, 35, 20, 20, 35, 35,
                         35, 35, 31, 31, 23, 23, 31, 31, 31, 31, 20, 20};
    global_adj_map[9] = {33, 33, 26, 26, 33, 33, 15, 15, 15, 15, 26, 26,
                         26, 26, 33, 33, 16, 16, 26, 26, 33, 33, 16, 16};
    global_adj_map[10] = {17, 17, 29, 29, 29, 29, 27, 27, 27, 27, 19, 19,
                          29, 29, 27, 27, 17, 17, 29, 29, 27, 27, 19, 19};
    global_adj_map[11] = {18, 18, 31, 31, 18, 18, 14, 14, 31, 31, 22, 22, 35, 35, 14, 14,
                          31, 31, 35, 35, 12, 12, 31, 31, 22, 22, 35, 35, 35, 35, 12, 12};
    global_adj_map[12] = {37, 37, 11, 11, 43, 43, 3, 3, 43, 43, 37, 37, 37, 37, 37, 37, 3, 3, 43, 43, 11, 11, 43, 43};
    global_adj_map[13] = {44, 44, 0, 0, 1, 1, 41, 41, 44, 44, 1, 1, 0, 0, 41, 41, 41, 41, 44, 44, 41, 41, 44, 44};
    global_adj_map[14] = {39, 39, 3, 3, 40, 40, 39, 39, 39, 39, 3, 3, 40, 40, 40, 40, 11, 11, 40, 40, 39, 39, 11, 11};
    global_adj_map[15] = {36, 36, 9, 9, 36, 36, 42, 42, 42, 42, 6, 6, 42, 42, 9, 9, 42, 42, 36, 36, 6, 6, 36, 36};
    global_adj_map[16] = {38, 38, 6, 6, 38, 38, 9, 9, 46, 46, 6, 6, 46, 46, 46, 46, 38, 38, 46, 46, 38, 38, 9, 9};
    global_adj_map[17] = {41, 41, 10, 10, 44, 44, 44, 44, 41, 41, 5, 5, 41, 41, 10, 10, 44, 44, 5, 5, 44, 44, 41, 41};
    global_adj_map[18] = {42, 42, 7, 7, 42, 42, 42, 42, 4,  4,  42, 42, 11, 11,
                          36, 36, 7, 7, 36, 36, 36, 36, 11, 11, 4,  4,  36, 36};
    global_adj_map[19] = {45, 45, 5, 5, 45, 45, 45, 45, 10, 10, 47, 47, 45, 45, 5, 5, 47, 47, 10, 10, 47, 47, 47, 47};
    global_adj_map[20] = {40, 40, 2, 2, 2, 2, 39, 39, 40, 40, 40, 40, 40, 40, 8, 8, 8, 8, 39, 39, 39, 39, 39, 39};
    global_adj_map[21] = {7, 7, 38, 38, 46, 46, 46, 46, 4, 4, 46, 46, 7, 7, 38, 38, 4, 4, 38, 38, 38, 38, 46, 46};
    global_adj_map[22] = {47, 47, 4,  4,  45, 45, 11, 11, 47, 47, 0,  0,  4, 4, 0,  0,
                          45, 45, 47, 47, 1,  1,  11, 11, 47, 47, 45, 45, 1, 1, 45, 45};
    global_adj_map[23] = {37, 37, 43, 43, 43, 43, 37, 37, 43, 43, 8, 8, 43, 43, 2, 2, 2, 2, 37, 37, 8, 8, 37, 37};
    global_adj_map[24] = {36, 36, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 36, 36, 38, 38, 4, 4, 4, 4, 6, 6, 38, 38};
    global_adj_map[25] = {6, 6, 4, 4, 6, 6, 42, 42, 6, 6, 4, 4, 46, 46, 6, 6, 4, 4, 42, 42, 4, 4, 46, 46};
    global_adj_map[26] = {42, 42, 9, 9, 7, 7, 46, 46, 46, 46, 9, 9, 7, 7, 7, 7, 42, 42, 9, 9, 9, 9, 7, 7};
    global_adj_map[27] = {10, 10, 45, 45, 0, 0, 0, 0, 10, 10, 45, 45, 41, 41, 0, 0, 10, 10, 41, 41, 0, 0, 10, 10};
    global_adj_map[28] = {1, 1, 5, 5, 45, 45, 41, 41, 1, 1, 45, 45, 1, 1, 5, 5, 41, 41, 5, 5, 1, 1, 5, 5};
    global_adj_map[29] = {0, 0, 0, 0, 44, 44, 47, 47, 10, 10, 10, 10, 0, 0, 47, 47, 0, 0, 10, 10, 44, 44, 10, 10};
    global_adj_map[30] = {40, 40, 3, 3, 2, 2, 3, 3, 43, 43, 3, 3, 2, 2, 40, 40, 2, 2, 2, 2, 43, 43, 3, 3};
    global_adj_map[31] = {8, 8, 40, 40, 8, 8, 8, 8, 43, 43, 11, 11, 40, 40, 11, 11, 8, 8, 43, 43, 11, 11, 11, 11};
    global_adj_map[32] = {1, 1, 47, 47, 5, 5, 5, 5, 47, 47, 5, 5, 1, 1, 1, 1, 44, 44, 1, 1, 44, 44, 5, 5};
    global_adj_map[33] = {9, 9, 7, 7, 36, 36, 7, 7, 9, 9, 36, 36, 9, 9, 38, 38, 7, 7, 9, 9, 38, 38, 7, 7};
    global_adj_map[34] = {2, 2, 2, 2, 3, 3, 3, 3, 39, 39, 3, 3, 37, 37, 39, 39, 2, 2, 3, 3, 37, 37, 2, 2};
    global_adj_map[35] = {11, 11, 11, 11, 11, 11, 39, 39, 37, 37, 8,  8,  8, 8,
                          47, 47, 8,  8,  11, 11, 37, 37, 47, 47, 39, 39, 8, 8};
    global_adj_map[36] = {33, 33, 18, 18, 24, 24, 18, 18, 18, 18, 18, 18,
                          15, 15, 15, 15, 24, 24, 15, 15, 15, 15, 33, 33};
    global_adj_map[37] = {12, 12, 35, 35, 23, 23, 34, 34, 23, 23, 12, 12,
                          35, 35, 12, 12, 34, 34, 23, 23, 12, 12, 23, 23};
    global_adj_map[38] = {24, 24, 21, 21, 16, 16, 16, 16, 21, 21, 21, 21,
                          16, 16, 33, 33, 16, 16, 24, 24, 33, 33, 21, 21};
    global_adj_map[39] = {14, 14, 35, 35, 14, 14, 34, 34, 34, 34, 20, 20,
                          20, 20, 14, 14, 20, 20, 35, 35, 20, 20, 14, 14};
    global_adj_map[40] = {30, 30, 14, 14, 14, 14, 20, 20, 20, 20, 20, 20,
                          30, 30, 20, 20, 31, 31, 31, 31, 14, 14, 14, 14};
    global_adj_map[41] = {17, 17, 28, 28, 27, 27, 13, 13, 13, 13, 17, 17,
                          28, 28, 13, 13, 17, 17, 27, 27, 17, 17, 13, 13};
    global_adj_map[42] = {15, 15, 26, 26, 15, 15, 25, 25, 15, 15, 18, 18,
                          25, 25, 18, 18, 26, 26, 18, 18, 15, 15, 18, 18};
    global_adj_map[43] = {12, 12, 23, 23, 30, 30, 23, 23, 31, 31, 30, 30,
                          12, 12, 23, 23, 31, 31, 12, 12, 12, 12, 23, 23};
    global_adj_map[44] = {13, 13, 32, 32, 13, 13, 13, 13, 29, 29, 13, 13,
                          17, 17, 17, 17, 32, 32, 17, 17, 29, 29, 17, 17};
    global_adj_map[45] = {27, 27, 22, 22, 22, 22, 19, 19, 19, 19, 22, 22,
                          28, 28, 22, 22, 19, 19, 28, 28, 19, 19, 27, 27};
    global_adj_map[46] = {21, 21, 26, 26, 21, 21, 16, 16, 26, 26, 16, 16,
                          21, 21, 25, 25, 16, 16, 16, 16, 21, 21, 25, 25};
    global_adj_map[47] = {22, 22, 19, 19, 22, 22, 29, 29, 22, 22, 32, 32, 35, 35,
                          29, 29, 19, 19, 22, 22, 32, 32, 19, 19, 35, 35, 19, 19};
}

bool model_to_assignment(
    const std::vector<lbool>& model, int num_vars, std::vector<signed char>& assignment, std::string& err) {
    if (static_cast<int>(model.size()) < num_vars) {
        err = "model shorter than n*n vars";
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

bool mapping_from_witness(
    int n, const std::vector<signed char>& assignment, std::vector<int>& tgt_to_global, std::string& err) {
    tgt_to_global.assign(static_cast<size_t>(n), -1);
    for (int t = 0; t < n; ++t) {
        int chosen = -1;
        int ntrue = 0;
        for (int g = 0; g < n; ++g) {
            int idx = t * n + g;
            if (assignment[static_cast<size_t>(idx)] == 1) {
                ++ntrue;
                chosen = g;
            }
        }
        if (ntrue != 1) {
            err = "target row " + std::to_string(t) + " has " + std::to_string(ntrue) + " positive x[t,*] (expected 1)";
            return false;
        }
        tgt_to_global[static_cast<size_t>(t)] = chosen;
    }
    std::vector<int> use_count(static_cast<size_t>(n), 0);
    for (int g : tgt_to_global) {
        if (g >= 0 && g < n) {
            use_count[static_cast<size_t>(g)]++;
        }
    }
    for (int g = 0; g < n; ++g) {
        if (use_count[static_cast<size_t>(g)] != 1) {
            err = "global column " + std::to_string(g) + " used " + std::to_string(use_count[static_cast<size_t>(g)]) +
                  " times (expected bijection)";
            return false;
        }
    }
    err.clear();
    return true;
}

void print_index_mapping(int n, const std::vector<int>& tgt_to_global) {
    std::cout << "\n--- 1:1 mapping (target index -> global index, 0-based) ---\n";
    for (int t = 0; t < n; ++t) {
        std::cout << "  " << t << " -> " << tgt_to_global[static_cast<size_t>(t)] << "\n";
    }
    std::cout << "--- compact (global image in target order 0.." << (n - 1) << ") ---\n  [";
    for (int t = 0; t < n; ++t) {
        if (t > 0) {
            std::cout << ", ";
        }
        std::cout << tgt_to_global[static_cast<size_t>(t)];
    }
    std::cout << "]\n---\n";
}

}  // namespace

int main() {
    constexpr int n = 48;

    AdjMap mesh_raw;
    fill_mesh48(mesh_raw);

    MultMatrix mult_g;
    build_global_mult_symmetric(mesh_raw, n, mult_g);
    scale_global_mult_by_half(n, mult_g);

    MultMatrix mult_t;
    // gcd(17, 48) == 1 -> bijection on target vertex names; logical multiplicities are 0 / 2 / 4 only.
    const std::vector<int> relabel = make_muladd_perm(n, 17, 11);
    relabel_multigraph(mult_g, n, relabel, mult_t);

    std::cout << "=== topology_mapping_sat_benchmark (CryptoMiniSat C++ API) n=" << n << " ===\n";
    std::cout << "logical = permuted physical (mult {0,2,4}); expecting SAT (non-unique if automorphisms exist)\n";
    std::cout << "CNF vars " << (n * n) << " clauses " << count_clauses(n, mult_t, mult_g) << "\n";

    const char* dump = std::getenv("TT_METAL_SAT_DUMP_CNF");
    if (dump != nullptr && dump[0] != '\0') {
        try {
            write_multigraph_iso_cnf_optional(dump, n, mult_t, mult_g);
            std::cout << "Wrote DIMACS to " << dump << " (TT_METAL_SAT_DUMP_CNF)\n";
        } catch (const std::exception& ex) {
            std::cerr << "DIMACS dump failed: " << ex.what() << "\n";
        }
    }

    try {
        SATSolver solver;
        solver.set_verbosity(0);
        auto t_c0 = std::chrono::steady_clock::now();
        add_multigraph_iso_clauses(solver, n, mult_t, mult_g);
        auto t_c1 = std::chrono::steady_clock::now();
        lbool ret = solver.solve();
        auto t_c2 = std::chrono::steady_clock::now();

        auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_c1 - t_c0).count();
        auto solve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_c2 - t_c1).count();

        std::cout << "CMS: clause build " << build_ms << " ms, solve " << solve_ms << " ms\n";
        if (ret == l_True) {
            std::cout << "Result: SAT\n";
            const int num_vars = n * n;
            std::vector<signed char> witness;
            std::string err;
            if (model_to_assignment(solver.get_model(), num_vars, witness, err)) {
                std::vector<int> perm;
                std::string merr;
                if (mapping_from_witness(n, witness, perm, merr)) {
                    print_index_mapping(n, perm);
                } else {
                    std::cout << "Model is not a bijection: " << merr << "\n";
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
