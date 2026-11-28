// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Work plan of one sparse-row launch (tt/custom_ops/sparse_rows/plan.py's SparseRowPlan, on the device-generated
// path): row ownership per core, x window pieces, neighbour exchange, and the per-core metadata rows.
#pragma once
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "device.hpp"
#include "level.hpp"

namespace hpcg_tt {

// ALIGN, MAX_NB, GRAN, WGAP, MAX_PIECES: tt/shared_constants.h (through level.hpp)

struct Meta {  // one launch's metadata rows, uploaded, and its per-core segment counts
    Tensor t;
    int64_t w = 0;  // words per row (one DRAM page per core)
    std::vector<int64_t> nseg, nseg_gs, nseg_r;
};

struct Plan {
    int nslot = 27;
    int64_t pout = 0;  // out rows' vector length
    int nc = 0, NB = 0, R = 1;
    std::vector<std::pair<int64_t, int64_t>> out_blocks;
    std::vector<int64_t> own, own_l1;                   // [nc][NB][2], [nc][NB]
    std::vector<int64_t> plo, pln, pl1, npc;            // [nc][R] window pieces (zero past npc[k]), pieces per core
    int64_t win_b = 0;                                  // window bytes (a multiple of 512)
    std::vector<int64_t> exch_rows, exch_at, nbr, nat;  // exchange entries (5 words) per (core, block); neighbours
    // transfers: kind 1 restrict, 2 prolong, between fine level F and coarse level C
    int kind = 0;
    const Level* F = nullptr;
    const Level* C = nullptr;
    const Level* L = nullptr;  // stencil plans: the level
    std::map<std::pair<std::vector<int64_t>, int64_t>, Meta> metas;

    // a Gauss-Seidel plan of level L on up to ncores cores (planned on fewer while a core has more than MAX_NB
    // neighbours); defer: exchange lists left for complete_exchange()
    static Plan stencil(const Level& L, int ncores, const Dev& D, bool defer);
    // a grid transfer plan (restriction or prolongation) between fine F and coarse C
    static Plan transfer(int kind, const Level& F, const Level& C, int ncores);
    bool complete_exchange(const Dev& D);  // false when a core has more than MAX_NB neighbours
    int max_neighbours() const;

    const Meta& meta(const Dev& D, const std::vector<int64_t>& phases, int64_t tpw);
    void segment_counts(
        const std::vector<int64_t>& phases,
        int64_t tpw,
        std::vector<int64_t>& nseg,
        std::vector<int64_t>& nseg_gs,
        std::vector<int64_t>& nseg_r) const;

private:
    void ownership(const std::vector<std::vector<int64_t>>* bands);
    void exchange(const Dev& D, int64_t maxp, const int64_t* lo, const int64_t* ln, const int64_t* l1);
    void compact(
        int64_t maxp, const std::vector<int64_t>& lo, const std::vector<int64_t>& ln, const std::vector<int64_t>& l1);
};

}  // namespace hpcg_tt
