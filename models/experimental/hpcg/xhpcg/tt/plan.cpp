// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "plan.hpp"

#include <algorithm>
#include <stdexcept>

#include "native.hpp"

namespace hpcg_tt {

namespace {
int clamp_cores(int ncores, const std::vector<std::pair<int64_t, int64_t>>& blocks) {
    int64_t most = 0;
    for (const auto& [s, e] : blocks) {
        most = std::max(most, (e - s + ALIGN - 1) / ALIGN);
    }
    return (int)std::max<int64_t>(1, std::min<int64_t>(ncores, most));
}

// bands are used only when they split every block and the cores divide evenly over each block's bands (plan.py's
// _init). Blocks can differ: on an odd nx the even-y colours have one band more than the odd-y ones (level.cpp)
const std::vector<std::vector<int64_t>>* usable_bands(const std::vector<std::vector<int64_t>>* bands, int NB, int nc) {
    if (bands == nullptr || (int)bands->size() != NB) {
        return nullptr;
    }
    for (const auto& b : *bands) {
        const int B = (int)b.size() - 1;
        if (nc % B || nc < B) {
            return nullptr;
        }
    }
    return bands;
}

std::vector<std::pair<int64_t, int64_t>> colour_blocks(const Level& L) {
    std::vector<std::pair<int64_t, int64_t>> b;
    for (int c = 0; c < 8; ++c) {
        b.emplace_back(L.starts[c], L.starts[c + 1]);
    }
    return b;
}
}  // namespace

void Plan::ownership(const std::vector<std::vector<int64_t>>* bands) {
    std::vector<int64_t> s, e, cuts, cat(NB + 1, 0);
    for (const auto& [a, b] : out_blocks) {
        s.push_back(a);
        e.push_back(b);
    }
    if (bands != nullptr) {
        for (int b = 0; b < NB; ++b) {
            cuts.insert(cuts.end(), (*bands)[b].begin(), (*bands)[b].end());
            cat[b + 1] = cat[b] + (int64_t)(*bands)[b].size();
        }
    }
    own.assign((size_t)nc * NB * 2, 0);
    hpcg_ownership(NB, s.data(), e.data(), cuts.data(), cat.data(), nc, ALIGN, own.data());
}

void Plan::compact(
    int64_t maxp, const std::vector<int64_t>& lo, const std::vector<int64_t>& ln, const std::vector<int64_t>& l1) {
    R = 1;
    for (int k = 0; k < nc; ++k) {
        R = (int)std::max<int64_t>(R, npc[k]);
    }
    plo.assign((size_t)nc * R, 0);
    pln.assign((size_t)nc * R, 0);
    pl1.assign((size_t)nc * R, 0);
    int64_t most = 0;
    for (int k = 0; k < nc; ++k) {
        int64_t words = 0;
        for (int64_t i = 0; i < npc[k]; ++i) {
            plo[k * R + i] = lo[k * maxp + i];
            pln[k * R + i] = ln[k * maxp + i];
            pl1[k * R + i] = l1[k * maxp + i];
            words += ln[k * maxp + i];
        }
        most = std::max(most, words);
    }
    win_b = (most * 4 + 511) / 512 * 512;  // ZEROX clears the window in 512-byte pieces
}

void Plan::exchange(const Dev& D, int64_t maxp, const int64_t* lo, const int64_t* ln, const int64_t* l1) {
    int64_t cap = 64LL * nc * NB;
    while (true) {
        std::vector<int64_t> rows(cap * 5), at((size_t)nc * NB + 1), nb((size_t)nc * nc), na(nc + 1);
        const int64_t n = hpcg_plan_exchange_grouped(
            nc,
            NB,
            own.data(),
            maxp,
            lo,
            ln,
            l1,
            npc.data(),
            D.noc_xy.data(),
            cap,
            rows.data(),
            at.data(),
            nb.data(),
            na.data());
        if (n >= 0) {
            rows.resize(n * 5);
            nb.resize(na[nc]);
            exch_rows = std::move(rows);
            exch_at = std::move(at);
            nbr = std::move(nb);
            nat = std::move(na);
            return;
        }
        cap = -n;
    }
}

int Plan::max_neighbours() const {
    int m = 0;
    for (int k = 0; k + 1 < (int)nat.size(); ++k) {
        m = std::max(m, (int)(nat[k + 1] - nat[k]));
    }
    return m;
}

bool Plan::complete_exchange(const Dev& D) {
    exchange(D, R, plo.data(), pln.data(), pl1.data());
    metas.clear();
    return max_neighbours() <= MAX_NB;
}

Plan Plan::stencil(const Level& L, int ncores, const Dev& D, bool defer) {
    const auto blocks = colour_blocks(L);
    int nreq = ncores;
    while (true) {
        Plan p;
        p.L = &L;
        p.nslot = 27;
        p.pout = L.P;
        p.out_blocks = blocks;
        p.NB = 8;
        p.nc = clamp_cores(nreq, blocks);
        p.ownership(usable_bands(L.bands > 1 ? &L.bstarts : nullptr, p.NB, p.nc));
        const int64_t ng = blocks.back().second / GRAN + 2;  // granules of the input vector (its blocks)
        const int64_t empty_lo = blocks[0].first;
        p.own_l1.assign((size_t)p.nc * p.NB, 0);
        std::vector<int64_t> lo, ln, l1;
        int64_t maxp = 0;
        for (int a = 0; a < 2; ++a) {
            maxp = MAX_PIECES[a];
            lo.assign(p.nc * maxp, 0);
            ln.assign(p.nc * maxp, 0);
            l1.assign(p.nc * maxp, 0);
            p.npc.assign(p.nc, 0);
            const int64_t r = hpcg_plan_windows_stencil(
                L.nx,
                L.nb,
                L.nl,
                L.line_band.data(),
                L.band_y0.data(),
                L.band_ny.data(),
                L.band_start.data(),
                p.nc,
                p.NB,
                p.own.data(),
                GRAN,
                WGAP,
                ng,
                empty_lo,
                maxp,
                lo.data(),
                ln.data(),
                l1.data(),
                p.npc.data(),
                p.own_l1.data());
            if (r >= 0) {
                break;
            }
            if (a == 1) {
                throw std::runtime_error("a core needs more than 4096 window pieces");
            }
        }
        if (!defer) {
            p.exchange(D, maxp, lo.data(), ln.data(), l1.data());
        } else {
            p.exch_at.assign((size_t)p.nc * p.NB + 1, 0);
            p.nat.assign(p.nc + 1, 0);
        }
        p.compact(maxp, lo, ln, l1);
        if (defer || p.max_neighbours() <= MAX_NB || p.nc == 1) {
            return p;
        }
        nreq = std::max(1, (int)(p.nc * 0.75));  // from the cores used: nreq may be above the block-size clamp
    }
}

Plan Plan::transfer(int kind, const Level& F, const Level& C, int ncores) {
    Plan p;
    p.kind = kind;
    p.F = &F;
    p.C = &C;
    p.nslot = 1;
    const int64_t nb0 = F.starts[1];
    const auto cblocks = colour_blocks(C);
    std::vector<std::pair<int64_t, int64_t>> src;
    std::vector<std::vector<int64_t>> fb;
    const std::vector<std::vector<int64_t>>* bands = nullptr;
    if (kind == 1) {  // restriction: out = the coarse level's blocks, in = fine block 0
        p.out_blocks = cblocks;
        src = {{0, nb0}};
        p.pout = C.P;
        bands = C.bands > 1 ? &C.bstarts : nullptr;
    } else {  // prolongation: out = fine block 0, in = the coarse level
        p.out_blocks = {{0, nb0}};
        src = cblocks;
        p.pout = nb0;
        if (F.bands > 1) {
            fb = {F.bstarts[0]};
            bands = &fb;
        }
    }
    p.NB = (int)p.out_blocks.size();
    p.nc = clamp_cores(ncores, p.out_blocks);
    p.ownership(usable_bands(bands, p.NB, p.nc));
    int64_t ng = 0;
    for (const auto& [s, e] : src) {
        ng = std::max(ng, e);
    }
    ng = ng / GRAN + 2;
    const int64_t empty_lo = src[0].first;
    std::vector<int64_t> lo, ln, l1;
    int64_t maxp = 0;
    for (int a = 0; a < 2; ++a) {
        maxp = MAX_PIECES[a];
        lo.assign(p.nc * maxp, 0);
        ln.assign(p.nc * maxp, 0);
        l1.assign(p.nc * maxp, 0);
        p.npc.assign(p.nc, 0);
        const int64_t r = hpcg_plan_windows_transfer(
            kind,
            F.nx,
            F.nb,
            F.nl,
            F.line_band.data(),
            F.band_y0.data(),
            F.band_ny.data(),
            F.band_start.data(),
            C.nx,
            C.nb,
            C.nl,
            C.line_band.data(),
            C.band_y0.data(),
            C.band_ny.data(),
            C.band_start.data(),
            p.nc,
            p.NB,
            p.own.data(),
            GRAN,
            WGAP,
            ng,
            empty_lo,
            maxp,
            lo.data(),
            ln.data(),
            l1.data(),
            p.npc.data());
        if (r >= 0) {
            break;
        }
        if (a == 1) {
            throw std::runtime_error("a core needs more than 4096 window pieces");
        }
    }
    p.own_l1.assign((size_t)p.nc * p.NB, 0);
    p.exch_at.assign((size_t)p.nc * p.NB + 1, 0);
    p.nat.assign(p.nc + 1, 0);
    p.compact(maxp, lo, ln, l1);
    return p;
}

void Plan::segment_counts(
    const std::vector<int64_t>& phases,
    int64_t tpw,
    std::vector<int64_t>& nseg,
    std::vector<int64_t>& nseg_gs,
    std::vector<int64_t>& nseg_r) const {
    nseg.assign(nc, 0);
    nseg_gs.assign(nc, 0);
    nseg_r.assign(nc, 0);
    for (int k = 0; k < nc; ++k) {
        auto segs = [&](int b) {
            const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
            return e > a ? (e - 1) / tpw - a / tpw + 1 : 0;  // segments never cross an L1 page of tpw words
        };
        for (int64_t ph : phases) {
            nseg[k] += segs((int)(ph & 7));
            if (ph < 8) {
                nseg_gs[k] += segs((int)ph);
            }
            if (ph == 8) {
                nseg_r[k] += segs(0);
            }
        }
    }
}

const Meta& Plan::meta(const Dev& D, const std::vector<int64_t>& phases, int64_t tpw) {
    const auto key = std::make_pair(phases, tpw);
    auto it = metas.find(key);
    if (it != metas.end()) {
        return it->second;
    }
    static std::vector<int32_t> buf(1 << 18);  // reused: TT-NN copies the rows on upload
    Meta m;
    m.nseg.assign(nc, 0);
    m.nseg_gs.assign(nc, 0);
    m.nseg_r.assign(nc, 0);
    int64_t longest = 0;
    while (true) {
        longest = hpcg_plan_meta_into(
            nc,
            NB,
            (int32_t)phases.size(),
            phases.data(),
            R,
            plo.data(),
            pln.data(),
            pl1.data(),
            npc.data(),
            nat.data(),
            nbr.data(),
            D.noc_xy.data(),
            own.data(),
            own_l1.data(),
            exch_at.data(),
            exch_rows.data(),
            tpw,
            16,
            (int64_t)buf.size(),
            buf.data(),
            m.nseg.data(),
            m.nseg_gs.data(),
            m.nseg_r.data());
        if (longest >= 0) {
            break;
        }
        buf.resize(-longest);
    }
    m.w = (longest + 15) / 16 * 16;  // one DRAM page per core: its row, a multiple of 64 bytes
    m.t = D.upload_rows(buf.data(), nc, m.w);
    return metas.emplace(key, std::move(m)).first->second;
}

}  // namespace hpcg_tt
