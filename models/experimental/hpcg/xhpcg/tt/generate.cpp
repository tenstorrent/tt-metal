// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generate.hpp"

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

#include "generic_op.hpp"

#include "native.hpp"

namespace hpcg_tt {

using tt::tt_metal::KernelDescriptor;

namespace {
const std::string KERNEL = "models/experimental/hpcg/tt/custom_ops/stencil_matrix/kernels/generate.cpp";
const std::string KERNEL_TRISC = "models/experimental/hpcg/tt/custom_ops/stencil_matrix/kernels/generate_trisc.cpp";
// DM_SLOTS, MB_BYTES, CHUNKS_RISC1_DEFAULT, XCAP, SET_PAGES, JOB_ARGS, MIN_META_PAGES: tt/shared_constants.h
const std::vector<std::pair<std::string, uint32_t>> GEN_CB = {HPCG_STENCIL_MATRIX_CBS(HPCG_CB_PAIR)};
uint32_t launch_id() {  // a random nonzero 32-bit nonce per launch: the TRISCs tell this launch's messages from
                        // words any earlier launch left in L1 (kernels/fill.h)
    static std::mt19937 gen{std::random_device{}()};
    uint32_t v = 0;
    while (v == 0) {
        v = (uint32_t)gen();
    }
    return v;
}

std::vector<uint32_t> chunk_shares(int rows) {  // a nibble per core row y, in 2 words
    std::vector<uint32_t> out = {0, 0};
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 8 && k * 8 + i < rows; ++i) {
            out[k] |= CHUNKS_RISC1_DEFAULT << (4 * i);
        }
    }
    return out;
}

std::vector<std::vector<int64_t>> level_tables(const Level& L) {  // line_band, band_y0, band_ny, band_start
    return {
        std::vector<int64_t>(L.line_band.begin(), L.line_band.end()),
        std::vector<int64_t>(L.band_y0.begin(), L.band_y0.end()),
        std::vector<int64_t>(L.band_ny.begin(), L.band_ny.end()),
        L.band_start};
}
}  // namespace

std::vector<Tensor> Job::tensors() const {
    std::vector<Tensor> t = {values, offsets};
    t.insert(t.end(), zeros.begin(), zeros.end());
    if (b) {
        t.push_back(*b);
    }
    return t;
}

std::pair<Tensor, Tensor> allocate_tables(const Dev& D, const Level& L) {
    const int64_t rows = 27 * (L.stride / TILE);
    return {D.alloc_rows(rows, TILE), D.alloc_rows(rows, TILE)};
}

Job stencil_job(
    const Dev& D,
    const Level& L,
    Plan& plan,
    const std::optional<Tensor>& b,
    const std::vector<Tensor>& zeros,
    const std::optional<std::pair<Tensor, Tensor>>& tables) {
    if (zeros.size() > 2) {
        throw std::invalid_argument("at most two zero vectors");
    }
    Job j;
    j.kind = 0;
    j.plan = &plan;
    j.ntlv = L.stride / TILE;
    const auto t = tables ? *tables : allocate_tables(D, L);
    j.values = t.first;
    j.offsets = t.second;
    j.b = b;
    j.zeros = zeros;
    j.levels = {&L};
    j.hw = 36;
    j.head[0] = {L.nx, L.nb, L.nl, j.ntlv};
    std::vector<int64_t> ends;
    for (int c = 0; c < 8; ++c) {
        ends.push_back(L.starts[c] + L.count(c));
    }
    j.head[22] = ends;
    return j;
}

Job transfer_job(const Dev& D, Plan& plan) {
    Job j;
    j.kind = plan.kind;
    j.plan = &plan;
    j.ntlv = plan.pout / TILE;
    j.values = D.alloc_rows(j.ntlv, TILE);
    j.offsets = D.alloc_rows(j.ntlv, TILE);
    const Level& F = *plan.F;
    const Level& C = *plan.C;
    const Level& O = j.kind == 1 ? C : F;  // the level whose rows are the out rows
    std::vector<int64_t> pad(plan.NB, 0);
    hpcg_transfer_pad(
        j.kind,
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
        plan.NB,
        pad.data());
    j.levels = {&F, &C};
    j.hw = 51;
    j.head[0] = {F.nx, F.nb, F.nl, j.ntlv};
    std::vector<int64_t> ends;
    for (int c = 0; c < (j.kind == 1 ? 8 : 1); ++c) {
        ends.push_back(O.starts[c] + O.count(c));
    }
    j.head[22] = ends;
    j.head[36] = {C.nx, C.nb, C.nl};
    for (auto& v : pad) {
        v = std::max<int64_t>(v, 0);
    }
    j.head[43] = pad;
    return j;
}

GenOp gen_op(const Dev& D, const std::vector<Job>& jobs, int ncores, int job_slots) {
    int nc = ncores;
    if (nc <= 0) {
        nc = 1;
        for (const Job& j : jobs) {
            nc = std::max(nc, j.plan->nc);
        }
    }
    // META: a shared region (each level's storage-order tables, once), then one row per core (the jobs' segments one
    // after another, 16-word aligned: header, then the core's window piece lo and l1 tables), all whole 4 KB pages
    std::map<const Level*, std::vector<int64_t>> toff;
    std::vector<int32_t> shared;
    int64_t sw = 0;
    for (const Job& j : jobs) {
        for (const Level* L : j.levels) {
            if (toff.count(L)) {
                continue;
            }
            const auto tabs = level_tables(*L);
            std::vector<int64_t> o = {sw};
            for (size_t i = 0; i + 1 < tabs.size(); ++i) {
                o.push_back(o.back() + (int64_t)tabs[i].size());
            }
            toff[L] = o;
            for (const auto& t : tabs) {
                for (int64_t v : t) {
                    shared.push_back((int32_t)v);
                }
                sw += (int64_t)t.size();
            }
        }
    }
    const int64_t S = jobs.empty() ? TILE : (sw + TILE - 1) / TILE * TILE;  // a tensor holds at least one page
    std::vector<int64_t> at;
    int64_t w = 0;
    for (const Job& j : jobs) {
        at.push_back(S + w);  // the segment's first word in L1
        w += (j.hw + 2 * j.plan->R + 15) / 16 * 16;
    }
    w = (w + TILE - 1) / TILE * TILE;
    std::vector<int32_t> meta(S + (int64_t)nc * w, 0);
    std::copy(shared.begin(), shared.end(), meta.begin());
    for (size_t ji = 0; ji < jobs.size(); ++ji) {
        const Job& j = jobs[ji];
        const Plan& p = *j.plan;
        const int64_t a = at[ji], R = p.R;
        for (int k = 0; k < p.nc; ++k) {
            int32_t* seg = meta.data() + S + (int64_t)k * w + (a - S);
            for (const auto& [o, v] : j.head) {
                for (size_t i = 0; i < v.size(); ++i) {
                    seg[o + i] = (int32_t)v[i];
                }
            }
            seg[4] = (int32_t)p.npc[k];
            for (int i = 0; i < p.NB * 2; ++i) {
                seg[6 + i] = (int32_t)p.own[(int64_t)k * p.NB * 2 + i];
            }
            const auto& t0 = toff[j.levels[0]];
            for (int i = 0; i < 4; ++i) {
                seg[30 + i] = (int32_t)t0[i];
            }
            seg[34] = (int32_t)(a + j.hw);
            seg[35] = (int32_t)(a + j.hw + R);
            if (j.levels.size() > 1) {
                const auto& t1 = toff[j.levels[1]];
                for (int i = 0; i < 4; ++i) {
                    seg[39 + i] = (int32_t)t1[i];
                }
            }
            for (int64_t i = 0; i < R; ++i) {  // zero past a core's own count
                seg[j.hw + i] = (int32_t)p.plo[k * R + i];
                seg[j.hw + R + i] = (int32_t)p.pl1[k * R + i];
            }
        }
    }
    GenOp g;
    g.meta = D.upload_rows(meta.data(), (int64_t)meta.size() / TILE, TILE);
    const int64_t shared_pages = S / TILE, meta_pages = w / TILE;
    g.cores = core_ranges(nc, D.gx);
    for (const auto& [k, v] : GEN_CB) {
        g.defs.emplace_back(k, std::to_string(v));
    }
    g.defs.emplace_back("DMSLOTS", std::to_string(DM_SLOTS));
    // one accessor serves every table and vector (interleaved DRAM, the same arguments for both layouts used)
    g.ct = accessor_args(jobs.empty() ? g.meta : jobs[0].values);
    const auto mct = accessor_args(g.meta);
    g.ct.insert(g.ct.end(), mct.begin(), mct.end());
    g.args = {addr(g.meta), (uint32_t)(jobs.empty() ? 0 : shared_pages), (uint32_t)meta_pages, (uint32_t)jobs.size()};
    for (size_t ji = 0; ji < jobs.size(); ++ji) {
        const Job& j = jobs[ji];
        const Tensor& bt = j.b ? *j.b : j.values;  // unused slots point at the values (never written)
        std::vector<Tensor> zt = j.zeros;
        while (zt.size() < 2) {
            zt.push_back(j.values);
        }
        g.args.insert(
            g.args.end(),
            {(uint32_t)j.kind,
             (uint32_t)j.plan->nc,
             (uint32_t)at[ji],
             addr(j.values),
             addr(j.offsets),
             addr(bt),
             (uint32_t)(j.b ? 1 : 0),
             (uint32_t)j.zeros.size(),
             addr(zt[0]),
             addr(zt[1])});
    }
    const int slots = job_slots < 0 ? (int)jobs.size() : job_slots;
    g.args.resize(g.args.size() + JOB_ARGS * (slots - (int)jobs.size()), 0);  // the kernel reads len(jobs) slots
    auto add = [&](uint32_t idx, uint32_t nbytes, uint32_t page) {
        g.cbs.push_back(cb(g.cores, (uint8_t)idx, nbytes, page, false));
    };
    add(0, 2 * SET_PAGES * 4096, 4096);
    add(1, 2 * SET_PAGES * 4096, 4096);
    // at least MIN_META_PAGES, so launches with small tables (the fine level's, the warm-up) share a program
    const uint32_t meta_cb = (uint32_t)std::max<int64_t>(shared_pages + meta_pages, MIN_META_PAGES) * 4096;
    add(2, meta_cb, 4096);
    add(3, meta_cb, 4096);
    add(4, 4096, 4096);
    add(5, 2 * XCAP * 4, 2 * XCAP * 4);
    add(6, 2 * XCAP * 4, 2 * XCAP * 4);
    add(7, MB_BYTES, MB_BYTES);
    g.io = {g.meta};
    for (const Job& j : jobs) {
        for (const Tensor& t : j.tensors()) {
            g.io.push_back(t);
        }
    }
    if (jobs.empty()) {
        g.io.push_back(g.meta);  // generic_op takes at least two tensors
    }
    g.build(D, launch_id());
    return g;
}

void GenOp::build(const Dev& D, uint32_t launch) {
    prog = tt::tt_metal::ProgramDescriptor{};
    const auto shares = chunk_shares(D.gy);
    for (int half = 0; half < 2; ++half) {
        KernelDescriptor k;
        k.kernel_source = KERNEL;
        k.source_type = KernelDescriptor::SourceType::FILE_PATH;
        k.core_ranges = cores;
        k.compile_time_args = ct;
        k.defines = defs;
        k.common_runtime_args = {(uint32_t)D.gx, (uint32_t)half};
        k.common_runtime_args.insert(k.common_runtime_args.end(), args.begin(), args.begin() + 4);
        k.common_runtime_args.push_back(launch);
        k.common_runtime_args.insert(k.common_runtime_args.end(), shares.begin(), shares.end());
        k.common_runtime_args.insert(k.common_runtime_args.end(), args.begin() + 4, args.end());
        if (half == 0) {
            k.config = tt::tt_metal::ReaderConfigDescriptor{};
        } else {
            k.config = tt::tt_metal::WriterConfigDescriptor{};
        }
        prog.kernels.push_back(k);
    }
    KernelDescriptor t;
    t.kernel_source = KERNEL_TRISC;
    t.source_type = KernelDescriptor::SourceType::FILE_PATH;
    t.core_ranges = cores;
    t.defines = defs;
    t.common_runtime_args = {launch};
    t.config = tt::tt_metal::ComputeConfigDescriptor{};
    prog.kernels.push_back(t);
    prog.cbs = cbs;
}

void GenOp::operator()(const Dev& D) {
    if (called) {  // every launch needs its own id (kernels/fill.h)
        build(D, launch_id());
    }
    called = true;
    ttnn::generic_op(io, prog);
}

void warm_up(const Dev& D, int ncores) {
    GenOp g = gen_op(D, {}, ncores, 1);
    g(D);
}

}  // namespace hpcg_tt
