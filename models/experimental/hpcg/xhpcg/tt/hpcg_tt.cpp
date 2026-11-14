// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The solver state on the card (tt/hpcg.py's HPCG, device-generated path) and the entry points HPCG 3.1's
// replaceable functions call.
#include "hpcg_tt.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/operations/trace.hpp>

#include "device.hpp"
#include "generate.hpp"
#include "hpcg.hpp"
#include "level.hpp"
#include "mytimer.hpp"
#include "ops.hpp"
#include "plan.hpp"

namespace hpcg_tt {

namespace {

const std::vector<int64_t> COLOUR_ORDER = {7, 6, 5, 4, 3, 2, 1, 0};  // forward-sweep colour order

struct DLevel {
    std::unique_ptr<Level> L;
    std::unique_ptr<Plan> plan;
    std::optional<Job> job;
    Tensor values, offsets, z, Az, r;
    std::unique_ptr<Plan> rp, pp;  // grid transfers to the next coarser level
    Tensor Rt, Roff, Pt, Poff;
    int64_t tpw = TILE;
    std::vector<int64_t> pos;  // HPCG's natural row index -> storage position q
    std::optional<Op> gs, gs_pre, restrict_, prolong, spmv;
    std::vector<float> buf;  // host staging for vector copies (storage order)
};

struct DotScratch {  // a dot product of two host vectors (its own partials: the finish adds whole pages)
    Tensor x, y, part, res;
    std::optional<VecOp> dot;
    std::optional<Op> fin;
};

struct State {
    Dev* D = nullptr;
    std::vector<std::unique_ptr<DLevel>> lv;
    std::map<const SparseMatrix*, int> level_of;  // HPCG's matrix of each level
    const SparseMatrix* A0 = nullptr;             // HPCG's fine matrix
    Tensor r, b, x, p, Ap, dbg, part, one, minus_one, inf_slot, rtz, rtz_old, pAp, normr, normr0;
    std::optional<Op> spmv_cg, spmv_start;
    std::map<std::string, Op> v;
    int nparts = 1;
    bool traced = false;
    tt::tt_metal::distributed::MeshTraceId tid_start{}, tid_iter{};
    std::vector<double> b_host, diag_host;  // what the card's b and level-0 diagonal hold, in natural order
    std::map<int, DotScratch> dots;
};

std::unique_ptr<State> g_state;

// HPCG_TT_TRACE=1: wait for the card after each stage and say which stage finished (finding a hang)
void stage(const char* what) {
    static const bool on = std::getenv("HPCG_TT_TRACE") != nullptr;
    if (on && Dev::get()) {
        Dev::get()->sync();
        std::fprintf(stderr, "hpcg_tt: %s done\n", what);
    }
}

// At exit: the traces, then every device tensor, then the card. HPCG's main.cpp returns without a call into the port,
// and by the time exit handlers run the main thread's thread_local objects are gone, TT-Metal's among them (its
// GraphTracker's processor list, which every buffer release reads). So the release runs on a new thread, whose
// thread_local objects are fresh.
void cleanup() {
    std::thread([] {
        if (g_state) {
            if (g_state->traced) {
                ttnn::operations::trace::release_trace(g_state->D->d, g_state->tid_start);
                ttnn::operations::trace::release_trace(g_state->D->d, g_state->tid_iter);
            }
            g_state.reset();
        }
        if (Dev* D = Dev::get()) {
            D->close();
        }
    }).join();
}

int level_index(const SparseMatrix& A) {
    if (!g_state) {
        return -1;
    }
    auto it = g_state->level_of.find(&A);
    return it == g_state->level_of.end() ? -1 : it->second;
}

// host vector (HPCG's natural order, fp64) -> device vector (storage order, fp32; padding rows 0)
void to_device(DLevel& d, const double* v, Tensor& t) {
    const Level& L = *d.L;
    d.buf.assign(L.P, 0.0f);
#pragma omp parallel for
    for (int64_t i = 0; i < L.n; ++i) {
        d.buf[d.pos[i]] = (float)v[i];
    }
    g_state->D->write(t, d.buf.data());
}

void from_device(DLevel& d, const Tensor& t, double* v) {
    const Level& L = *d.L;
    d.buf.resize(L.P);
    g_state->D->read(t, d.buf.data());
#pragma omp parallel for
    for (int64_t i = 0; i < L.n; ++i) {
        v[i] = d.buf[d.pos[i]];
    }
}

float read_scalar(const Tensor& t) {  // word 0 of a one-page device scalar (reads are whole pages)
    static std::vector<float> page(TILE);
    g_state->D->read(t, page.data());
    return page[0];
}

void run_mg(State& S, int l) {  // z = M^-1 r on level l: tt/hpcg.py's mg
    DLevel& d = *S.lv[l];
    (*d.gs_pre)();  // z = 0, pre-smoother, then Az = A z (block 0: r - A z)
    if (l + 1 < (int)S.lv.size()) {
        (*d.restrict_)();
        run_mg(S, l + 1);
        (*d.prolong)();
        (*d.gs)();
    }
}

void run_start(State& S) {  // r = b - A x, |r0|; r.z = inf (tt/hpcg.py's start)
    (*S.spmv_start)();
    S.v.at("r_b")();
    S.v.at("rr")();
    S.v.at("nr0_fin")();
    S.v.at("rtz_inf")();
}

void run_iteration(State& S, bool precondition) {  // one CG iteration (tt/hpcg.py's iteration)
    if (precondition) {
        run_mg(S, 0);
    } else {
        ttnn::copy(S.r, S.lv[0]->z);  // z = r (TestCG's unpreconditioned runs only)
    }
    S.v.at("rz")();
    S.v.at("rz_fin")();
    S.v.at("p")();
    (*S.spmv_cg)();
    S.v.at("pAp")();
    S.v.at("pAp_fin")();
    S.v.at("xr")();
    S.v.at("nr_fin")();
}

void capture(State& S) {
    // the first launches build every program (TT-NN's program cache); the traces then replay the same launches
    S.v.at("x0")();
    stage("x0");
    if (std::getenv("HPCG_TT_TRACE")) {  // the CG start launch by launch
        (*S.spmv_start)();
        stage("spmv_start");
        for (const char* k : {"r_b", "rr", "nr0_fin", "rtz_inf"}) {
            S.v.at(k)();
            stage(k);
        }
    } else {
        run_start(S);
    }
    stage("CG start");
    if (std::getenv("HPCG_TT_TRACE")) {  // the iteration launch by launch
        DLevel& d0 = *S.lv[0];
        (*d0.gs_pre)();
        stage("gs_pre 0");
        for (int l = 0; l + 1 < (int)S.lv.size(); ++l) {
            (*S.lv[l]->restrict_)();
            stage("restrict");
            (*S.lv[l + 1]->gs_pre)();
            stage("gs_pre coarse");
        }
        for (int l = (int)S.lv.size() - 2; l >= 0; --l) {
            (*S.lv[l]->prolong)();
            stage("prolong");
            (*S.lv[l]->gs)();
            stage("gs post");
        }
        for (const char* k : {"rz", "rz_fin", "p"}) {
            S.v.at(k)();
            stage(k);
        }
        (*S.spmv_cg)();
        stage("spmv_cg");
        for (const char* k : {"pAp", "pAp_fin", "xr", "nr_fin"}) {
            S.v.at(k)();
            stage(k);
        }
    } else {
        run_iteration(S, true);
    }
    S.D->sync();
    S.tid_start = ttnn::operations::trace::begin_trace_capture(S.D->d, std::nullopt);
    run_start(S);
    ttnn::operations::trace::end_trace_capture(S.D->d, S.tid_start, std::nullopt);
    S.tid_iter = ttnn::operations::trace::begin_trace_capture(S.D->d, std::nullopt);
    run_iteration(S, true);
    ttnn::operations::trace::end_trace_capture(S.D->d, S.tid_iter, std::nullopt);
    S.traced = true;
}

// the card's level-0 diagonal follows the host's (TestCG scales it through ReplaceMatrixDiagonal, then restores it)
void sync_diagonal(State& S, const SparseMatrix& A) {
    DLevel& d = *S.lv[0];
    const int64_t n = d.L->n;
    int dchanged = 0;
#pragma omp parallel for reduction(| : dchanged)
    for (int64_t i = 0; i < n; ++i) {
        dchanged |= *A.matrixDiagonal[i] != S.diag_host[i];
    }
    if (!dchanged) {
        return;
    }
    const Level& L = *d.L;
    std::vector<float> row(L.P, 1.0f);  // slot 13 (the diagonal); padding rows keep 1, so their update leaves 0
#pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        S.diag_host[i] = *A.matrixDiagonal[i];
        row[d.pos[i]] = (float)S.diag_host[i];
    }
    S.D->write(d.values, row.data(), 13 * L.stride * 4, L.P * 4);
}

// b on the card follows the host's b; when b changed, the diagonal is checked too (TestCG changes both together)
void sync_b_and_diagonal(State& S, const SparseMatrix& A, const Vector& b) {
    DLevel& d = *S.lv[0];
    const int64_t n = d.L->n;
    int changed = 0;
#pragma omp parallel for reduction(| : changed)
    for (int64_t i = 0; i < n; ++i) {
        changed |= b.values[i] != S.b_host[i];
    }
    if (!changed) {
        return;
    }
    std::copy(b.values, b.values + n, S.b_host.begin());
    to_device(d, b.values, S.b);
    sync_diagonal(S, A);
}

void set_x(State& S, const Vector& x) {
    DLevel& d = *S.lv[0];
    int nonzero = 0;
#pragma omp parallel for reduction(| : nonzero)
    for (int64_t i = 0; i < d.L->n; ++i) {
        nonzero |= x.values[i] != 0.0;
    }
    // x = 0 and p = 0 first, whatever x is: iteration 1 needs p = 0 (ComputeSPMV borrows p for its input), and zeros
    // are written, so a NaN or inf left by a diverged solve does not carry over
    S.v.at("x0")();
    if (nonzero) {
        to_device(d, x.values, S.x);
    }
}

std::vector<int64_t> natural_to_q(const Level& L) {
    std::vector<int64_t> pos(L.n);
    const int32_t nx = L.nx;
#pragma omp parallel for
    for (int64_t i = 0; i < L.n; ++i) {
        const int32_t ix = (int32_t)(i % nx), iy = (int32_t)((i / nx) % nx), iz = (int32_t)(i / ((int64_t)nx * nx));
        pos[i] = L.q_of(ix, iy, iz);
    }
    return pos;
}

}  // namespace

int optimize(const SparseMatrix& A, const Vector& b) {
    const Geometry& g = *A.geom;
    if (g.size != 1 || g.nx != g.ny || g.nx != g.nz || g.nx % 8 != 0) {
        HPCG_fout << "hpcg_tt: needs one process and an nx = ny = nz grid with nx a multiple of 8" << std::endl;
        std::fprintf(stderr, "hpcg_tt: needs one process and an nx = ny = nz grid with nx a multiple of 8\n");
        // HPCG's main.cpp ignores OptimizeProblem's return value: going on, every CG call would return at once with no
        // time, and main.cpp's number of CG sets (total_runtime / worst CG time) would divide by 0
        std::exit(1);
    }
    g_state = std::make_unique<State>();
    State& S = *g_state;
    Dev& D = Dev::open();
    S.D = &D;
    const int nx = g.nx, nc = D.nc;
    int bands[NLEV];
    for (int l = 0; l < NLEV; ++l) {
        bands[l] = default_bands(nx >> l);
    }
    // generation: a launch with no jobs that loads the kernel binaries and builds the fine level's program, then the
    // fine level's launch, then the coarser levels' and the grid transfers' (tt/hpcg.py's order)
    warm_up(D, nc);
    stage("warm-up");
    auto d0 = std::make_unique<DLevel>();
    d0->L = std::make_unique<Level>(nx, bands[0]);
    const Level& L0 = *d0->L;
    const auto tables0 = allocate_tables(D, L0);
    S.r = D.alloc_tile(L0.P);
    S.b = D.alloc_tile(L0.P);
    S.x = D.alloc_tile(L0.P);
    S.p = D.alloc_tile(L0.P);
    S.Ap = D.alloc_tile(L0.P);
    d0->plan = std::make_unique<Plan>(Plan::stencil(L0, nc, D, true));
    d0->job = stencil_job(D, L0, *d0->plan, S.b, {S.x, S.p}, tables0);  // x = 0, p = 0 on the fine level
    d0->values = d0->job->values;
    d0->offsets = d0->job->offsets;
    gen_op(D, {*d0->job})(D);
    stage("fine-level generation");
    if (!d0->plan->complete_exchange(D)) {  // too many neighbours: re-plan (fewer cores) and generate again
        d0->plan = std::make_unique<Plan>(Plan::stencil(L0, nc, D, false));
        d0->job = stencil_job(D, L0, *d0->plan, S.b, {S.x, S.p}, std::make_pair(d0->values, d0->offsets));
        gen_op(D, {*d0->job})(D);
    }
    d0->z = D.alloc_tile(L0.P);
    d0->Az = D.alloc_tile(L0.P);
    d0->r = S.r;  // the fine level shares CG's r
    S.lv.push_back(std::move(d0));
    std::vector<Job> jobs;
    for (int l = 1; l < NLEV; ++l) {
        auto d = std::make_unique<DLevel>();
        d->L = std::make_unique<Level>(nx >> l, bands[l]);
        d->plan = std::make_unique<Plan>(Plan::stencil(*d->L, nc, D, false));
        d->job = stencil_job(D, *d->L, *d->plan, std::nullopt, {}, std::nullopt);
        d->values = d->job->values;
        d->offsets = d->job->offsets;
        d->z = D.alloc_tile(d->L->P);
        d->Az = D.alloc_tile(d->L->P);
        d->r = D.alloc_tile(d->L->P);
        jobs.push_back(*d->job);
        S.lv.push_back(std::move(d));
    }
    for (int l = 0; l + 1 < NLEV; ++l) {  // grid transfers l -> l + 1, planned from both levels' formulas
        DLevel& d = *S.lv[l];
        d.rp = std::make_unique<Plan>(Plan::transfer(1, *d.L, *S.lv[l + 1]->L, nc));
        d.pp = std::make_unique<Plan>(Plan::transfer(2, *d.L, *S.lv[l + 1]->L, nc));
        const Job jr = transfer_job(D, *d.rp), jp = transfer_job(D, *d.pp);
        d.Rt = jr.values;
        d.Roff = jr.offsets;
        d.Pt = jp.values;
        d.Poff = jp.offsets;
        jobs.push_back(jr);
        jobs.push_back(jp);
    }
    gen_op(D, jobs)(D);
    stage("coarse-level and transfer generation");
    S.dbg = D.alloc_rows(2 * nc, 16);  // per-core cycle counters (written only by KDEBUG builds)

    std::vector<int64_t> phases = COLOUR_ORDER;  // forward then backward sweep
    phases.insert(phases.end(), COLOUR_ORDER.rbegin(), COLOUR_ORDER.rend());
    for (int l = 0; l < NLEV; ++l) {
        DLevel& d = *S.lv[l];
        const Level& L = *d.L;
        // L1 page: one 16x16 face when each core owns <= 256 rows of a colour
        d.tpw = (double)(L.starts[1] - L.starts[0]) / d.plan->nc <= 256 ? 256 : TILE;
        std::vector<int64_t> pre = phases;
        if (l < NLEV - 1) {
            for (int b = 0; b < 8; ++b) {
                pre.push_back(8 + b);
            }
        }
        SparseRowArgs post;  // post-smoother: the first 16 phases of the pre-smoother's table
        post.gs = true;
        post.phases = phases;
        post.b = d.r;
        post.tpw = d.tpw;
        post.wsplit = l == 0;
        post.table_phases = pre;
        d.gs = sparse_row_op(D, *d.plan, d.values, d.offsets, d.z, d.z, post, S.dbg);
        SparseRowArgs prea;  // pre-smoother: z = 0 on entry, then Az = A z with block 0 = r - A z
        prea.gs = true;
        prea.phases = pre;
        prea.b = d.r;
        if (l < NLEV - 1) {
            prea.out2 = d.Az;
        }
        prea.zerox = true;
        prea.restf = true;
        prea.tpw = d.tpw;
        d.gs_pre = sparse_row_op(D, *d.plan, d.values, d.offsets, d.z, d.z, prea, S.dbg);
        if (l + 1 < NLEV) {
            DLevel& dc = *S.lv[l + 1];
            SparseRowArgs ra;
            ra.gs = false;
            ra.phases = {0, 1, 2, 3, 4, 5, 6, 7};
            d.restrict_ = sparse_row_op(D, *d.rp, d.Rt, d.Roff, d.Az, dc.r, ra, S.dbg);
            SparseRowArgs pa;  // prolongation: gather and add in one launch
            pa.gs = false;
            pa.phases = {0};
            pa.b = d.z;
            pa.addb = true;
            d.prolong = sparse_row_op(D, *d.pp, d.Pt, d.Poff, dc.z, d.z, pa, S.dbg);
        }
    }
    DLevel& d0r = *S.lv[0];
    SparseRowArgs sa;  // CG SpMV: an SpMV-only gs launch on the smoother plan
    sa.gs = true;
    sa.phases = {8, 9, 10, 11, 12, 13, 14, 15};
    sa.b = S.p;
    sa.out2 = S.Ap;
    S.spmv_cg = sparse_row_op(D, *d0r.plan, d0r.values, d0r.offsets, S.p, S.p, sa, S.dbg);
    SparseRowArgs st = sa;  // CG start: A x written into r
    st.b = S.x;
    st.out2 = S.r;
    S.spmv_start = sparse_row_op(D, *d0r.plan, d0r.values, d0r.offsets, S.x, S.x, st, S.dbg);

    const int64_t ncv = std::min<int64_t>(nc, L0.P / TILE);
    {
        std::vector<float> zeros(ncv * TILE, 0.0f);
        S.part = D.upload_rows(zeros.data(), ncv, TILE);  // per-core dot partials, zero beyond the slots
        std::vector<float> page(TILE, 1.0f);
        S.one = D.upload_rows(page.data(), 1, TILE);
        std::fill(page.begin(), page.end(), -1.0f);
        S.minus_one = D.upload_rows(page.data(), 1, TILE);
        std::fill(page.begin(), page.end(), 0.0f);
        page[0] = std::numeric_limits<float>::infinity();
        S.inf_slot = D.upload_rows(page.data(), 1, TILE);
    }
    S.rtz = D.alloc_rows(1, TILE);
    S.rtz_old = D.alloc_rows(1, TILE);
    S.pAp = D.alloc_rows(1, TILE);
    S.normr = D.alloc_rows(1, TILE);
    S.normr0 = D.alloc_rows(1, TILE);
    const Tensor& z = S.lv[0]->z;
    const int64_t P0 = L0.P;
    VecOp rz = cg_vector_op(D, VecMode::dot, S.r, z, nullptr, nullptr, nullptr, nullptr, &S.part, P0);
    S.nparts = (int)((rz.nc * 64 + 4 * TILE - 1) / (4 * TILE));  // pages holding one 64-byte slot per core
    const int n = S.nparts;
    S.v["rr"] = cg_vector_op(D, VecMode::dot, S.r, S.r, nullptr, nullptr, nullptr, nullptr, &S.part, P0);
    S.v["nr0_fin"] = dot_finish_op(D, S.part, n, S.normr0, nullptr, false, true);
    S.v["rz"] = rz;
    S.v["rz_fin"] = dot_finish_op(D, S.part, n, S.rtz, &S.rtz_old, true, false);
    S.v["pAp"] = cg_vector_op(D, VecMode::dot, S.p, S.Ap, nullptr, nullptr, nullptr, nullptr, &S.part, P0);
    S.v["pAp_fin"] = dot_finish_op(D, S.part, n, S.pAp, nullptr, false, false);
    S.v["p"] = cg_vector_op(D, VecMode::axpy, z, S.p, nullptr, nullptr, &S.rtz, &S.rtz_old, nullptr, P0);
    S.v["xr"] = cg_vector_op(D, VecMode::xr, S.x, S.p, &S.r, &S.Ap, &S.rtz, &S.pAp, &S.part, P0);
    S.v["nr_fin"] = dot_finish_op(D, S.part, n, S.normr, nullptr, false, true);
    // reset and CG start with the same binaries: x = 0 and p = 0 (written, not computed), r = b - A x; r.z = inf so
    // iteration 1's p is z
    S.v["x0"] = cg_vector_op(D, VecMode::zero, S.x, S.p, nullptr, nullptr, nullptr, nullptr, nullptr, P0);
    S.v["r_b"] = cg_vector_op(D, VecMode::axpy, S.b, S.r, nullptr, nullptr, &S.minus_one, &S.one, nullptr, P0);
    S.v["rtz_inf"] = dot_finish_op(D, S.inf_slot, 1, S.rtz, nullptr, false, false);

    // HPCG's matrices of the levels, and each level's natural -> storage map
    S.A0 = &A;
    const SparseMatrix* Am = &A;
    for (int l = 0; l < NLEV && Am != nullptr; ++l, Am = Am->Ac) {
        S.level_of[Am] = l;
        S.lv[l]->pos = natural_to_q(*S.lv[l]->L);
    }
    // what the card holds: the generated b and diagonal equal HPCG's (CheckProblem checked HPCG's on the host)
    S.b_host.assign(b.values, b.values + L0.n);
    S.diag_host.assign(L0.n, 0.0);
#pragma omp parallel for
    for (int64_t i = 0; i < L0.n; ++i) {
        S.diag_host[i] = *A.matrixDiagonal[i];
    }
    D.sync();
    std::atexit(cleanup);
    return 0;
}

int cg(
    const SparseMatrix& A,
    const Vector& b,
    Vector& x,
    int max_iter,
    double tolerance,
    int& niters,
    double& normr,
    double& normr0,
    double* times,
    bool doPreconditioning) {
    const double t_begin = mytimer();
    if (level_index(A) != 0) {
        return 1;
    }
    State& S = *g_state;
    if (!S.traced) {
        capture(S);
    }
    sync_b_and_diagonal(S, A, b);
    set_x(S, x);
    ttnn::operations::trace::execute_trace(S.D->d, S.tid_start, std::nullopt, false);
    niters = 0;
    if (tolerance <= 0.0 && doPreconditioning) {  // HPCG's timed sets: every iteration, nothing read back
        for (int k = 1; k <= max_iter; ++k) {
            ttnn::operations::trace::execute_trace(S.D->d, S.tid_iter, std::nullopt, false);
        }
        niters = max_iter;
        normr0 = read_scalar(S.normr0);
        normr = read_scalar(S.normr);
    } else {  // a residual target (or no preconditioner): |r| read after each iteration, as CG.cpp checks it
        normr0 = read_scalar(S.normr0);
        normr = normr0;
        for (int k = 1; k <= max_iter && normr / normr0 > tolerance; ++k) {
            if (doPreconditioning) {
                ttnn::operations::trace::execute_trace(S.D->d, S.tid_iter, std::nullopt, false);
            } else {
                run_iteration(S, false);
            }
            normr = read_scalar(S.normr);
            niters = k;
        }
    }
    from_device(*S.lv[0], S.x, x.values);
    times[0] += mytimer() - t_begin;
    return 0;
}

int spmv(const SparseMatrix& A, const Vector& x, Vector& y) {
    const int l = level_index(A);
    if (l < 0) {
        return 1;
    }
    State& S = *g_state;
    sync_diagonal(S, *S.A0);  // TestCG may have changed and restored it since the last call
    DLevel& d = *S.lv[l];
    if (l == 0) {  // Ap = A p, the CG SpMV
        to_device(d, x.values, S.p);
        (*S.spmv_cg)();
        from_device(d, S.Ap, y.values);
        return 0;
    }
    if (!d.spmv) {  // a plain z -> Az launch of a coarse level (built on first use)
        SparseRowArgs a;
        a.gs = false;
        a.phases = {0, 1, 2, 3, 4, 5, 6, 7};
        a.tpw = d.tpw;
        d.spmv = sparse_row_op(*S.D, *d.plan, d.values, d.offsets, d.z, d.Az, a, S.dbg);
    }
    to_device(d, x.values, d.z);
    (*d.spmv)();
    from_device(d, d.Az, y.values);
    return 0;
}

int mg(const SparseMatrix& A, const Vector& r, Vector& x) {
    const int l = level_index(A);
    if (l < 0) {
        return 1;
    }
    State& S = *g_state;
    sync_diagonal(S, *S.A0);  // TestCG may have changed and restored it since the last call
    DLevel& d = *S.lv[l];
    to_device(d, r.values, d.r);
    run_mg(S, l);
    from_device(d, d.z, x.values);
    return 0;
}

int symgs(const SparseMatrix& A, const Vector& r, Vector& x) {
    const int l = level_index(A);
    if (l < 0) {
        return 1;
    }
    State& S = *g_state;
    sync_diagonal(S, *S.A0);  // TestCG may have changed and restored it since the last call
    DLevel& d = *S.lv[l];
    to_device(d, r.values, d.r);
    to_device(d, x.values, d.z);
    (*d.gs)();  // forward and backward sweep, x updated in place
    from_device(d, d.z, x.values);
    return 0;
}

double device_bytes() { return Dev::get() ? Dev::get()->bytes : 0.0; }

bool dot(local_int_t n, const Vector& x, const Vector& y, double& result) {
    if (!g_state) {
        return false;
    }
    State& S = *g_state;
    int l = -1;
    for (int i = 0; i < (int)S.lv.size(); ++i) {
        if (S.lv[i]->L->n == n) {
            l = i;
        }
    }
    if (l < 0) {
        return false;
    }
    DLevel& d = *S.lv[l];
    DotScratch& s = S.dots[l];
    if (!s.dot) {
        s.x = S.D->alloc_tile(d.L->P);
        s.y = S.D->alloc_tile(d.L->P);
        s.res = S.D->alloc_rows(1, TILE);
        const int64_t ncv = std::min<int64_t>(S.D->nc, d.L->P / TILE);
        std::vector<float> zeros(ncv * TILE, 0.0f);
        s.part = S.D->upload_rows(zeros.data(), ncv, TILE);
        s.dot = cg_vector_op(*S.D, VecMode::dot, s.x, s.y, nullptr, nullptr, nullptr, nullptr, &s.part, d.L->P);
        const int parts = (int)((s.dot->nc * 64 + 4 * TILE - 1) / (4 * TILE));
        s.fin = dot_finish_op(*S.D, s.part, parts, s.res, nullptr, false, false);
    }
    to_device(d, x.values, s.x);
    to_device(d, y.values, s.y);
    (*s.dot)();
    (*s.fin)();
    result = read_scalar(s.res);
    if (std::getenv("HPCG_TT_TRACE")) {
        std::fprintf(stderr, "hpcg_tt: dot n=%d -> %.9g\n", (int)n, result);
    }
    return true;
}

}  // namespace hpcg_tt
