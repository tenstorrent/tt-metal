// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_solver_sat_solver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <unistd.h>
#include <string>
#include <string_view>

#include <cadical.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "fabric_host_utils.hpp"  // humanize()

namespace tt::tt_fabric::detail {

namespace {

// Local env helper (topology_sat_env_long lives in topology_solver_sat.cpp; keep this facade self-contained).
inline long solver_env_long(const char* name, long def) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return def;
    }
    char* end = nullptr;
    const long r = std::strtol(v, &end, 10);
    return (end == v) ? def : r;
}

// Formula-family gate for the distilled pool: only the big inter-mesh solve (same threshold as the DIMACS dump
// hooks in topology_solver_sat.cpp) participates. The tiny intra-mesh per-stage solves are DIFFERENT formulas in
// the same process; pooling their clauses into the inter-mesh dump would be unsound.
constexpr int kDistillMinVars = 5000;

// CaDiCaL calls terminate() frequently during solve(); we use it purely as a heartbeat (always returns false, so it
// never actually stops the search). Every 15s it logs a short, plain-English INFO line (which solution we're after +
// a live "dead ends ruled out per second" rate, so a long silent solve is clearly alive and making progress), plus a
// DEBUG line with the full SAT stat breakdown for experts.
//
// "% of the search space searched" has no honest literal value for a SAT solver: the space is 2^vars and CDCL does not
// traverse it uniformly. The meaningful proxy is how many variables are PERMANENTLY pinned -- fixed (proven to a value)
// + eliminated (removed by preprocessing) -- out of total vars. That ratio is bounded [0,100], monotonic, and directly
// reflects how much of the free space has collapsed. During the long first solve, that climbing pinned% (plus elapsed)
// is the "progress to first solution" signal; once a solution is found the enumeration loop's own per-solution log
// continues the count. The clock is only sampled every 4096 calls to keep the hot search loop cheap.
class HeartbeatTerminator : public CaDiCaL::Terminator {
public:
    void bind(CaDiCaL::Solver* solver) { solver_ = solver; }

    void reset() {
        start_ = std::chrono::steady_clock::now();
        last_log_ = start_;
        calls_ = 0;
        prev_conflicts_ = 0;
    }

    // Session-level progress fed by the caller (survives reset(), which only clears the per-solve timers). `phase` is a
    // short free-text stage tag (e.g. "descent", "enumerate"); `found`/`target` drive the "progress to first solution"
    // and enumeration counters. target <= 0 means "not an enumeration" (single solve).
    void set_phase(std::string_view phase) { phase_.assign(phase); }
    void set_solution_progress(int64_t found, int64_t target) {
        sols_found_ = found;
        sols_target_ = target;
    }

    void set_cancel(std::atomic<bool>* flag) { cancel_ = flag; }

    bool terminate() override {
        // Portfolio cancellation: once another worker has won, stop this solve promptly (checked every call; a
        // relaxed atomic load is ~1ns). Kept before the sample-mask early-out so cancellation is responsive.
        if (cancel_ != nullptr && cancel_->load(std::memory_order_relaxed)) {
            return true;
        }
        if ((++calls_ & kSampleMask) != 0) {
            return false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_log_ < kInterval) {
            return false;
        }
        const auto since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_).count();
        last_log_ = now;
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_).count();

        if (solver_ == nullptr) {
            log_info(tt::LogFabric, "SAT solve {}s in progress", elapsed);
            return false;
        }

        const int64_t conflicts = solver_->get_statistic_value("conflicts");
        const int64_t decisions = solver_->get_statistic_value("decisions");
        const int64_t propagations = solver_->get_statistic_value("propagations");
        const int64_t irredundant = solver_->get_statistic_value("irredundant");
        const int64_t redundant = solver_->get_statistic_value("redundant");
        const int64_t fixed = solver_->get_statistic_value("fixed");
        const int64_t eliminated = solver_->get_statistic_value("eliminated");
        const int vars = solver_->vars();
        const int64_t d_conf = (conflicts >= 0 && conflicts >= prev_conflicts_) ? conflicts - prev_conflicts_ : 0;
        prev_conflicts_ = conflicts < 0 ? 0 : conflicts;
        const int64_t conf_rate = since_last > 0 ? (d_conf * 1000) / since_last : 0;

        // Search-space-progress proxy: fraction of variables permanently pinned (fixed + eliminated) out of all vars.
        // Bounded [0,100], monotonic; the honest stand-in for "how much of the space is searched" (see class comment).
        const int64_t pinned = (fixed < 0 ? 0 : fixed) + (eliminated < 0 ? 0 : eliminated);
        const double pinned_pct = vars > 0 ? (100.0 * static_cast<double>(pinned)) / static_cast<double>(vars) : 0.0;

        // Plain-English progress for the INFO line. Before the first solution lands it reads "searching for solution 1
        // of N"; afterwards it says how many are found and which one it is hunting next.
        std::string sols;
        if (sols_target_ > 0) {
            sols = (sols_found_ <= 0)
                       ? fmt::format("searching for solution 1 of {}", sols_target_)
                       : fmt::format("found {} of {}, searching for #{}", sols_found_, sols_target_, sols_found_ + 1);
        } else {
            sols = (sols_found_ <= 0) ? std::string("searching for a valid solution")
                                      : fmt::format("found {} so far, searching for more", sols_found_);
        }
        const std::string phase_tag = phase_.empty() ? std::string() : fmt::format(" [{}]", phase_);

        // INFO: plain language anyone can read. A "dead end" = the solver tried a combination, hit a contradiction, and
        // backed out (a conflict); the rate is the best "it's alive and working" signal, and the running total is how
        // much of the space it has ruled out. (The static % locked-down figure lives in the DEBUG line below.)
        log_info(
            tt::LogFabric,
            "Solver running {}s | {} | ruling out {} dead ends/s ({} ruled out so far)",
            elapsed,
            sols,
            humanize(conf_rate),
            humanize(conflicts));

        // DEBUG: the full stat breakdown, with each figure spelled out. space = the formula and how much of it has
        // permanently collapsed (fixed = proven to a value, eliminated = removed by preprocessing; together = pinned);
        // clauses = original (irredundant) + learned (redundant); work = raw CDCL effort spent since the solve started.
        log_debug(
            tt::LogFabric,
            "SAT solve {}s{} detail | space: {} vars ({} fixed + {} eliminated = {:.1f}% pinned), {} irredundant + {} "
            "learned clauses | work: {} conflicts (+{}/s), {} decisions, {} propagations",
            elapsed,
            phase_tag,
            humanize(vars),
            humanize(fixed),
            humanize(eliminated),
            pinned_pct,
            humanize(irredundant),
            humanize(redundant),
            humanize(conflicts),
            humanize(conf_rate),
            humanize(decisions),
            humanize(propagations));
        return false;
    }

private:
    static constexpr std::uint64_t kSampleMask = 0xFFF;   // sample the clock every 4096 checks
    static constexpr std::chrono::seconds kInterval{15};  // one heartbeat line every 15s
    CaDiCaL::Solver* solver_ = nullptr;
    std::chrono::steady_clock::time_point start_{};
    std::chrono::steady_clock::time_point last_log_{};
    std::uint64_t calls_ = 0;
    int64_t prev_conflicts_ = 0;
    std::string phase_;
    int64_t sols_found_ = 0;
    int64_t sols_target_ = 0;
    std::atomic<bool>* cancel_ = nullptr;
};

// Clause-sharing export adapter. CaDiCaL invokes learning(size) as each clause is learned; if it returns true, the
// clause's literals stream via learn(lit) terminated by learn(0). We keep only short clauses (size <= max_size: short
// clauses are the high-value, low-volume ones portfolio solvers share) and publish each to the shared pool. Pure
// export -- import is done by the portfolio driver via add() between conflict-budget windows, so the worker stays a
// plain incremental CaDiCaL (no observed-var freezing).
class ClauseExportLearner : public CaDiCaL::Learner {
public:
    // Dual-sink export: `pool` (clause-sharing portfolio, may be null) and/or the process-wide DistilledClausePool
    // (TT_TOPO_SAT_POOL=1). The distilled sink is gated on the producer being the BIG inter-mesh formula
    // (*known_vars > kDistillMinVars -- see the family-soundness note on kDistillMinVars) and stamps each clause
    // with the producer's current permanent host-cap tag (*cap_tag).
    ClauseExportLearner(
        ClauseSharingPool* pool,
        int producer_id,
        int max_size,
        bool distill,
        int distill_max,
        const int* cap_tag,
        const int* known_vars) :
        pool_(pool),
        producer_id_(producer_id),
        max_size_(max_size),
        distill_(distill),
        distill_max_(distill_max),
        cap_tag_(cap_tag),
        known_vars_(known_vars) {}

    bool learning(int size) override {
        if (size <= 0) {
            return false;
        }
        const bool share_wants = pool_ != nullptr && size <= max_size_;
        const bool distill_wants = distill_wants_size(size);
        return share_wants || distill_wants;
    }

    void learn(int lit) override {
        if (lit != 0) {
            buf_.push_back(lit);
            return;
        }
        if (!buf_.empty()) {
            const int sz = static_cast<int>(buf_.size());
            if (pool_ != nullptr && sz <= max_size_) {
                pool_->publish(producer_id_, buf_);
            }
            if (distill_wants_size(sz)) {
                DistilledClausePool::instance().publish(
                    buf_, cap_tag_ != nullptr ? *cap_tag_ : DistilledClausePool::kNoCap);
            }
            buf_.clear();
        }
    }

private:
    bool distill_wants_size(int size) const {
        return distill_ && size <= distill_max_ && known_vars_ != nullptr && *known_vars_ > kDistillMinVars;
    }

    ClauseSharingPool* pool_ = nullptr;
    int producer_id_ = 0;
    int max_size_ = 0;
    bool distill_ = false;
    int distill_max_ = 0;
    const int* cap_tag_ = nullptr;
    const int* known_vars_ = nullptr;
    std::vector<int> buf_;
};

}  // namespace

// ── DistilledClausePool ───────────────────────────────────────────────────────

DistilledClausePool& DistilledClausePool::instance() {
    static DistilledClausePool pool;
    return pool;
}

void DistilledClausePool::publish(const std::vector<int>& lits, int cap_tag) {
    if (lits.empty()) {
        return;
    }
    std::vector<int> key = lits;
    std::sort(key.begin(), key.end());
    std::lock_guard<std::mutex> lk(m_);
    const auto it = entries_.find(key);
    if (it != entries_.end()) {
        it->second = std::max(it->second, cap_tag);  // duplicate: keep the loosest (most widely valid) tag
        return;
    }
    if (entries_.size() >= kMaxStored) {
        return;  // safety valve: stop growing (dedup/tag updates above still apply)
    }
    entries_.emplace(std::move(key), cap_tag);
}

std::vector<std::vector<int>> DistilledClausePool::collect(int run_cap, std::size_t max_clauses, int max_var) const {
    std::vector<std::vector<int>> out;
    {
        std::lock_guard<std::mutex> lk(m_);
        out.reserve(entries_.size());
        for (const auto& [lits, tag] : entries_) {
            if (tag < run_cap) {
                continue;  // learned under a TIGHTER permanent cap than this run enforces -> not entailed here
            }
            bool in_range = true;
            for (const int l : lits) {
                if (l > max_var || l < -max_var) {
                    in_range = false;
                    break;
                }
            }
            if (in_range) {
                out.push_back(lits);
            }
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        return a.size() < b.size();
    });
    if (out.size() > max_clauses) {
        out.resize(max_clauses);
    }
    return out;
}

std::size_t DistilledClausePool::size() const {
    std::lock_guard<std::mutex> lk(m_);
    return entries_.size();
}

struct TopologySatSolver::Impl {
    mutable CaDiCaL::Solver solver;
    HeartbeatTerminator heartbeat;
    std::unique_ptr<ClauseExportLearner> export_learner;
    // Clause-sharing export config (portfolio) + distilled-pool export config (TT_TOPO_SAT_POOL=1). Both feed one
    // ClauseExportLearner (CaDiCaL supports a single Learner), rebuilt whenever either sink is (re)configured.
    ClauseSharingPool* share_pool = nullptr;
    int share_id = 0;
    int share_max = 0;
    bool distill = false;
    int distill_max = 8;
    // Permanent host-cap tag for pool exports/dumps (kNoCap until a cap unit clause is added). Lives in Impl (stable
    // across TopologySatSolver moves) because the learner holds a pointer to it.
    int pool_cap_tag = DistilledClausePool::kNoCap;
    // Mirror of the facade's next_var_ (Impl-stable pointer for the learner's size gate).
    int known_vars = 0;

    Impl() {
        solver.set("quiet", 1);
        heartbeat.bind(&solver);
        solver.connect_terminator(&heartbeat);
    }

    ~Impl() {
        if (export_learner) {
            solver.disconnect_learner();
        }
    }

    void reconnect_learner() {
        if (share_pool == nullptr && !distill) {
            return;
        }
        if (export_learner) {
            solver.disconnect_learner();
        }
        export_learner = std::make_unique<ClauseExportLearner>(
            share_pool, share_id, share_max, distill, distill_max, &pool_cap_tag, &known_vars);
        solver.connect_learner(export_learner.get());
    }

    void enable_clause_export(ClauseSharingPool* pool, int producer_id, int max_size) {
        if (pool == nullptr) {
            return;
        }
        share_pool = pool;
        share_id = producer_id;
        share_max = max_size;
        reconnect_learner();
    }

    // Root-level fixed units are formula-entailed (assumptions never fix at root), so they are pool-exportable under
    // the current permanent cap tag. O(vars) scan of CaDiCaL's fixed() -- cheap; dedup happens in the pool.
    void harvest_fixed_units() {
        if (!distill || known_vars <= kDistillMinVars) {
            return;
        }
        auto& dp = DistilledClausePool::instance();
        std::vector<int> unit(1);
        for (int v = 1; v <= known_vars; ++v) {
            const int f = solver.fixed(v);
            if (f == 0) {
                continue;
            }
            unit[0] = (f > 0) ? v : -v;
            dp.publish(unit, pool_cap_tag);
        }
    }

    void reserve(int max_var) {
        if (max_var > 0) {
            known_vars = std::max(known_vars, max_var);
            solver.reserve(max_var);
        }
    }

    void add(int lit) { solver.add(lit); }

    void set_cancel(std::atomic<bool>* flag) { heartbeat.set_cancel(flag); }

    void set_progress_phase(std::string_view phase) { heartbeat.set_phase(phase); }

    void set_solution_progress(int64_t found, int64_t target) { heartbeat.set_solution_progress(found, target); }

    void assume(int lit) { solver.assume(lit); }

    void phase(int lit) { solver.phase(lit); }

    void unphase(int lit) { solver.unphase(lit); }

    int solve() {
        heartbeat.reset();
        return solver.solve();
    }

    int solve_limited(int max_conflicts) {
        heartbeat.reset();
        solver.limit("conflicts", max_conflicts);
        const int r = solver.solve();
        solver.limit("conflicts", -1);  // -1 == unlimited; clear so later solve() calls are unbounded
        return r;
    }

    int val(int lit) const {
        const int a = std::abs(lit);
        const int r = solver.val(a);
        if (r == 0) {
            return 0;
        }
        if (lit > 0) {
            return (r > 0) ? lit : -lit;
        }
        return (r < 0) ? lit : -lit;
    }
};

TopologySatSolver::TopologySatSolver() : impl_(std::make_unique<Impl>()) {
    // Turn on the faithful DIMACS tee when a dump is requested OR the gimsatul hybrid is active (it exports the tape
    // to feed gimsatul). Production (neither set) pays nothing.
    const char* dp = std::getenv("TT_TOPO_SAT_DUMP_DIMACS");
    const char* gm = std::getenv("TT_TOPO_SAT_GIMSATUL");
    if ((dp != nullptr && dp[0] != '\0') || (gm != nullptr && gm[0] != '\0' && gm[0] != '0')) {
        dump_record_ = true;
    }
    // Distilled pool (TT_TOPO_SAT_POOL=1): every solver exports short learned clauses + harvested root-fixed units
    // to the process-wide pool (family-gated on >kDistillMinVars vars inside the learner/harvest, so the tiny
    // intra-mesh solves stay out). Production (unset) pays nothing.
    if (solver_env_long("TT_TOPO_SAT_POOL", 0) != 0) {
        impl_->distill = true;
        impl_->distill_max = static_cast<int>(solver_env_long("TT_TOPO_SAT_POOL_MAX_LITS", 8));
        impl_->reconnect_learner();
    }
}

void TopologySatSolver::configure_for_blocking_clause_enumeration() {
    // Only valid in CONFIGURING state (before the first non-config add()).
    // ILB: incremental lazy backtracking — reuse trail across incremental clause additions (CaDiCaL NEWS 1.7.3+).
    // Fixed at 2 ("reuse everything") on purpose, used for iterative solving without re-encoding.
    (void)impl_->solver.set("ilb", 2);
}

bool TopologySatSolver::set_option(const std::string& name, int value) { return impl_->solver.set(name.c_str(), value); }

void TopologySatSolver::set_cancel_flag(std::atomic<bool>* flag) { impl_->set_cancel(flag); }

void TopologySatSolver::enable_clause_export(ClauseSharingPool* pool, int producer_id, int max_size) {
    impl_->enable_clause_export(pool, producer_id, max_size);
}

bool TopologySatSolver::write_dimacs(const std::string& path) {
    log_info(
        tt::LogFabric, "[topo-sat] write_dimacs tee: record={} tape_clauses~={}", dump_record_, num_clauses_);
    // Faithful export from our own clause tape (CaDiCaL::write_dimacs drops post-solve incremental clauses).
    if (!dump_record_) {
        return impl_->solver.write_dimacs(path.c_str()) == nullptr;  // fallback (base encode only)
    }
    FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        return false;
    }
    std::fprintf(f, "p cnf %d %zu\n", next_var_ < 0 ? 0 : next_var_, num_clauses_);
    for (const int lit : dump_tape_) {
        if (lit == 0) {
            std::fputs("0\n", f);
        } else {
            std::fprintf(f, "%d ", lit);
        }
    }
    std::fclose(f);
    return true;
}

int TopologySatSolver::gimsatul_solve(int threads, const std::vector<int>& assumption_units, std::size_t run_cap_k) {
    have_gimsatul_model_ = false;
    const char* bin = std::getenv("TT_TOPO_SAT_GIMSATUL_BIN");
    if (bin == nullptr || bin[0] == '\0' || !dump_record_) {
        return 0;  // no binary / no tape -> unknown; caller falls back to native solve
    }
    const auto t0 = std::chrono::steady_clock::now();
    // Distilled pool injection (TT_TOPO_SAT_POOL=1): append pool entries valid under this run's host cap
    // (min(permanent cap tag, run_cap_k)), shortest-first, capped at TT_TOPO_SAT_POOL_MAX_CLAUSES. Harvest the
    // driver's root-fixed units first so anything proven since the last native solve rides along too.
    std::vector<std::vector<int>> pool_clauses;
    std::size_t pool_total = 0;
    int run_cap = impl_->pool_cap_tag;
    if (impl_->distill) {
        impl_->harvest_fixed_units();
        if (run_cap_k > 0 && run_cap_k < static_cast<std::size_t>(DistilledClausePool::kNoCap)) {
            run_cap = std::min(run_cap, static_cast<int>(run_cap_k));
        }
        const auto max_cl =
            static_cast<std::size_t>(solver_env_long("TT_TOPO_SAT_POOL_MAX_CLAUSES", 20'000));
        auto& dp = DistilledClausePool::instance();
        pool_total = dp.size();
        pool_clauses = dp.collect(run_cap, max_cl, next_var_ < 0 ? 0 : next_var_);
    }
    const std::string base = std::string("/tmp/tt_gimsatul_") + std::to_string(static_cast<long>(::getpid()));
    const std::string cnf = base + ".cnf";
    const std::string out = base + ".out";
    // Write the faithful CNF (recorded tape: base encode + occupancy/cap clauses + blocking clauses) + the
    // assumption units (gimsatul has no assume()) + the distilled pool clauses.
    FILE* f = std::fopen(cnf.c_str(), "w");
    if (f == nullptr) {
        return 0;
    }
    std::fprintf(
        f, "p cnf %d %zu\n", next_var_ < 0 ? 0 : next_var_, num_clauses_ + assumption_units.size() + pool_clauses.size());
    for (const int lit : dump_tape_) {
        if (lit == 0) {
            std::fputs("0\n", f);
        } else {
            std::fprintf(f, "%d ", lit);
        }
    }
    for (const int u : assumption_units) {
        std::fprintf(f, "%d 0\n", u);
    }
    for (const auto& cl : pool_clauses) {
        for (const int lit : cl) {
            std::fprintf(f, "%d ", lit);
        }
        std::fputs("0\n", f);
    }
    std::fclose(f);

    const std::string cmd =
        std::string(bin) + " " + cnf + " --threads=" + std::to_string(threads) + " > " + out + " 2>/dev/null";
    (void)std::system(cmd.c_str());

    FILE* r = std::fopen(out.c_str(), "r");
    if (r == nullptr) {
        std::remove(cnf.c_str());
        return 0;
    }
    int status = 0;
    gimsatul_model_.assign(static_cast<size_t>(next_var_ < 0 ? 0 : next_var_) + 1, 0);
    static thread_local std::vector<char> buf(1 << 16);
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), r) != nullptr) {
        const char* line = buf.data();
        if (line[0] == 's') {
            if (std::strstr(line, "UNSATISFIABLE") != nullptr) {
                status = kUnsat;
            } else if (std::strstr(line, "SATISFIABLE") != nullptr) {
                status = kSat;
            }
        } else if (line[0] == 'v') {
            const char* p = line + 1;
            char* end = nullptr;
            for (long v = std::strtol(p, &end, 10); p != end; v = std::strtol(p, &end, 10)) {
                p = end;
                if (v == 0) {
                    break;
                }
                const long a = v < 0 ? -v : v;
                if (a >= 1 && a < static_cast<long>(gimsatul_model_.size())) {
                    gimsatul_model_[static_cast<size_t>(a)] = (v < 0) ? -1 : 1;
                }
            }
        }
    }
    std::fclose(r);
    std::remove(cnf.c_str());
    std::remove(out.c_str());
    if (status == kSat) {
        have_gimsatul_model_ = true;
    }
    // Always-on path marker: proves the gimsatul delegation ran and how many pool clauses each dump carried.
    log_info(
        tt::LogFabric,
        "[topo-sat] gimsatul solve: {} vars, {} clauses, {} assumption unit(s), {} pool clauses injected (pool={}, "
        "run_cap={}), threads={} -> {} in {:.1f} ms",
        next_var_ < 0 ? 0 : next_var_,
        num_clauses_,
        assumption_units.size(),
        pool_clauses.size(),
        pool_total,
        run_cap == DistilledClausePool::kNoCap ? std::string("none") : std::to_string(run_cap),
        threads,
        status == kSat        ? "SAT"
        : status == kUnsat    ? "UNSAT"
                              : "unknown",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return status;
}

void TopologySatSolver::note_permanent_cap(std::size_t cap) {
    const int c = cap >= static_cast<std::size_t>(DistilledClausePool::kNoCap)
                      ? DistilledClausePool::kNoCap
                      : static_cast<int>(cap);
    impl_->pool_cap_tag = std::min(impl_->pool_cap_tag, c);
}

void TopologySatSolver::phase_hint_from_last_gimsatul_model() {
    if (!have_gimsatul_model_) {
        return;
    }
    for (size_t v = 1; v < gimsatul_model_.size(); ++v) {
        const signed char s = gimsatul_model_[v];
        if (s > 0) {
            impl_->phase(static_cast<int>(v));
        } else if (s < 0) {
            impl_->phase(-static_cast<int>(v));
        }
    }
}

TopologySatSolver::~TopologySatSolver() = default;

TopologySatSolver::TopologySatSolver(TopologySatSolver&&) noexcept = default;

TopologySatSolver& TopologySatSolver::operator=(TopologySatSolver&&) noexcept = default;

int TopologySatSolver::declare_one_more_variable() {
    ++next_var_;
    impl_->reserve(next_var_);
    return next_var_;
}

void TopologySatSolver::add(int lit) {
    if (lit == 0) {
        ++num_clauses_;
    } else {
        ++num_literals_;
    }
    if (dump_record_) {
        dump_tape_.push_back(lit);
    }
    impl_->add(lit);
}

void TopologySatSolver::assume(int lit) { impl_->assume(lit); }

void TopologySatSolver::phase(int lit) { impl_->phase(lit); }

void TopologySatSolver::unphase(int lit) { impl_->unphase(lit); }

int TopologySatSolver::solve() {
    have_gimsatul_model_ = false;  // a native solve supersedes any prior gimsatul model
    const int r = impl_->solve();
    impl_->harvest_fixed_units();  // distilled pool: root-fixed units proven by this solve (no-op unless POOL=1)
    return r;
}

int TopologySatSolver::solve_limited(int max_conflicts) {
    have_gimsatul_model_ = false;
    const int r = impl_->solve_limited(max_conflicts);
    impl_->harvest_fixed_units();
    return r;
}

void TopologySatSolver::set_progress_phase(std::string_view phase) { impl_->set_progress_phase(phase); }

void TopologySatSolver::set_solution_progress(std::int64_t found, std::int64_t target) {
    impl_->set_solution_progress(found, target);
}

int TopologySatSolver::val(int lit) const {
    // If a gimsatul model is live, answer from it (matches CaDiCaL val semantics: return `lit` if the literal is
    // satisfied, `-lit` if falsified, 0 if unassigned).
    if (have_gimsatul_model_) {
        const long a = lit < 0 ? -static_cast<long>(lit) : static_cast<long>(lit);
        if (a >= 1 && a < static_cast<long>(gimsatul_model_.size())) {
            const signed char s = gimsatul_model_[static_cast<size_t>(a)];
            if (s == 0) {
                return 0;
            }
            const bool lit_true = (lit > 0) ? (s > 0) : (s < 0);
            return lit_true ? lit : -lit;
        }
        return 0;
    }
    return impl_->val(lit);
}

}  // namespace tt::tt_fabric::detail
