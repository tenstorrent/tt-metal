// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_solver_sat_solver.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

#include <cadical.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric::detail {

namespace {

// Format a large count with a k/M/G suffix for the human-readable progress line (e.g. 2500 -> "2.5k").
std::string humanize(int64_t n) {
    if (n < 0) {
        return "0";
    }
    if (n < 1000) {
        return std::to_string(n);
    }
    const char* suffix = "kMGT";
    double v = static_cast<double>(n);
    int i = -1;
    while (v >= 1000.0 && i < 3) {
        v /= 1000.0;
        ++i;
    }
    return fmt::format("{:.1f}{}", v, suffix[i]);
}

// Terminator: CaDiCaL calls terminate() frequently during solve(). It serves two purposes here:
//  1. Cancellation -- returns true once a shared cancel flag is set, so the first clause-sharing worker to find a
//     model stops its peers promptly (a relaxed atomic load is ~1ns; nullptr means "never cancel").
//  2. Progress heartbeat -- every 15s it logs one plain-English INFO line so a long silent solve is visibly alive
//     (elapsed, what it is searching for, and a live "dead ends ruled out per second" conflict rate). The clock is
//     sampled only every 4096 calls to keep the hot search loop cheap. Never actually terminates for the heartbeat.
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
    // short free-text stage tag; `found`/`target` drive the enumeration counters (target <= 0 means a single solve).
    void set_phase(std::string_view phase) { phase_.assign(phase); }
    void set_solution_progress(int64_t found, int64_t target) {
        sols_found_ = found;
        sols_target_ = target;
    }

    void set_cancel(std::atomic<bool>* flag) { cancel_ = flag; }

    bool terminate() override {
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
        const int64_t d_conf = (conflicts >= 0 && conflicts >= prev_conflicts_) ? conflicts - prev_conflicts_ : 0;
        prev_conflicts_ = conflicts < 0 ? 0 : conflicts;
        const int64_t conf_rate = since_last > 0 ? (d_conf * 1000) / since_last : 0;

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

        // A "dead end" = the solver tried a combination, hit a contradiction, and backed out (a conflict); the rate is
        // the best "it's alive and working" signal, and the running total is how much of the space it has ruled out.
        log_info(
            tt::LogFabric,
            "Solver running {}s{} | {} | ruling out {} dead ends/s ({} ruled out so far)",
            elapsed,
            phase_tag,
            sols,
            humanize(conf_rate),
            humanize(conflicts));
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
    ClauseExportLearner(ClauseSharingPool* pool, int producer_id, int max_size) :
        pool_(pool), producer_id_(producer_id), max_size_(max_size) {}

    bool learning(int size) override { return pool_ != nullptr && size > 0 && size <= max_size_; }

    void learn(int lit) override {
        if (lit != 0) {
            buf_.push_back(lit);
            return;
        }
        if (!buf_.empty()) {
            pool_->publish(producer_id_, buf_);
            buf_.clear();
        }
    }

private:
    ClauseSharingPool* pool_ = nullptr;
    int producer_id_ = 0;
    int max_size_ = 0;
    std::vector<int> buf_;
};

}  // namespace

struct TopologySatSolver::Impl {
    mutable CaDiCaL::Solver solver;
    HeartbeatTerminator heartbeat;
    std::unique_ptr<ClauseExportLearner> export_learner;

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

    void reserve(int max_var) {
        if (max_var > 0) {
            solver.reserve(max_var);
        }
    }

    void add(int lit) { solver.add(lit); }

    void assume(int lit) { solver.assume(lit); }

    int solve() {
        heartbeat.reset();
        return solver.solve();
    }

    void set_cancel(std::atomic<bool>* flag) { heartbeat.set_cancel(flag); }

    void set_progress_phase(std::string_view phase) { heartbeat.set_phase(phase); }

    void set_solution_progress(int64_t found, int64_t target) { heartbeat.set_solution_progress(found, target); }

    void enable_clause_export(ClauseSharingPool* pool, int producer_id, int max_size) {
        if (pool == nullptr) {
            return;
        }
        export_learner = std::make_unique<ClauseExportLearner>(pool, producer_id, max_size);
        solver.connect_learner(export_learner.get());
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

TopologySatSolver::TopologySatSolver() : impl_(std::make_unique<Impl>()) {}

void TopologySatSolver::configure_for_blocking_clause_enumeration() {
    // Only valid in CONFIGURING state (before the first non-config add()).
    // ILB: incremental lazy backtracking — reuse trail across incremental clause additions (CaDiCaL NEWS 1.7.3+).
    (void)impl_->solver.set("ilb", 2);
}

bool TopologySatSolver::set_option(const std::string& name, int value) { return impl_->solver.set(name.c_str(), value); }

void TopologySatSolver::set_cancel_flag(std::atomic<bool>* flag) { impl_->set_cancel(flag); }

void TopologySatSolver::set_progress_phase(std::string_view phase) { impl_->set_progress_phase(phase); }

void TopologySatSolver::set_solution_progress(std::int64_t found, std::int64_t target) {
    impl_->set_solution_progress(found, target);
}

void TopologySatSolver::enable_clause_export(ClauseSharingPool* pool, int producer_id, int max_size) {
    impl_->enable_clause_export(pool, producer_id, max_size);
}

TopologySatSolver::~TopologySatSolver() = default;

TopologySatSolver::TopologySatSolver(TopologySatSolver&&) noexcept = default;

TopologySatSolver& TopologySatSolver::operator=(TopologySatSolver&&) noexcept = default;

int TopologySatSolver::declare_one_more_variable() {
    ++next_var_;
    impl_->reserve(next_var_);
    return next_var_;
}

void TopologySatSolver::add(int lit) { impl_->add(lit); }

void TopologySatSolver::assume(int lit) { impl_->assume(lit); }

int TopologySatSolver::solve() { return impl_->solve(); }

int TopologySatSolver::solve_limited(int max_conflicts) { return impl_->solve_limited(max_conflicts); }

int TopologySatSolver::val(int lit) const { return impl_->val(lit); }

}  // namespace tt::tt_fabric::detail
