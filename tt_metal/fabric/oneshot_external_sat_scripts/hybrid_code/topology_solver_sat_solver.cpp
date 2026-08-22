// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topology_solver_sat_solver.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

}  // namespace

struct TopologySatSolver::Impl {
    mutable CaDiCaL::Solver solver;
    HeartbeatTerminator heartbeat;

    Impl() {
        solver.set("quiet", 1);
        heartbeat.bind(&solver);
        solver.connect_terminator(&heartbeat);
    }

    void reserve(int max_var) {
        if (max_var > 0) {
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
}

void TopologySatSolver::configure_for_blocking_clause_enumeration() {
    // Only valid in CONFIGURING state (before the first non-config add()).
    // ILB: incremental lazy backtracking — reuse trail across incremental clause additions (CaDiCaL NEWS 1.7.3+).
    // Fixed at 2 ("reuse everything") on purpose, used for iterative solving without re-encoding.
    (void)impl_->solver.set("ilb", 2);
}

bool TopologySatSolver::set_option(const std::string& name, int value) { return impl_->solver.set(name.c_str(), value); }

void TopologySatSolver::set_cancel_flag(std::atomic<bool>* flag) { impl_->set_cancel(flag); }

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

int TopologySatSolver::gimsatul_solve(int threads, const std::vector<int>& assumption_units) {
    have_gimsatul_model_ = false;
    const char* bin = std::getenv("TT_TOPO_SAT_GIMSATUL_BIN");
    if (bin == nullptr || bin[0] == '\0' || !dump_record_) {
        return 0;  // no binary / no tape -> unknown; caller falls back to native solve
    }
    const std::string base = std::string("/tmp/tt_gimsatul_") + std::to_string(static_cast<long>(::getpid()));
    const std::string cnf = base + ".cnf";
    const std::string out = base + ".out";
    // Write the faithful CNF (recorded tape) + the assumption units (gimsatul has no assume()).
    FILE* f = std::fopen(cnf.c_str(), "w");
    if (f == nullptr) {
        return 0;
    }
    std::fprintf(f, "p cnf %d %zu\n", next_var_ < 0 ? 0 : next_var_, num_clauses_ + assumption_units.size());
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
    return status;
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
    return impl_->solve();
}

int TopologySatSolver::solve_limited(int max_conflicts) {
    have_gimsatul_model_ = false;
    return impl_->solve_limited(max_conflicts);
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
