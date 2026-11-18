// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/compile_session.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>

#include "impl/jit_server/jit_compile_rpc_client.hpp"
#include "impl/program/kernel_build.hpp"

namespace tt::tt_metal {

namespace {
thread_local experimental::CompileSession* t_build_sink = nullptr;
thread_local bool t_build_sink_scope = false;
}  // namespace

detail::BuildSinkScope::BuildSinkScope(bool active) : previous_(t_build_sink_scope) { t_build_sink_scope = active; }

detail::BuildSinkScope::~BuildSinkScope() { t_build_sink_scope = previous_; }

bool detail::BuildSinkScope::active() { return t_build_sink_scope; }

namespace experimental {

struct CompileSession::Impl {
    // One queue entry: the kernels of one program that were not queued before.
    using Unit = std::vector<detail::KernelBuild>;

    // Queues the program's builds whose kernels were not queued before, as one entry.
    void enqueue(std::vector<detail::KernelBuild> builds) {
        std::size_t queued = 0;
        {
            std::lock_guard lock(mutex);
            if (closing) {
                return;
            }
            if (!builds.empty()) {
                ++stats.programs_added;
            }
            Unit unit;  // KernelBuild is not assignable, so the new kernels move into a fresh vector
            for (auto& build : builds) {
                if (queued_kernels.insert(build.kernel_hash).second) {
                    unit.push_back(std::move(build));
                }
            }
            queued = unit.size();
            if (queued != 0) {
                queue.push_back(std::move(unit));
            }
            stats.kernels_queued += queued;
            stats.kernels_pending += queued;
        }
        if (queued != 0) {
            work_cv.notify_all();
        }
    }

    CompileSessionOptions options;

    mutable std::mutex mutex;
    std::condition_variable work_cv;  // workers wait for work or shutdown
    std::condition_variable idle_cv;  // wait() waits for an empty queue and idle workers
    std::deque<Unit> queue;
    std::unordered_set<std::size_t> queued_kernels;
    bool closing = false;
    // Set when the compile server could not be reached or stopped replying; the session's remaining kernels
    // build locally rather than each waiting out the failure again.
    bool remote_down = false;
    CompileSessionStats stats;
    std::vector<std::thread> workers;
    std::mutex join_mutex;  // a second close() waits here until the first has joined the workers

    explicit Impl(const CompileSessionOptions& opts) : options(opts) {
        std::size_t n = options.workers != 0 ? options.workers : std::max(1u, std::thread::hardware_concurrency());
        workers.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            workers.emplace_back([this] { worker_loop(); });
        }
    }

    bool idle() const { return stats.kernels_running == 0 && (closing || queue.empty()); }

    // Builds unit through the compile server, then locally whatever that did not build. Never throws.
    std::vector<detail::KernelBuildStatus> build(Unit& unit, bool remote, std::size_t& remote_fallbacks) {
        std::vector<detail::KernelBuildStatus> statuses;
        try {
            if (remote) {
                statuses = detail::build_kernels(unit, options.remote_endpoints);
                std::exception_ptr remote_error;
                for (const auto& status : statuses) {
                    if (status.result == detail::KernelBuildResult::Failed) {
                        remote_error = status.error;
                        ++remote_fallbacks;
                    }
                }
                if (remote_error) {
                    note_remote_failure(remote_error, unit);
                }
            }
            if (statuses.empty() || remote_fallbacks != 0) {
                // Local build of the kernels the server did not build (all of them without a server).
                Unit rest;
                std::vector<std::size_t> rest_index;
                for (std::size_t i = 0; i < unit.size(); ++i) {
                    if (statuses.empty() || statuses[i].result == detail::KernelBuildResult::Failed) {
                        rest.push_back(unit[i]);
                        rest_index.push_back(i);
                    }
                }
                auto local = detail::build_kernels(rest, {});
                statuses.resize(unit.size());
                for (std::size_t j = 0; j < rest.size(); ++j) {
                    statuses[rest_index[j]] = std::move(local[j]);
                }
            }
        } catch (const std::exception& e) {
            log_warning(
                tt::LogBuildKernels,
                "CompileSession: build of {} kernel(s) failed; they compile inline when needed: {}",
                unit.size(),
                e.what());
            statuses.assign(unit.size(), {});
        }
        for (std::size_t i = 0; i < statuses.size(); ++i) {
            if (statuses[i].result == detail::KernelBuildResult::Failed && statuses[i].error) {
                try {
                    std::rethrow_exception(statuses[i].error);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogBuildKernels,
                        "CompileSession: build of kernel {} failed; it compiles inline when needed: {}",
                        unit[i].kernel_hash,
                        e.what());
                } catch (...) {
                    log_warning(
                        tt::LogBuildKernels,
                        "CompileSession: build of kernel {} failed; it compiles inline when needed",
                        unit[i].kernel_hash);
                }
            }
        }
        return statuses;
    }

    void note_remote_failure(const std::exception_ptr& error, const Unit& unit) {
        try {
            std::rethrow_exception(error);
        } catch (const jit_server::RemoteCompileTransportError& e) {
            std::lock_guard lock(mutex);
            if (!remote_down) {
                remote_down = true;
                log_warning(
                    tt::LogBuildKernels,
                    "CompileSession: {}; this session builds its remaining kernels locally",
                    e.what());
            }
        } catch (const std::exception& e) {
            log_warning(
                tt::LogBuildKernels,
                "CompileSession: remote build of {} kernel(s) failed, building them locally: {}",
                unit.size(),
                e.what());
        } catch (...) {
        }
    }

    void worker_loop() {
        while (true) {
            Unit unit;
            bool remote = false;
            {
                std::unique_lock lock(mutex);
                work_cv.wait(lock, [this] { return closing || !queue.empty(); });
                if (queue.empty()) {
                    return;  // closing
                }
                unit = std::move(queue.front());
                queue.pop_front();
                stats.kernels_pending -= unit.size();
                stats.kernels_running += unit.size();
                remote = !options.remote_endpoints.empty() && !remote_down &&
                         stats.kernels_queued >= options.remote_min_kernels;
            }
            const auto start = std::chrono::steady_clock::now();
            std::size_t remote_fallbacks = 0;
            const auto statuses = build(unit, remote, remote_fallbacks);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const std::size_t size = unit.size();
            unit.clear();  // release the kernels and their build env outside the lock
            {
                std::lock_guard lock(mutex);
                stats.kernels_running -= size;
                stats.busy_seconds += seconds;
                ++stats.programs_built;
                stats.kernels_remote_fallback += remote_fallbacks;
                for (const auto& status : statuses) {
                    switch (status.result) {
                        case detail::KernelBuildResult::Built: ++stats.kernels_built; break;
                        case detail::KernelBuildResult::BuiltRemotely:
                            ++stats.kernels_built;
                            ++stats.kernels_remote;
                            break;
                        case detail::KernelBuildResult::AlreadyBuilt:
                        case detail::KernelBuildResult::BuildingElsewhere: ++stats.kernels_preempted; break;
                        case detail::KernelBuildResult::Failed: ++stats.kernels_failed; break;
                    }
                }
                if (idle()) {
                    idle_cv.notify_all();
                }
            }
        }
    }

    void close() {
        std::lock_guard join_lock(join_mutex);
        {
            std::lock_guard lock(mutex);
            if (!closing) {
                closing = true;
                stats.kernels_dropped += stats.kernels_pending;
                stats.kernels_pending = 0;
                queue.clear();
            }
        }
        work_cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        idle_cv.notify_all();
    }
};

CompileSession::CompileSession(const CompileSessionOptions& options) : impl_(std::make_unique<Impl>(options)) {}

CompileSession::~CompileSession() {
    set_build_sink(false);
    impl_->close();
}

void CompileSession::wait() {
    std::unique_lock lock(impl_->mutex);
    impl_->idle_cv.wait(lock, [this] { return impl_->idle(); });
}

void CompileSession::close() { impl_->close(); }

CompileSessionStats CompileSession::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}

void CompileSession::set_build_sink(bool on) {
    if (on) {
        t_build_sink = this;
    } else if (t_build_sink == this) {
        t_build_sink = nullptr;
    }
}

CompileSession* CompileSession::build_sink() { return t_build_sink; }

bool CompileSession::takes_kernel_builds() { return t_build_sink != nullptr && detail::BuildSinkScope::active(); }

void CompileSession::add_kernel_builds(std::vector<detail::KernelBuild> builds) {
    t_build_sink->impl_->enqueue(std::move(builds));
}

}  // namespace experimental
}  // namespace tt::tt_metal
