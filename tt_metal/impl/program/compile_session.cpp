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
#include "impl/program/kernel_build_job.hpp"
#include "impl/program/program_impl.hpp"

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
    // Queues the jobs whose kernels were not queued before.
    void enqueue(std::vector<detail::KernelBuildJob> jobs) {
        bool queued_any = false;
        {
            std::lock_guard lock(mutex);
            if (closing) {
                return;
            }
            if (!jobs.empty()) {
                ++stats.programs_added;
            }
            for (auto& job : jobs) {
                if (!queued_kernels.insert(job.kernel_hash).second) {
                    continue;
                }
                queue.push_back(std::make_unique<detail::KernelBuildJob>(std::move(job)));
                ++stats.kernels_queued;
                ++stats.kernels_pending;
                queued_any = true;
            }
        }
        if (queued_any) {
            work_cv.notify_all();
        }
    }

    CompileSessionOptions options;

    mutable std::mutex mutex;
    std::condition_variable work_cv;  // workers wait for work or shutdown
    std::condition_variable idle_cv;  // wait() waits for an empty queue and idle workers
    std::deque<std::unique_ptr<detail::KernelBuildJob>> queue;
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

    void worker_loop() {
        while (true) {
            std::unique_ptr<detail::KernelBuildJob> job;
            bool remote = false;
            {
                std::unique_lock lock(mutex);
                work_cv.wait(lock, [this] { return closing || !queue.empty(); });
                if (queue.empty()) {
                    return;  // closing
                }
                job = std::move(queue.front());
                queue.pop_front();
                --stats.kernels_pending;
                ++stats.kernels_running;
                remote = !options.remote_endpoints.empty() && !remote_down &&
                         stats.kernels_queued >= options.remote_min_kernels;
            }
            const auto start = std::chrono::steady_clock::now();
            std::optional<detail::KernelBuildResult> result;
            bool remote_failed = false;
            try {
                if (remote) {
                    try {
                        result = detail::run_kernel_build_job(*job, options.remote_endpoints);
                    } catch (const jit_server::RemoteCompileTransportError& e) {
                        remote_failed = true;
                        std::lock_guard lock(mutex);
                        if (!remote_down) {
                            remote_down = true;
                            log_warning(
                                tt::LogBuildKernels,
                                "CompileSession: {}; this session builds its remaining kernels locally",
                                e.what());
                        }
                    } catch (const std::exception& e) {
                        remote_failed = true;
                        log_warning(
                            tt::LogBuildKernels,
                            "CompileSession: remote build of kernel {} failed, building it locally: {}",
                            job->kernel_hash,
                            e.what());
                    }
                }
                if (!result) {
                    result = detail::run_kernel_build_job(*job, {});
                }
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogBuildKernels,
                    "CompileSession: build of kernel {} failed; it compiles inline when needed: {}",
                    job->kernel_hash,
                    e.what());
            } catch (...) {
                log_warning(
                    tt::LogBuildKernels,
                    "CompileSession: build of kernel {} failed; it compiles inline when needed",
                    job->kernel_hash);
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            job.reset();  // release the kernel and its build env outside the lock
            {
                std::lock_guard lock(mutex);
                --stats.kernels_running;
                stats.busy_seconds += seconds;
                if (!result) {
                    ++stats.kernels_failed;
                } else {
                    stats.kernels_remote_fallback += remote_failed ? 1 : 0;
                    switch (*result) {
                        case detail::KernelBuildResult::Built: ++stats.kernels_built; break;
                        case detail::KernelBuildResult::BuiltRemotely:
                            ++stats.kernels_built;
                            ++stats.kernels_remote;
                            break;
                        case detail::KernelBuildResult::AlreadyBuilt:
                        case detail::KernelBuildResult::BuildingElsewhere: ++stats.kernels_preempted; break;
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
                stats.kernels_dropped += queue.size();
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

bool CompileSession::sink_kernel_builds(detail::ProgramImpl& program, IDevice* device) {
    if (t_build_sink == nullptr || !detail::BuildSinkScope::active()) {
        return false;
    }
    // Resolves build inputs for every kernel; needs the device, so it runs on the caller's thread.
    t_build_sink->impl_->enqueue(program.prepare_kernel_builds(device));
    return true;
}

}  // namespace experimental
}  // namespace tt::tt_metal
