// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/compile_session.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/reflection.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/jit_server/jit_compile_rpc_client.hpp"
#include "impl/program/kernel_build_job.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"

namespace tt::tt_metal::experimental {

namespace {
thread_local CompileSession* t_deferring_session = nullptr;
}  // namespace

struct CompileSession::Impl {
    struct Queued {
        std::size_t priority;
        std::uint64_t seq;
        std::unique_ptr<detail::KernelBuildJob> job;
    };
    struct LaterFirst {
        bool operator()(const Queued& a, const Queued& b) const {
            return a.priority != b.priority ? a.priority > b.priority : a.seq > b.seq;
        }
    };

    CompileSessionOptions options;

    mutable std::mutex mutex;
    std::condition_variable work_cv;  // workers wait for work or shutdown
    std::condition_variable idle_cv;  // wait() waits for an empty queue and idle workers
    std::priority_queue<Queued, std::vector<Queued>, LaterFirst> queue;
    std::unordered_set<std::uint64_t> added_keys;
    std::unordered_set<std::size_t> queued_kernels;
    std::size_t priority = 0;
    std::uint64_t seq = 0;
    bool closing = false;
    // Set when the compile server could not be reached or stopped replying; the session's remaining kernels
    // build locally rather than each waiting out the failure again.
    bool remote_down = false;
    CompileSessionStats stats;
    std::vector<std::thread> workers;
    std::mutex join_mutex;  // a second close() waits here until the first has joined the workers
    std::atomic<int> deferring_threads = 0;

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
                // priority_queue::top() is const; the element is popped right after, so moving out is safe.
                job = std::move(const_cast<Queued&>(queue.top()).job);
                queue.pop();
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
                queue = {};
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
    set_deferral(false);
    if (int n = impl_->deferring_threads.load(); n > 0) {
        log_warning(
            tt::LogBuildKernels,
            "CompileSession destroyed while deferral is still on for {} other thread(s); turn it off first",
            n);
    }
    impl_->close();
}

void CompileSession::add(distributed::MeshDevice& device, Program& program, std::uint64_t dedup_key) {
    std::uint64_t key = 0;
    if (dedup_key != 0) {
        // The same op on a device with another configuration builds different kernels.
        const auto build_key = MetalContext::instance(extract_context_id(&device))
                                   .build_env_manager()
                                   .get_device_build_env(device.build_id())
                                   .build_key();
        key = ttsl::hash::hash_objects_with_default_seed(dedup_key, build_key);
        key = key != 0 ? key : 1;
        std::lock_guard lock(impl_->mutex);
        if (impl_->closing) {
            return;
        }
        if (impl_->added_keys.contains(key)) {
            ++impl_->stats.programs_deduped;
            return;
        }
    }
    // Resolves build inputs for every kernel; needs the device, so it runs on the caller's thread.
    std::vector<detail::KernelBuildJob> jobs = program.impl().prepare_kernel_builds(&device);
    bool queued_any = false;
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->closing) {
            return;
        }
        // Recorded only once prepare succeeded, so a program that failed to prepare does not hide later ones.
        if (key != 0) {
            impl_->added_keys.insert(key);
        }
        if (!jobs.empty()) {
            ++impl_->stats.programs_added;
        }
        for (auto& job : jobs) {
            if (!impl_->queued_kernels.insert(job.kernel_hash).second) {
                continue;
            }
            impl_->queue.push(
                {impl_->priority, impl_->seq++, std::make_unique<detail::KernelBuildJob>(std::move(job))});
            ++impl_->stats.kernels_queued;
            ++impl_->stats.kernels_pending;
            queued_any = true;
        }
    }
    if (queued_any) {
        impl_->work_cv.notify_all();
    }
}

void CompileSession::set_priority(std::size_t priority) {
    std::lock_guard lock(impl_->mutex);
    impl_->priority = priority;
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

void CompileSession::set_deferral(bool on) {
    if (on && t_deferring_session != this) {
        if (t_deferring_session != nullptr) {
            t_deferring_session->set_deferral(false);
        }
        t_deferring_session = this;
        ++impl_->deferring_threads;
    } else if (!on && t_deferring_session == this) {
        t_deferring_session = nullptr;
        --impl_->deferring_threads;
    }
}

CompileSession* CompileSession::deferring_session() { return t_deferring_session; }

}  // namespace tt::tt_metal::experimental
