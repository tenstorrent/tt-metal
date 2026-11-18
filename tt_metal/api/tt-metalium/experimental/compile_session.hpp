// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tt::tt_metal {
namespace detail {
class ProgramImpl;
struct KernelBuild;
}  // namespace detail
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

// What a CompileSession queues and builds as one unit.
enum class CompileSessionHandoff {
    // Each kernel on its own: one queue entry, and one compile-server request and reply, per kernel.
    Kernel,
    // A program's kernels together, as ProgramImpl::compile builds them: one queue entry per program, whose kernels
    // build in parallel locally or go to the compile server as one batch.
    Program,
};

struct CompileSessionOptions {
    // Threads that take queued units and build them. A local build runs its kernels on the JIT build threads, so
    // more workers than cores only helps when the builds wait on the compile server. 0 uses the hardware concurrency.
    std::size_t workers = 0;
    // JIT compile-server endpoints ("host:port"). Empty means every kernel builds locally. If a server cannot be
    // reached or a reply does not arrive within TT_METAL_JIT_SERVER_TIMEOUT_S, the session stops using them.
    std::vector<std::string> remote_endpoints;
    // Kernels go to the compile server only once the session has queued at least this many;
    // below that, the round trips cost more than they save.
    std::size_t remote_min_kernels = 10;
    CompileSessionHandoff handoff = CompileSessionHandoff::Kernel;
};

struct CompileSessionStats {
    std::size_t programs_added = 0;           // programs that queued at least one kernel
    std::size_t units_built = 0;              // queue entries built (kernels or programs, see handoff)
    std::size_t kernels_queued = 0;           // distinct kernels queued
    std::size_t kernels_built = 0;            // built by a session worker (on-disk cache hits included)
    std::size_t kernels_preempted = 0;        // built or being built by another thread when a worker reached them
    std::size_t kernels_remote = 0;           // of kernels_built, the ones the compile server built
    std::size_t kernels_remote_fallback = 0;  // remote build failed and the kernel was built here instead
    std::size_t kernels_failed = 0;           // build threw; the kernel compiles inline later
    std::size_t kernels_dropped = 0;          // still queued when the session closed
    std::size_t kernels_pending = 0;          // queued, not started yet
    std::size_t kernels_running = 0;          // being built right now
    double busy_seconds = 0.0;                // worker time spent building, summed over workers
};

// Builds, in the background, the kernels of programs compiled for a mock device, so that a later run on the real
// device finds them in the on-disk kernel cache.
//
// While the build sink is on for a thread (set_build_sink), a workload that thread enqueues on a mock device is
// compiled as usual up to the build: ProgramImpl::compile resolves each kernel's build inputs on that thread, then
// hands them to this session instead of building them, and the program counts as compiled; a mock device never loads
// binaries. Programs compiled any other way (a direct CompileProgram) build normally, since their callers read the
// binaries. Each kernel is queued once, and workers build the queued units in order with the same code
// ProgramImpl::compile uses, without touching the device again; the device may be closed and reopened meanwhile. A
// normal compile of the same kernel later in this process waits for a build that is running, and builds the kernel
// itself if no worker has reached it yet.
//
// The session only fills the kernel cache, so it never changes results. A failed build is counted and left to the
// normal compile path. Remote builds that fail are retried locally. The device's MetalContext must outlive the
// session (true for devices opened through ttnn).
class CompileSession {
public:
    explicit CompileSession(const CompileSessionOptions& options = {});
    ~CompileSession();  // drops kernels not started yet and joins the workers

    CompileSession(const CompileSession&) = delete;
    CompileSession& operator=(const CompileSession&) = delete;

    // Blocks until nothing is queued or being built (or, once closed, until nothing is being built).
    void wait();

    // Drops kernels not started yet and waits for the ones being built. Idempotent; a concurrent call waits too.
    void close();

    CompileSessionStats stats() const;

    // While the build sink is on for the calling thread, a workload enqueued on a mock device queues its kernels in
    // this session and counts as compiled at once: a mock device never loads binaries, so the run goes on without
    // waiting for them. Programs are otherwise built, cached and enqueued as usual, so buffers get the addresses they
    // get on the real device. Turn it off on every thread before destroying the session.
    void set_build_sink(bool on);
    static CompileSession* build_sink();

private:
    friend class ::tt::tt_metal::detail::ProgramImpl;
    // True while the calling thread's build sink takes the kernels of the program it compiles: a sink is on, inside
    // the detail::BuildSinkScope that a mock-device workload compile opens.
    static bool takes_kernel_builds();
    // Queues one program's kernels in the calling thread's build sink.
    static void add_kernel_builds(std::vector<::tt::tt_metal::detail::KernelBuild> builds);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
