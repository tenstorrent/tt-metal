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
class Program;
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

struct CompileSessionOptions {
    // Worker threads that build queued kernels. 0 uses the hardware concurrency.
    std::size_t workers = 0;
    // JIT compile-server endpoints ("host:port"). Empty means every kernel builds locally.
    std::vector<std::string> remote_endpoints;
    // Kernels go to the compile server only once the session has queued at least this many;
    // below that, the round trips cost more than they save.
    std::size_t remote_min_kernels = 10;
};

struct CompileSessionStats {
    std::size_t programs_added = 0;     // add() calls that resolved at least one kernel
    std::size_t programs_deduped = 0;   // add() calls skipped because their key was already added
    std::size_t kernels_queued = 0;     // distinct kernels queued
    std::size_t kernels_built = 0;      // built by a session worker (on-disk cache hits included)
    std::size_t kernels_preempted = 0;  // built or being built by another thread when a worker reached them
    std::size_t kernels_remote = 0;     // of kernels_built, the ones the compile server built
    std::size_t kernels_failed = 0;     // build threw; the kernel compiles inline later
    std::size_t kernels_dropped = 0;    // still queued when the session closed
    std::size_t kernels_pending = 0;    // queued, not started yet
    std::size_t kernels_running = 0;    // being built right now
    double busy_seconds = 0.0;          // worker time spent building, summed over workers
};

// Builds the kernels of programs that were constructed but not compiled, ahead of the run that needs them.
//
// add() resolves each kernel's build inputs on the calling thread, which needs the device open. The
// kernel is then queued and a worker builds it into the on-disk kernel cache, in priority order (lower
// first), without touching the device again; the device may be closed and reopened meanwhile. A normal
// compile of the same kernel later in this process waits for a build that is running, and builds the
// kernel itself if no worker has reached it yet.
//
// The session only fills the kernel cache, so it never changes results. A failed build is counted and
// left to the normal compile path. Remote builds that fail are retried locally. The device's MetalContext
// must outlive the session (true for devices opened through ttnn).
class CompileSession {
public:
    explicit CompileSession(const CompileSessionOptions& options = {});
    ~CompileSession();  // drops kernels not started yet and joins the workers

    CompileSession(const CompileSession&) = delete;
    CompileSession& operator=(const CompileSession&) = delete;

    // Queues every kernel of program that has not been queued before. The session takes the program's kernels
    // over: the program must not be compiled or enqueued afterwards. A nonzero dedup_key (for example a program
    // cache hash) skips programs whose key was already added for a device with the same build key, before any
    // work is done for them.
    void add(distributed::MeshDevice& device, Program& program, std::uint64_t dedup_key = 0);

    // Priority of kernels queued from now on; lower builds first.
    void set_priority(std::size_t priority);

    // Blocks until nothing is queued or being built (or, once closed, until nothing is being built).
    void wait();

    // Drops kernels not started yet and waits for the ones being built. Idempotent; a concurrent call waits too.
    void close();

    CompileSessionStats stats() const;

    // While deferral is on for the calling thread, MakeProgramFromSpec and MakeMeshWorkloadFromSpec(s)
    // return programs without compiling them, and deferring_session() returns this session so that callers
    // building programs can hand them to add(). Turn it off on every thread before destroying the session.
    void set_deferral(bool on);
    static CompileSession* deferring_session();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
