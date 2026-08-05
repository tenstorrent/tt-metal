// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Cooperative stackful-fiber scheduler for the emule program runner: each
// (core, RISC) kernel runs on a ucontext fiber multiplexed onto a persistent
// pool of K worker threads; a fiber that blocks at a sync point parks and is
// woken on its predicate. Process-global singleton reached from jit_hw sync
// primitives via the extern-C bridge.
// See tt-emule docs/fiber-engine.md.

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "jit_hw/internal/emule_thread_ctx.h"  // ThreadCommonCtx (the fiber-owned ctx)

namespace tt::tt_metal::emule_fiber {

/// A quiescent deadlock, or a host-wait sequence that pumped past its liveness bound. Distinct from
/// a kernel fault, which propagates as whatever the kernel threw — catch this, never std::exception.
class FiberEngineStall : public std::runtime_error {
public:
    explicit FiberEngineStall(const std::string& what) : std::runtime_error(what) {}
};

struct FiberSchedulerImpl;  // defined in the .cpp (namespace-scope so the ucontext
                            // trampoline + per-worker thread_locals can access it)

// Per-fiber identity — used to restore the silicon-named my_x/my_y globals on
// swap-in (they cannot move into the ctx) and for the hang-detection dump.
struct FiberIdentity {
    uint8_t  phys_x = 0;
    uint8_t  phys_y = 0;
    uint32_t logical_x = 0;
    uint32_t logical_y = 0;
    uint8_t  proc_id = 0;
    const char* kernel_src = nullptr;  // static string (kernel source path), for diagnostics
};

// Outcome of a resumable launch. Completed = every fiber ran to Done (registry torn down).
// HostWait = the run quiesced with a host-facing socket wait parked; the fibers stay ALIVE
// (no teardown) and the run resumes via pump() as the host feeds the socket. See run_persistent().
enum class RunOutcome { Completed, HostWait };

class FiberScheduler {
public:
    static FiberScheduler& instance();

    // ---- Runner-facing C++ API (register/run split) ----
    // Register a fiber; does NOT run it. The fiber takes ownership of `ctx`
    // (the per-RISC ThreadCommonCtx). `entry` is the kernel body.
    void spawn(std::function<void()> entry, std::unique_ptr<ThreadCommonCtx> ctx, const FiberIdentity& id);

    // Run all registered fibers to completion on K workers, then clear the
    // registry. Rethrows the first fiber exception; throws on a quiescent
    // deadlock (tier 1); aborts with a diagnostic dump on livelock/hang (tier 2).
    void run_until_idle();

    // Host-interleaved (persistent) dispatch — for a program whose kernels block on a
    // host-fed socket. Like run_until_idle(), but if the run quiesces with a host-facing
    // socket wait parked it returns HostWait WITHOUT tearing down (fibers stay alive);
    // pump() then resumes them one quantum per host socket call, and Completed tears down.
    // A quiescent deadlock with NO socket wait parked still throws (diagnostics preserved).
    // Teardown is automatic: the pump() that observes every fiber Done returns Completed and clears
    // the registry (driven by the host's final socket barrier / synchronize). No separate join hook.
    // A HostWait return leaves the tier-2 watchdog RUNNING, so TT_EMULE_FIBER_WATCHDOG_SEC covers
    // the gap until the host pumps again and a never-pumped run cannot end as a silent success.
    RunOutcome run_persistent();     // initial launch
    RunOutcome pump();               // resume the SAME parked fibers one quantum (host advanced a credit word)

    // Consecutive no-progress pumps before pump() escalates to the tier-1 deadlock
    // (TT_EMULE_HOST_WAIT_STALL_LIMIT, default 8192). Exposed so the drain backstop sizes above it.
    static uint64_t host_wait_stall_limit();

    // Per-dispatch generation for runner-side keepalives; begin_ returns the one it opened. Reclaim
    // needs the per-generation query: a parked run holds ONE generation, pinning any age bound.
    uint64_t begin_spawn_generation();
    uint64_t oldest_live_spawn_generation() const;
    bool spawn_generation_is_live(uint64_t gen) const;
    // Every generation with a live fiber, in ONE registry scan. Reclaiming against this beats a
    // per-candidate spawn_generation_is_live(), which rescans thousands of fibers per keepalive.
    std::vector<uint64_t> live_spawn_generations() const;

    // Tear down a run that will not drain, throwing FiberEngineStall(`why` + parked-fiber dump). For
    // a caller with its own bound: the engine's key on GLOBAL progress, which a stuck socket outlasts.
    void abandon_host_wait(const std::string& why);

    // ---- Bridge ops (called by the runner's extern-C thunks from a running fiber) ----
    void lock();
    void unlock();
    void park_locked(const void* key);         // pre: lock held; post: lock released
    void park_locked_socket(const void* key);  // as park_locked, tagged as a host-fed socket wait
    // Tag the current fiber as spin-polling a socket with no data (miss sets, hit clears); untagged, a
    // poll-only program reads as a livelock. `host_fed` = HOST sender; d2d is dumped but not HostWait.
    void note_socket_poll_wait(bool waiting, bool host_fed);
    // CB analogue for the non-blocking cb_pages_available_at_front probe. The producer is a peer, so
    // this only arms the stall check and names the wait in the dump; never host-resumable. n==0 clears.
    void note_cb_poll_wait(unsigned cb_id, unsigned n);
    void quiescence_park();              // defer to quiescence: re-queue lowest-priority, released at quiescence
    void wake(const void* key);
    void yield();
    void note_publish(unsigned pages);

    FiberScheduler(const FiberScheduler&) = delete;
    FiberScheduler& operator=(const FiberScheduler&) = delete;

private:
    FiberScheduler();
    ~FiberScheduler();
    void launch_and_wait(bool initial);  // shared launch+wait tail (run_until_idle/run_persistent/pump)
    void teardown_and_throw();           // clear the registry; rethrow eptr / throw on deadlock
    RunOutcome finish_or_host_wait();    // shared outcome tail of run_persistent()/pump()
    std::unique_ptr<FiberSchedulerImpl> p_;
};

}  // namespace tt::tt_metal::emule_fiber
