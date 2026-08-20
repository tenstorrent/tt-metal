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

/// An engine stall (quiescent deadlock, or a host wait past its bound). Catch this, not std::exception.
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
// PeerWait = the run quiesced but a PEER RANK may still deliver into this rank's L1; fibers stay
// alive on the same contract as HostWait, and the caller re-pumps once a delivery lands.
enum class RunOutcome { Completed, HostWait, PeerWait };

// Installed by the runner when multi-rank is active; absent (and never consulted) single-rank.
// Returns true while some peer rank could still deliver here — i.e. this rank's local quiescence is
// not yet a global one, so the stall is resumable rather than a deadlock. A rank cannot classify a
// wait as peer-fed on its own (every worker L1 is a legal fabric destination), which is why the
// question is asked globally instead of per-fiber. See tt-emule docs/multi-rank-emulation.md.
void set_peer_progress_probe(std::function<bool()> probe);

// Read-only companion for the tier-2 watchdog: true while some peer rank could still act. The probe
// above publishes this rank's quiescence, so the watchdog thread must not call it. Without this, a
// rank that legitimately waits for a peer keeps being resumed, its resumption count climbs, and the
// livelock trip fires on a run that is merely early. See tt-emule docs/multi-rank-emulation.md.
void set_peer_liveness_probe(std::function<bool()> probe);

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

    // run_until_idle(), but a quiescence with a host-fed socket wait parked returns HostWait, fibers ALIVE.
    RunOutcome run_persistent();     // initial launch
    // Resume the SAME parked fibers one quantum; Completed here tears the registry down.
    RunOutcome pump();
    // True when a fresh peer-fed socket poll waiter is present.
    bool has_peer_fed_waiter() const;
    // Across a suspended gap the tier-2 watchdog stays up on TT_EMULE_HOST_WAIT_WATCHDOG_SEC, clocked at the park.

    // No-progress pumps before pump() escalates (TT_EMULE_HOST_WAIT_STALL_LIMIT, 8192); the drain sizes above it.
    static uint64_t host_wait_stall_limit();

    // Per-dispatch generation for runner keepalives. Per-generation, not an age bound: a parked run pins one.
    uint64_t begin_spawn_generation();
    uint64_t oldest_live_spawn_generation() const;
    bool spawn_generation_is_live(uint64_t gen) const;
    // Every live generation in ONE scan; per-candidate spawn_generation_is_live() rescans all_ each time.
    std::vector<uint64_t> live_spawn_generations() const;

    // Tear down a run that will not drain: throws FiberEngineStall(`why` + the parked-fiber dump).
    void abandon_host_wait(const std::string& why);

    // ---- Bridge ops (called by the runner's extern-C thunks from a running fiber) ----
    void lock();
    void unlock();
    void park_locked(const void* key);         // pre: lock held; post: lock released
    void park_locked_socket(const void* key);  // as park_locked, tagged as a host-fed socket wait
    // Tag a socket spin-poll (miss sets, hit clears); only host_fed can resolve a stall to HostWait.
    void note_socket_poll_wait(bool waiting, bool host_fed);
    // CB analogue: arms the stall check and names the wait, never host-resumable (peer producer). n==0 clears.
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
