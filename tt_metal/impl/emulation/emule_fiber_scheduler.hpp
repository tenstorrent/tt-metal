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

#include "jit_hw/internal/emule_thread_ctx.h"  // ThreadCommonCtx (the fiber-owned ctx)

namespace tt::tt_metal::emule_fiber {

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

    // ---- Bridge ops (called by the runner's extern-C thunks from a running fiber) ----
    void lock();
    void unlock();
    void park_locked(const void* key);  // pre: lock held; post: lock released
    void latency_park();                 // model NOC read latency; released at quiescence
    void wake(const void* key);
    void yield();
    void note_publish(unsigned pages);

    FiberScheduler(const FiberScheduler&) = delete;
    FiberScheduler& operator=(const FiberScheduler&) = delete;

private:
    FiberScheduler();
    ~FiberScheduler();
    std::unique_ptr<FiberSchedulerImpl> p_;
};

}  // namespace tt::tt_metal::emule_fiber
