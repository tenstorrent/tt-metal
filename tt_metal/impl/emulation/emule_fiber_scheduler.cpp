// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emule_fiber_scheduler.hpp"

#include "tt_emule/cb_sync_state.hpp"  // tt_emule::CBSyncState (sizeof, for the dump)

#include <ucontext.h>
#include <sys/mman.h>
#include <dlfcn.h>   // dladdr, for the parked-fiber op census
#include <cxxabi.h>  // demangle the op symbol
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <iterator>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Silicon-named per-RISC globals (read by unmodified upstream); defined in
// emulated_program_runner.cpp. The scheduler restores them on every swap-in
// since one worker hosts many fibers. See tt-emule docs/fiber-engine.md.
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];
// Blaze-only experimental firmware-global shim (issue #50953) — global-scope,
// unmangled names required by dlopen(-rdynamic) JIT-kernel symbol resolution.
extern thread_local uint8_t my_logical_x_;
extern thread_local uint8_t my_logical_y_;

namespace tt::tt_metal::emule_fiber {

namespace {

enum class FiberState : uint8_t { Ready, Running, Parked, QuiescenceDeferred, Done };

struct Fiber {
    ucontext_t ctx{};
    void* map_base = nullptr;   // mmap'd region (guard page + stack); munmap on destroy
    size_t map_bytes = 0;
    std::unique_ptr<ThreadCommonCtx> owned_ctx;
    std::function<void()> entry;
    FiberIdentity id{};
    FiberState state = FiberState::Ready;
    const void* park_key = nullptr;
    Fiber* park_link = nullptr;     // intrusive parked-list
    bool wait_is_socket = false;    // parked on a host-fed socket credit word (park_locked_socket):
                                    // marks a quiescence as "waiting for host I/O", not a deadlock
    bool in_ready = false;          // already in ready_[home]; unguarded, interleaved wakes park it twice
    // Poll tags: written only by this fiber's worker, read by peers and the watchdog, so all atomic.
    std::atomic<bool> socket_poll_waiting{false};  // spin-polling an empty socket. Ready, not parked
    std::atomic<bool> poll_is_host_fed{false};     // sender is the HOST (h2d); a d2d poll is never host-resumable
    // Refreshed unlocked by this fiber's worker; locking a hot spin loop would serialize the pool.
    std::atomic<uint64_t> own_resumes{0};      // times THIS fiber ran; freshness measures against it, not resumptions_
    std::atomic<uint64_t> poll_wait_stamp{0};  // own_resumes at last refresh; sticky, so staleness retires the tag
    std::atomic<bool> cb_poll_waiting{false};  // as above for cb_pages_available_at_front: peer-fed, never a host wait
    std::atomic<uint32_t> cb_poll_id{0};
    std::atomic<uint32_t> cb_poll_n{0};
    std::atomic<uint64_t> cb_poll_stamp{0};
    uint64_t spawn_gen = 0;  // dispatch generation spawned in; gates keepalive reclaim (oldest_live_spawn_generation)
    std::exception_ptr eptr;
    unsigned home = 0;              // pinned worker — a fiber NEVER migrates (the JIT kernel
                                    // caches the thread_local __emule_self address)

    ~Fiber() {
        if (map_base) {
            ::munmap(map_base, map_bytes);
        }
    }
};

// Per-worker state. Fibers are pinned (Fiber::home), so a fiber always runs on the
// same worker — these are read/written only by that worker.
thread_local ucontext_t t_sched;          // the worker loop's context (swap target)
thread_local Fiber*     t_current = nullptr;
thread_local struct FiberSchedulerImpl* t_impl = nullptr;
thread_local unsigned   t_worker = 0;      // this worker's index (== home of its fibers)

size_t env_size(const char* name, size_t dflt) {
    if (const char* s = std::getenv(name)) {
        char* end = nullptr;
        errno = 0;
        unsigned long long v = std::strtoull(s, &end, 10);
        // Reject out-of-range: callers derive bounds from these, and a saturated ULLONG_MAX wraps them.
        if (end != s && errno != ERANGE && v > 0) {
            return static_cast<size_t>(v);
        }
    }
    return dflt;
}

// Is a sticky poll tag current? Guarded: the watchdog samples both atomics unlocked, so resumes < stamp.
bool poll_tag_is_fresh(uint64_t resumes, uint64_t stamp, uint64_t staleness) {
    return resumes < stamp || (resumes - stamp) <= staleness;
}

}  // namespace

struct FiberSchedulerImpl {
    std::mutex mu_;
    std::condition_variable cv_;                       // workers wait here for ready fibers
    std::mutex wd_mu_;                                 // guards the watchdog's shutdown wait
    std::condition_variable wd_cv_;                    // wakes the watchdog promptly on teardown
    std::vector<std::deque<Fiber*>> ready_;            // per-worker ready queues (fibers are
                                                       // pinned: ready_[w] holds only home==w)
    std::unordered_map<const void*, Fiber*> parked_;   // key -> intrusive list head
    std::vector<Fiber*> quiescence_deferred_;          // fibers deferred to quiescence: re-queued
                                                       // at lowest priority, released only once the
                                                       // scheduler reaches quiescence (below)
    std::vector<std::unique_ptr<Fiber>> all_;          // ownership of every spawned fiber
    size_t launched_watermark_ = 0;                    // prefix of all_ integrated; only all_[watermark..] may enqueue

    unsigned K_ = 1;           // persistent pool size (read once at pool creation)
    unsigned W_ = 0;           // workers ACTIVE this program = min(K_, fiber count); only
                               // ready_[0..W_) is used, fibers home to [0..W_), surplus
                               // workers stay parked on start_cv_ (no per-fiber wakeups)
    unsigned workers_done_ = 0;// active workers that finished the current run (under mu_)
    unsigned idle_ = 0;        // workers waiting on cv_ (under mu_)
    unsigned running_ = 0;     // fibers currently executing on a worker (under mu_)
    unsigned active_ = 0;      // fibers not yet Done (under mu_)
    bool deadlock_ = false;
    bool abort_flag_ = false;
    bool persistent_ = false;  // run_persistent/pump in flight: a host-fed socket wait quiescing is
                               // a resumable HostWait, not a tier-1 deadlock. See run_persistent().
    bool host_wait_ = false;   // set by inner_loop when it broke out for host I/O (vs Done/deadlock)
    unsigned socket_poll_waiters_ = 0;   // fibers TAGGED spin-polling (under mu_). Sticky gate; freshness decides
    uint64_t poll_wait_staleness_ = 64;  // resumes a tag stays credible unrefreshed; per-fiber, so peers can't age it
    unsigned cb_poll_waiters_ = 0;       // as socket_poll_waiters_, but for CB probes (peer-fed)
    std::exception_ptr first_eptr_;

    std::atomic<uint64_t> progress_{0};      // fiber completions + published pages (tier 2)
    std::atomic<uint64_t> resumptions_{0};   // swap-ins (tier 2 livelock signal)

    // Raw-L1-store lost-wakeup recovery watermarks (guarded by mu_; per-run — reset
    // in run_until_idle alongside progress_/resumptions_). A kernel that advances a
    // same-core handshake word with a raw L1 store (`*ptr = v`, no noc_semaphore_set)
    // issues no __emule_fiber_wake, so a peer's noc_semaphore_wait can miss it. This
    // surfaces two ways: livelock while busy-waiters churn (spin-starvation release),
    // or true quiescence if every fiber parks at once (one-shot re-poll before the
    // tier-1 deadlock abort). Both wake all parked fibers to re-check predicates;
    // spurious-wake-safe (__emule_fiber_wait re-checks under lock), gated so healthy
    // runs never trigger. See tt-emule docs/fiber-engine.md.
    uint64_t last_progress_val_ = 0;
    uint64_t last_progress_resump_ = 0;
    // Force-releases that moved neither progress_ nor the fingerprint: not lost-wakeup victims.
    unsigned barren_releases_ = 0;
    uint64_t last_parked_sig_ = 0;
    bool last_parked_sig_valid_ = false;
    uint64_t last_deadlock_repoll_progress_ = UINT64_MAX;  // sentinel = never re-polled

    // Host-wait liveness bound (reset per persistent SEQUENCE in run_persistent, NOT per launch).
    // Counts consecutive pumps that returned HostWait having advanced nothing (progress_ == 0). A
    // wedged kernel whose socket the host never feeds would otherwise loop host<->pump forever with
    // no diagnostic; pump() escalates to a tier-1 deadlock after kHostWaitNoProgressLimit such pumps.
    // See tt-emule docs/socket-emulation.md.
    uint64_t host_wait_no_progress_pumps_ = 0;

    // Tier-2 watchdog. A HostWait return leaves it RUNNING (the gap's only guard); launch/teardown reaps.
    std::thread wd_;
    // Parked between pumps: HOST latency, so the watchdog swaps bounds and re-clocks at the park.
    std::atomic<bool> host_wait_parked_{false};

    // Stamped onto each spawned fiber; bumped by begin_spawn_generation() per dispatch register phase.
    uint64_t spawn_gen_ = 0;

    // Reason for the FiberEngineStall a teardown throws; empty = tier-1 quiescent deadlock.
    std::string stall_reason_;

    size_t stack_bytes_ = 1u << 20;          // 1 MB default

    // Persistent worker pool, created once and reused: threads block on start_cv_
    // between programs; run_until_idle bumps generation_ + notify_all to launch and
    // waits on done_cv_ for workers_done_ == W_. See tt-emule docs/fiber-engine.md.
    std::vector<std::thread> pool_;
    std::condition_variable start_cv_;       // pool waits here between programs
    std::condition_variable done_cv_;        // run_until_idle waits here for run completion
    uint64_t generation_ = 0;                // bumped per program (under mu_); workers detect a new run
    bool shutdown_ = false;                  // set by ~FiberScheduler to drain the pool

    void worker_main(unsigned w);            // persistent outer loop (one per pool thread)
    void inner_loop(unsigned w);             // per-program body; runs until active_==0 / abort
    void install_fiber(Fiber* f);
    bool any_ready() const {                 // any runnable fiber in any worker's queue?
        for (const auto& q : ready_) {
            if (!q.empty()) return true;
        }
        return false;
    }
    bool any_parked_is_socket_wait() const {  // any parked fiber blocked on a host-fed socket wait?
        for (const auto& kv : parked_) {      // cold path: only called at quiescence
            for (Fiber* f = kv.second; f; f = f->park_link) {
                if (f->wait_is_socket) return true;
            }
        }
        return false;
    }
    // The tag is sticky, so filter by freshness: one dead poller would hide every later deadlock.
    bool any_fresh_socket_poll_waiter() const {
        if (socket_poll_waiters_ == 0) {
            return false;
        }
        for (const auto& up : all_) {
            const Fiber* f = up.get();
            if (f->socket_poll_waiting.load(std::memory_order_relaxed) &&
                f->poll_is_host_fed.load(std::memory_order_relaxed) && f->state != FiberState::Done &&
                poll_tag_is_fresh(
                    f->own_resumes.load(std::memory_order_relaxed),
                    f->poll_wait_stamp.load(std::memory_order_relaxed),
                    poll_wait_staleness_)) {
                return true;
            }
        }
        return false;
    }
    bool any_waiting_on_host() const { return any_fresh_socket_poll_waiter() || any_parked_is_socket_wait(); }
    // Waker is a peer, not the host: a raw-NOC d2d publish runs no __emule_fiber_wake, hence the release.
    bool any_parked_non_socket() const {
        for (const auto& kv : parked_) {
            for (Fiber* f = kv.second; f; f = f->park_link) {
                if (!f->wait_is_socket) {
                    return true;
                }
            }
        }
        return false;
    }
    // "Who is parked on what": progress_ misses a recovery that only moves a raw L1 word. pre: mu_ held.
    uint64_t parked_signature() const {
        // Order-independent: every release refills parked_, so iteration order changes without motion.
        auto mix = [](uint64_t v) {
            v ^= v >> 33;
            v *= 0xff51afd7ed558ccdull;
            v ^= v >> 33;
            return v;
        };
        uint64_t sig = mix(parked_.size());
        for (const auto& kv : parked_) {
            const uint64_t k = reinterpret_cast<uintptr_t>(kv.first);
            for (Fiber* f = kv.second; f; f = f->park_link) {
                // Pair fiber with key: a move between two existing keys changes neither size nor key set.
                sig += mix(k ^ mix(reinterpret_cast<uintptr_t>(f)));
            }
        }
        return sig;
    }
    // Single idempotent funnel onto the pinned ready queue: a twice-queued fiber corrupts parked_.
    void enqueue_ready(Fiber* f) {
        // A queued fiber is always Ready; Parked, the write below would run it still linked into parked_.
        assert(!f->in_ready || f->state == FiberState::Ready);
        f->state = FiberState::Ready;
        if (f->in_ready) {
            return;
        }
        f->in_ready = true;
        ready_[f->home].push_back(f);
    }
    // Wake EVERY parked fiber to re-check its predicate; spurious-wake-safe. pre: mu_ held.
    void release_all_parked() {
        for (auto& kv : parked_) {
            Fiber* f = kv.second;
            while (f) {
                Fiber* nx = f->park_link;
                f->park_link = nullptr;
                f->park_key = nullptr;
                f->wait_is_socket = false;
                enqueue_ready(f);  // sets Ready; idempotent if already queued
                f = nx;
            }
        }
        parked_.clear();
    }
    // Release the quiescence-deferred set back to ready (spin release, or quiescence). pre: mu_ held.
    void release_quiescence_deferred() {
        for (Fiber* f : quiescence_deferred_) {
            enqueue_ready(f);
        }
        quiescence_deferred_.clear();
    }
    std::string dump_parked();               // single-threaded (post-join)
    void watchdog();                         // tier-2
    void stop_watchdog();                    // clear run_active_ + join wd_; no-op if not running
    // Retire this fiber's poll tags: they age on its own resumes, so an unresumed one stays fresh forever.
    void retire_poll_tags(Fiber* f) {
        if (f->socket_poll_waiting.load(std::memory_order_relaxed)) {
            f->socket_poll_waiting.store(false, std::memory_order_relaxed);
            f->poll_is_host_fed.store(false, std::memory_order_relaxed);
            --socket_poll_waiters_;
        }
        if (f->cb_poll_waiting.load(std::memory_order_relaxed)) {
            f->cb_poll_waiting.store(false, std::memory_order_relaxed);
            --cb_poll_waiters_;
        }
    }
    std::atomic<bool> run_active_{false};
};

// ---- the makecontext trampoline (first entry of a fiber) ----
static void fiber_trampoline() {
    Fiber* f = t_current;                    // set by the worker before swap-in
    FiberSchedulerImpl* impl = t_impl;
    try {
        f->entry();
    } catch (...) {
        f->eptr = std::current_exception();
    }
    impl->mu_.lock();                        // re-lock so the worker loop resumes mu_-held
    impl->retire_poll_tags(f);               // a fiber that exits mid-poll must not leak its tally
    f->state = FiberState::Done;
    swapcontext(&f->ctx, &t_sched);          // -> worker loop (never returns here)
}

void FiberSchedulerImpl::install_fiber(Fiber* f) {
    __emule_self = f->owned_ctx.get();       // the single thread_local repoint
    my_x[0] = my_x[1] = f->id.phys_x;        // restore the silicon-named coords
    my_y[0] = my_y[1] = f->id.phys_y;
    my_logical_x_ = static_cast<uint8_t>(f->id.logical_x);  // firmware LOGICAL coords (issue #50953)
    my_logical_y_ = static_cast<uint8_t>(f->id.logical_y);
    // Per-fiber ASAN state (e.g. the Object-Intent resolved-range log) lives in the ctx
    // above, so it swaps in with __emule_self — no separate restore needed here. See tt-emule #241.
}

void FiberSchedulerImpl::worker_main(unsigned w) {
    // Persistent worker: parks on start_cv_ between programs, participates only when
    // this program activated it (w < W_). Reuse is safe because all per-RISC state is
    // per-fiber or restored per swap-in. See tt-emule docs/fiber-engine.md.
    t_impl = this;
    t_worker = w;
    uint64_t seen = 0;
    mu_.lock();
    for (;;) {
        {
            std::unique_lock<std::mutex> wl(mu_, std::adopt_lock);
            start_cv_.wait(wl, [&] { return shutdown_ || generation_ != seen; });
            wl.release();                        // mu_ stays locked
        }
        if (shutdown_) {
            mu_.unlock();
            return;
        }
        seen = generation_;
        if (w < W_) {
            inner_loop(w);                       // enters + returns with mu_ held
            ++workers_done_;
            if (workers_done_ == W_) {
                done_cv_.notify_one();           // last active worker wakes run_until_idle
            }
        }
        // surplus workers (w >= W_) loop back to wait for the next generation
    }
}

// Per-program scheduling loop. Pre: mu_ held. Post: mu_ held. Runs this program's fibers to
// completion (active_ == 0) or to an abort (deadlock / kernel exception).
void FiberSchedulerImpl::inner_loop(unsigned w) {
    // Force-complete pending in-flight reads + wake all parked fibers once the
    // scheduler churns this many resumptions with zero progress. Below the tier-2
    // livelock backstop (TT_EMULE_FIBER_PROGRESS_WINDOW) and above healthy churn.
    static const uint64_t spin_release_window = env_size("TT_EMULE_SPIN_RELEASE_WINDOW", 4096);
    // Barren releases before HostWait stops deferring to them; each costs a full window, two overrun tier-2.
    static const uint64_t barren_release_limit = env_size("TT_EMULE_BARREN_RELEASE_LIMIT", 1);
    for (;;) {
        if (abort_flag_) break;
        // Spin-starvation release: a kernel busy-wait (do{invalidate_l1_cache();}while
        // (<raw L1 word>)) yields every iteration, so the ready queue never empties and
        // full quiescence is never reached — quiescence-deferred reads never "complete" and a
        // raw-store lost wakeup never re-checks. On churn past the window with zero
        // progress, force-complete the reads AND wake every parked fiber to re-poll its
        // predicate. Gated so healthy runs (progress advancing) never trigger.
        // socket_poll_waiters_ is in the guard: a poll-only program parks and defers nothing.
        if (!quiescence_deferred_.empty() || !parked_.empty() || socket_poll_waiters_ != 0 || cb_poll_waiters_ != 0) {
            uint64_t p = progress_.load(std::memory_order_relaxed);
            uint64_t r = resumptions_.load(std::memory_order_relaxed);
            if (p != last_progress_val_) {
                last_progress_val_ = p;
                last_progress_resump_ = r;
                barren_releases_ = 0;  // the last release (or normal execution) got somewhere
                last_parked_sig_valid_ = false;
            } else if (r - last_progress_resump_ > spin_release_window) {
                // Was the PREVIOUS release barren? Sample pre-release; afterwards parked_ says nothing.
                const uint64_t sig = parked_signature();
                if (last_parked_sig_valid_ && sig != last_parked_sig_) {
                    barren_releases_ = 0;  // something moved; the release earned another round
                }
                last_parked_sig_ = sig;
                last_parked_sig_valid_ = true;
                // Yield-spin HostWait trigger: all waits host-fed, or peer-parked but N releases moved nothing.
                if (persistent_ && any_waiting_on_host() &&
                    (!any_parked_non_socket() || barren_releases_ >= barren_release_limit)) {
                    host_wait_ = true;
                    abort_flag_ = true;
                    cv_.notify_all();
                    break;
                }
                release_quiescence_deferred();
                release_all_parked();
                cv_.notify_all();
                ++barren_releases_;         // provisional; cleared next round if anything moved
                last_progress_resump_ = r;  // re-arm; don't re-fire until more churn
            }
        }
        if (ready_[w].empty()) {
            if (active_ == 0) break;
            ++idle_;
            // Quiescence: nothing executing, nothing runnable in any queue. The
            // `!any_ready()` term is essential — `idle_ == W_` alone is a false positive
            // at W>1 (a worker counts itself idle before re-observing a just-enqueued
            // fiber). See tt-emule docs/fiber-engine.md.
            if (idle_ == W_ && running_ == 0 && !any_ready()) {
                // Quiescence-defer release: a deferred fiber was re-queued at lowest
                // priority to run only once every other runnable fiber has. We are at
                // that quiescence point now — release them all back to ready. Clients
                // use this to reproduce a silicon ordering (e.g. argmax's first read
                // barrier, or a two-producer cb_wait_front letting its co-producer run).
                // See tt-emule docs/fiber-engine.md.
                if (!quiescence_deferred_.empty()) {
                    release_quiescence_deferred();
                    cv_.notify_all();
                    --idle_;
                    continue;
                }
                // Fibers still sync-parked at quiescence. Before declaring a tier-1
                // deadlock, do a one-shot re-poll: wake every parked fiber so it
                // re-checks its __emule_fiber_wait predicate (recovers a raw-L1-store
                // lost wakeup when every fiber parked at once). Keyed off a progress
                // watermark: if progress advanced since the last re-poll (or this is
                // the first), the wake-all may unblock a waiter whose word a raw store
                // already set — try it; if a re-poll yields no new progress it is a
                // genuine deadlock. Spurious-wake-safe: __emule_fiber_wait re-checks
                // under the lock and re-parks. See tt-emule docs/fiber-engine.md.
                if (!parked_.empty()) {
                    uint64_t p = progress_.load(std::memory_order_relaxed);
                    if (p != last_deadlock_repoll_progress_) {
                        last_deadlock_repoll_progress_ = p;
                        release_all_parked();
                        cv_.notify_all();
                        --idle_;
                        continue;
                    }
                    // Re-poll made no new progress: genuinely stuck. Under a persistent run, a
                    // host-fed socket wait parked here is not a deadlock — it is a resumable
                    // HostWait: hand control back so the host can feed the socket and pump().
                    // With no socket wait parked, it is a real deadlock (diagnostics unchanged).
                    if (persistent_ && any_waiting_on_host()) {
                        host_wait_ = true;
                        abort_flag_ = true;
                        --idle_;
                        break;
                    }
                    deadlock_ = true;
                    abort_flag_ = true;
                    --idle_;
                    break;
                }
            }
            {
                std::unique_lock<std::mutex> wl(mu_, std::adopt_lock);
                cv_.wait(wl, [&] { return !ready_[w].empty() || active_ == 0 || abort_flag_; });
                wl.release();                // mu_ stays locked, back to manual management
            }
            --idle_;
            continue;
        }
        Fiber* f = ready_[w].front();
        ready_[w].pop_front();
        f->in_ready = false;  // off the queue; enqueue_ready may admit it again
        // A queue entry is a HINT: drop one no longer Ready, since the next wake re-enqueues it.
        if (f->state != FiberState::Ready) {
            continue;
        }
        f->state = FiberState::Running;
        ++running_;
        t_current = f;
        install_fiber(f);
        resumptions_.fetch_add(1, std::memory_order_relaxed);
        f->own_resumes.fetch_add(1, std::memory_order_relaxed);
        mu_.unlock();
        swapcontext(&t_sched, &f->ctx);      // run/resume f; returns with mu_ LOCKED
        --running_;
        if (f->state == FiberState::Done) {
            if (f->eptr && !first_eptr_) {
                first_eptr_ = f->eptr;
            }
            progress_.fetch_add(1, std::memory_order_relaxed);
            --active_;
        }
        // Parked/Ready already placed by park_locked/yield under mu_.
        t_current = nullptr;
        __emule_self = nullptr;
    }
    cv_.notify_all();                        // wake active peers to observe active_==0 / abort
    // returns with mu_ held — worker_main increments workers_done_ under it
}

// ---- bridge ops (called from a running fiber via the runner's extern-C thunks) ----

void FiberScheduler::lock() { p_->mu_.lock(); }
void FiberScheduler::unlock() { p_->mu_.unlock(); }

static void park_current(FiberSchedulerImpl* p, const void* key, bool is_socket) {
    // pre: mu_ held by this thread (the .so's __emule_fiber_lock). Register parked and
    // hand the lock to the worker loop across the switch. is_socket tags a host-fed socket
    // wait so quiescence-with-parked is treated as a resumable HostWait, not a deadlock.
    Fiber* f = t_current;
    f->state = FiberState::Parked;
    // Leaving Ready: drop the flag so a later wake re-enqueues; stale entries are discarded on pop.
    f->in_ready = false;
    p->retire_poll_tags(f);  // a parking fiber is no longer spin-polling
    f->park_key = key;
    f->wait_is_socket = is_socket;
    Fiber*& head = p->parked_[key];          // inserts nullptr if absent
    f->park_link = head;
    head = f;
    swapcontext(&f->ctx, &t_sched);          // -> worker loop (mu_ held); resumes mu_-UNLOCKED
}

void FiberScheduler::park_locked(const void* key) { park_current(p_.get(), key, /*is_socket=*/false); }

// Tag/untag a socket spin-poll. Edge-triggered: only transitions take mu_ to move the tally.
void FiberScheduler::note_socket_poll_wait(bool waiting, bool host_fed) {
    Fiber* f = t_current;
    if (!f) {
        return;
    }
    if (waiting) {
        // Refresh EVERY iteration, not just the transition: the stamp is what proves it is still looping.
        f->poll_wait_stamp.store(f->own_resumes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        f->poll_is_host_fed.store(host_fed, std::memory_order_relaxed);
    }
    if (f->socket_poll_waiting.load(std::memory_order_relaxed) == waiting) {
        return;
    }
    std::lock_guard<std::mutex> g(p_->mu_);
    f->socket_poll_waiting.store(waiting, std::memory_order_relaxed);
    if (waiting) {
        ++p_->socket_poll_waiters_;
    } else {
        f->poll_is_host_fed.store(false, std::memory_order_relaxed);
        --p_->socket_poll_waiters_;
    }
}

// Tag a CB spin-poll (n != 0) or clear it (n == 0). Never host-resumable: the producer is a peer.
void FiberScheduler::note_cb_poll_wait(unsigned cb_id, unsigned n) {
    Fiber* f = t_current;
    if (!f) {
        return;
    }
    const bool waiting = (n != 0);
    if (waiting) {
        // One slot per fiber: overwriting on every miss would name the CB probed LAST, not the starving one.
        const uint64_t resumes = f->own_resumes.load(std::memory_order_relaxed);
        const bool own_slot = !f->cb_poll_waiting.load(std::memory_order_relaxed) ||
                              f->cb_poll_id.load(std::memory_order_relaxed) == cb_id;
        const bool slot_stale =
            !poll_tag_is_fresh(resumes, f->cb_poll_stamp.load(std::memory_order_relaxed), p_->poll_wait_staleness_);
        if (own_slot || slot_stale) {
            // Atomic: a plain store lets the hang report print a torn cb id / page count never awaited.
            f->cb_poll_id.store(cb_id, std::memory_order_relaxed);
            f->cb_poll_n.store(n, std::memory_order_relaxed);
            f->cb_poll_stamp.store(resumes, std::memory_order_relaxed);
        }
    } else if (
        f->cb_poll_waiting.load(std::memory_order_relaxed) && f->cb_poll_id.load(std::memory_order_relaxed) != cb_id) {
        // A hit on a DIFFERENT CB must not clear: only the awaited one, or the stall check stays unarmed.
        return;
    }
    if (f->cb_poll_waiting.load(std::memory_order_relaxed) == waiting) {
        return;
    }
    std::lock_guard<std::mutex> g(p_->mu_);
    f->cb_poll_waiting.store(waiting, std::memory_order_relaxed);
    if (waiting) {
        ++p_->cb_poll_waiters_;
    } else {
        --p_->cb_poll_waiters_;
    }
}

// Same as park_locked, but tags the park as a host-fed socket wait (see park_current / inner_loop).
void FiberScheduler::park_locked_socket(const void* key) { park_current(p_.get(), key, /*is_socket=*/true); }

void FiberScheduler::quiescence_park() {
    // Defer the current fiber to scheduler quiescence: re-queue it at lowest priority,
    // released only once every other runnable fiber has run (see worker_loop). No key, no
    // predicate; takes mu_ itself and hands it to the worker loop across the switch (mirrors
    // park_locked). Clients use it to reproduce a silicon ordering (argmax's first read
    // barrier; a two-producer cb_wait_front letting its co-producer run).
    Fiber* f = t_current;
    if (!f) {
        return;  // only meaningful inside a fiber; insurance against a host-side call
    }
    p_->mu_.lock();
    f->state = FiberState::QuiescenceDeferred;
    f->in_ready = false;  // same as park_current: leaving Ready releases the queue flag
    // As in park_current: a deferred fiber is not resumed, so its tags would stop ageing.
    p_->retire_poll_tags(f);
    p_->quiescence_deferred_.push_back(f);
    swapcontext(&f->ctx, &t_sched);          // -> worker loop (mu_ held); resumes mu_-UNLOCKED
}

void FiberScheduler::wake(const void* key) {
    std::lock_guard<std::mutex> g(p_->mu_);
    auto it = p_->parked_.find(key);
    if (it == p_->parked_.end()) {
        return;
    }
    Fiber* f = it->second;
    while (f) {
        Fiber* nx = f->park_link;
        f->park_link = nullptr;
        f->park_key = nullptr;
        f->state = FiberState::Ready;
        p_->enqueue_ready(f);  // back to its pinned worker
        f = nx;
    }
    p_->parked_.erase(it);
    p_->cv_.notify_all();
}

void FiberScheduler::yield() {
    Fiber* f = t_current;
    p_->mu_.lock();
    f->state = FiberState::Ready;
    p_->enqueue_ready(f);  // back to its pinned worker (== this worker)
    p_->cv_.notify_all();
    swapcontext(&f->ctx, &t_sched);          // mu_ held -> worker loop; resumes mu_-UNLOCKED
}

void FiberScheduler::note_publish(unsigned pages) {
    p_->progress_.fetch_add(pages ? pages : 1, std::memory_order_relaxed);
}

// ---- register / run ----

void FiberScheduler::spawn(std::function<void()> entry, std::unique_ptr<ThreadCommonCtx> ctx,
                           const FiberIdentity& id) {
    auto f = std::make_unique<Fiber>();
    const size_t pg = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t usable = p_->stack_bytes_;
    const size_t total = usable + pg;
    void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        throw std::runtime_error("EMULE fiber: stack mmap failed");
    }
    if (mprotect(base, pg, PROT_NONE) != 0) {  // guard page at the low (overflow) end
        munmap(base, total);
        throw std::runtime_error("EMULE fiber: stack guard-page mprotect failed");
    }
    getcontext(&f->ctx);
    f->ctx.uc_stack.ss_sp = static_cast<char*>(base) + pg;
    f->ctx.uc_stack.ss_size = usable;
    f->ctx.uc_link = nullptr;                // we switch explicitly (trampoline swaps out)
    makecontext(&f->ctx, fiber_trampoline, 0);
    f->map_base = base;
    f->map_bytes = total;
    f->entry = std::move(entry);
    f->owned_ctx = std::move(ctx);
    f->id = id;
    f->state = FiberState::Ready;

    std::lock_guard<std::mutex> g(p_->mu_);
    f->spawn_gen = p_->spawn_gen_;      // stamp the dispatch generation (runner keepalive reclaim)
    p_->all_.push_back(std::move(f));   // home + ready-queue placement happens in run_until_idle
}

// Conservative stack scan naming the blaze op a parked fiber sits in.
// Walks the fiber's saved stack for words that dladdr resolves to a symbol, and
// returns the first demangled blaze::...::Op<...> frame. Env-gated
// (EMULE_PARK_STACKS=1) and best-effort: a false positive is a stale return address,
// never a crash (every deref is bounds-checked against the fiber's own mapping).
static std::string emule_park_op_name(const Fiber* f) {
    if (!f || !f->map_base || f->map_bytes == 0) {
        return {};
    }
    auto sp = static_cast<uintptr_t>(f->ctx.uc_mcontext.gregs[REG_RSP]);
    auto lo = reinterpret_cast<uintptr_t>(f->map_base);
    auto hi = lo + f->map_bytes;
    if (sp < lo || sp >= hi) {
        return {};
    }
    for (uintptr_t p = sp; p + sizeof(void*) <= hi; p += sizeof(void*)) {
        void* word = *reinterpret_cast<void* const*>(p);
        if (!word) {
            continue;
        }
        Dl_info info{};
        if (!dladdr(word, &info) || !info.dli_sname) {
            continue;
        }
        int status = 0;
        char* dem = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
        std::string name = (status == 0 && dem) ? dem : info.dli_sname;
        std::free(dem);
        if (name.find("blaze::") != std::string::npos && name.find("::Op<") != std::string::npos) {
            auto cut = name.find(">::");
            return cut == std::string::npos ? name : name.substr(0, cut + 1);
        }
    }
    return {};
}

uint64_t FiberScheduler::begin_spawn_generation() {
    std::lock_guard<std::mutex> g(p_->mu_);
    return ++p_->spawn_gen_;
}

uint64_t FiberScheduler::oldest_live_spawn_generation() const {
    std::lock_guard<std::mutex> g(p_->mu_);
    uint64_t oldest = UINT64_MAX;
    for (const auto& up : p_->all_) {
        if (up->state != FiberState::Done && up->spawn_gen < oldest) {
            oldest = up->spawn_gen;
        }
    }
    return oldest;
}

std::vector<uint64_t> FiberScheduler::live_spawn_generations() const {
    std::lock_guard<std::mutex> g(p_->mu_);
    std::vector<uint64_t> gens;
    for (const auto& up : p_->all_) {
        if (up->state != FiberState::Done) {
            gens.push_back(up->spawn_gen);
        }
    }
    std::sort(gens.begin(), gens.end());
    gens.erase(std::unique(gens.begin(), gens.end()), gens.end());
    return gens;
}

bool FiberScheduler::spawn_generation_is_live(uint64_t gen) const {
    std::lock_guard<std::mutex> g(p_->mu_);
    for (const auto& up : p_->all_) {
        if (up->state != FiberState::Done && up->spawn_gen == gen) {
            return true;
        }
    }
    return false;
}

void FiberScheduler::abandon_host_wait(const std::string& why) {
    {
        std::lock_guard<std::mutex> g(p_->mu_);
        if (p_->all_.empty()) {
            return;  // no registry to abandon
        }
        p_->deadlock_ = true;
        p_->stall_reason_ = why;
    }
    teardown_and_throw();
}

std::string FiberSchedulerImpl::dump_parked() {
    std::ostringstream os;
    os << "  " << parked_.size() << " distinct wait-key(s); parked fibers:\n";
    const bool park_stacks = std::getenv("EMULE_PARK_STACKS") != nullptr;
    for (auto& [key, head] : parked_) {
        if (park_stacks) {
            std::string op = emule_park_op_name(head);
            if (!op.empty()) {
                os << "    [op] " << op << "\n";
            }
        }
        for (Fiber* f = head; f; f = f->park_link) {
            os << "    core(log " << f->id.logical_x << "," << f->id.logical_y
               << " phys " << int(f->id.phys_x) << "," << int(f->id.phys_y) << ")"
               << " risc/proc " << int(f->id.proc_id);
            if (f->id.kernel_src) {
                os << " kernel " << f->id.kernel_src;
            }
            // Best-effort key naming: a CB if the key lands in this fiber's cbs[] array.
            const auto* ctx = f->owned_ctx.get();
            const char* name = nullptr;
            char buf[192];
            if (ctx && ctx->cbs) {
                auto base = reinterpret_cast<uintptr_t>(ctx->cbs);
                auto k = reinterpret_cast<uintptr_t>(key);
                if (k >= base && k < base + sizeof(tt_emule::CBSyncState) * __EMULE_CTX_MAX_CBS) {
                    const size_t cbid = (k - base) / sizeof(tt_emule::CBSyncState);
                    // Append the ASAN-recorded call site so a parked CB names the exact
                    // kernel line, not just the op. A producer (cb_reserve_back) and a
                    // consumer (cb_wait_front) park on the SAME address, so pick the side
                    // that is actually blocked: ASAN sets the dangling flag on entry, before
                    // the blocking wait, and clears it on the matching push/pop. Reporting
                    // the wait site unconditionally mislabels a stuck producer with a
                    // consumer line, usually a stale one from a call that already completed.
                    const bool res_blocked = ctx->san.cb_reserve_dangling[cbid];
                    const bool wait_blocked = ctx->san.cb_wait_dangling[cbid];
                    auto basename = [](const char* p) {
                        const char* s = std::strrchr(p, '/');
                        return s ? s + 1 : p;
                    };
                    // Both can be outstanding when one fiber is both producer and consumer
                    // of this CB; name both rather than guessing which one is stuck.
                    if (res_blocked && wait_blocked && ctx->san.cb_reserve_file[cbid] && ctx->san.cb_wait_file[cbid]) {
                        std::snprintf(buf, sizeof(buf), "CB %zu @ reserve %s:%u / wait %s:%u", cbid,
                                      basename(ctx->san.cb_reserve_file[cbid]), ctx->san.cb_reserve_line[cbid],
                                      basename(ctx->san.cb_wait_file[cbid]), ctx->san.cb_wait_line[cbid]);
                    } else if (res_blocked && ctx->san.cb_reserve_file[cbid]) {
                        std::snprintf(buf, sizeof(buf), "CB %zu @ reserve %s:%u", cbid,
                                      basename(ctx->san.cb_reserve_file[cbid]), ctx->san.cb_reserve_line[cbid]);
                    } else if (wait_blocked && ctx->san.cb_wait_file[cbid]) {
                        std::snprintf(buf, sizeof(buf), "CB %zu @ wait %s:%u", cbid,
                                      basename(ctx->san.cb_wait_file[cbid]), ctx->san.cb_wait_line[cbid]);
                    } else {
                        // Neither side outstanding: any recorded site is stale, so don't
                        // print one — a wrong line is worse than no line.
                        std::snprintf(buf, sizeof(buf), "CB %zu", cbid);
                    }
                    name = buf;
                }
            }
            if (!name && ctx && ctx->bridge_l1) {
                auto base = reinterpret_cast<uintptr_t>(ctx->bridge_l1);
                auto k = reinterpret_cast<uintptr_t>(key);
                if (k >= base) {
                    std::snprintf(buf, sizeof(buf), "L1 sem @ 0x%lx (cur=%u)",
                                  (unsigned long)(k - base),
                                  *reinterpret_cast<const volatile uint32_t*>(key));
                    name = buf;
                }
            }
            os << " waiting on " << (name ? name : "sync object") << " (key " << key << ")\n";
        }
    }
    if (!quiescence_deferred_.empty()) {
        os << "  " << quiescence_deferred_.size() << " fiber(s) deferred to quiescence\n";
    }
    if (socket_poll_waiters_ != 0) {
        // Split: a stale tag means the fiber left the loop, so reporting it sends triage after nothing.
        unsigned fresh = 0, stale = 0, peer = 0;
        for (const auto& up : all_) {
            const Fiber* f = up.get();
            if (!f->socket_poll_waiting.load(std::memory_order_relaxed) || f->state == FiberState::Done) {
                continue;
            }
            if (!f->poll_is_host_fed.load(std::memory_order_relaxed)) {
                ++peer;  // d2d: sender is a peer fiber, so this is a device-side wait
                continue;
            }
            (poll_tag_is_fresh(
                 f->own_resumes.load(std::memory_order_relaxed),
                 f->poll_wait_stamp.load(std::memory_order_relaxed),
                 poll_wait_staleness_)
                 ? fresh
                 : stale)++;
        }
        if (fresh != 0) {
            os << "  " << fresh << " fiber(s) spin-polling a host-fed socket with no data (awaiting host stream)\n";
        }
        if (stale != 0) {
            os << "  " << stale
               << " fiber(s) hold a stale host-socket poll tag (left the poll loop; NOT awaiting the host)\n";
        }
        if (peer != 0) {
            os << "  " << peer
               << " fiber(s) spin-polling a d2d socket (peer sender has not published; NOT a host wait)\n";
        }
    }
    if (cb_poll_waiters_ != 0) {
        // Never in parked_ (a CB probe spins), so without this a CB-stuck run dumps an empty list.
        for (const auto& up : all_) {
            const Fiber* f = up.get();
            if (!f->cb_poll_waiting.load(std::memory_order_relaxed) || f->state == FiberState::Done ||
                f->own_resumes.load(std::memory_order_relaxed) - f->cb_poll_stamp.load(std::memory_order_relaxed) >
                    poll_wait_staleness_) {
                continue;
            }
            os << "    core(log " << f->id.logical_x << "," << f->id.logical_y << " phys " << int(f->id.phys_x) << ","
               << int(f->id.phys_y) << ")"
               << " risc/proc " << int(f->id.proc_id);
            if (f->id.kernel_src) {
                os << " kernel " << f->id.kernel_src;
            }
            os << " spin-polling CB " << f->cb_poll_id.load(std::memory_order_relaxed) << " for "
               << f->cb_poll_n.load(std::memory_order_relaxed) << " page(s) (peer producer has not published)\n";
        }
    }
    return os.str();
}

// Stop and reap the tier-2 watchdog. Idempotent, and NOT under mu_: its abort path dumps under mu_.
void FiberSchedulerImpl::stop_watchdog() {
    host_wait_parked_.store(false, std::memory_order_release);
    if (!wd_.joinable()) {
        return;
    }
    {  // Clear under wd_mu_ + notify so the watchdog wakes at once (no lost wakeup).
        std::lock_guard<std::mutex> lk(wd_mu_);
        run_active_.store(false, std::memory_order_release);
    }
    wd_cv_.notify_all();
    wd_.join();
}

void FiberSchedulerImpl::watchdog() {
    const auto interval = std::chrono::milliseconds(250);
    const uint64_t window = env_size("TT_EMULE_FIBER_PROGRESS_WINDOW", 200000);
    const auto backstop = std::chrono::seconds(env_size("TT_EMULE_FIBER_WATCHDOG_SEC", 120));
    // Parked time measures the HOST; still finite, since a run nobody pumps must not read as success.
    const auto host_backstop = std::chrono::seconds(env_size("TT_EMULE_HOST_WAIT_WATCHDOG_SEC", 900));
    uint64_t last_progress = progress_.load();
    uint64_t last_resump = resumptions_.load();
    auto last_advance = std::chrono::steady_clock::now();
    bool was_parked = host_wait_parked_.load(std::memory_order_acquire);
    while (run_active_.load(std::memory_order_acquire)) {
        {
            // Interruptible sleep: wake immediately when run_until_idle clears run_active_
            // (set under wd_mu_, so the notify can't be lost), else time out after `interval`
            // to do the progress check. Avoids a ~`interval` join stall at every program end.
            std::unique_lock<std::mutex> lk(wd_mu_);
            if (wd_cv_.wait_for(lk, interval,
                                [this] { return !run_active_.load(std::memory_order_acquire); })) {
                break;
            }
        }
        const bool parked = host_wait_parked_.load(std::memory_order_acquire);
        if (parked != was_parked) {
            was_parked = parked;
            last_advance = std::chrono::steady_clock::now();  // the host gap starts (or ends) here
        }
        uint64_t p = progress_.load();
        uint64_t r = resumptions_.load();
        if (p != last_progress) {
            last_progress = p;
            last_resump = r;
            last_advance = std::chrono::steady_clock::now();
            continue;
        }
        // Fast livelock trip: many resumptions, zero progress. Never while parked (counters are frozen).
        bool livelock = !parked && (r - last_resump) > window;
        bool wall = (std::chrono::steady_clock::now() - last_advance) > (parked ? host_backstop : backstop);
        if (livelock || wall) {
            std::fprintf(
                stderr,
                "[EMULE] fiber engine: no global progress (%s) — suspected %s.\n%s",
                livelock ? "resumption window"
                : parked ? "host-wait backstop, TT_EMULE_HOST_WAIT_WATCHDOG_SEC"
                         : "wall-clock backstop",
                livelock ? "livelock / wake-cycle"
                : parked ? "the host never pumped this parked run again"
                         : "lost wakeup / hang",
                [this] {
                    std::lock_guard<std::mutex> g(mu_);
                    return dump_parked();
                }()
                    .c_str());
            std::abort();
        }
        last_resump = r;
    }
}

// Shared launch+wait: spawn/arm the workers, run one quantum on the pool, and block the dispatch
// thread until every active worker has finished (done_cv_). initial=true assigns homes from all_
// (a fresh program); initial=false (pump) reuses the existing homing — the caller has already put
// the fibers to resume back into ready_. On return the run has reached a boundary: every fiber Done,
// a deadlock, or (persistent) a host-wait. Teardown/throw is the caller's (teardown_and_throw).
void FiberScheduler::launch_and_wait(bool initial) {
    // Reap a watchdog left over a HostWait gap. Ahead of the early returns, so no exit leaves it ticking.
    p_->stop_watchdog();
    // Lazily create the persistent worker pool on the first run (K is process-constant);
    // threads live until ~FiberScheduler. See tt-emule docs/fiber-engine.md.
    if (p_->pool_.empty()) {
        p_->K_ = static_cast<unsigned>(env_size("TT_EMULE_FIBER_WORKERS", 64));
        p_->pool_.reserve(p_->K_);
        for (unsigned i = 0; i < p_->K_; ++i) {
            p_->pool_.emplace_back([this, i] { p_->worker_main(i); });
        }
    }

    unsigned W = 0;
    {   // Publish counters + progress, W_, the ready queues, and ++generation_ in ONE mu_ critical
        // section, so a worker released for a generation never pairs a new W_ with a stale generation_.
        // See tt-emule docs/fiber-engine.md §9.4 (the workers_done_ overshoot / done_cv_ wedge it avoids).
        std::lock_guard<std::mutex> g(p_->mu_);
        // `initial` means "a program was just registered", not "the registry is empty".
        const bool resuming = initial && p_->launched_watermark_ > 0;
        // Reset outcome flags BEFORE either early return: a stale host_wait_ reports a run that is gone.
        p_->idle_ = 0;
        p_->running_ = 0;
        p_->workers_done_ = 0;
        p_->deadlock_ = false;
        p_->abort_flag_ = false;
        p_->host_wait_ = false;
        if (initial && !resuming) {
            p_->active_ = static_cast<unsigned>(p_->all_.size());
            if (p_->active_ == 0) {
                return;   // nothing to run; the pool stays parked
            }
            // Clear the captured kernel exception only on a fresh run; a HostWait return bypasses
            // teardown_and_throw (its sole consumer), so it must survive across pump quanta.
            p_->first_eptr_ = nullptr;
        }
        // Per-run recovery watermarks — reset alongside progress_/resumptions_ (below).
        // A stale last_deadlock_repoll_progress_ from a prior run can collide with this
        // run's quiescence progress and skip the recovery re-poll → spurious deadlock.
        p_->last_progress_val_ = 0;
        p_->last_progress_resump_ = 0;
        p_->barren_releases_ = 0;
        p_->last_parked_sig_valid_ = false;
        p_->last_deadlock_repoll_progress_ = UINT64_MAX;
        if (initial && !resuming) {
            p_->quiescence_deferred_.clear();
            // Activate W = min(K, fiber count) workers; pin each fiber round-robin across [0,W).
            // Surplus workers (>= W) stay parked on start_cv_, so a tiny program pays no herd.
            // See tt-emule docs/fiber-engine.md.
            W = std::min<unsigned>(p_->K_, p_->active_);
            p_->W_ = W;
            p_->ready_.assign(W, {});  // discards the queues, so the in_ready flags go with them
            for (size_t i = 0; i < p_->all_.size(); ++i) {
                Fiber* f = p_->all_[i].get();
                f->home = static_cast<unsigned>(i % W);
                f->in_ready = false;
                p_->enqueue_ready(f);
            }
        } else if (resuming) {
            // Integrate ONLY the new fibers: re-queueing a Parked one parks it twice, re-homing breaks pinning.
            size_t retired = 0;
            {
                for (auto& q : p_->ready_) {
                    q.erase(
                        std::remove_if(q.begin(), q.end(), [](Fiber* f) { return f->state == FiberState::Done; }),
                        q.end());
                }
                auto is_done = [](const std::unique_ptr<Fiber>& f) { return f->state == FiberState::Done; };
                const size_t before = p_->all_.size();
                // Only the launched prefix is eligible; [watermark, size) has not run and is not Done.
                auto prefix_end = p_->all_.begin() + static_cast<std::ptrdiff_t>(p_->launched_watermark_);
                auto keep_end = std::remove_if(p_->all_.begin(), prefix_end, is_done);
                retired = static_cast<size_t>(std::distance(keep_end, prefix_end));
                if (retired != 0) {
                    // Close the gap: remove_if already compacted the prefix, so shift the new gen down.
                    std::move(prefix_end, p_->all_.end(), keep_end);
                    p_->all_.resize(before - retired);
                    p_->launched_watermark_ -= retired;
                }
            }
            const size_t n_new = p_->all_.size() - p_->launched_watermark_;
            W = std::min<unsigned>(p_->K_, p_->active_ + static_cast<unsigned>(n_new));
            if (W < p_->W_) {
                W = p_->W_;  // never shrink: live fibers are pinned to homes in [0, W_)
            }
            p_->W_ = W;
            p_->ready_.resize(W);
            for (size_t i = p_->launched_watermark_; i < p_->all_.size(); ++i) {
                Fiber* f = p_->all_[i].get();
                f->home = static_cast<unsigned>(i % W);
                p_->enqueue_ready(f);
                ++p_->active_;
            }
            if (p_->active_ == 0) {
                return;  // nothing live and nothing new
            }
        } else {
            W = p_->W_;   // pump: reuse homing; ready_ already refilled by pump()'s re-poll
        }
        // Reset run counters (a fresh run and each pump re-poll both start from zero, matching the
        // per-run watermarks above).
        p_->progress_.store(0);
        p_->resumptions_.store(0);
        ++p_->generation_;
        if (initial) {
            p_->launched_watermark_ = p_->all_.size();
        }
    }
    if (std::getenv("TT_EMULE_FIBER_LOG_N")) {
        std::fprintf(stderr, "[EMULE FIBER] program: %u fibers on W=%u of K=%u workers\n",
                     p_->active_, W, p_->K_);
    }

    // Watchdog before notify_all so it covers the run; created after the dispatch block so a throw
    // there can't leave a joinable thread. (A spurious start_cv_ wakeup could start a worker in the
    // brief gap before this line — benign: the watchdog spawns at once and a hang can't complete that fast.)
    p_->run_active_.store(true, std::memory_order_release);
    p_->wd_ = std::thread([this] { p_->watchdog(); });

    p_->start_cv_.notify_all();
    {   // Block the dispatch thread until every active worker has finished this run.
        std::unique_lock<std::mutex> lk(p_->mu_);
        p_->done_cv_.wait(lk, [&] { return p_->workers_done_ == W; });
    }

    bool host_wait = false;
    {  // A captured kernel fault outranks a host wait: HostWait would defer the rethrow indefinitely.
        std::lock_guard<std::mutex> g(p_->mu_);
        if (p_->first_eptr_ && p_->host_wait_) {
            p_->host_wait_ = false;
        }
        host_wait = p_->host_wait_;
    }
    // A HostWait return leaves the watchdog RUNNING (the gap's only guard); anything else ends the run.
    p_->host_wait_parked_.store(host_wait, std::memory_order_release);
    if (!host_wait) {
        p_->stop_watchdog();
    }
    if (std::getenv("TT_EMULE_FIBER_LOG_N")) {
        std::lock_guard<std::mutex> g(p_->mu_);
        std::fprintf(
            stderr,
            "[EMULE FIBER] %s -> %s: active=%u parked_keys=%zu poll_waiters=%u progress=%llu\n",
            initial ? "launch" : "pump",
            p_->host_wait_ ? "HostWait" : (p_->deadlock_ ? "DEADLOCK" : "Completed"),
            p_->active_,
            p_->parked_.size(),
            p_->socket_poll_waiters_,
            (unsigned long long)p_->progress_.load());
    }
}

// Collect results + clear the registry for the next program / mesh. Rethrows the first fiber
// exception; throws on a quiescent deadlock. Called only after a run reaches Completed (never on a
// resumable HostWait — then the fibers must stay alive).
void FiberScheduler::teardown_and_throw() {
    // Before mu_: the watchdog may still be up, and its abort path dumps under mu_.
    p_->stop_watchdog();
    std::exception_ptr eptr;
    bool deadlock;
    std::string dump;
    std::string reason;
    {   // workers are parked on start_cv_ now, but take mu_ anyway for clean ordering.
        std::lock_guard<std::mutex> g(p_->mu_);
        eptr = p_->first_eptr_;
        // Consume it: latched, the resuming path (which clears only on a fresh launch) rethrows it later.
        p_->first_eptr_ = nullptr;
        deadlock = p_->deadlock_;
        if (deadlock) {
            dump = p_->dump_parked();
            reason = p_->stall_reason_.empty() ? "EMULE fiber engine: quiescent deadlock — all workers idle, fibers "
                                                 "parked, none runnable."
                                               : p_->stall_reason_;
        }
        p_->stall_reason_.clear();
        p_->ready_.clear();
        p_->parked_.clear();
        p_->quiescence_deferred_.clear();
        p_->socket_poll_waiters_ = 0;  // tallies are per-registry; all_ is about to go
        p_->cb_poll_waiters_ = 0;
        p_->all_.clear();   // frees Fiber stacks via ~Fiber
        p_->launched_watermark_ = 0;  // registry is empty again: the next launch is a fresh one
        p_->host_wait_ = false;       // per-registry, and the registry is gone
    }

    // A real kernel exception is the root cause; report it before any deadlock symptom.
    if (eptr) {
        std::rethrow_exception(eptr);
    }
    if (deadlock) {
        throw FiberEngineStall(reason + "\n" + dump);
    }
}

void FiberScheduler::run_until_idle() {
    p_->persistent_ = false;   // non-persistent: a quiescence-with-parked is a tier-1 deadlock
    launch_and_wait(/*initial=*/true);
    teardown_and_throw();
}

// The engine's own bound, read once; the drain backstop reads it back rather than re-parsing the env.
static uint64_t host_wait_no_progress_limit() {
    static const uint64_t v = env_size("TT_EMULE_HOST_WAIT_STALL_LIMIT", 8192);
    return v;
}

uint64_t FiberScheduler::host_wait_stall_limit() { return host_wait_no_progress_limit(); }

RunOutcome FiberScheduler::run_persistent() {
    p_->persistent_ = true;
    // Reset only for a FRESH sequence: re-arming per dispatch means a wedged run never escalates.
    const bool resuming = p_->launched_watermark_ > 0;
    if (!resuming) {
        p_->host_wait_no_progress_pumps_ = 0;
    }
    launch_and_wait(/*initial=*/true);
    return finish_or_host_wait();
}

// Shared tail of run_persistent()/pump(); pump()'s no-progress accounting must run BEFORE it.
RunOutcome FiberScheduler::finish_or_host_wait() {
    if (p_->host_wait_) {
        return RunOutcome::HostWait;   // fibers parked awaiting host socket I/O — ALIVE, no teardown
    }
    teardown_and_throw();
    return RunOutcome::Completed;
}

RunOutcome FiberScheduler::pump() {
    p_->persistent_ = true;
    {
        std::lock_guard<std::mutex> g(p_->mu_);
        if (p_->all_.empty()) {
            return RunOutcome::Completed;   // no persistent run in flight — no-op
        }
        // The host's credit store was a raw L1 write with no wake, so re-poll every parked fiber.
        p_->release_all_parked();
    }
    launch_and_wait(/*initial=*/false);
    if (p_->host_wait_) {
        // Liveness bound: a pump that advanced nothing (progress_ == 0) while still parked on a socket
        // wait is a no-progress cycle. If the host keeps pumping a wedged kernel whose socket never
        // advances, escalate to the tier-1 deadlock after kHostWaitNoProgressLimit consecutive
        // no-progress pumps instead of looping host<->pump forever with no diagnostic. Any forward
        // progress resets the count, so a slow-but-advancing feed is unaffected. Keying on global
        // progress_ (not the awaited socket) leaves a residual masking window — WA-3, see
        // tt-emule-blaze/.claude/skills/workarounds.
        if (p_->progress_.load() == 0) {
            if (++p_->host_wait_no_progress_pumps_ >= host_wait_no_progress_limit()) {
                p_->deadlock_ = true;
                teardown_and_throw();  // reuses the tier-1 quiescent-deadlock diagnostic (dump_parked)
            }
        } else {
            p_->host_wait_no_progress_pumps_ = 0;
        }
    }
    return finish_or_host_wait();
}

FiberScheduler::FiberScheduler() : p_(std::make_unique<FiberSchedulerImpl>()) {
    // Read the fiber stack size once, at construction: spawn() consumes stack_bytes_ to size each
    // fiber's mmap'd stack, and spawn() always runs before run_until_idle (single-device launch spawns
    // then runs; the mesh path spawns all devices in the defer phase, then runs once). Reading it in
    // run_until_idle would miss the first program. See tt-emule docs/fiber-engine.md.
    p_->stack_bytes_ = env_size("TT_EMULE_FIBER_STACK_BYTES", 1u << 20);
    p_->poll_wait_staleness_ = env_size("TT_EMULE_POLL_TAG_STALENESS", 64);
}

FiberScheduler::~FiberScheduler() {
    if (p_) {
        p_->stop_watchdog();  // a HostWait gap watchdog can still be up if the run never completed
    }
    if (p_ && !p_->pool_.empty()) {
        {
            std::lock_guard<std::mutex> g(p_->mu_);
            p_->shutdown_ = true;
        }
        p_->start_cv_.notify_all();
        for (auto& t : p_->pool_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }
}

FiberScheduler& FiberScheduler::instance() {
    static FiberScheduler s;
    return s;
}

}  // namespace tt::tt_metal::emule_fiber
