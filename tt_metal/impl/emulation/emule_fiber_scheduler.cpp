// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emule_fiber_scheduler.hpp"

#include "tt_emule/cb_sync_state.hpp"  // tt_emule::CBSyncState (sizeof, for the dump)

#include <ucontext.h>
#include <sys/mman.h>
#include <unistd.h>
#include <execinfo.h>  // backtrace() — busy-wait call-stack capture (TT_EMULE_TRACE_BUSYWAIT)
#include <dlfcn.h>     // dladdr()

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
    std::exception_ptr eptr;
    unsigned home = 0;              // pinned worker — a fiber NEVER migrates (the JIT kernel
                                    // caches the thread_local __emule_self address)
    uint64_t yields = 0;            // [SCHEDSTATE] cumulative yield() count — a large value on a
                                    // still-Running fiber marks a busy yield-spinner (vs a stuck non-yielder)

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
        unsigned long long v = std::strtoull(s, &end, 10);
        if (end != s && v > 0) return static_cast<size_t>(v);
    }
    return dflt;
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

    // Low-perturbation occ-wipe probe (TT_EMULE_TRACE_WAKECOUNT). Per wake-key: total
    // wake() calls over the run, and how many of those found occupied==0 at wake time
    // (a reset/pop wipe rather than a push). Dumped per parked key at the deadlock to
    // split "CB never produced" (total==0) from "produced then wiped" (total>0, occ=0).
    std::unordered_map<const void*, uint64_t> wake_total_;
    std::unordered_map<const void*, uint64_t> wake_occzero_;

    unsigned K_ = 1;           // persistent pool size (read once at pool creation)
    unsigned W_ = 0;           // workers ACTIVE this program = min(K_, fiber count); only
                               // ready_[0..W_) is used, fibers home to [0..W_), surplus
                               // workers stay parked on start_cv_ (no per-fiber wakeups)
    unsigned workers_done_ = 0;// active workers that finished the current run (under mu_)
    unsigned idle_ = 0;        // workers waiting on cv_ (under mu_)
    unsigned running_ = 0;     // fibers currently executing on a worker (under mu_)
    unsigned active_ = 0;      // fibers not yet Done (under mu_)
    std::vector<Fiber*> running_fiber_;  // [SCHEDSTATE] per-worker currently-running fiber (nullptr = none)
    bool deadlock_ = false;
    bool abort_flag_ = false;
    bool persistent_ = false;  // run_persistent/pump in flight: a host-fed socket wait quiescing is
                               // a resumable HostWait, not a tier-1 deadlock. See run_persistent().
    bool host_wait_ = false;   // set by inner_loop when it broke out for host I/O (vs Done/deadlock)
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
    uint64_t last_deadlock_repoll_progress_ = UINT64_MAX;  // sentinel = never re-polled

    // Host-wait liveness bound (reset per persistent SEQUENCE in run_persistent, NOT per launch).
    // Counts consecutive pumps that returned HostWait having advanced nothing (progress_ == 0). A
    // wedged kernel whose socket the host never feeds would otherwise loop host<->pump forever with
    // no diagnostic; pump() escalates to a tier-1 deadlock after kHostWaitNoProgressLimit such pumps.
    // See tt-emule docs/socket-emulation.md.
    uint64_t host_wait_no_progress_pumps_ = 0;

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
    std::string dump_parked();               // single-threaded (post-join)
    std::string dump_all();                  // full fiber census incl Done (TT_EMULE_TRACE_CENSUS)
    void watchdog();                         // tier-2
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
    f->state = FiberState::Done;
    swapcontext(&f->ctx, &t_sched);          // -> worker loop (never returns here)
}

void FiberSchedulerImpl::install_fiber(Fiber* f) {
    __emule_self = f->owned_ctx.get();       // the single thread_local repoint
    my_x[0] = my_x[1] = f->id.phys_x;        // restore the silicon-named coords
    my_y[0] = my_y[1] = f->id.phys_y;
    my_logical_x_ = static_cast<uint8_t>(f->id.logical_x);  // firmware LOGICAL coords
    my_logical_y_ = static_cast<uint8_t>(f->id.logical_y);
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
    for (;;) {
        if (abort_flag_) break;
        // Spin-starvation release: a kernel busy-wait (do{invalidate_l1_cache();}while
        // (<raw L1 word>)) yields every iteration, so the ready queue never empties and
        // full quiescence is never reached — quiescence-deferred reads never "complete" and a
        // raw-store lost wakeup never re-checks. On churn past the window with zero
        // progress, force-complete the reads AND wake every parked fiber to re-poll its
        // predicate. Gated so healthy runs (progress advancing) never trigger.
        if (!quiescence_deferred_.empty() || !parked_.empty()) {
            uint64_t p = progress_.load(std::memory_order_relaxed);
            uint64_t r = resumptions_.load(std::memory_order_relaxed);
            if (p != last_progress_val_) {
                last_progress_val_ = p;
                last_progress_resump_ = r;
            } else if (r - last_progress_resump_ > spin_release_window) {
                // Persistent (host-interleaved) run: churn with zero progress while a host-fed socket
                // wait is parked means the device is blocked on the host (a peer RISC yield-spins on a
                // barrier whose other side awaits a socket token). Hand control back so the host can
                // stream + pump(), rather than force-releasing (which only re-churns). This is the
                // yield-spin HostWait trigger (vs the quiescence-parked one below). See run_persistent.
                if (persistent_ && any_parked_is_socket_wait()) {
                    host_wait_ = true;
                    abort_flag_ = true;
                    cv_.notify_all();
                    break;
                }
                for (Fiber* f : quiescence_deferred_) {
                    f->state = FiberState::Ready;
                    ready_[f->home].push_back(f);
                }
                quiescence_deferred_.clear();
                for (auto& kv : parked_) {
                    Fiber* f = kv.second;
                    while (f) {
                        Fiber* nx = f->park_link;
                        f->park_link = nullptr;
                        f->park_key = nullptr;
                        f->state = FiberState::Ready;
                        ready_[f->home].push_back(f);
                        f = nx;
                    }
                }
                parked_.clear();
                cv_.notify_all();
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
                    for (Fiber* f : quiescence_deferred_) {
                        f->state = FiberState::Ready;
                        ready_[f->home].push_back(f);
                    }
                    quiescence_deferred_.clear();
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
                        for (auto& kv : parked_) {
                            Fiber* f = kv.second;
                            while (f) {
                                Fiber* nx = f->park_link;
                                f->park_link = nullptr;
                                f->park_key = nullptr;
                                f->state = FiberState::Ready;
                                ready_[f->home].push_back(f);
                                f = nx;
                            }
                        }
                        parked_.clear();
                        cv_.notify_all();
                        --idle_;
                        continue;
                    }
                    // Re-poll made no new progress: genuinely stuck. Under a persistent run, a
                    // host-fed socket wait parked here is not a deadlock — it is a resumable
                    // HostWait: hand control back so the host can feed the socket and pump().
                    // With no socket wait parked, it is a real deadlock (diagnostics unchanged).
                    if (persistent_ && any_parked_is_socket_wait()) {
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
        f->state = FiberState::Running;
        ++running_;
        t_current = f;
        running_fiber_[w] = f;               // [SCHEDSTATE] record the fiber this worker runs
        install_fiber(f);
        resumptions_.fetch_add(1, std::memory_order_relaxed);
        mu_.unlock();
        swapcontext(&t_sched, &f->ctx);      // run/resume f; returns with mu_ LOCKED
        --running_;
        running_fiber_[w] = nullptr;         // [SCHEDSTATE] worker no longer running that fiber
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
    f->park_key = key;
    f->wait_is_socket = is_socket;
    Fiber*& head = p->parked_[key];          // inserts nullptr if absent
    f->park_link = head;
    head = f;
    swapcontext(&f->ctx, &t_sched);          // -> worker loop (mu_ held); resumes mu_-UNLOCKED
}

void FiberScheduler::park_locked(const void* key) { park_current(p_.get(), key, /*is_socket=*/false); }

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
    p_->quiescence_deferred_.push_back(f);
    swapcontext(&f->ctx, &t_sched);          // -> worker loop (mu_ held); resumes mu_-UNLOCKED
}

void FiberScheduler::wake(const void* key) {
    std::lock_guard<std::mutex> g(p_->mu_);
    auto it = p_->parked_.find(key);
    static const bool trace_wc = std::getenv("TT_EMULE_TRACE_WAKECOUNT") != nullptr;
    if (trace_wc) {
        p_->wake_total_[key]++;
        // Classify occ-at-wake only when a parked waiter proves the key is a CB in its
        // cbs[] range — memory-safe (never deref an arbitrary sem key as a CBSyncState).
        if (it != p_->parked_.end() && it->second) {
            const auto* ctx = it->second->owned_ctx.get();
            if (ctx && ctx->cbs) {
                auto base = reinterpret_cast<uintptr_t>(ctx->cbs);
                auto k = reinterpret_cast<uintptr_t>(key);
                if (k >= base && k < base + sizeof(tt_emule::CBSyncState) * __EMULE_CTX_MAX_CBS) {
                    if (reinterpret_cast<const tt_emule::CBSyncState*>(key)->occupied.load(
                            std::memory_order_acquire) == 0) {
                        p_->wake_occzero_[key]++;
                    }
                }
            }
        }
    }
    if (it == p_->parked_.end()) {
        return;
    }
    Fiber* f = it->second;
    while (f) {
        Fiber* nx = f->park_link;
        f->park_link = nullptr;
        f->park_key = nullptr;
        f->state = FiberState::Ready;
        p_->ready_[f->home].push_back(f);    // back to its pinned worker
        f = nx;
    }
    p_->parked_.erase(it);
    p_->cv_.notify_all();
}

void FiberScheduler::yield() {
    Fiber* f = t_current;
    // Busy-wait call-stack probe: a raw `while(*ptr) invalidate_l1_cache()` kernel loop yields
    // here every iteration (never parks). Sample the fiber's backtrace on cores (9,8)/(10,8) —
    // the deadlock busy-spin runs the whole watchdog window, so it dominates the samples. Map the
    // JIT-.so frame offsets with addr2line to find the exact busy-loop. TT_EMULE_TRACE_BUSYWAIT.
    static const bool trace_bw = std::getenv("TT_EMULE_TRACE_BUSYWAIT") != nullptr;
    if (trace_bw && f &&
        ((f->id.logical_x == 9 && f->id.logical_y == 8) || (f->id.logical_x == 10 && f->id.logical_y == 8))) {
        static std::atomic<uint64_t> ctr{0};
        static const uint64_t period = env_size("TT_EMULE_BUSYWAIT_PERIOD", 500000);
        if (ctr.fetch_add(1, std::memory_order_relaxed) % period == 0) {
            void* bt[20];
            int n = backtrace(bt, 20);
            const auto* ctx = f->owned_ctx.get();
            std::fprintf(stderr, "[BUSYWAIT] dev %d core(%d,%d) proc %d kind=%s\n",
                         ctx ? int(ctx->chip_id) : -1, int(f->id.logical_x), int(f->id.logical_y),
                         int(f->id.proc_id),
                         (ctx && ctx->kind == ThreadCommonCtx::Kind::Compute) ? "CMP" : "DM");
            for (int i = 0; i < n; ++i) {
                Dl_info di;
                if (dladdr(bt[i], &di) && di.dli_fname) {
                    std::fprintf(stderr, "    #%d %p %s+0x%lx  [%s]\n", i, bt[i],
                                 di.dli_sname ? di.dli_sname : "?",
                                 di.dli_saddr ? (unsigned long)((char*)bt[i] - (char*)di.dli_saddr) : 0ul,
                                 di.dli_fname);
                } else {
                    std::fprintf(stderr, "    #%d %p ?\n", i, bt[i]);
                }
            }
        }
    }
    p_->mu_.lock();
    f->yields++;                             // [SCHEDSTATE] spinner detector
    f->state = FiberState::Ready;
    p_->ready_[f->home].push_back(f);        // back to its pinned worker (== this worker)
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
    p_->all_.push_back(std::move(f));   // home + ready-queue placement happens in run_until_idle
}

std::string FiberSchedulerImpl::dump_parked() {
    std::ostringstream os;
    os << "  " << parked_.size() << " distinct wait-key(s); parked fibers:\n";
    for (auto& [key, head] : parked_) {
        for (Fiber* f = head; f; f = f->park_link) {
            const auto* ctx = f->owned_ctx.get();
            os << "    dev " << (ctx ? int(ctx->chip_id) : -1)
               << " core(log " << f->id.logical_x << "," << f->id.logical_y
               << " phys " << int(f->id.phys_x) << "," << int(f->id.phys_y) << ")"
               << " risc/proc " << int(f->id.proc_id)
               << " kind=" << (ctx && ctx->kind == ThreadCommonCtx::Kind::Compute ? "CMP" : "DM");
            if (f->id.kernel_src) {
                os << " kernel " << f->id.kernel_src;
            }
            // Best-effort key naming: a CB if the key lands in this fiber's cbs[] array.
            const char* name = nullptr;
            char buf[224];
            if (ctx && ctx->cbs) {
                auto base = reinterpret_cast<uintptr_t>(ctx->cbs);
                auto k = reinterpret_cast<uintptr_t>(key);
                if (k >= base && k < base + sizeof(tt_emule::CBSyncState) * __EMULE_CTX_MAX_CBS) {
                    const auto* cb = reinterpret_cast<const tt_emule::CBSyncState*>(key);
                    // [CBPROD] resolve the CB's registered producer ctx -> its core/proc (who fills it).
                    const void* prod = cb->producer.load(std::memory_order_acquire);
                    bool multi = cb->multi_producer.load(std::memory_order_acquire);
                    int pdev = -1, plx = -1, ply = -1, pproc = -1;
                    const char* pkind = "?";
                    if (prod) {
                        for (auto& up : all_) {
                            if (up && up->owned_ctx.get() == prod) {
                                pdev = int(up->owned_ctx->chip_id);
                                plx = up->id.logical_x; ply = up->id.logical_y; pproc = int(up->id.proc_id);
                                pkind = (up->owned_ctx->kind == ThreadCommonCtx::Kind::Compute) ? "CMP" : "DM";
                                break;
                            }
                        }
                    }
                    std::snprintf(buf, sizeof(buf),
                                  "CB %zu occ=%u rcv_cmp=%u ackd=%u npages=%u prod=dev%d(log%d,%d)p%d/%s%s",
                                  (k - base) / sizeof(tt_emule::CBSyncState),
                                  cb->occupied.load(std::memory_order_acquire),
                                  cb->received_compute.load(std::memory_order_acquire),
                                  cb->acked.load(std::memory_order_acquire),
                                  cb->num_pages, pdev, plx, ply, pproc, pkind, multi ? "[multi]" : "");
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
            os << " waiting on " << (name ? name : "sync object") << " (key " << key << ")";
            auto wt = wake_total_.find(key);
            if (wt != wake_total_.end()) {
                auto wz = wake_occzero_.find(key);
                os << " [wakes=" << wt->second
                   << " occ0-wakes=" << (wz != wake_occzero_.end() ? wz->second : 0) << "]";
            } else if (!wake_total_.empty()) {
                os << " [wakes=0]";  // probe active, this key never signaled → never produced
            }
            os << "\n";
        }
    }
    if (!quiescence_deferred_.empty()) {
        os << "  " << quiescence_deferred_.size() << " fiber(s) deferred to quiescence\n";
    }
    return os.str();
}

// Full census of EVERY spawned fiber (Ready/Running/Parked/QuiescenceDeferred/Done), not just
// the parked ones dump_parked() shows. The hang investigation needs to see a fiber that
// FINISHED without producing (invisible in dump_parked). Single-threaded (post-join / at abort).
std::string FiberSchedulerImpl::dump_all() {
    static const char* const kState[] = {"Ready", "Running", "Parked", "QuiescenceDeferred", "Done"};
    std::ostringstream os;
    os << "  fiber census (" << all_.size() << " fibers):\n";
    for (auto& up : all_) {
        Fiber* f = up.get();
        if (!f) continue;
        const auto* ctx = f->owned_ctx.get();
        const bool is_cmp = ctx && ctx->kind == ThreadCommonCtx::Kind::Compute;
        os << "    [CENSUS] dev " << (ctx ? int(ctx->chip_id) : -1)
           << " core(log " << f->id.logical_x << "," << f->id.logical_y
           << " phys " << int(f->id.phys_x) << "," << int(f->id.phys_y) << ")"
           << " proc " << int(f->id.proc_id)
           << " kind=" << (is_cmp ? "CMP" : "DM")
           << " state=" << kState[static_cast<int>(f->state)];
        if (f->state == FiberState::Parked && f->park_key) {
            const void* key = f->park_key;
            const char* name = nullptr;
            char buf[80];
            if (ctx && ctx->cbs) {
                auto base = reinterpret_cast<uintptr_t>(ctx->cbs);
                auto k = reinterpret_cast<uintptr_t>(key);
                if (k >= base && k < base + sizeof(tt_emule::CBSyncState) * __EMULE_CTX_MAX_CBS) {
                    const auto* cb = reinterpret_cast<const tt_emule::CBSyncState*>(key);
                    std::snprintf(buf, sizeof(buf), "CB %zu occ=%u rcv_cmp=%u npages=%u",
                                  (k - base) / sizeof(tt_emule::CBSyncState),
                                  cb->occupied.load(std::memory_order_acquire),
                                  cb->received_compute.load(std::memory_order_acquire),
                                  cb->num_pages);
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
            os << " @ " << (name ? name : "sync object");
        }
        os << "\n";
    }
    return os.str();
}

extern "C" void __emule_semw_dump();  // [SEMWATCH] ring-buffer dump (defined in the runner TU)
void FiberSchedulerImpl::watchdog() {
    const auto interval = std::chrono::milliseconds(250);
    const uint64_t window = env_size("TT_EMULE_FIBER_PROGRESS_WINDOW", 200000);
    const auto backstop = std::chrono::seconds(env_size("TT_EMULE_FIBER_WATCHDOG_SEC", 120));
    uint64_t last_progress = progress_.load();
    uint64_t last_resump = resumptions_.load();
    auto last_advance = std::chrono::steady_clock::now();
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
        uint64_t p = progress_.load();
        uint64_t r = resumptions_.load();
        if (p != last_progress) {
            last_progress = p;
            last_resump = r;
            last_advance = std::chrono::steady_clock::now();
            continue;
        }
        // progress stalled. Fast livelock trip: many resumptions, zero progress.
        bool livelock = (r - last_resump) > window;
        bool wall = (std::chrono::steady_clock::now() - last_advance) > backstop;
        if (livelock || wall) {
            std::fprintf(stderr,
                "[EMULE] fiber engine: no global progress (%s) — suspected %s.\n%s",
                livelock ? "resumption window" : "wall-clock backstop",
                livelock ? "livelock / wake-cycle" : "lost wakeup / hang",
                [this] { std::lock_guard<std::mutex> g(mu_); return dump_parked(); }().c_str());
            if (std::getenv("TT_EMULE_TRACE_CENSUS")) {
                std::fprintf(stderr, "%s",
                             [this] { std::lock_guard<std::mutex> g(mu_); return dump_all(); }().c_str());
            }
            {  // [SCHEDSTATE] discriminate cv-lost-wakeup (a Ready fiber sits in a sleeping worker's
               // queue) vs busy-yield starvation (running_>0 from a yield()-spinner). Snapshot the
               // worker-scheduling counters + every READY (non-parked) fiber and its pinned home worker.
                std::lock_guard<std::mutex> g(mu_);
                std::string rdump;
                size_t total_ready = 0;
                for (unsigned w = 0; w < ready_.size(); ++w) {
                    if (ready_[w].empty()) {
                        continue;
                    }
                    total_ready += ready_[w].size();
                    char b[96];
                    std::snprintf(b, sizeof(b), "    ready_[%u]: %zu fiber(s)\n", w, ready_[w].size());
                    rdump += b;
                    for (Fiber* f : ready_[w]) {
                        const auto* ctx = f->owned_ctx.get();
                        char c[256];
                        std::snprintf(c, sizeof(c),
                            "      dev %d core(log %d,%d phys %d,%d) proc %d kind=%s home=%u kernel %s\n",
                            ctx ? int(ctx->chip_id) : -1, f->id.logical_x, f->id.logical_y, f->id.phys_x,
                            f->id.phys_y, int(f->id.proc_id),
                            (ctx && ctx->kind == ThreadCommonCtx::Kind::Compute) ? "CMP" : "DM", f->home,
                            f->id.kernel_src ? f->id.kernel_src : "?");
                        rdump += c;
                    }
                }
                std::fprintf(stderr,
                    "[SCHEDSTATE] W_=%u idle_=%u running_=%u active_=%u any_ready=%d total_ready=%zu\n%s",
                    W_, idle_, running_, active_, any_ready() ? 1 : 0, total_ready, rdump.c_str());
                std::string run_d;
                for (unsigned w = 0; w < running_fiber_.size(); ++w) {
                    Fiber* f = running_fiber_[w];
                    if (!f) {
                        char b[48];
                        std::snprintf(b, sizeof(b), "    worker[%u] running: <none>\n", w);
                        run_d += b;
                        continue;
                    }
                    const auto* ctx = f->owned_ctx.get();
                    char c[288];
                    std::snprintf(c, sizeof(c),
                        "    worker[%u] RUNNING dev %d core(log %d,%d phys %d,%d) proc %d kind=%s state=%d yields=%llu kernel %s\n",
                        w, ctx ? int(ctx->chip_id) : -1, f->id.logical_x, f->id.logical_y, f->id.phys_x,
                        f->id.phys_y, int(f->id.proc_id),
                        (ctx && ctx->kind == ThreadCommonCtx::Kind::Compute) ? "CMP" : "DM", int(f->state),
                        (unsigned long long)f->yields, f->id.kernel_src ? f->id.kernel_src : "?");
                    run_d += c;
                }
                std::fprintf(stderr, "[SCHEDSTATE-RUNNING]\n%s", run_d.c_str());
            }
            __emule_semw_dump();  // [SEMWATCH] print recorded watched-sem events at the watchdog abort
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
    // Lazily create the persistent worker pool on the first run (K is process-constant);
    // threads live until ~FiberScheduler. See tt-emule docs/fiber-engine.md.
    if (p_->pool_.empty()) {
        p_->K_ = static_cast<unsigned>(env_size("TT_EMULE_FIBER_WORKERS", 64));
        p_->running_fiber_.assign(p_->K_, nullptr);  // [SCHEDSTATE] per-worker current fiber
        p_->pool_.reserve(p_->K_);
        for (unsigned i = 0; i < p_->K_; ++i) {
            p_->pool_.emplace_back([this, i] { p_->worker_main(i); });
        }
    }

    unsigned W = 0;
    {
        std::lock_guard<std::mutex> g(p_->mu_);
        if (initial) {
            p_->active_ = static_cast<unsigned>(p_->all_.size());
            if (p_->active_ == 0) {
                return;   // nothing to run; the pool stays parked
            }
            // Clear the captured kernel exception only on a fresh run; a HostWait return bypasses
            // teardown_and_throw (its sole consumer), so it must survive across pump quanta.
            p_->first_eptr_ = nullptr;
            p_->wake_total_.clear();     // occ-wipe probe: per-program (keys are reused across programs)
            p_->wake_occzero_.clear();
        }
        p_->idle_ = 0;
        p_->running_ = 0;
        p_->workers_done_ = 0;
        p_->deadlock_ = false;
        p_->abort_flag_ = false;
        p_->host_wait_ = false;
        // Per-run recovery watermarks — reset alongside progress_/resumptions_ (below).
        // A stale last_deadlock_repoll_progress_ from a prior run can collide with this
        // run's quiescence progress and skip the recovery re-poll → spurious deadlock.
        p_->last_progress_val_ = 0;
        p_->last_progress_resump_ = 0;
        p_->last_deadlock_repoll_progress_ = UINT64_MAX;
        if (initial) {
            p_->quiescence_deferred_.clear();
            // Activate W = min(K, fiber count) workers; pin each fiber round-robin across [0,W).
            // Surplus workers (>= W) stay parked on start_cv_, so a tiny program pays no herd.
            // See tt-emule docs/fiber-engine.md.
            W = std::min<unsigned>(p_->K_, p_->active_);
            p_->W_ = W;
            p_->ready_.assign(W, {});
            for (size_t i = 0; i < p_->all_.size(); ++i) {
                Fiber* f = p_->all_[i].get();
                f->home = static_cast<unsigned>(i % W);
                p_->ready_[f->home].push_back(f);
            }
        } else {
            W = p_->W_;   // pump: reuse homing; ready_ already refilled by pump()'s re-poll
        }
    }
    p_->progress_.store(0);
    p_->resumptions_.store(0);
    if (std::getenv("TT_EMULE_FIBER_LOG_N")) {
        std::fprintf(stderr, "[EMULE FIBER] program: %u fibers on W=%u of K=%u workers\n",
                     p_->active_, W, p_->K_);
    }

    p_->run_active_.store(true, std::memory_order_release);
    std::thread wd([this] { p_->watchdog(); });

    // Launch: bump the generation under mu_ (after the watchdog is up), then wake the pool.
    {
        std::lock_guard<std::mutex> g(p_->mu_);
        ++p_->generation_;
    }
    p_->start_cv_.notify_all();
    {   // Block the dispatch thread until every active worker has finished this run.
        std::unique_lock<std::mutex> lk(p_->mu_);
        p_->done_cv_.wait(lk, [&] { return p_->workers_done_ == W; });
    }

    {   // Clear under wd_mu_ + notify so the watchdog wakes at once (no lost wakeup).
        std::lock_guard<std::mutex> lk(p_->wd_mu_);
        p_->run_active_.store(false, std::memory_order_release);
    }
    p_->wd_cv_.notify_all();
    wd.join();
}

// Collect results + clear the registry for the next program / mesh. Rethrows the first fiber
// exception; throws on a quiescent deadlock. Called only after a run reaches Completed (never on a
// resumable HostWait — then the fibers must stay alive).
extern "C" void __emule_semw_dump();  // [SEMWATCH] ring-buffer dump (defined in the runner TU)
void FiberScheduler::teardown_and_throw() {
    std::exception_ptr eptr;
    bool deadlock;
    std::string dump;
    {   // workers are parked on start_cv_ now, but take mu_ anyway for clean ordering.
        std::lock_guard<std::mutex> g(p_->mu_);
        eptr = p_->first_eptr_;
        deadlock = p_->deadlock_;
        if (deadlock) {
            dump = p_->dump_parked();
            __emule_semw_dump();  // [SEMWATCH] print recorded watched-sem events at the deadlock census
            if (std::getenv("TT_EMULE_TRACE_CENSUS")) {
                std::fprintf(stderr, "%s", p_->dump_all().c_str());
            }
        }
        p_->ready_.clear();
        p_->parked_.clear();
        p_->quiescence_deferred_.clear();
        p_->all_.clear();   // frees Fiber stacks via ~Fiber
    }

    // A real kernel exception is the root cause; report it before any deadlock symptom.
    if (eptr) {
        std::rethrow_exception(eptr);
    }
    if (deadlock) {
        throw std::runtime_error("EMULE fiber engine: quiescent deadlock — all workers idle, "
                                 "fibers parked, none runnable.\n" + dump);
    }
}

void FiberScheduler::run_until_idle() {
    p_->persistent_ = false;   // non-persistent: a quiescence-with-parked is a tier-1 deadlock
    launch_and_wait(/*initial=*/true);
    teardown_and_throw();
}

RunOutcome FiberScheduler::run_persistent() {
    p_->persistent_ = true;
    p_->host_wait_no_progress_pumps_ = 0;   // start of a fresh host-wait sequence
    launch_and_wait(/*initial=*/true);
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
        // The host advanced a credit word with a raw L1 store (no __emule_fiber_wake), so wake
        // every parked fiber to re-check its predicate. This is the quiescence re-poll, host-driven;
        // spurious-wake-safe (__emule_fiber_wait re-checks under the lock and re-parks).
        for (auto& kv : p_->parked_) {
            for (Fiber* f = kv.second; f;) {
                Fiber* nx = f->park_link;
                f->park_link = nullptr;
                f->park_key = nullptr;
                f->wait_is_socket = false;
                f->state = FiberState::Ready;
                p_->ready_[f->home].push_back(f);
                f = nx;
            }
        }
        p_->parked_.clear();
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
        static const uint64_t kHostWaitNoProgressLimit = env_size("TT_EMULE_HOST_WAIT_STALL_LIMIT", 8192);
        if (p_->progress_.load() == 0) {
            if (++p_->host_wait_no_progress_pumps_ >= kHostWaitNoProgressLimit) {
                p_->deadlock_ = true;
                teardown_and_throw();  // reuses the tier-1 quiescent-deadlock diagnostic (dump_parked)
            }
        } else {
            p_->host_wait_no_progress_pumps_ = 0;
        }
        return RunOutcome::HostWait;
    }
    teardown_and_throw();
    return RunOutcome::Completed;
}

FiberScheduler::FiberScheduler() : p_(std::make_unique<FiberSchedulerImpl>()) {
    // Read the fiber stack size once, at construction: spawn() consumes stack_bytes_ to size each
    // fiber's mmap'd stack, and spawn() always runs before run_until_idle (single-device launch spawns
    // then runs; the mesh path spawns all devices in the defer phase, then runs once). Reading it in
    // run_until_idle would miss the first program. See tt-emule docs/fiber-engine.md.
    p_->stack_bytes_ = env_size("TT_EMULE_FIBER_STACK_BYTES", 1u << 20);
}

FiberScheduler::~FiberScheduler() {
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
