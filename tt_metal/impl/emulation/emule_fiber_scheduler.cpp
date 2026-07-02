// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emule_fiber_scheduler.hpp"

#include "tt_emule/cb_sync_state.hpp"  // tt_emule::CBSyncState (sizeof, for the dump)

#include <ucontext.h>
#include <sys/mman.h>
#include <unistd.h>

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

namespace tt::tt_metal::emule_fiber {

namespace {

enum class FiberState : uint8_t { Ready, Running, Parked, LatencyParked, Done };

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
    std::vector<Fiber*> latency_parked_;               // fibers modeling NOC read latency:
                                                       // released only at quiescence (below)
    std::vector<std::unique_ptr<Fiber>> all_;          // ownership of every spawned fiber

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
    std::exception_ptr first_eptr_;

    std::atomic<uint64_t> progress_{0};      // fiber completions + published pages (tier 2)
    std::atomic<uint64_t> resumptions_{0};   // swap-ins (tier 2 livelock signal)

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
    std::string dump_parked();               // single-threaded (post-join)
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
    for (;;) {
        if (abort_flag_) break;
        if (ready_[w].empty()) {
            if (active_ == 0) break;
            ++idle_;
            // Quiescence: nothing executing, nothing runnable in any queue. The
            // `!any_ready()` term is essential — `idle_ == W_` alone is a false positive
            // at W>1 (a worker counts itself idle before re-observing a just-enqueued
            // fiber). See tt-emule docs/fiber-engine.md.
            if (idle_ == W_ && running_ == 0 && !any_ready()) {
                // Read-latency model: a latency-parked fiber is an in-flight read. At
                // quiescence the read "completes" — release them all (lowest priority),
                // reproducing the silicon ordering some kernels lean on.
                // See tt-emule docs/fiber-engine.md.
                if (!latency_parked_.empty()) {
                    for (Fiber* f : latency_parked_) {
                        f->state = FiberState::Ready;
                        ready_[f->home].push_back(f);
                    }
                    latency_parked_.clear();
                    cv_.notify_all();
                    --idle_;
                    continue;
                }
                // Tier-1 quiescent deadlock: nothing to release and fibers still sync-parked.
                if (!parked_.empty()) {
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
        install_fiber(f);
        resumptions_.fetch_add(1, std::memory_order_relaxed);
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

void FiberScheduler::park_locked(const void* key) {
    // pre: mu_ held by this thread (the .so's __emule_fiber_lock). Register parked and
    // hand the lock to the worker loop across the switch.
    Fiber* f = t_current;
    f->state = FiberState::Parked;
    f->park_key = key;
    Fiber*& head = p_->parked_[key];         // inserts nullptr if absent
    f->park_link = head;
    head = f;
    swapcontext(&f->ctx, &t_sched);          // -> worker loop (mu_ held); resumes mu_-UNLOCKED
}

void FiberScheduler::latency_park() {
    // Model NOC read latency: defer the current fiber as lowest-priority "in-flight" work,
    // released only at quiescence (see worker_loop). No key, no predicate; takes mu_ itself
    // and hands it to the worker loop across the switch (mirrors park_locked).
    Fiber* f = t_current;
    if (!f) {
        return;  // read barriers only run inside a fiber; insurance against a host-side call
    }
    p_->mu_.lock();
    f->state = FiberState::LatencyParked;
    p_->latency_parked_.push_back(f);
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
        p_->ready_[f->home].push_back(f);    // back to its pinned worker
        f = nx;
    }
    p_->parked_.erase(it);
    p_->cv_.notify_all();
}

void FiberScheduler::yield() {
    Fiber* f = t_current;
    p_->mu_.lock();
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
            os << "    core(log " << f->id.logical_x << "," << f->id.logical_y
               << " phys " << int(f->id.phys_x) << "," << int(f->id.phys_y) << ")"
               << " risc/proc " << int(f->id.proc_id);
            if (f->id.kernel_src) {
                os << " kernel " << f->id.kernel_src;
            }
            // Best-effort key naming: a CB if the key lands in this fiber's cbs[] array.
            const auto* ctx = f->owned_ctx.get();
            const char* name = nullptr;
            char buf[64];
            if (ctx && ctx->cbs) {
                auto base = reinterpret_cast<uintptr_t>(ctx->cbs);
                auto k = reinterpret_cast<uintptr_t>(key);
                if (k >= base && k < base + sizeof(tt_emule::CBSyncState) * 32) {
                    std::snprintf(buf, sizeof(buf), "CB %zu",
                                  (k - base) / sizeof(tt_emule::CBSyncState));
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
    if (!latency_parked_.empty()) {
        os << "  " << latency_parked_.size() << " fiber(s) in read-latency flight\n";
    }
    return os.str();
}

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
            std::abort();
        }
        last_resump = r;
    }
}

void FiberScheduler::run_until_idle() {
    // Lazily create the persistent worker pool on the first run (K is process-constant);
    // threads live until ~FiberScheduler. See tt-emule docs/fiber-engine.md.
    if (p_->pool_.empty()) {
        p_->K_ = static_cast<unsigned>(env_size("TT_EMULE_FIBER_WORKERS", 64));
        p_->pool_.reserve(p_->K_);
        for (unsigned i = 0; i < p_->K_; ++i) {
            p_->pool_.emplace_back([this, i] { p_->worker_main(i); });
        }
    }

    unsigned W;
    {
        std::lock_guard<std::mutex> g(p_->mu_);
        p_->active_ = static_cast<unsigned>(p_->all_.size());
        if (p_->active_ == 0) {
            return;   // nothing to run; the pool stays parked
        }
        p_->idle_ = 0;
        p_->running_ = 0;
        p_->workers_done_ = 0;
        p_->deadlock_ = false;
        p_->abort_flag_ = false;
        p_->first_eptr_ = nullptr;
        p_->latency_parked_.clear();
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

    std::exception_ptr eptr;
    bool deadlock;
    std::string dump;
    {   // Collect results + clear the registry for the next program / mesh (workers are parked
        // on start_cv_ now, but take mu_ anyway for clean ordering).
        std::lock_guard<std::mutex> g(p_->mu_);
        eptr = p_->first_eptr_;
        deadlock = p_->deadlock_;
        if (deadlock) {
            dump = p_->dump_parked();
        }
        p_->ready_.clear();
        p_->parked_.clear();
        p_->latency_parked_.clear();
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
