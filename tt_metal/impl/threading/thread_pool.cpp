// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <linux/futex.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/resource.h>
#include <numa.h>
#include <tt-metalium/device.hpp>
#include "impl/device/device_impl.hpp"
#include <tt_stl/small_vector.hpp>
#include <tt_stl/tt_pause.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/context_types.hpp"
#include "impl/threading/thread_pool.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace thread_binding {

std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node() {
    std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = {};
    if (numa_available() != -1) {
        for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
            int node = numa_node_of_cpu(cpu);
            cpu_cores_per_numa_node[node].push_back(cpu);
        }
    }
    return cpu_cores_per_numa_node;
}

bool balanced_physical_device_numa(ContextId context_id) {
    if (numa_available() != -1) {
        int num_nodes = numa_max_node() + 1;
        std::unordered_set<int> numa_nodes_for_cluster = {};
        for (auto device_id : MetalContext::instance(context_id).get_cluster().user_exposed_chip_ids()) {
            auto numa_node_for_device =
                MetalContext::instance(context_id).get_cluster().get_numa_node_for_device(device_id);
            numa_nodes_for_cluster.insert(numa_node_for_device);
        }
        return numa_nodes_for_cluster.size() == num_nodes;
    }
    return false;
}

uint32_t get_cpu_core_for_physical_device(ContextId context_id, uint32_t physical_device_id) {
    static std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = get_cpu_cores_per_numa_node();
    static std::unordered_map<int, int> logical_cpu_id_per_numa_node = {};
    // Initialize to an invalid value. Determine the NUMA Node based on the physical device id.
    // If a NUMA Node is not found, use a round robin policy.
    int numa_node = -1;
    if (physical_device_id < MetalContext::instance(context_id).get_cluster().number_of_devices() &&
        !tt::tt_metal::MetalContext::instance(context_id).rtoptions().get_simulator_enabled()) {
        // If the cluster uses all NUMA nodes, assign worker threads to CPU cores based
        // on the NUMA layout. If not, balance the worker threads across all NUMA Nodes/
        // CPU cores to minimize resource contention.
        static std::unordered_map<uint64_t, bool> balanced_cache;
        auto& cluster = MetalContext::instance(context_id).get_cluster();
        uint64_t cache_key = static_cast<uint64_t>(cluster.arch()) |
                             (static_cast<uint64_t>(cluster.get_target_device_type()) << 8) |
                             (static_cast<uint64_t>(cluster.number_of_devices()) << 16);
        auto [it, inserted] = balanced_cache.try_emplace(cache_key, false);
        if (inserted) {
            it->second = balanced_physical_device_numa(context_id);
        }
        numa_node = it->second
                        ? MetalContext::instance(context_id).get_cluster().get_numa_node_for_device(physical_device_id)
                        : physical_device_id % 2;
    }
    if (cpu_cores_per_numa_node.contains(numa_node)) {
        auto& cpu_cores_on_node = cpu_cores_per_numa_node[numa_node];
        return cpu_cores_on_node[(logical_cpu_id_per_numa_node[numa_node]++) % cpu_cores_on_node.size()];
    }
    uint32_t num_threads = std::thread::hardware_concurrency();
    TT_FATAL(num_threads, "Could not detect the number of CPU cores on host.");
    return physical_device_id % num_threads;
}

void bind_memory_to_numa_node(void* base, size_t bytes, int numa_node) {
    if (base == nullptr || bytes == 0 || numa_node < 0 || numa_available() == -1) {
        return;
    }
    if (numa_node > numa_max_node()) {
        return;
    }
    numa_tonode_memory(base, bytes, numa_node);
}

void set_worker_affinity(std::thread& worker, uint32_t cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    int rc = pthread_setaffinity_np(worker.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind worker thread to CPU Core. May see performance degradation. Error Code: {}",
            rc);
    }
}

void set_process_priority(int requested_priority) {
    // Get priority for calling process
    int process_priority = getpriority(PRIO_PROCESS, 0);
    log_debug(tt::LogMetal, "Initial Process Priority: {}", process_priority);
    if (process_priority == requested_priority) {
        return;
    }
    // Set priority for calling process to user specified value
    int rc = setpriority(PRIO_PROCESS, 0, requested_priority);
    if (rc) {
        log_warning(tt::LogMetal, "Unable to set process priority to {}, error code: {}", requested_priority, rc);
    }
}

}  // namespace thread_binding

void bind_memory_to_numa_node(void* base, size_t bytes, int numa_node) {
    thread_binding::bind_memory_to_numa_node(base, bytes, numa_node);
}

namespace threading_primitives {

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free);

// Used instead of std::atomic::wait/notify: libstdc++ tracks waiters in a 16-slot process-wide table, so
// notify enters the kernel whenever any thread is parked on an address in the same slot.
void futex_wait(std::atomic<uint32_t>& word, uint32_t expected) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
}

void futex_wake_one(std::atomic<uint32_t>& word) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

void futex_wake_all(std::atomic<uint32_t>& word) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(&word), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

// Spins like std::atomic::wait before it parks (after 100 polls), so that callers can park on their own futex.
template <typename Ready>
bool spin_until(Ready ready) {
    constexpr uint32_t POLLS = 100, PAUSES = 12, YIELDS = 4;
    for (uint32_t i = 0; i < POLLS; i++) {
        if (ready()) {
            return true;
        }
    }
    for (uint32_t i = 0; i < PAUSES + YIELDS; i++) {
        if (ready()) {
            return true;
        }
        i < PAUSES ? ttsl::pause() : static_cast<void>(sched_yield());
    }
    return false;
}

// Polls for up to `window`. The pauses between polls leave most of the core to its SMT sibling, which often runs
// the thread the poll is waiting for.
template <typename Ready>
bool spin_for(std::chrono::nanoseconds window, Ready ready) {
    constexpr uint32_t POLLS_PER_CLOCK_READ = 32, PAUSES_PER_POLL = 8;
    const auto deadline = std::chrono::steady_clock::now() + window;
    do {
        for (uint32_t i = 0; i < POLLS_PER_CLOCK_READ; i++) {
            if (ready()) {
                return true;
            }
            for (uint32_t pause = 0; pause < PAUSES_PER_POLL; pause++) {
                ttsl::pause();
            }
        }
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

// Tasks in flight across all workers of a pool, so that joining waits on one counter.
class Completion {
public:
    void add(int64_t n = 1) { pending_.fetch_add(n, std::memory_order_relaxed); }

    bool finished() const { return pending_.load(std::memory_order_acquire) == 0; }

    void done() {
        // A worker wakes the joining threads only if one is parked, and only when the count reaches zero.
        if (pending_.fetch_sub(1, std::memory_order_seq_cst) == 1 && parked_.load(std::memory_order_seq_cst) != 0) {
            generation_.fetch_add(1, std::memory_order_release);
            futex_wake_all(generation_);
        }
    }

    void wait() {
        // Longer than a fan-out to parked workers takes to drain, so the joining thread rarely parks.
        constexpr auto JOIN_SPIN = std::chrono::microseconds(50);
        const auto deadline = std::chrono::steady_clock::now() + JOIN_SPIN;
        while (pending_.load(std::memory_order_acquire) != 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            ttsl::pause();
        }
        if (pending_.load(std::memory_order_acquire) == 0) {
            return;
        }
        // seq_cst pairs with done(): either done() sees the waiter registered, or the waiter sees zero.
        parked_.fetch_add(1, std::memory_order_seq_cst);
        while (true) {
            // A wake after this load changes the generation, so futex_wait returns at once.
            const uint32_t generation = generation_.load(std::memory_order_acquire);
            if (pending_.load(std::memory_order_seq_cst) == 0) {
                break;
            }
            futex_wait(generation_, generation);
        }
        parked_.fetch_sub(1, std::memory_order_relaxed);
    }

private:
    alignas(64) std::atomic<int64_t> pending_ = 0;
    // Joining threads past the spin, and the futex word they park on.
    alignas(64) std::atomic<uint32_t> parked_ = 0;
    std::atomic<uint32_t> generation_ = 0;
};

// Single-producer, single-consumer ring of tasks.
class TaskQueue {
public:
    // Producer. Stalls while the ring is full.
    void push(std::function<void()>&& task) {
        const uint64_t tail = tail_.load(std::memory_order_relaxed);
        if (tail - head_.load(std::memory_order_acquire) == CAPACITY) {
            ttsl::nice_spin_until([&] { return tail - head_.load(std::memory_order_acquire) < CAPACITY; });
        }
        slots_[tail % CAPACITY] = std::move(task);
        // seq_cst pairs with the worker's check before it parks (NumaAwareExecutor::wait_for_work).
        tail_.store(tail + 1, std::memory_order_seq_cst);
    }

    // Consumer.
    bool empty() const { return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_seq_cst); }

    // Consumer. The queue must not be empty.
    std::function<void()> pop() {
        const uint64_t head = head_.load(std::memory_order_relaxed);
        auto task = std::move(slots_[head % CAPACITY]);
        slots_[head % CAPACITY] = nullptr;
        head_.store(head + 1, std::memory_order_release);
        return task;
    }

private:
    static constexpr uint64_t CAPACITY = 65536;
    std::array<std::function<void()>, CAPACITY> slots_;
    alignas(64) std::atomic<uint64_t> head_ = 0;
    alignas(64) std::atomic<uint64_t> tail_ = 0;
};

class ParallelJob;

// NUMA + CPU Affinity aware executor, used by custom thread-pool implementations.
// Contains:
//  1. A TaskQueue where tasks can be submitted by the user, to be asynchronously executed
//  2. A worker thread to asynchronously execute tasks
//  3. Primitives to synchronize the application and worker thread
// Usage:
// This executor should only be used to asynchronously process tasks for a specific TT-Device
// (specified through the physical_device_id constructor argument).
// The executor is NUMA aware, i.e. it will bind its worker thread to a NUMA node that is "closest"
// to its physical device.
// If all physical devices are on the same NUMA node, CPU cores will be assigned to minimize contention
// across threads.
class NumaAwareExecutor {
public:
    // `index` is the executor's place in its pool; `published` is the pool's published parallel_for job, a slot of
    // `jobs`. After its last task or call, the worker stays awake for `spin` before it parks, unless `park_after`
    // names the job it last saw.
    NumaAwareExecutor(
        ContextId context_id,
        uint32_t physical_device_id,
        Completion& completion,
        uint32_t index,
        const std::atomic<uint64_t>& published,
        const std::atomic<uint64_t>& park_after,
        const std::unique_ptr<ParallelJob>* jobs,
        std::chrono::nanoseconds spin) :
        completion_(completion),
        index_(index),
        published_(published),
        park_after_(park_after),
        jobs_(jobs),
        spin_(spin) {
        // Set the priority for this process to 0 (niceness value in linux)
        thread_binding::set_process_priority(0);
        worker = std::thread([this]() { worker_loop(); });

        auto cpu_core_for_worker = thread_binding::get_cpu_core_for_physical_device(context_id, physical_device_id);
        thread_binding::set_worker_affinity(worker, cpu_core_for_worker);
    }

    // Delete copy and move operations as this class manages a thread and should not be copied or moved
    NumaAwareExecutor(const NumaAwareExecutor&) = delete;
    NumaAwareExecutor& operator=(const NumaAwareExecutor&) = delete;
    NumaAwareExecutor(NumaAwareExecutor&&) = delete;
    NumaAwareExecutor& operator=(NumaAwareExecutor&&) = delete;

    ~NumaAwareExecutor() { stop(); }

    // Joins the worker.
    void stop() {
        if (!worker.joinable()) {
            return;
        }
        shutdown_.store(true, std::memory_order_seq_cst);
        wake();
        worker.join();
    }

    void enqueue(std::function<void()>&& f) {
        tasks_.push(std::move(f));
        wake();
    }

    // Enters the kernel only if the worker is parked.
    void wake() noexcept {
        if (parked_.load(std::memory_order_seq_cst) != 0 && parked_.exchange(0, std::memory_order_seq_cst) != 0) {
            futex_wake_one(parked_);
        }
    }

    uint32_t index() const { return index_; }

    // Returns the first exception thrown by a task since the last call, once the pool has joined.
    std::exception_ptr take_exception() { return std::exchange(stored_exception_, nullptr); }

private:
    void worker_loop();

    // Takes part in the job `published` names, if that is still the job's current use.
    void join(uint64_t published) noexcept;

    bool has_work() const { return !tasks_.empty() || published_.load(std::memory_order_seq_cst) != seen_; }

    // Stays awake for the spin window so back-to-back work is picked up without entering the kernel, then parks
    // until a producer publishes work or the executor shuts down.
    void wait_for_work() {
        if (spin_until([this] { return has_work(); })) {
            return;
        }
        // Stop spinning once the pool asks its workers to park after the job this one last saw.
        auto park_requested = [this] { return park_after_.load(std::memory_order_relaxed) == seen_; };
        if (spin_.count() > 0 && !park_requested() &&
            spin_for(
                spin_, [&] { return has_work() || shutdown_.load(std::memory_order_relaxed) || park_requested(); }) &&
            !park_requested()) {
            return;
        }
        // seq_cst pairs with wake(): either the producer sees the worker parked, or the worker sees the work.
        parked_.store(1, std::memory_order_seq_cst);
        if (!has_work() && !shutdown_.load(std::memory_order_seq_cst)) {
            futex_wait(parked_, 1);
        }
        parked_.store(0, std::memory_order_relaxed);
    }

    TaskQueue tasks_;
    Completion& completion_;
    std::thread worker;
    alignas(64) std::atomic<uint32_t> parked_ = 0;
    std::atomic<bool> shutdown_ = false;
    alignas(64) uint32_t index_ = 0;
    const std::atomic<uint64_t>& published_;
    const std::atomic<uint64_t>& park_after_;
    const std::unique_ptr<ParallelJob>* jobs_ = nullptr;
    std::chrono::nanoseconds spin_{0};
    uint64_t seen_ = 0;  // The published word this worker last acted on; worker thread only.
    std::exception_ptr stored_exception_;
};

// State of one parallel_for call. Jobs live in their pool's ring and are reused: workers find the published job
// through the pool and may still hold it after its call has returned, so each use is a generation and a job is
// reused only once nobody holds it.
class ParallelJob {
public:
    // Each worker that takes part wakes this many others before running its own calls, so that the caller
    // enters the kernel once per fan-out instead of once per worker.
    static constexpr size_t WAKE_FANOUT = 4;
    static constexpr uint32_t NONE = UINT32_MAX;

    using Participants = ttsl::SmallVector<NumaAwareExecutor*, 32>;  // 32: one per Galaxy device

    // Sized for a pool of `capacity` workers; `slot` is the job's place in the pool's ring.
    ParallelJob(size_t capacity, uint32_t slot) :
        slot_(slot), claims_(capacity), participant_of_worker_(capacity, NONE) {}

    ParallelJob(const ParallelJob&) = delete;
    ParallelJob& operator=(const ParallelJob&) = delete;
    ParallelJob(ParallelJob&&) = delete;
    ParallelJob& operator=(ParallelJob&&) = delete;
    ~ParallelJob() = default;

    // Caller: takes the job if nobody holds it, starting its next generation in the same step.
    bool try_acquire() noexcept {
        uint64_t state = state_.load(std::memory_order_relaxed);
        if ((state & HOLDERS) != 0) {
            return false;
        }
        const uint64_t next = (((state >> 32) + 1) << 32) | 1;
        if (!state_.compare_exchange_strong(state, next, std::memory_order_acquire, std::memory_order_relaxed)) {
            return false;
        }
        generation_ = static_cast<uint32_t>(next >> 32);
        return true;
    }

    // Worker: holds the job if `generation` is still its current use; a held job is not reused.
    bool hold(uint32_t generation) noexcept {
        if ((state_.fetch_add(1, std::memory_order_acq_rel) >> 32) == generation) {
            return true;
        }
        release();
        return false;
    }

    void release() noexcept { state_.fetch_sub(1, std::memory_order_acq_rel); }

    // Caller: plans one call of fn per device id, each for the executor `executor_of` returns. A plan for the same
    // devices as the job's last use is kept, so that the caller does not rewrite lines the workers read.
    template <typename ExecutorOf>
    void plan(const std::function<void(size_t)>& fn, ttsl::Span<const uint32_t> device_ids, ExecutorOf executor_of) {
        if (fn_ != &fn) {
            fn_ = &fn;
        }
        if (exception_) {
            exception_ = nullptr;
        }
        auto& devices = plan_devices_.ids;
        if (planned_ && std::equal(device_ids.begin(), device_ids.end(), devices.begin(), devices.end())) {
            return;
        }
        planned_ = false;
        for (auto* executor : participants_) {
            participant_of_worker_[executor->index()] = NONE;
        }
        participants_.clear();
        first_call_.clear();
        last_call_.clear();
        next_call_.assign(device_ids.size(), NONE);
        for (uint32_t call = 0; call < device_ids.size(); call++) {
            auto* executor = executor_of(device_ids[call]);
            uint32_t& participant = participant_of_worker_[executor->index()];
            if (participant == NONE) {
                participant = static_cast<uint32_t>(participants_.size());
                participants_.push_back(executor);
                first_call_.push_back(call);
                last_call_.push_back(call);
            } else {
                next_call_[last_call_[participant]] = call;
                last_call_[participant] = call;
            }
        }
        devices.assign(device_ids.begin(), device_ids.end());
        planned_ = true;
    }

    const Participants& participants() const { return participants_; }

    // Caller: counts the planned calls in; workers may take part from now on.
    void start() noexcept { remaining_.add(static_cast<int64_t>(next_call_.size())); }

    // What the pool publishes for this use of the job: (generation << 32) | slot.
    uint64_t published() const { return (static_cast<uint64_t>(generation_) << 32) | slot_; }

    // Worker, holding the job: wakes its children, runs its calls and releases the job. A finished job is only
    // released: its executors may be stopping.
    void participate(NumaAwareExecutor* executor) noexcept {
        const uint32_t node = remaining_.finished() ? NONE : participant_of_worker_[executor->index()];
        if (node != NONE) {
            for_each_child(node, [this](size_t child) { wake_subtree(child); });
            run_if_unclaimed(node);
        }
        release();
    }

    // Runs all calls of participant `node` in order, so that calls for one worker never overlap.
    void run_if_unclaimed(size_t node) noexcept {
        if (claims_[node].generation.exchange(generation_, std::memory_order_acq_rel) == generation_) {
            return;
        }
        for (uint32_t call = first_call_[node]; call != NONE; call = next_call_[call]) {
            try {
                (*fn_)(call);
            } catch (...) {
                std::lock_guard lock(exception_mutex_);
                if (!exception_) {
                    exception_ = std::current_exception();
                }
            }
            remaining_.done();
        }
    }

    // Caller: waits for every call, returns the first exception, and releases the job.
    std::exception_ptr finish() noexcept {
        remaining_.wait();
        auto exception = exception_;
        release();
        return exception;
    }

private:
    template <typename Fn>
    void for_each_child(size_t node, Fn fn) const noexcept {
        for (size_t child = (node * WAKE_FANOUT) + 1;
             child <= (node * WAKE_FANOUT) + WAKE_FANOUT && child < participants_.size();
             child++) {
            fn(child);
        }
    }

    // Wakes `node`, or its children in its place if the caller already claimed it.
    void wake_subtree(size_t node) noexcept {
        if (claims_[node].generation.load(std::memory_order_relaxed) != generation_) {
            participants_[node]->wake();
            return;
        }
        for_each_child(node, [this](size_t child) { wake_subtree(child); });
    }

    // Holds the generation of the last use that claimed the participant, so a new use starts unclaimed.
    struct alignas(64) Claim {
        std::atomic<uint32_t> generation = 0;
    };

    static constexpr uint64_t HOLDERS = 0xffffffffULL;

    // (generation << 32) | holders: the caller that acquired the job and the workers taking part.
    std::atomic<uint64_t> state_ = 0;
    uint32_t generation_ = 0;  // Of the current use; set by the caller on acquire, read by holders.
    uint32_t slot_ = 0;
    // Valid until the caller returns, which happens only after every claimed call has finished.
    const std::function<void(size_t)>* fn_ = nullptr;
    std::vector<Claim> claims_;  // Indexed by participant.
    // Wake tree in heap order: the caller wakes [0], and [i] wakes [i * WAKE_FANOUT + 1 .. (i + 1) * WAKE_FANOUT].
    Participants participants_;
    ttsl::SmallVector<uint32_t, 64> participant_of_worker_;  // Indexed by executor index; NONE if not taking part.
    // Each participant's calls in call order: first_call_ and last_call_ by participant, next_call_ by call.
    ttsl::SmallVector<uint32_t, 32> first_call_;
    ttsl::SmallVector<uint32_t, 32> last_call_;
    ttsl::SmallVector<uint32_t, 32> next_call_;
    // The devices of the current plan; read by the caller only, so on lines of their own.
    struct alignas(64) {
        ttsl::SmallVector<uint32_t, 32> ids;
    } plan_devices_;
    bool planned_ = false;
    Completion remaining_;
    std::mutex exception_mutex_;
    std::exception_ptr exception_;
};

inline void NumaAwareExecutor::join(uint64_t published) noexcept {
    auto* job = jobs_[published & 0xffffffffULL].get();
    if (job->hold(static_cast<uint32_t>(published >> 32))) {
        job->participate(this);
    }
}

inline void NumaAwareExecutor::worker_loop() {
    while (true) {
        if (!tasks_.empty()) {
            {
                auto task = tasks_.pop();
                try {
                    task();
                } catch (...) {
                    if (!stored_exception_) {
                        stored_exception_ = std::current_exception();
                    }
                }
            }
            completion_.done();
            continue;
        }
        if (const uint64_t published = published_.load(std::memory_order_acquire); published != seen_) {
            seen_ = published;
            join(published);
            continue;
        }
        if (shutdown_.load(std::memory_order_acquire)) {
            return;
        }
        wait_for_work();
    }
}

}  // namespace threading_primitives

namespace thread_pool_impls {
// Implementations conforming to the ThreadPool interface.
using threading_primitives::Completion;
using threading_primitives::NumaAwareExecutor;
using threading_primitives::ParallelJob;

void run_on_caller(size_t num_calls, const std::function<void(size_t)>& fn) {
    std::exception_ptr exception;
    for (size_t call = 0; call < num_calls; call++) {
        try {
            fn(call);
        } catch (...) {
            if (!exception) {
                exception = std::current_exception();
            }
        }
    }
    if (exception) {
        std::rethrow_exception(exception);
    }
}

// Custom Thread-Pool using the threading::Executor class.
// Allows enqueuing tasks tied to specific devices.
class DeviceBoundThreadPool : public ThreadPool {
public:
    // Constructor accepting the physical device IDs this pool is bound to. Each thread will be tied to a device, and is
    // guaranteed to be bound to a CPU core on a NUMA Node "closest" to that device.
    // All physical devices must belong to the same context ID.
    DeviceBoundThreadPool(
        ContextId context_id,
        const std::vector<tt::tt_metal::Device*>& physical_devices,
        std::chrono::nanoseconds spin = std::chrono::nanoseconds(0)) :
        num_workers_(physical_devices.size()) {
        allocate_jobs();
        workers_.reserve(num_workers_);
        for (uint32_t i = 0; i < num_workers_; i++) {
            workers_.emplace_back(std::make_unique<NumaAwareExecutor>(
                context_id, physical_devices[i]->id(), completion_, i, published_, park_after_, jobs_.data(), spin));
            phys_device_to_thread_id_[physical_devices[i]->id()] = i;
            map_device(physical_devices[i]->id(), i);
        }
    }
    // Constructor accepting the number of threads to spawn. The threads in this pool will be bound to a specific CPU
    // core but they are not guaranteed to be "close" to any physical device.
    DeviceBoundThreadPool(ContextId context_id, uint32_t thread_count) : num_workers_(thread_count) {
        allocate_jobs();
        workers_.reserve(thread_count);

        for (uint32_t i = 0; i < thread_count; i++) {
            workers_.emplace_back(std::make_unique<NumaAwareExecutor>(
                context_id, i, completion_, i, published_, park_after_, jobs_.data(), std::chrono::nanoseconds(0)));
            phys_device_to_thread_id_[i] = i;
            map_device(i, i);
        }
    }

    DeviceBoundThreadPool(const DeviceBoundThreadPool&) = delete;
    DeviceBoundThreadPool& operator=(const DeviceBoundThreadPool&) = delete;
    DeviceBoundThreadPool(DeviceBoundThreadPool&&) = delete;
    DeviceBoundThreadPool& operator=(DeviceBoundThreadPool&&) = delete;

    ~DeviceBoundThreadPool() override {
        completion_.wait();
        // Stop every executor before destroying any: a parallel_for job on one worker can wake the others.
        for (auto& worker : workers_) {
            worker->stop();
        }
    }

    void enqueue(std::function<void()>&& f, std::optional<uint32_t> device_idx = std::nullopt) override {
        // If the user does not provide the Device ID tied to this task, determine the thread to use
        // based on the internally stored thread_idx. Tasks will get round-robined across threads,
        // when relying on the thread_idx.
        // If the device id is specified, use the thread tied to the device.
        uint32_t thread_id = device_idx.has_value()
                                 ? phys_device_to_thread_id_[device_idx.value()]
                                 : (thread_idx_.fetch_add(1, std::memory_order_relaxed) % num_workers_);
        completion_.add();
        workers_[thread_id]->enqueue(std::move(f));
    }

    void parallel_for(ttsl::Span<const uint32_t> device_ids, const std::function<void(size_t)>& fn) override {
        if (device_ids.empty()) {
            return;
        }
        auto* job = acquire_job();
        try {
            job->plan(fn, device_ids, [this](uint32_t device_id) { return workers_[worker_of(device_id)].get(); });
        } catch (...) {
            job->release();
            throw;
        }
        const auto& participants = job->participants();
        if (participants.size() == 1) {
            job->release();
            run_on_caller(device_ids.size(), fn);
            return;
        }
        // Nothing below throws: once published, workers hold the job and the job refers to fn.
        job->start();
        // seq_cst pairs with the workers' check before they park: either the first participant sees the job, or
        // wake() sees it parked. It wakes the others as it starts; awake workers see the job on their own.
        published_.store(job->published(), std::memory_order_seq_cst);
        participants[0]->wake();
        // The workers are woken first to last, so take participants from the back. The caller runs the calls of
        // every participant no worker has claimed, so the job finishes even if some workers are never woken.
        for (size_t node = participants.size(); node-- > 0;) {
            job->run_if_unclaimed(node);
        }
        if (auto exception = job->finish()) {
            std::rethrow_exception(exception);
        }
    }

    void park_workers() override {
        park_after_.store(published_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    void wait() override {
        thread_idx_.store(0, std::memory_order_relaxed);  // Reset thread_idx for next call without Device ID specified.
        completion_.wait();
        // Rethrow the first exception in the calling thread. Several threads may wait at once.
        std::exception_ptr exception;
        {
            std::lock_guard lock(exception_mutex_);
            for (auto& worker : workers_) {
                auto temp_exception = worker->take_exception();
                if (!exception && temp_exception) {
                    exception = temp_exception;
                }
            }
        }
        if (exception) {
            std::rethrow_exception(exception);
        }
    }

private:
    void allocate_jobs() {
        // A worker holds at most one job, while it takes part; each concurrent caller holds one more.
        jobs_.resize(num_workers_ + 2);
        for (uint32_t slot = 0; slot < jobs_.size(); slot++) {
            jobs_[slot] = std::make_unique<ParallelJob>(num_workers_, slot);
        }
    }

    // A job nobody holds, starting from the one used last: it is the most likely to still be in cache.
    ParallelJob* acquire_job() {
        const auto size = static_cast<uint32_t>(jobs_.size());
        uint32_t slot = last_job_.load(std::memory_order_relaxed);
        for (uint32_t probes = 1; !jobs_[slot]->try_acquire(); probes++) {
            slot = slot + 1 == size ? 0 : slot + 1;
            // All held: workers still taking part release theirs shortly.
            if (probes % size == 0) {
                ttsl::pause();
            }
        }
        last_job_.store(slot, std::memory_order_relaxed);
        return jobs_[slot].get();
    }

    void map_device(uint32_t device_id, uint32_t worker) {
        if (worker_of_device_.size() <= device_id) {
            worker_of_device_.resize(device_id + 1, ParallelJob::NONE);
        }
        worker_of_device_[device_id] = worker;
    }

    uint32_t worker_of(uint32_t device_id) const {
        if (device_id >= worker_of_device_.size() || worker_of_device_[device_id] == ParallelJob::NONE) {
            throw std::out_of_range("parallel_for: no worker is bound to this device");
        }
        return worker_of_device_[device_id];
    }

    // Declared before the executors so that it outlives them.
    Completion completion_;
    // Store the number of workers to repeated lookups
    uint32_t num_workers_ = 0;
    // parallel_for jobs, reused; declared before the executors, which refer to them.
    std::vector<std::unique_ptr<ParallelJob>> jobs_;
    std::atomic<uint32_t> last_job_ = 0;
    // The job of the current parallel_for, (generation << 32) | slot; every worker watches it.
    alignas(64) std::atomic<uint64_t> published_ = 0;
    // The published word after which idle workers park instead of spinning (park_workers).
    alignas(64) std::atomic<uint64_t> park_after_ = UINT64_MAX;
    // Worker by physical device id (ParallelJob::NONE if none).
    std::vector<uint32_t> worker_of_device_;
    // Executors backing this pool.
    std::vector<std::unique_ptr<NumaAwareExecutor>> workers_;
    // Used to pick threads when device_idx is not specified in the enqueue API
    std::atomic<uint32_t> thread_idx_ = 0;
    std::mutex exception_mutex_;
    // Mapping between the physical device id and its associated thread
    std::unordered_map<uint32_t, uint32_t> phys_device_to_thread_id_;
};

// Pass Through - This data structure is not backed by any worker threads. When a user enqueues a task,
// it is immediately executed.
// Primary Use Case: Single Device/Unit Mesh Dispatch.
class PassThroughThreadPool : public ThreadPool {
public:
    PassThroughThreadPool() = default;
    void enqueue(std::function<void()>&& f, std::optional<uint32_t> /*device_idx*/ = std::nullopt) override { f(); }
    void wait() override {}
    void parallel_for(ttsl::Span<const uint32_t> device_ids, const std::function<void(size_t)>& fn) override {
        run_on_caller(device_ids.size(), fn);
    }
};

}  // namespace thread_pool_impls

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(ContextId context_id, int num_threads) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(context_id, num_threads);
}

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(
    ContextId context_id, const std::vector<tt::tt_metal::Device*>& physical_devices) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(context_id, physical_devices);
}

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(
    ContextId context_id, const std::vector<tt::tt_metal::Device*>& physical_devices, std::chrono::microseconds spin) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(context_id, physical_devices, spin);
}

std::shared_ptr<ThreadPool> create_passthrough_thread_pool(ContextId /*context_id*/) {
    return std::make_shared<thread_pool_impls::PassThroughThreadPool>();
}

}  // namespace tt::tt_metal
