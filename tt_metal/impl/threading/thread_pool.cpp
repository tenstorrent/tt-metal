// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <linux/futex.h>
#include <sched.h>  // Needed for setting process priorities
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/resource.h>  // Needed for setting process priorities
#include <numa.h>
#include <tt-metalium/device.hpp>
#include <tt_stl/tt_pause.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/context_types.hpp"
#include "impl/threading/thread_pool.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace thread_binding {

// Physical core of a logical CPU, as (package, core). Returns nullopt when sysfs does not say.
std::optional<std::pair<int, int>> physical_core_of_cpu(int cpu) {
    const std::string topology = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    int package = -1;
    int core = -1;
    std::ifstream(topology + "physical_package_id") >> package;
    std::ifstream(topology + "core_id") >> core;
    if (package < 0 || core < 0) {
        return std::nullopt;
    }
    return std::pair{package, core};
}

// Drops the logical CPUs of the last `reserved_cores` physical cores of a NUMA node, so that threads outside the
// pools always have whole cores to run on. Keeps at least one core for the pools.
std::vector<uint32_t> without_reserved_cores(const std::vector<uint32_t>& cpus, uint32_t reserved_cores) {
    std::vector<std::pair<int, int>> cores;
    std::vector<std::optional<std::pair<int, int>>> core_of_cpu;
    for (auto cpu : cpus) {
        auto core = physical_core_of_cpu(cpu);
        if (!core) {
            return cpus;
        }
        core_of_cpu.push_back(core);
        if (std::find(cores.begin(), cores.end(), *core) == cores.end()) {
            cores.push_back(*core);
        }
    }
    const size_t kept_cores = cores.size() - std::min<size_t>(reserved_cores, cores.size() - 1);
    std::vector<uint32_t> kept;
    for (size_t i = 0; i < cpus.size(); i++) {
        if (std::find(cores.begin(), cores.begin() + kept_cores, *core_of_cpu[i]) != cores.begin() + kept_cores) {
            kept.push_back(cpus[i]);
        }
    }
    return kept;
}

std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node(uint32_t reserved_cores) {
    std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = {};
    if (numa_available() != -1) {
        for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
            int node = numa_node_of_cpu(cpu);
            cpu_cores_per_numa_node[node].push_back(cpu);
        }
    }
    if (reserved_cores > 0) {
        for (auto& [node, cpus] : cpu_cores_per_numa_node) {
            cpus = without_reserved_cores(cpus, reserved_cores);
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
    static std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node =
        get_cpu_cores_per_numa_node(MetalContext::instance(context_id).rtoptions().get_thread_pool_reserved_cores());
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

// Polls `ready` for the same window the pool spun before it parked on std::atomic::wait:
// 100 polls, then libstdc++'s 12 pause and 4 yield iterations. Returns whether `ready` became true.
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

// Tasks in flight across all workers of a pool, so that joining waits on one counter.
// A worker wakes the joining thread only if it is parked, and only when the count reaches zero.
class Completion {
public:
    void add(int64_t n = 1) { pending_.fetch_add(n, std::memory_order_relaxed); }

    bool finished() const { return pending_.load(std::memory_order_acquire) == 0; }

    void done(int64_t n = 1) {
        if (pending_.fetch_sub(n, std::memory_order_seq_cst) == n &&
            waiter_parked_.exchange(0, std::memory_order_seq_cst) != 0) {
            futex_wake_one(waiter_parked_);
        }
    }

    void wait() {
        // The joining thread has nothing else to do, so it spins for up to a long task before parking.
        constexpr auto JOIN_SPIN = std::chrono::microseconds(20);
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
        while (true) {
            // seq_cst pairs with done(): either done() sees the waiter parked, or the waiter sees zero.
            waiter_parked_.store(1, std::memory_order_seq_cst);
            if (pending_.load(std::memory_order_seq_cst) == 0) {
                waiter_parked_.store(0, std::memory_order_relaxed);
                return;
            }
            futex_wait(waiter_parked_, 1);
        }
    }

private:
    alignas(64) std::atomic<int64_t> pending_ = 0;
    alignas(64) std::atomic<uint32_t> waiter_parked_ = 0;
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
    NumaAwareExecutor(
        ContextId context_id,
        uint32_t physical_device_id,
        Completion& completion,
        std::chrono::nanoseconds active_spin = {}) :
        completion_(completion), active_spin_(active_spin), spin_window_(active_spin) {
        // Set the priority for this process to 0 (niceness value in linux)
        thread_binding::set_process_priority(0);
        worker = std::thread([this]() { run(); });

        auto cpu_core_for_worker = thread_binding::get_cpu_core_for_physical_device(context_id, physical_device_id);
        thread_binding::set_worker_affinity(worker, cpu_core_for_worker);
        core_ = thread_binding::physical_core_of_cpu(static_cast<int>(cpu_core_for_worker));
    }

    // Delete copy and move operations as this class manages a thread and should not be copied or moved
    NumaAwareExecutor(const NumaAwareExecutor&) = delete;
    NumaAwareExecutor& operator=(const NumaAwareExecutor&) = delete;
    NumaAwareExecutor(NumaAwareExecutor&&) = delete;
    NumaAwareExecutor& operator=(NumaAwareExecutor&&) = delete;

    // Drops a job offered after the worker stopped: while the pool stops its executors one by one, a worker
    // still running a job can hand it to one that has already stopped.
    ~NumaAwareExecutor();

    // Joins the worker. The owning pool waits for all tasks first, and stops every executor before destroying
    // any, because a parallel_for job on one worker can wake the others.
    void stop();

    void enqueue(std::function<void()>&& f) {
        tasks_.push(std::move(f));
        wake();
    }

    // Hands the worker a parallel_for job, taking over one of the job's references. A job the worker has not
    // picked up yet is dropped: its caller runs any calls it still has.
    void offer(ParallelJob* job);

    // Enters the kernel only if the worker is parked.
    void wake() {
        if (parked_.load(std::memory_order_seq_cst) != 0 && parked_.exchange(0, std::memory_order_seq_cst) != 0) {
            futex_wake_one(parked_);
        }
    }

    // The physical core the worker is pinned to, if sysfs says.
    const std::optional<std::pair<int, int>>& core() const { return core_; }

    // Returns the first exception thrown by a task since the last call, once the pool has joined.
    std::exception_ptr take_exception() { return std::exchange(stored_exception_, nullptr); }

private:
    void run();

    void note_work_done() {
        if (active_spin_.count() > 0) {
            last_work_ = std::chrono::steady_clock::now();
        }
    }

    bool has_work() const { return !tasks_.empty() || job_.load(std::memory_order_seq_cst) != nullptr; }

    // Spin briefly so back-to-back tasks are picked up without entering the kernel, then park until a
    // producer publishes work or the executor shuts down.
    void wait_for_work() {
        if (active_spin_.count() > 0) {
            const auto deadline = last_work_ + spin_window_;
            while (std::chrono::steady_clock::now() < deadline) {
                if (has_work() || shutdown_.load(std::memory_order_relaxed)) {
                    return;
                }
                ttsl::pause();
            }
        }
        if (spin_until([this] { return has_work(); })) {
            return;
        }
        // seq_cst pairs with wake(): either the producer sees the worker parked, or the worker sees the work.
        parked_.store(1, std::memory_order_seq_cst);
        if (!has_work() && !shutdown_.load(std::memory_order_seq_cst)) {
            futex_wait(parked_, 1);
        }
        parked_.store(0, std::memory_order_relaxed);
        // Work that comes back soon after the worker parked finds its core in a deep idle state, and the slow wake
        // can keep stretching the gap to the next work past the spin window. So the next spin covers twice such a
        // gap, up to MAX_SPIN_STRETCH times the configured window.
        if (active_spin_.count() > 0) {
            const auto max_window = MAX_SPIN_STRETCH * active_spin_;
            const std::chrono::nanoseconds gap = std::chrono::steady_clock::now() - last_work_;
            spin_window_ = gap < max_window ? std::clamp(2 * gap, active_spin_, max_window) : active_spin_;
        }
    }

    TaskQueue tasks_;
    Completion& completion_;
    std::thread worker;
    alignas(64) std::atomic<uint32_t> parked_ = 0;
    std::atomic<bool> shutdown_ = false;
    alignas(64) std::atomic<ParallelJob*> job_ = nullptr;
    static constexpr int MAX_SPIN_STRETCH = 8;
    const std::chrono::nanoseconds active_spin_;
    std::chrono::nanoseconds spin_window_;
    std::chrono::steady_clock::time_point last_work_;
    std::exception_ptr stored_exception_;
    std::optional<std::pair<int, int>> core_;
};

// State of one parallel_for call, shared by the caller and the participating workers. Workers can still hold
// it after the call has returned, so it is reference counted and deletes itself.
class ParallelJob {
public:
    // Each worker that takes part wakes this many others before running its own calls, so that the caller
    // enters the kernel once per fan-out instead of once per worker.
    static constexpr size_t WAKE_FANOUT = 2;

    ParallelJob(const std::function<void(size_t)>& fn, size_t num_calls) :
        fn_(fn), claims_(num_calls), executor_of_call_(num_calls) {
        remaining_.add(static_cast<int64_t>(num_calls));
    }

    ParallelJob(const ParallelJob&) = delete;
    ParallelJob& operator=(const ParallelJob&) = delete;
    ParallelJob(ParallelJob&&) = delete;
    ParallelJob& operator=(ParallelJob&&) = delete;
    ~ParallelJob() = default;

    // A call with no executor is left to the caller.
    void assign(size_t call, NumaAwareExecutor* executor) {
        if (executor == nullptr) {
            return;
        }
        executor_of_call_[call] = executor;
        if (std::find(participants_.begin(), participants_.end(), executor) == participants_.end()) {
            participants_.push_back(executor);
        }
    }

    const std::vector<NumaAwareExecutor*>& participants() const { return participants_; }

    // Takes one reference for the caller and one for the first participant, which the caller offers the job to.
    // Each participant offers it to the ones it wakes, taking a reference for each.
    void start() { refs_.store(2, std::memory_order_relaxed); }

    // Worker: hands the job to the participants below this one that still have calls to run, runs this worker's
    // calls, and drops the worker's reference. A job that has finished may point at executors being stopped, so it
    // is only dropped.
    void run_participant(NumaAwareExecutor* executor) {
        if (!remaining_.finished()) {
            const size_t position =
                std::find(participants_.begin(), participants_.end(), executor) - participants_.begin();
            for_each_child(position, [this](size_t child) { wake_subtree(child); });
            int64_t ran = 0;
            for (size_t call = 0; call < executor_of_call_.size(); call++) {
                if (executor_of_call_[call] == executor && run(call)) {
                    ran++;
                }
            }
            completed(ran);
        }
        release();
    }

    // Runs `call` unless another thread has claimed it, and returns whether it did. The caller reports the calls
    // it ran with completed(), once, so that threads do not contend on the count for every call.
    bool run(size_t call) {
        if (claims_[call].claimed.exchange(true, std::memory_order_acq_rel)) {
            return false;
        }
        try {
            fn_(call);
        } catch (...) {
            std::lock_guard lock(exception_mutex_);
            if (!exception_) {
                exception_ = std::current_exception();
            }
        }
        return true;
    }

    void completed(int64_t calls) {
        if (calls > 0) {
            remaining_.done(calls);
        }
    }

    // Caller: waits for every call, returns the first exception, and drops the caller's reference.
    std::exception_ptr finish() {
        remaining_.wait();
        auto exception = exception_;
        release();
        return exception;
    }

    void release() {
        if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

private:
    template <typename Fn>
    void for_each_child(size_t position, Fn fn) const {
        for (size_t child = (position * WAKE_FANOUT) + 1;
             child <= (position * WAKE_FANOUT) + WAKE_FANOUT && child < participants_.size();
             child++) {
            fn(child);
        }
    }

    // Offers the job to the participant at `position` and wakes it if the caller has not already claimed all of
    // its calls, and otherwise does the same for its children in its place.
    void wake_subtree(size_t position) {
        for (size_t call = 0; call < executor_of_call_.size(); call++) {
            if (executor_of_call_[call] == participants_[position] &&
                !claims_[call].claimed.load(std::memory_order_relaxed)) {
                refs_.fetch_add(1, std::memory_order_relaxed);
                participants_[position]->offer(this);
                participants_[position]->wake();
                return;
            }
        }
        for_each_child(position, [this](size_t child) { wake_subtree(child); });
    }

    struct alignas(64) Claim {
        std::atomic<bool> claimed = false;
    };

    // Valid until the caller returns, which happens only after every claimed call has finished.
    const std::function<void(size_t)>& fn_;
    std::vector<Claim> claims_;
    std::vector<NumaAwareExecutor*> executor_of_call_;
    std::vector<NumaAwareExecutor*> participants_;
    Completion remaining_;
    std::atomic<size_t> refs_ = 0;
    std::mutex exception_mutex_;
    std::exception_ptr exception_;
};

inline void NumaAwareExecutor::stop() {
    if (!worker.joinable()) {
        return;
    }
    shutdown_.store(true, std::memory_order_seq_cst);
    wake();
    worker.join();
}

inline NumaAwareExecutor::~NumaAwareExecutor() {
    stop();
    if (auto* job = job_.exchange(nullptr, std::memory_order_acquire)) {
        job->release();
    }
}

inline void NumaAwareExecutor::offer(ParallelJob* job) {
    // seq_cst pairs with the worker's check before it parks.
    if (auto* stale = job_.exchange(job, std::memory_order_seq_cst)) {
        stale->release();
    }
}

inline void NumaAwareExecutor::run() {
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
            note_work_done();
            continue;
        }
        if (job_.load(std::memory_order_relaxed) != nullptr) {
            if (auto* job = job_.exchange(nullptr, std::memory_order_acquire)) {
                job->run_participant(this);
                note_work_done();
            }
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

// Custom Thread-Pool using the threading::Executor class.
// Allows enqueuing tasks tied to specific devices.
class DeviceBoundThreadPool : public ThreadPool {
public:
    // Constructor accepting the physical device IDs this pool is bound to. Each thread will be tied to a device, and is
    // guaranteed to be bound to a CPU core on a NUMA Node "closest" to that device.
    // All physical devices must belong to the same context ID.
    DeviceBoundThreadPool(
        ContextId context_id,
        const std::vector<tt::tt_metal::IDevice*>& physical_devices,
        std::chrono::microseconds active_spin = {}) :
        num_workers_(physical_devices.size()) {
        workers_.reserve(num_workers_);
        for (uint32_t i = 0; i < num_workers_; i++) {
            workers_.emplace_back(
                std::make_unique<NumaAwareExecutor>(context_id, physical_devices[i]->id(), completion_, active_spin));
            phys_device_to_thread_id_[physical_devices[i]->id()] = i;
        }
        record_worker_cores();
    }
    // Constructor accepting the number of threads to spawn. The threads in this pool will be bound to a specific CPU
    // core but they are not guaranteed to be "close" to any physical device.
    DeviceBoundThreadPool(ContextId context_id, uint32_t thread_count) : num_workers_(thread_count) {
        workers_.reserve(thread_count);

        for (uint32_t i = 0; i < thread_count; i++) {
            workers_.emplace_back(std::make_unique<NumaAwareExecutor>(context_id, i, completion_));
            phys_device_to_thread_id_[i] = i;
        }
        record_worker_cores();
    }

    DeviceBoundThreadPool(const DeviceBoundThreadPool&) = delete;
    DeviceBoundThreadPool& operator=(const DeviceBoundThreadPool&) = delete;
    DeviceBoundThreadPool(DeviceBoundThreadPool&&) = delete;
    DeviceBoundThreadPool& operator=(DeviceBoundThreadPool&&) = delete;

    ~DeviceBoundThreadPool() override {
        completion_.wait();
        for (auto& worker : workers_) {
            worker->stop();
        }
    }

    void enqueue(std::function<void()>&& f, std::optional<uint32_t> device_idx = std::nullopt) override {
        // If the user does not provide the Device ID tied to this task, determine the thread to use
        // based on the internally stored thread_idx. Tasks will get round-robined across threads,
        // when relying on the thread_idx.
        // If the device id is specified, use the thread tied to the device.
        uint32_t thread_id =
            device_idx.has_value() ? phys_device_to_thread_id_[device_idx.value()] : ((thread_idx_++) % num_workers_);
        completion_.add();
        workers_[thread_id]->enqueue(std::move(f));
    }

    void parallel_for(ttsl::Span<const uint32_t> device_ids, const std::function<void(size_t)>& fn) override {
        if (device_ids.empty()) {
            return;
        }
        // Waking a worker for a single call can only cost time.
        if (device_ids.size() == 1) {
            fn(0);
            return;
        }
        auto* job = new ParallelJob(fn, device_ids.size());
        // A worker pinned to the caller's physical core would take CPU time from the caller, so the caller runs its
        // calls instead. An unpinned caller can land on a pool core, for example when woken by a pinned thread.
        const auto* caller_core = core_of_calling_thread();
        for (size_t call = 0; call < device_ids.size(); call++) {
            const uint32_t thread_id = phys_device_to_thread_id_.at(device_ids[call]);
            const bool shares_core = caller_core != nullptr && worker_cores_[thread_id] == *caller_core;
            job->assign(call, shares_core ? nullptr : workers_[thread_id].get());
        }
        job->start();
        const auto& participants = job->participants();
        if (participants.empty()) {
            job->release();  // the first participant's reference
        } else {
            // The first participant hands the job on to the others as it starts, so that the caller touches one
            // worker's state rather than every worker's.
            participants[0]->offer(job);
            participants[0]->wake();
        }
        // The workers are woken first to last, so take calls from the back. The caller runs every call no worker
        // has claimed, so the job finishes even if some workers are never woken.
        int64_t ran = 0;
        for (size_t call = device_ids.size(); call-- > 0;) {
            if (job->run(call)) {
                ran++;
            }
        }
        job->completed(ran);
        if (auto exception = job->finish()) {
            std::rethrow_exception(exception);
        }
    }

    void wait() override {
        thread_idx_ = 0;  // Reset thread_idx for next call without Device ID specified.
        completion_.wait();
        // Rethrow the first exception in the calling thread.
        std::exception_ptr exception;
        for (auto& worker : workers_) {
            auto temp_exception = worker->take_exception();
            if (!exception && temp_exception) {
                exception = temp_exception;
            }
        }
        if (exception) {
            std::rethrow_exception(exception);
        }
    }

private:
    const std::pair<int, int>* core_of_calling_thread() const {
        const int cpu = sched_getcpu();
        if (cpu < 0 || static_cast<size_t>(cpu) >= core_of_cpu_.size() || !core_of_cpu_[cpu]) {
            return nullptr;
        }
        return &*core_of_cpu_[cpu];
    }

    // Kept apart from the executors, whose cache lines their workers write.
    void record_worker_cores() {
        worker_cores_.reserve(workers_.size());
        for (const auto& worker : workers_) {
            worker_cores_.push_back(worker->core());
        }
    }

    static std::vector<std::optional<std::pair<int, int>>> cores_of_cpus() {
        std::vector<std::optional<std::pair<int, int>>> cores(std::max<long>(sysconf(_SC_NPROCESSORS_CONF), 0));
        for (size_t cpu = 0; cpu < cores.size(); cpu++) {
            cores[cpu] = thread_binding::physical_core_of_cpu(static_cast<int>(cpu));
        }
        return cores;
    }

    // Declared before the executors so that it outlives them.
    Completion completion_;
    // Physical core of each logical CPU, indexed by CPU.
    const std::vector<std::optional<std::pair<int, int>>> core_of_cpu_ = cores_of_cpus();
    // Executors backing this pool.
    std::vector<std::unique_ptr<NumaAwareExecutor>> workers_;
    // Physical core of each executor's worker, by thread id.
    std::vector<std::optional<std::pair<int, int>>> worker_cores_;
    // Used to pick threads when device_idx is not specified in the enqueue API
    uint32_t thread_idx_ = 0;
    // Store the number of workers to repeated lookups
    uint32_t num_workers_ = 0;
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
        for (size_t call = 0; call < device_ids.size(); call++) {
            fn(call);
        }
    }
};

}  // namespace thread_pool_impls

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(ContextId context_id, int num_threads) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(context_id, num_threads);
}

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(
    ContextId context_id,
    const std::vector<tt::tt_metal::IDevice*>& physical_devices,
    std::chrono::microseconds active_spin) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(context_id, physical_devices, active_spin);
}

std::shared_ptr<ThreadPool> create_passthrough_thread_pool(ContextId /*context_id*/) {
    return std::make_shared<thread_pool_impls::PassThroughThreadPool>();
}

}  // namespace tt::tt_metal
