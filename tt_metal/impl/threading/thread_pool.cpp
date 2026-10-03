// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <future>
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

// Spins like std::atomic::wait before it parks (after 100 polls), so that callers can park on their own futex.
// Returns whether `ready` became true.
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
    void add() { pending_.fetch_add(1, std::memory_order_relaxed); }

    void done() {
        if (pending_.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
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
    NumaAwareExecutor(ContextId context_id, uint32_t physical_device_id, Completion& completion) :
        completion_(completion) {
        // Set the priority for this process to 0 (niceness value in linux)
        thread_binding::set_process_priority(0);
        worker = std::thread([this]() { run(); });

        auto cpu_core_for_worker = thread_binding::get_cpu_core_for_physical_device(context_id, physical_device_id);
        thread_binding::set_worker_affinity(worker, cpu_core_for_worker);
    }

    // Delete copy and move operations as this class manages a thread and should not be copied or moved
    NumaAwareExecutor(const NumaAwareExecutor&) = delete;
    NumaAwareExecutor& operator=(const NumaAwareExecutor&) = delete;
    NumaAwareExecutor(NumaAwareExecutor&&) = delete;
    NumaAwareExecutor& operator=(NumaAwareExecutor&&) = delete;

    // The owning pool waits for all tasks before destroying its executors.
    ~NumaAwareExecutor() {
        shutdown_.store(true, std::memory_order_seq_cst);
        wake();
        worker.join();
    }

    void enqueue(std::function<void()>&& f) {
        tasks_.push(std::move(f));
        wake();
    }

    // Returns the first exception thrown by a task since the last call, once the pool has joined.
    std::exception_ptr take_exception() { return std::exchange(stored_exception_, nullptr); }

private:
    void run() {
        while (true) {
            if (tasks_.empty()) {
                if (shutdown_.load(std::memory_order_acquire)) {
                    return;
                }
                wait_for_work();
                continue;
            }
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
        }
    }

    // Spin briefly so back-to-back tasks are picked up without entering the kernel, then park until a
    // producer publishes work or the executor shuts down.
    void wait_for_work() {
        if (spin_until([this] { return !tasks_.empty(); })) {
            return;
        }
        // seq_cst pairs with wake(): either the producer sees the worker parked, or the worker sees the task.
        parked_.store(1, std::memory_order_seq_cst);
        if (tasks_.empty() && !shutdown_.load(std::memory_order_seq_cst)) {
            futex_wait(parked_, 1);
        }
        parked_.store(0, std::memory_order_relaxed);
    }

    // Enters the kernel only if the worker is parked.
    void wake() {
        if (parked_.load(std::memory_order_seq_cst) != 0 && parked_.exchange(0, std::memory_order_seq_cst) != 0) {
            futex_wake_one(parked_);
        }
    }

    TaskQueue tasks_;
    Completion& completion_;
    std::thread worker;
    alignas(64) std::atomic<uint32_t> parked_ = 0;
    std::atomic<bool> shutdown_ = false;
    std::exception_ptr stored_exception_;
};

}  // namespace threading_primitives

namespace thread_pool_impls {
// Implementations conforming to the ThreadPool interface.
using threading_primitives::Completion;
using threading_primitives::NumaAwareExecutor;

// Custom Thread-Pool using the threading::Executor class.
// Allows enqueuing tasks tied to specific devices.
class DeviceBoundThreadPool : public ThreadPool {
public:
    // Constructor accepting the physical device IDs this pool is bound to. Each thread will be tied to a device, and is
    // guaranteed to be bound to a CPU core on a NUMA Node "closest" to that device.
    // All physical devices must belong to the same context ID.
    DeviceBoundThreadPool(ContextId context_id, const std::vector<tt::tt_metal::IDevice*>& physical_devices) :
        num_workers_(physical_devices.size()) {
        workers_.reserve(num_workers_);
        for (uint32_t i = 0; i < num_workers_; i++) {
            workers_.emplace_back(
                std::make_unique<NumaAwareExecutor>(context_id, physical_devices[i]->id(), completion_));
            phys_device_to_thread_id_[physical_devices[i]->id()] = i;
        }
    }
    // Constructor accepting the number of threads to spawn. The threads in this pool will be bound to a specific CPU
    // core but they are not guaranteed to be "close" to any physical device.
    DeviceBoundThreadPool(ContextId context_id, uint32_t thread_count) : num_workers_(thread_count) {
        workers_.reserve(thread_count);

        for (uint32_t i = 0; i < thread_count; i++) {
            workers_.emplace_back(std::make_unique<NumaAwareExecutor>(context_id, i, completion_));
            phys_device_to_thread_id_[i] = i;
        }
    }

    DeviceBoundThreadPool(const DeviceBoundThreadPool&) = delete;
    DeviceBoundThreadPool& operator=(const DeviceBoundThreadPool&) = delete;
    DeviceBoundThreadPool(DeviceBoundThreadPool&&) = delete;
    DeviceBoundThreadPool& operator=(DeviceBoundThreadPool&&) = delete;

    ~DeviceBoundThreadPool() override { completion_.wait(); }

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
    // Declared before the executors so that it outlives them.
    Completion completion_;
    // Executors backing this pool.
    std::vector<std::unique_ptr<NumaAwareExecutor>> workers_;
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
};

}  // namespace thread_pool_impls

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(ContextId context_id, int num_threads) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(context_id, num_threads);
}

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(
    ContextId context_id, const std::vector<tt::tt_metal::IDevice*>& physical_devices) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(context_id, physical_devices);
}

std::shared_ptr<ThreadPool> create_passthrough_thread_pool(ContextId /*context_id*/) {
    return std::make_shared<thread_pool_impls::PassThroughThreadPool>();
}

}  // namespace tt::tt_metal
