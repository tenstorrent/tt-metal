// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <tuple>
#include <fmt/format.h>
#include <tt-metalium/host_api.hpp>
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common_types.hpp"
#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_memory_map.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>

namespace tt::tt_fabric::fabric_tests {

// ======================================================================================
// Memory Management
// ======================================================================================

/**
 * Manages memory resources for a single worker core
 */
class CoreResources {
public:
    CoreResources(
        uint32_t l1_alignment,
        uint32_t payload_chunk_size,
        const BaseMemoryRegion& payload_region,
        const BaseMemoryRegion& atomic_region) :
        payload_chunk_size(payload_chunk_size),
        l1_alignment_(l1_alignment),
        payload_region_(payload_region),
        atomic_region_(atomic_region),
        next_atomic_addr_(this->atomic_region_.start) {
        init_payload_buffer_allocator();
    }

    bool has_available_payload_chunk() const { return !available_payload_chunks_.empty(); }

    uint32_t allocate_payload_chunk() {
        if (!has_available_payload_chunk()) {
            TT_THROW("Out of payload memory on core.");
        }
        uint32_t addr = available_payload_chunks_.back();
        available_payload_chunks_.pop_back();
        return addr;
    }

    void reserve_payload_chunk(uint32_t addr) {
        auto it = std::find(available_payload_chunks_.begin(), available_payload_chunks_.end(), addr);
        if (it == available_payload_chunks_.end()) {
            TT_THROW("Attempting to reserve a payload chunk that is not available or already allocated.");
        }
        available_payload_chunks_.erase(it);
    }

    uint32_t allocate_atomic_counter() {
        uint32_t addr = next_atomic_addr_;
        if (addr + l1_alignment_ > atomic_region_.end()) {
            TT_THROW("Out of atomic counter memory on core.");
        }
        next_atomic_addr_ += l1_alignment_;
        return addr;
    }

    void reserve_atomic_counter(uint32_t addr) {
        if (addr + l1_alignment_ > atomic_region_.end()) {
            TT_THROW("Out of atomic counter memory on core.");
        }
        // validate alignment
        TT_FATAL(addr % l1_alignment_ == 0, "Atomic counter address {} is not properly aligned", addr);

        next_atomic_addr_ = addr + l1_alignment_;
    }

    uint32_t get_num_available_payload_chunks() const { return available_payload_chunks_.size(); }

    const std::vector<uint32_t>& get_available_payload_chunks() const { return available_payload_chunks_; }

    std::vector<uint32_t> get_available_atomic_counters() const {
        const uint32_t available_space = atomic_region_.end() - next_atomic_addr_;
        const uint32_t num_counters = available_space / l1_alignment_;

        std::vector<uint32_t> counters;
        counters.reserve(num_counters);

        for (uint32_t addr = next_atomic_addr_; addr + l1_alignment_ <= atomic_region_.end(); addr += l1_alignment_) {
            counters.push_back(addr);
        }
        return counters;
    }

    const uint32_t payload_chunk_size;

private:
    void init_payload_buffer_allocator() {
        uint32_t chunk_size = this->payload_chunk_size;
        TT_FATAL(
            chunk_size > 0 && chunk_size % l1_alignment_ == 0,
            "Payload chunk_size must be positive and a multiple of L1 alignment");

        available_payload_chunks_.clear();
        for (uint32_t addr = payload_region_.start; addr + chunk_size <= payload_region_.end(); addr += chunk_size) {
            available_payload_chunks_.push_back(addr);
        }
        // Allocate from the end of the buffer first
        std::reverse(available_payload_chunks_.begin(), available_payload_chunks_.end());
    }

    uint32_t l1_alignment_;
    BaseMemoryRegion payload_region_;
    BaseMemoryRegion atomic_region_;
    uint32_t next_atomic_addr_;
    std::vector<uint32_t> available_payload_chunks_;
};

// ======================================================================================
// Device-Local Resource Management
// ======================================================================================

struct CorePool {
    CorePool(const CoreAllocationConfig& policy) : policy(policy) {}

    std::vector<CoreCoord> get_available_cores(
        const std::unordered_map<CoreCoord, uint32_t>& core_workload, uint32_t partition_id = 0) const {
        std::vector<CoreCoord> available;

        // Use partition-local cores if partitioned
        const auto& cores_to_check = (num_partitions > 1) ? partition_cores.at(partition_id) : active_pool;

        available.reserve(cores_to_check.size());
        for (const auto& core : cores_to_check) {
            auto it = core_workload.find(core);
            if (it == core_workload.end() || it->second < policy.max_configs_per_core) {
                available.push_back(core);
            }
        }
        return available;
    }

    void initialize_with_cores(const std::vector<CoreCoord>& cores, uint32_t num_partitions = 1);
    void add_cores(const std::vector<CoreCoord>& new_cores);

    const CoreAllocationConfig& policy;
    std::vector<CoreCoord> active_pool;
    size_t next_pool_idx = 0;
    bool initialized = false;

    // Partition support
    uint32_t num_partitions = 1;
    std::unordered_map<CoreCoord, uint32_t> core_to_partition;
    std::unordered_map<uint32_t, std::vector<CoreCoord>> partition_cores;
    uint32_t next_partition_id = 0;
    std::unordered_map<uint32_t, size_t> next_pool_idx_per_partition;
};

// CorePool method implementations
inline void CorePool::initialize_with_cores(const std::vector<CoreCoord>& cores, uint32_t num_partitions) {
    TT_FATAL(!initialized, "Cannot re-initialize CorePool");

    // Sanity check - need at least 1 core per partition
    TT_FATAL(
        cores.size() >= num_partitions,
        "Cannot partition {} cores into {} partitions - need at least 1 core per partition",
        cores.size(),
        num_partitions);

    this->num_partitions = num_partitions;
    this->next_partition_id = 0;

    // Initialize per-partition tracking
    for (uint32_t p = 0; p < num_partitions; p++) {
        next_pool_idx_per_partition[p] = 0;
    }

    // Delegate to add_cores for partition assignment
    add_cores(cores);

    initialized = true;
}

inline void CorePool::add_cores(const std::vector<CoreCoord>& new_cores) {
    if (new_cores.empty()) {
        return;
    }

    active_pool.insert(active_pool.end(), new_cores.begin(), new_cores.end());
    std::sort(active_pool.begin(), active_pool.end());

    // Assign partitions and update both maps
    for (const auto& core : new_cores) {
        uint32_t pid = next_partition_id;
        core_to_partition[core] = pid;
        partition_cores[pid].push_back(core);
        next_partition_id = (next_partition_id + 1) % num_partitions;
    }
}

class TestDeviceResources {
public:
    TestDeviceResources(
        const FabricNodeId& node_id,
        const CoreCoord& worker_grid_size,
        uint32_t l1_alignment,
        uint32_t payload_chunk_size,
        const CoreAllocationConfig& sender_policy,
        const CoreAllocationConfig& receiver_policy,
        const BaseMemoryRegion& payload_region,
        const BaseMemoryRegion& atomic_region);

    void initialize_receiver_pool(uint32_t num_partitions = 1);
    CoreCoord reserve_sync_core();
    CoreCoord reserve_sender_core(const std::optional<CoreCoord>& specified_core);
    CoreCoord reserve_receiver_core(
        const std::optional<CoreCoord>& specified_core, uint32_t partition_id = 0, uint32_t num_partitions = 1);
    CoreResources& get_or_create_core_resources(const CoreCoord& core, CoreType core_type);

    const FabricNodeId node_id_;
    uint32_t l1_alignment_;
    uint32_t payload_chunk_size_;
    BaseMemoryRegion payload_region_;
    BaseMemoryRegion atomic_region_;
    std::vector<CoreCoord> pristine_cores_;                        // Cores not yet used at all.
    std::array<CorePool, 2> core_pools_;                           // Indexed by CoreType
    std::unordered_map<CoreCoord, uint32_t> core_workload_;        // map core -> num_configs
    std::unordered_map<CoreCoord, CoreResources> core_resources_;  // map core -> its memory allocator

    // Credit allocation: per-sender-core (needed for multi-link scenarios)
    std::unordered_map<CoreCoord, DynamicMemoryRegion> credit_allocators_;

    // Collect remaining pristine cores from all pools (for mux allocation)
    std::vector<CoreCoord> collect_remaining_pristine_cores() const;

    // Initialize credit allocator for a specific sender core (lazy)
    void initialize_credit_allocator(const CoreCoord& sender_core, const SenderMemoryMap& sender_memory_map);

    // Allocate credit chunk from a specific sender core's L1 credit region
    // Returns the base address of the allocated chunk
    // Throws if allocation would exceed region bounds
    uint32_t allocate_credit_chunk(
        const CoreCoord& sender_core, uint32_t num_receivers, const SenderMemoryMap& sender_memory_map);

    // Reset credit allocators
    void reset_credit_allocators();

private:
    void reserve_core_internal(const CoreCoord& core, CoreType core_type, uint32_t partition_id = 0);
    void assign_core_to_partition(CorePool& pool, const CoreCoord& core, uint32_t partition_id);
    CoreCoord find_next_available_core(CorePool& pool, uint32_t partition_id = 0);
    CoreCoord find_next_available_sync_core(CorePool& pool);
    void refill_pool(CorePool& pool);
};

// ======================================================================================
// Implementations for TestDeviceResources
// ======================================================================================

inline TestDeviceResources::TestDeviceResources(
    const FabricNodeId& node_id,
    const CoreCoord& worker_grid_size,
    uint32_t l1_alignment,
    uint32_t payload_chunk_size,
    const CoreAllocationConfig& sender_policy,
    const CoreAllocationConfig& receiver_policy,
    const BaseMemoryRegion& payload_region,
    const BaseMemoryRegion& atomic_region) :
    node_id_(node_id),
    l1_alignment_(l1_alignment),
    payload_chunk_size_(payload_chunk_size),
    payload_region_(payload_region),
    atomic_region_(atomic_region),
    core_pools_{CorePool(sender_policy), CorePool(receiver_policy)} {
    for (size_t y = 0; y < worker_grid_size.y; ++y) {
        for (size_t x = 0; x < worker_grid_size.x; ++x) {
            pristine_cores_.emplace_back(x, y);
        }
    }
    // Sort to ensure canonical order for determinism
    std::sort(pristine_cores_.begin(), pristine_cores_.end());
}

inline void TestDeviceResources::initialize_receiver_pool(uint32_t num_partitions) {
    CorePool& receiver_pool = core_pools_[RECEIVER_TYPE_IDX];
    if (!receiver_pool.initialized) {
        // All cores not used for senders are available for receivers.
        receiver_pool.initialize_with_cores(pristine_cores_, num_partitions);
        pristine_cores_.clear();

        if (num_partitions > 1) {
            log_debug(
                tt::LogTest,
                "Device {}: Initialized receiver pool with {} cores, {} partitions",
                node_id_,
                receiver_pool.active_pool.size(),
                num_partitions);
        }
    }
}

inline CoreCoord TestDeviceResources::reserve_sync_core() {
    CorePool& pool = core_pools_[SENDER_TYPE_IDX];
    if (!pool.initialized) {
        refill_pool(pool);
        pool.initialized = true;
    }

    CoreCoord core = find_next_available_sync_core(pool);
    reserve_core_internal(core, CoreType::SENDER);
    return core;
}

inline CoreCoord TestDeviceResources::reserve_sender_core(const std::optional<CoreCoord>& specified_core) {
    if (specified_core.has_value()) {
        reserve_core_internal(specified_core.value(), CoreType::SENDER);
        return specified_core.value();
    }

    CorePool& pool = core_pools_[SENDER_TYPE_IDX];
    if (!pool.initialized) {
        refill_pool(pool);
        pool.initialized = true;
    }

    CoreCoord core = find_next_available_core(pool);
    reserve_core_internal(core, CoreType::SENDER);
    return core;
}

inline CoreCoord TestDeviceResources::reserve_receiver_core(
    const std::optional<CoreCoord>& specified_core, uint32_t partition_id, uint32_t num_partitions) {
    // Lazy init with correct partition count
    if (!core_pools_[RECEIVER_TYPE_IDX].initialized) {
        initialize_receiver_pool(num_partitions);
    }

    CoreCoord core;
    if (specified_core.has_value()) {
        core = specified_core.value();
    } else {
        core = find_next_available_core(core_pools_[RECEIVER_TYPE_IDX], partition_id);
    }

    // reserve_core_internal handles ALL partition logic
    reserve_core_internal(core, CoreType::RECEIVER, partition_id);
    return core;
}

inline CoreResources& TestDeviceResources::get_or_create_core_resources(const CoreCoord& core, CoreType /*core_type*/) {
    if (!core_resources_.contains(core)) {
        core_resources_.emplace(
            core,
            CoreResources(
                l1_alignment_,
                payload_chunk_size_,  // Use constant from receiver memory map
                payload_region_,
                atomic_region_));
    }
    return core_resources_.at(core);
}

inline void TestDeviceResources::refill_pool(CorePool& pool) {
    uint32_t refill_count = pool.active_pool.empty() ? pool.policy.initial_pool_size : pool.policy.pool_refill_size;

    std::vector<CoreCoord> new_cores;

    for (uint32_t i = 0; i < refill_count; ++i) {
        if (pristine_cores_.empty()) {
            if (pool.active_pool.empty()) {
                TT_THROW("No more pristine cores available to create a new active pool.");
            }
            break;
        }
        new_cores.push_back(pristine_cores_.back());
        pristine_cores_.pop_back();
    }

    // Use CorePool::add_cores to handle partition assignment
    pool.add_cores(new_cores);
}

inline void TestDeviceResources::initialize_credit_allocator(
    const CoreCoord& sender_core, const SenderMemoryMap& sender_memory_map) {
    auto it = credit_allocators_.find(sender_core);
    if (it == credit_allocators_.end()) {
        credit_allocators_.emplace(
            sender_core,
            DynamicMemoryRegion(
                sender_memory_map.get_credit_addresses_base(),
                sender_memory_map.get_credit_addresses_size(),
                SenderMemoryMap::CREDIT_ADDRESS_STRIDE));
    }
}

inline uint32_t TestDeviceResources::allocate_credit_chunk(
    const CoreCoord& sender_core, uint32_t num_receivers, const SenderMemoryMap& sender_memory_map) {
    // Lazy initialization per sender core
    auto it = credit_allocators_.find(sender_core);
    if (it == credit_allocators_.end()) {
        initialize_credit_allocator(sender_core, sender_memory_map);
        it = credit_allocators_.find(sender_core);
    }

    // Allocate from this sender core's allocator
    uint32_t chunk_base = it->second.allocate_chunk(num_receivers);
    return chunk_base;
}

inline void TestDeviceResources::reset_credit_allocators() {
    for (auto& [core, allocator] : credit_allocators_) {
        allocator.reset();
    }
}

inline std::vector<CoreCoord> TestDeviceResources::collect_remaining_pristine_cores() const {
    std::vector<CoreCoord> remaining_cores;

    // Collect from pristine_cores_ (if not yet moved to pools)
    remaining_cores.insert(remaining_cores.end(), pristine_cores_.begin(), pristine_cores_.end());

    // Collect from sender pool (cores that haven't been allocated yet)
    const CorePool& sender_pool = core_pools_[SENDER_TYPE_IDX];
    if (sender_pool.initialized) {
        for (const auto& core : sender_pool.active_pool) {
            auto it = core_workload_.find(core);
            if (it == core_workload_.end()) {
                // Core is in pool but has never been allocated
                remaining_cores.push_back(core);
            }
        }
    }

    // Collect from receiver pool (cores that haven't been allocated yet)
    const CorePool& receiver_pool = core_pools_[RECEIVER_TYPE_IDX];
    if (receiver_pool.initialized) {
        for (const auto& core : receiver_pool.active_pool) {
            auto it = core_workload_.find(core);
            if (it == core_workload_.end()) {
                // Core is in pool but has never been allocated
                remaining_cores.push_back(core);
            }
        }
    }

    // Pools are mutually exclusive, no duplicates possible
    return remaining_cores;
}

inline CoreCoord TestDeviceResources::find_next_available_sync_core(CorePool& pool) {
    size_t current_pool_idx = pool.next_pool_idx;
    const CoreCoord& core = pool.active_pool[current_pool_idx];
    pool.next_pool_idx = (current_pool_idx + 1) % pool.active_pool.size();
    return core;
}

inline CoreCoord TestDeviceResources::find_next_available_core(CorePool& pool, uint32_t partition_id) {
    // Use per-partition tracking
    size_t& next_idx = pool.next_pool_idx_per_partition[partition_id];
    const auto& partition_cores = pool.partition_cores[partition_id];

    while (true) {
        // Search partition-local cores
        for (size_t i = 0; i < partition_cores.size(); ++i) {
            size_t idx_to_check = (next_idx + i) % partition_cores.size();
            const CoreCoord& core = partition_cores[idx_to_check];

            // Check availability
            auto it = core_workload_.find(core);
            if (it == core_workload_.end() || it->second < pool.policy.max_configs_per_core) {
                // Found available core. Update index based on policy
                if (pool.policy.policy == CoreAllocationPolicy::ExhaustFirst) {
                    if (it == core_workload_.end() || (it->second + 1) < pool.policy.max_configs_per_core) {
                        next_idx = idx_to_check;
                    } else {
                        next_idx = (idx_to_check + 1) % partition_cores.size();
                    }
                } else {  // RoundRobin
                    next_idx = (idx_to_check + 1) % partition_cores.size();
                }
                return core;
            }
        }

        // Pool exhausted, try to refill
        size_t old_pool_size = pool.active_pool.size();
        refill_pool(pool);

        if (pool.active_pool.size() == old_pool_size) {
            TT_THROW("No available core found and the pool could not be refilled. All cores are at max workload.");
        }

        // Reset to start of partition (refill already updated partition_cores)
        next_idx = 0;
    }
}

inline void TestDeviceResources::assign_core_to_partition(
    CorePool& pool, const CoreCoord& core, uint32_t partition_id) {
    auto partition_it = pool.core_to_partition.find(core);

    if (partition_it != pool.core_to_partition.end()) {
        // Core already in pool
        if (partition_it->second == partition_id) {
            return;  // Already in correct partition, nothing to do
        }

        // Move from old partition to new partition
        uint32_t old_partition = partition_it->second;

        // Remove from old partition's vector
        auto& old_partition_cores = pool.partition_cores[old_partition];
        old_partition_cores.erase(
            std::remove(old_partition_cores.begin(), old_partition_cores.end(), core), old_partition_cores.end());
    }

    // Add to new partition (works for both move and fresh assignment)
    pool.core_to_partition[core] = partition_id;
    pool.partition_cores[partition_id].push_back(core);
    std::sort(pool.partition_cores[partition_id].begin(), pool.partition_cores[partition_id].end());
}

inline void TestDeviceResources::reserve_core_internal(
    const CoreCoord& core, CoreType core_type, uint32_t partition_id) {
    CorePool& pool = core_pools_[static_cast<size_t>(core_type)];
    CorePool& other_pool =
        core_pools_[static_cast<size_t>(core_type == CoreType::SENDER ? CoreType::RECEIVER : CoreType::SENDER)];

    // Check if the core is already active in the OTHER pool. This is a critical exclusivity check.
    auto other_it = std::find(other_pool.active_pool.begin(), other_pool.active_pool.end(), core);
    if (other_it != other_pool.active_pool.end()) {
        TT_THROW(
            "Cannot reserve core [{}, {}] on device {} for type {}: It is already active in the pool for type {}.",
            core.x,
            core.y,
            node_id_,
            (core_type == CoreType::SENDER ? "SENDER" : "RECEIVER"),
            (core_type == CoreType::SENDER ? "RECEIVER" : "SENDER"));
    }

    // Check partition conflict (only for RECEIVER with partitions)
    if (core_type == CoreType::RECEIVER && pool.num_partitions > 1) {
        auto partition_it = pool.core_to_partition.find(core);
        if (partition_it != pool.core_to_partition.end() && partition_it->second != partition_id) {
            // Core is in different partition - check if it's active
            auto workload_it = core_workload_.find(core);
            if (workload_it != core_workload_.end() && workload_it->second > 0) {
                TT_THROW(
                    "Cannot reserve core {} on device {} for partition {}: "
                    "core is already in use by partition {} with {} configs",
                    core.str(),
                    node_id_,
                    partition_id,
                    partition_it->second,
                    workload_it->second);
            }
        }
    }

    // If core is pristine, it must be added to the active pool first.
    auto pristine_it = std::find(pristine_cores_.begin(), pristine_cores_.end(), core);
    if (pristine_it != pristine_cores_.end()) {
        pristine_cores_.erase(pristine_it);
        // Put it in the active pool so it's not "lost" from tracking.
        pool.active_pool.push_back(core);
        std::sort(pool.active_pool.begin(), pool.active_pool.end());
    }

    // Assign/move to correct partition (only for RECEIVER with partitions)
    if (core_type == CoreType::RECEIVER && pool.num_partitions > 1) {
        assign_core_to_partition(pool, core, partition_id);
    }

    if (core_workload_[core] >= pool.policy.max_configs_per_core) {
        TT_THROW(
            "Cannot reserve core [{}, {}] on device {}: It is already at its maximum workload of {} for this core "
            "type.",
            core.x,
            core.y,
            node_id_,
            pool.policy.max_configs_per_core);
    }
    core_workload_[core]++;

    get_or_create_core_resources(core, core_type);
}

// ======================================================================================
// Credit Allocation Helpers
// ======================================================================================

/**
 * Calculate credit configuration: the sender starts with the receive ring's full capacity, and the receiver returns
 * credits every half ring. Each return is a header-only packet through the fabric, and the routers are limited by
 * packet count rather than bytes, so larger batches leave more of the fabric for data.
 */
inline static std::pair<uint32_t, uint32_t> calculate_credit_config(
    uint32_t buffer_capacity_bytes, uint32_t packet_size_bytes, uint32_t num_packets) {
    uint32_t buffer_capacity_packets = buffer_capacity_bytes / packet_size_bytes;
    uint32_t initial_credits = std::min(buffer_capacity_packets, num_packets);
    uint32_t batch_size = std::max(1u, initial_credits / 2);
    return {initial_credits, batch_size};
}

// ======================================================================================
// Global Allocator
// ======================================================================================

class GlobalAllocator {
public:
    GlobalAllocator(
        const IDeviceInfoProvider& device_info_provider,
        const IRouteManager& route_manager,
        const AllocatorPolicies& policies,
        const SenderMemoryMap& sender_memory_map,
        const ReceiverMemoryMap& receiver_memory_map);

    void allocate_resources(TestConfig& test_config);
    void reset();

    // Get pristine cores for a device (for local mux allocation)
    std::vector<CoreCoord> get_pristine_cores_for_device(const FabricNodeId& node_id) const;

private:
    TestDeviceResources& get_or_create_device_resources(const FabricNodeId& node_id);

    const IDeviceInfoProvider& device_info_provider_;
    const IRouteManager& route_manager_;
    AllocatorPolicies policies_;
    const SenderMemoryMap& sender_memory_map_;
    const ReceiverMemoryMap& receiver_memory_map_;
    std::optional<CoreCoord> worker_grid_size_;
    std::unordered_map<FabricNodeId, std::unique_ptr<TestDeviceResources>> all_device_resources_;
    bool enable_flow_control_ = false;  // Set during allocate_resources, used during device creation
};

inline GlobalAllocator::GlobalAllocator(
    const IDeviceInfoProvider& device_info_provider,
    const IRouteManager& route_manager,
    const AllocatorPolicies& policies,
    const SenderMemoryMap& sender_memory_map,
    const ReceiverMemoryMap& receiver_memory_map) :
    device_info_provider_(device_info_provider),
    route_manager_(route_manager),
    policies_(policies),
    sender_memory_map_(sender_memory_map),
    receiver_memory_map_(receiver_memory_map) {}

inline TestDeviceResources& GlobalAllocator::get_or_create_device_resources(const FabricNodeId& node_id) {
    auto it = all_device_resources_.find(node_id);
    if (it != all_device_resources_.end()) {
        return *it->second;
    }

    // Create new device resources
    if (!worker_grid_size_.has_value()) {
        worker_grid_size_ = device_info_provider_.get_worker_grid_size();
    }

    auto [inserted_it, success] = all_device_resources_.emplace(
        node_id,
        std::make_unique<TestDeviceResources>(
            node_id,
            worker_grid_size_.value(),
            device_info_provider_.get_l1_alignment(),  // Get directly from device info provider
            policies_.default_payload_chunk_size,
            policies_.sender_config,
            policies_.receiver_config,
            receiver_memory_map_.payload_chunks,
            receiver_memory_map_.atomic_counters));

    return *inserted_it->second;
}

/**
 * Manages dynamic allocation policy computation and caching.
 *
 * Responsibilities:
 * - Compute optimal allocation policies for a given test configuration
 * - Cache policies based on topology key (name + num_links)
 * - Determine when reconstruction is needed (iteration 0 or topology change)
 */
class DynamicPolicyManager {
public:
    DynamicPolicyManager(const IDeviceInfoProvider& device_info_provider, const IRouteManager& route_manager) :
        device_info_provider_(device_info_provider), route_manager_(route_manager) {}

    /**
     * Get new policy for a test configuration if recomputation is needed.
     *
     * Returns std::nullopt if the cached policy can be reused (same topology, not iteration 0).
     * Returns a new policy if:
     * - First iteration (iteration_number == 0), OR
     * - Topology changed (different name or num_links)
     *
     */
    std::optional<AllocatorPolicies> get_new_policy_for_test(const TestConfig& config) {
        // Build topology key from parameters that affect allocation policy:
        // - config.name: Different tests have different sender/receiver patterns
        // - num_links: Link duplication creates additional sender cores, affecting reserved cores
        //
        // Parameters that DON'T affect policy (and thus share the same key):
        // - size/num_packets: Only affects validation, not resource allocation
        // - ftype/ntype: Only affects routing, not allocation
        //
        // Format: "test_name:links_N" (e.g., "FlowControlMesh:links_2")
        std::string topology_key = fmt::format("{}:links_{}", config.name, config.fabric_setup.num_links);

        bool needs_recomputation = (config.iteration_number == 0) || (topology_key != last_topology_key_);

        if (needs_recomputation) {
            // Compute fresh policy
            cached_policy_ = compute_policy(config);
            last_topology_key_ = topology_key;

            return cached_policy_;  // Return new policy
        }
        return std::nullopt;  // Signal to reuse existing policy
    }

    void reset() {
        last_topology_key_.clear();
        cached_policy_ = AllocatorPolicies{};
    }

    /**
     * Split senders with more configs than one core runs on its fast paths across several sender cores.
     *
     * A sender core runs at most get_max_hot_state_configs() configs from its hot path and generates payloads on its
     * compute engine for at most SFPU_PAYLOAD_MAX_CONFIGS; past either it falls back to slower paths. Each piece costs
     * a sender core and a full-size channel on every mux its flows go through. A mux serves at most
     * get_max_mux_channels() channels, and its channel buffers must fit in its core's L1
     * (FabricMuxConfig::compute_memory_map_end_address()), so for each (device, link) this picks the smallest piece
     * size that still fits both:
     * 1. The hot-path cap, if the mux still fits receiver cores with at most SFPU_VERIFY_MAX_CONFIGS configs each, so
     *    receivers keep deep rings and can verify on their compute engine.
     * 2. Otherwise SFPU_PAYLOAD_MAX_CONFIGS, if the mux fits receiver cores at the most configs per core.
     * 3. Otherwise the largest pieces that fit, or no split at all.
     * Splitting applies only to validated traffic: kernel benchmark mode and latency tests run neither fast path.
     *
     * Flows are dealt out round-robin (with K pieces, piece p takes flows p, p + K, p + 2K, ...), so the pieces are
     * near-equal in size and each holds a mix of near and far destinations. Senders with a pinned core are not split.
     *
     * fixed_receiver_configs_per_core: the receivers' configs per core when the allocation policy is given rather than
     * computed per test.
     */
    static void split_senders_to_fit_core_limits(
        TestConfig& config,
        const IRouteManager& route_manager,
        const IDeviceInfoProvider& device_info_provider,
        std::optional<uint32_t> fixed_receiver_configs_per_core = std::nullopt) {
        // Latency / bandwidth test mode and non-packet-validation mode don't use hot state, so don't benefit here
        if (config.performance_test_mode != PerformanceTestMode::NONE || config.skip_packet_validation) {
            return;
        }
        const uint32_t hot_state_max_configs = get_max_hot_state_configs();
        const uint32_t max_mux_channels = get_max_mux_channels();
        // A test mux is a worker core whose memory map starts at the unreserved base
        const auto& mesh_device = device_info_provider.get_mesh_device();
        const size_t mux_base_address = device_info_provider.get_l1_unreserved_base();
        const size_t full_size_buffer_bytes = device_info_provider.get_channel_buffer_size_bytes();
        const size_t mux_l1_end = tt::tt_fabric::FabricMuxConfig::compute_l1_end_address(mesh_device);
        const auto receive_configs = count_receive_configs_per_device_and_link(config, route_manager);

        // Create ({device id, routing plane (aka link id)}, sender config indices) map
        std::map<std::pair<FabricNodeId, uint32_t>, std::vector<size_t>> senders_per_device_and_link;
        for (size_t i = 0; i < config.senders.size(); i++) {
            senders_per_device_and_link[{config.senders[i].device, config.senders[i].link_id}].push_back(i);
        }

        // Most configs per piece for each sender; 0 leaves the sender whole.
        std::vector<size_t> max_configs_per_piece(config.senders.size(), 0);
        for (const auto& [device_and_link, sender_indices] : senders_per_device_and_link) {
            std::vector<size_t> splittable;
            size_t largest = 0;
            for (const size_t i : sender_indices) {
                const auto& sender = config.senders[i];

                // We split senders that don't have a pinned core + have more configs than the hot state cap
                if (!sender.core.has_value() && sender.patterns.size() > hot_state_max_configs) {
                    splittable.push_back(i);
                    largest = std::max(largest, sender.patterns.size());
                }
            }

            // No senders to split so continue
            if (splittable.empty()) {
                continue;
            }

            // Counted as if every sender, receiver and sync core on this link used one mux (they may spread over
            // several directions), as the allocator's receiver policy does. Receivers take channels only to return
            // credits.
            const auto receive_it = receive_configs.find(device_and_link);
            // Receivers only use the mux when returning credits
            const uint32_t num_receive_configs =
                (!config.enable_flow_control || receive_it == receive_configs.end()) ? 0 : receive_it->second;
            // Senders that don't need to be split keep one full-size channel each
            const size_t unsplit_senders = sender_indices.size() - splittable.size();

            // Check if the given configs per piece + receiver configs per core fit the mux: its channel cap, and its
            // channel buffers in L1. Sender pieces take full-size channels; sync and receivers (returning credits)
            // take header-only channels.
            const auto fits = [&](size_t configs_per_piece, uint32_t receiver_configs_per_core) {
                size_t pieces = 0;
                for (const size_t i : splittable) {
                    pieces += tt::div_up(config.senders[i].patterns.size(), configs_per_piece);
                }
                const uint32_t configs_per_core = fixed_receiver_configs_per_core.value_or(receiver_configs_per_core);
                const size_t receiver_cores = tt::div_up(num_receive_configs, std::max(configs_per_core, 1u));
                const size_t full_size_channels = unsplit_senders + pieces;
                const size_t header_only_channels = 1 /* sync */ + receiver_cores;
                if (full_size_channels + header_only_channels > max_mux_channels) {
                    return false;
                }
                return tt::tt_fabric::FabricMuxConfig::compute_memory_map_end_address(
                           mesh_device,
                           static_cast<uint8_t>(full_size_channels),
                           static_cast<uint8_t>(header_only_channels),
                           MUX_BUFFERS_PER_CHANNEL,
                           MUX_BUFFERS_PER_CHANNEL,
                           full_size_buffer_bytes,
                           mux_base_address) <= mux_l1_end;
            };

            // Two cases:
            // 1. The hot state max configs fit the mux channel cap, so receivers hold a few configs each and can verify
            // on sfpu.
            // 2. Else, fallback to packing receiver configs on receivers on densely as possible. This uses the fewest
            // receiver cores (more cores freed up for senders), but not all receiver configs will be verified by sfpu.
            size_t chosen = 0;
            if (fits(hot_state_max_configs, SFPU_VERIFY_MAX_CONFIGS)) {
                chosen = hot_state_max_configs;
            } else {
                // Find the smallest piece size that fits the mux channel cap
                for (size_t configs_per_piece = std::max<size_t>(SFPU_PAYLOAD_MAX_CONFIGS, hot_state_max_configs + 1);
                     configs_per_piece < largest;
                     configs_per_piece++) {
                    if (fits(configs_per_piece, MAX_CONFIGS_PER_CORE_CEILING)) {
                        chosen = configs_per_piece;
                        break;
                    }
                }
            }

            // Set the chosen configs per piece for each sender that needs to be split
            for (const size_t i : splittable) {
                max_configs_per_piece[i] = chosen;
            }
        }

        keep_senders_whole_where_cores_run_out(
            config, receive_configs, device_info_provider, fixed_receiver_configs_per_core, max_configs_per_piece);

        const size_t num_senders_before = config.senders.size();
        size_t largest_before = 0;
        size_t largest_after = 0;
        std::vector<SenderConfig> senders;
        senders.reserve(config.senders.size());
        for (size_t s = 0; s < config.senders.size(); s++) {
            auto& sender = config.senders[s];
            const size_t num_patterns = sender.patterns.size();
            largest_before = std::max(largest_before, num_patterns);

            // No need to split if # patterns for this sender < chosen configs per piece
            if (max_configs_per_piece[s] == 0 || num_patterns <= max_configs_per_piece[s]) {
                largest_after = std::max(largest_after, num_patterns);
                senders.push_back(std::move(sender));
                continue;
            }

            // Split the sender into pieces, each with at most max_configs_per_piece[s] patterns.
            // We use a round-robin approach to distribute the patterns (by distance) evenly across the pieces.
            const size_t num_pieces = tt::div_up(num_patterns, max_configs_per_piece[s]);
            auto patterns = std::move(sender.patterns);
            sender.patterns.clear();
            sender.split_into = static_cast<uint32_t>(num_pieces);
            std::vector<SenderConfig> pieces(num_pieces, sender);
            for (size_t i = 0; i < num_patterns; i++) {
                pieces[i % num_pieces].patterns.push_back(std::move(patterns[i]));
            }
            for (auto& piece : pieces) {
                largest_after = std::max(largest_after, piece.patterns.size());
                senders.push_back(std::move(piece));
            }
        }
        config.senders = std::move(senders);
        if (config.senders.size() != num_senders_before) {
            log_info(
                tt::LogTest,
                "Split {} senders (largest {} configs) into {} senders (largest {} configs) to fit the per-core limits "
                "(hot path {}, SFPU payload generation {}, mux channels {})",
                num_senders_before,
                largest_before,
                config.senders.size(),
                largest_after,
                hot_state_max_configs,
                SFPU_PAYLOAD_MAX_CONFIGS,
                max_mux_channels);
        }
    }

    // The links on which some router channel (direction, link, VC) of a device has two or more sender cores (devices
    // without any are left out). The connection manager gives those senders a mux on that channel, which every other
    // client of the channel must also go through.
    static std::map<FabricNodeId, std::set<uint32_t>> find_links_shared_by_senders(
        const TestConfig& config, const IRouteManager& route_manager, const IDeviceInfoProvider& device_info_provider) {
        using RouterChannel = std::tuple<FabricNodeId, RoutingDirection, uint32_t /* link */, uint8_t /* vc */>;
        std::map<RouterChannel, /* num sender cores */ uint32_t> sender_cores_per_router_channel;
        for (const auto& sender : config.senders) {
            const uint8_t vc_id = sender.vc_id.value_or(default_worker_vc_id);
            // A sender core is one client of each channel, however many of its configs use it
            std::set<RoutingDirection> directions;
            for (const auto& pattern : sender.patterns) {
                for (const auto direction : get_pattern_outgoing_directions(
                         sender.device, pattern, config.fabric_setup.topology, route_manager, device_info_provider)) {
                    directions.insert(direction);
                }
            }
            for (const auto direction : directions) {
                sender_cores_per_router_channel[{sender.device, direction, sender.link_id, vc_id}]++;
            }
        }
        std::map<FabricNodeId, std::set<uint32_t>> shared_links;
        for (const auto& [router_channel, num_sender_cores] : sender_cores_per_router_channel) {
            if (num_sender_cores > 1) {
                shared_links[/* device id*/std::get<0>(router_channel)].insert(/* link id */ std::get<2>(router_channel));
            }
        }
        return shared_links;
    }

    // The directions in which a sender's packets for one pattern leave its chip. The pattern's hops and first
    // destination are resolved as TestContext::add_traffic_config resolves them for the sender.
    static std::vector<RoutingDirection> get_pattern_outgoing_directions(
        const FabricNodeId& src_node_id,
        const TrafficPatternConfig& pattern,
        tt::tt_fabric::Topology topology,
        const IRouteManager& route_manager,
        const IDeviceInfoProvider& device_info_provider) {
        const auto& dest = pattern.destination.value();
        std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops;
        FabricNodeId dst_node_id = src_node_id;

        // Get the hops for the pattern + the destination node id
        if (dest.hops.has_value()) { // multicast case: keep the hops + first landing destination
            hops = dest.hops;
            auto hops_copy = dest.hops.value();
            dst_node_id = route_manager.get_dst_node_ids_from_hops(src_node_id, hops_copy, pattern.ftype.value())[0];
        } else {
            dst_node_id = dest.device.value();
            if (src_node_id.mesh_id == dst_node_id.mesh_id) {
                hops = route_manager.get_hops_to_chip(src_node_id, dst_node_id);
            }
        }

        // Find the actual directions in which the pattern's packets leave the source device
        return get_sender_outgoing_directions(
            route_manager,
            src_node_id,
            hops,
            dst_node_id,
            pattern.ftype.value(),
            device_info_provider.is_2D_routing_enabled(),
            topology);
    }

private:
    /**
     * Undo the sender split on any device whose worker grid can't hold the result of the split. Each resulting piece of
     * the split takes a sender core, and two senders on one link need a mux core in every direction they send in.
     * A device that doesn't fit the split keeps all its senders whole.
     */
    static void keep_senders_whole_where_cores_run_out(
        const TestConfig& config,
        const std::map<std::pair<FabricNodeId, uint32_t>, uint32_t>& receive_configs,
        const IDeviceInfoProvider& device_info_provider,
        std::optional<uint32_t> fixed_receiver_configs_per_core,
        std::vector<size_t>& max_configs_per_piece) {
        const CoreCoord worker_grid = device_info_provider.get_worker_grid_size();
        const uint32_t total_worker_cores = worker_grid.x * worker_grid.y;
        const uint32_t mux_cores_per_link = device_info_provider.get_max_connections_per_device();
        const uint32_t receiver_configs_per_core =
            std::max(fixed_receiver_configs_per_core.value_or(MAX_CONFIGS_PER_CORE_CEILING), 1u);

        // Sender cores per (device, link) after the split, and which senders of each device would be split
        std::map<std::pair<FabricNodeId, uint32_t>, uint32_t> sender_cores_per_device_and_link;
        std::map<FabricNodeId, std::vector<size_t>> split_senders_per_device;
        for (size_t s = 0; s < config.senders.size(); s++) {
            const auto& sender = config.senders[s];
            const size_t num_patterns = sender.patterns.size();
            const bool split = max_configs_per_piece[s] != 0 && num_patterns > max_configs_per_piece[s];
            sender_cores_per_device_and_link[{sender.device, sender.link_id}] +=
                split ? tt::div_up(num_patterns, max_configs_per_piece[s]) : 1;
            if (split) {
                split_senders_per_device[sender.device].push_back(s);
            }
        }

        // Go through each device that had a split sender, and ensure that the worker grid can hold the result of the
        // split (enough cores exist for the sender cores, mux cores, receiver cores, sync core, and safety margin).
        for (const auto& [device, split_senders] : split_senders_per_device) {
            uint32_t sender_cores = 0;
            uint32_t mux_cores = 0;
            for (const auto& [device_and_link, cores] : sender_cores_per_device_and_link) {
                // Only count cores for this device
                if (device_and_link.first != device) {
                    continue;
                }
                // Increment # of mux cores needed
                sender_cores += cores;
                if (cores > 1 || config.enable_flow_control) {
                    mux_cores += mux_cores_per_link;
                }
            }

            // Count receiver cores needed for this device
            uint32_t receiver_cores = 0;
            for (const auto& [device_and_link, num_configs] : receive_configs) {
                if (device_and_link.first == device) {
                    receiver_cores += tt::div_up(num_configs, receiver_configs_per_core);
                }
            }

            // Count sync core needed for this device
            const uint32_t sync_cores = config.global_sync ? 1 : 0;

            // Grid can fit all the cores needed so we can fit the split
            const uint32_t cores_needed = sender_cores + mux_cores + receiver_cores + sync_cores + SAFETY_MARGIN_CORES;
            if (cores_needed <= total_worker_cores) {
                continue;
            }

            // Undo the split for this device
            log_info(
                tt::LogTest,
                "Keeping {} senders on device {} whole: split, they would need {} of its {} worker cores ({} sender, "
                "{} mux, {} receiver, {} sync, {} margin)",
                split_senders.size(),
                device,
                cores_needed,
                total_worker_cores,
                sender_cores,
                mux_cores,
                receiver_cores,
                sync_cores,
                SAFETY_MARGIN_CORES);
            for (const size_t s : split_senders) {
                max_configs_per_piece[s] = 0;
            }
        }
    }

    // Receive configs landing on each device, keyed by (device, link of the sending core). A multicast counts once on
    // every device it reaches.
    static std::map<std::pair<FabricNodeId, uint32_t>, uint32_t> count_receive_configs_per_device_and_link(
        const TestConfig& config, const IRouteManager& route_manager) {
        std::map<std::pair<FabricNodeId, uint32_t>, uint32_t> receive_configs;
        for (const auto& sender : config.senders) {
            for (const auto& pattern : sender.patterns) {
                const auto& dest = pattern.destination.value();
                if (dest.hops.has_value()) {
                    auto dst_node_ids = route_manager.get_dst_node_ids_from_hops(
                        sender.device,
                        const_cast<std::unordered_map<RoutingDirection, uint32_t>&>(dest.hops.value()),
                        pattern.ftype.value());
                    for (const auto& dst_id : dst_node_ids) {
                        receive_configs[{dst_id, sender.link_id}]++;
                    }
                } else if (dest.device.has_value()) {
                    receive_configs[{dest.device.value(), sender.link_id}]++;
                }
            }
        }
        return receive_configs;
    }

    // Mux channels on one (device, link) kept for its sender and sync cores: one per sender core + one for sync.
    // FabricConnectionManager checks each mux's actual total when it builds it.
    // NOTE: Can overestimate # channels used by senders if they send to different directions, but this doesn't break
    // correctness.
    static uint32_t count_mux_channels_for_senders_and_sync(
        const TestConfig& config, const FabricNodeId& device, uint32_t link) {
        uint32_t sender_channels = 0;
        for (const auto& sender : config.senders) {
            if (sender.device == device && sender.link_id == link) {
                sender_channels++;
            }
        }
        return std::max(sender_channels, 1u) + 1 /* sync */;
    }

    /**
     * Compute dynamic allocation policies based on test configuration.
     *
     * Analyzes the test topology (senders, receivers, devices, flow control) and computes
     * optimal allocation policies:
     * - max_configs_per_core: How many receiver configs can share a single core
     *
     * The receive ring size is left to the test context, which fills the receiver core's L1.
     */
    AllocatorPolicies compute_policy(const TestConfig& config) {
        // 1. Query system parameters
        CoreCoord worker_grid = device_info_provider_.get_worker_grid_size();
        uint32_t total_worker_cores = worker_grid.x * worker_grid.y;

        // 2. Determine num_links and where muxes go. With flow control every device has a mux per direction on every
        // link (receivers return credits over the senders' links). Otherwise a device has them only on the links where
        // two or more of its senders send in the same direction (e.g. the pieces of a split sender).
        uint32_t max_link_id = 0;
        for (const auto& sender : config.senders) {
            max_link_id = std::max(max_link_id, sender.link_id);
        }
        const uint32_t num_links = max_link_id + 1;  // link_id is 0-indexed
        const uint32_t mux_cores_per_link = device_info_provider_.get_max_connections_per_device();
        const auto links_shared_by_senders =
            find_links_shared_by_senders(config, route_manager_, device_info_provider_);
        const auto count_mux_cores_on_device = [&](const FabricNodeId& device_id) -> uint32_t {
            if (config.enable_flow_control) {
                return mux_cores_per_link * num_links;
            }
            const auto shared_it = links_shared_by_senders.find(device_id);
            return shared_it == links_shared_by_senders.end()
                       ? 0
                       : mux_cores_per_link * static_cast<uint32_t>(shared_it->second.size());
        };

        // 3. Build per-device receiver load histogram
        std::unordered_map<FabricNodeId, uint32_t> receiver_load_per_device;
        for (const auto& [device_and_link, num_configs] :
             count_receive_configs_per_device_and_link(config, route_manager_)) {
            receiver_load_per_device[device_and_link.first] += num_configs;
        }

        // 4. Per-device analysis - find worst case
        uint32_t max_configs_per_core_needed = DEFAULT_MIN_CONFIGS_PER_CORE;
        std::optional<FabricNodeId> worst_case_device;

        for (const auto& [device_id, num_receivers] : receiver_load_per_device) {
            // Count reserved cores on this device
            uint32_t sender_cores_on_device = 0;
            for (const auto& sender : config.senders) {
                if (sender.device == device_id) {
                    sender_cores_on_device++;
                }
            }

            bool has_sync = false;
            for (const auto& sync : config.sync_configs) {
                const auto& sender_config = sync.sender_config;
                if (sender_config.device == device_id) {
                    has_sync = true;
                    break;
                }
            }

            const uint32_t mux_cores_on_device = count_mux_cores_on_device(device_id);
            uint32_t reserved_cores =
                (has_sync ? 1 : 0) + sender_cores_on_device + mux_cores_on_device + SAFETY_MARGIN_CORES;

            // Feasibility check 1: No cores left for receivers
            if (reserved_cores >= total_worker_cores) {
                log_fatal(
                    tt::LogTest,
                    "Device [mesh={}, chip={}] allocation is INFEASIBLE!\n"
                    "  Reserved cores: {} >= Total cores: {}\n"
                    "  Breakdown: sync={}, senders={}, mux={}, safety={}\n"
                    "  No cores left for {} receiver configs!\n"
                    "  Suggestions: Reduce link count, disable flow control, or use larger core grid.",
                    device_id.mesh_id,
                    device_id.chip_id,
                    reserved_cores,
                    total_worker_cores,
                    (has_sync ? 1 : 0),
                    sender_cores_on_device,
                    mux_cores_on_device,
                    SAFETY_MARGIN_CORES,
                    num_receivers);
                TT_FATAL(false, "Infeasible allocation configuration");
            }

            uint32_t available_for_receivers = total_worker_cores - reserved_cores;

            // Feasibility check 2: Insufficient cores for minimum buffer size
            uint32_t min_cores_needed =
                (num_receivers + MAX_CONFIGS_PER_CORE_CEILING - 1) / MAX_CONFIGS_PER_CORE_CEILING;

            if (available_for_receivers < min_cores_needed) {
                log_fatal(
                    tt::LogTest,
                    "Device [mesh={}, chip={}] allocation is INFEASIBLE!\n"
                    "  Receiver configs: {}\n"
                    "  Minimum cores needed: {} (to provide 16KB per receiver)\n"
                    "  Available cores: {}\n"
                    "  Reserved cores: {}\n"
                    "  The test requires more receiver cores than available even at maximum sharing.\n"
                    "  Suggestions: Reduce test scale, links, or use larger core grid.",
                    device_id.mesh_id,
                    device_id.chip_id,
                    num_receivers,
                    min_cores_needed,
                    available_for_receivers,
                    reserved_cores);
                TT_FATAL(false, "Infeasible allocation: insufficient receiver cores");
            }

            // Compute required configs per core based on available cores
            uint32_t required = (num_receivers + available_for_receivers - 1) / available_for_receivers;

            // Apply mux client cap if flow control is enabled
            // Since link duplication uniformly distributes receivers across links,
            // we divide total receivers by num_links to get receivers per link
            if (config.enable_flow_control && num_links > 0) {
                uint32_t receivers_per_link = num_receivers / num_links;
                uint32_t channels_for_senders_and_sync = 0;
                for (uint32_t link = 0; link < num_links; link++) {
                    channels_for_senders_and_sync = std::max(
                        channels_for_senders_and_sync,
                        count_mux_channels_for_senders_and_sync(config, device_id, link));
                }
                TT_FATAL(
                    channels_for_senders_and_sync < get_max_mux_channels(),
                    "Device [mesh={}, chip={}] needs {} mux channels for its sender and sync cores on one link, "
                    "leaving "
                    "none of the {} for receivers. Use fewer senders per link.",
                    device_id.mesh_id,
                    device_id.chip_id,
                    channels_for_senders_and_sync,
                    get_max_mux_channels());
                const uint32_t max_recv_cores_per_link = get_max_mux_channels() - channels_for_senders_and_sync;

                if (receivers_per_link > max_recv_cores_per_link) {
                    // Calculate minimum sharing factor needed to satisfy mux client cap
                    uint32_t sharing_factor_per_link =
                        (receivers_per_link + max_recv_cores_per_link - 1) / max_recv_cores_per_link;

                    // Apply the stricter constraint (mux cap vs available cores)
                    required = std::max(required, sharing_factor_per_link);
                }
            }

            if (required > max_configs_per_core_needed) {
                max_configs_per_core_needed = required;
                worst_case_device = device_id;
            }
        }

        // 5. Apply bounds
        uint32_t max_configs_per_core = std::min(max_configs_per_core_needed, MAX_CONFIGS_PER_CORE_CEILING);

        // 6. Build and return policies. The test context sizes the receive rings to fill L1.
        AllocatorPolicies computed_policies;
        computed_policies.receiver_config.max_configs_per_core = max_configs_per_core;

        return computed_policies;
    }

    // Dependencies
    const IDeviceInfoProvider& device_info_provider_;
    const IRouteManager& route_manager_;

    // Caching state
    std::string last_topology_key_;
    AllocatorPolicies cached_policy_;

    // Constants for dynamic policy computation
    // Most receive configs on one core: 64 rings still get at least 16 KB each of a receiver core's L1 on Wormhole and
    // Blackhole.
    static constexpr uint32_t MAX_CONFIGS_PER_CORE_CEILING = 64;
    static constexpr uint32_t SAFETY_MARGIN_CORES = 2;
    static constexpr uint32_t DEFAULT_MIN_CONFIGS_PER_CORE = 1;
};

inline void GlobalAllocator::allocate_resources(TestConfig& test_config) {
    // Store flow control flag for use during device creation
    enable_flow_control_ = test_config.enable_flow_control;

    // PASS 0: Reserve sync cores for synchronization
    for (auto& sync_config : test_config.sync_configs) {
        auto& sync_sender = sync_config.sender_config;
        auto& device_resources = get_or_create_device_resources(sync_sender.device);
        sync_sender.core = device_resources.reserve_sync_core();
    }

    // PASS 1: Reserve all specified sender cores first. This establishes the pool of
    // cores that are *not* available for receivers.
    uint32_t receiver_partitions = 1;  // Compute number of link partitions needed
    for (const auto& sender : test_config.senders) {
        receiver_partitions = std::max(receiver_partitions, sender.link_id + 1);

        if (sender.core.has_value()) {
            auto& device_resources = get_or_create_device_resources(sender.device);
            device_resources.reserve_sender_core(sender.core);
        }
    }

    // PASS 1.5: Allocate cores for any senders that don't have one specified.
    for (auto& sender : test_config.senders) {
        if (!sender.core.has_value()) {
            auto& device_resources = get_or_create_device_resources(sender.device);
            sender.core = device_resources.reserve_sender_core(std::nullopt);
        }
    }

    // PASS 2: Allocate receivers and their memory. Now that all senders are known, the
    // remaining pristine cores on all devices can form the receiver pool.
    for (auto& sender : test_config.senders) {
        for (auto& pattern : sender.patterns) {
            auto& dest = pattern.destination.value();

            if (dest.core.has_value() && dest.target_address.has_value() &&
                (!pattern_needs_atomic_inc(pattern) || dest.atomic_inc_address.has_value())) {
                // Fully specified, just book-keep.
                auto& device_resources = get_or_create_device_resources(dest.device.value());
                device_resources.reserve_receiver_core(dest.core, sender.link_id, receiver_partitions);
                // We assume the pre-specified address is valid.
                continue;
            }

            uint32_t num_receivers_for_credits = 0;

            if (dest.hops.has_value()) {  // process based on hops
                std::vector<FabricNodeId> dst_node_ids =
                    route_manager_.get_dst_node_ids_from_hops(sender.device, dest.hops.value(), pattern.ftype.value());

                uint32_t chunk_size = policies_.default_payload_chunk_size;
                TT_FATAL(
                    pattern.size.value() <= chunk_size,
                    "Requested payload size {} exceeds the per-worker buffer chunk size of {}",
                    pattern.size.value(),
                    chunk_size);

                bool needs_payload =
                    (pattern.ntype->type == NocSendType::NOC_UNICAST_WRITE ||
                     pattern.ntype->type == NocSendType::NOC_UNICAST_SCATTER_WRITE ||
                     pattern.ntype->type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC);
                bool needs_atomic = pattern_needs_atomic_inc(pattern);

                // Count on how many destination devices each receiver core is available. A multicast needs one
                // core (and addresses on it) that is free on every destination device.
                std::map<CoreCoord, uint32_t> core_counts;
                std::vector<TestDeviceResources*> dst_device_resources;
                dst_device_resources.reserve(dst_node_ids.size());
                for (const auto& device_id : dst_node_ids) {
                    auto& device_resources = get_or_create_device_resources(device_id);
                    dst_device_resources.push_back(&device_resources);

                    const auto& receiver_pool = device_resources.core_pools_[RECEIVER_TYPE_IDX];
                    if (!receiver_pool.initialized) {
                        device_resources.initialize_receiver_pool(receiver_partitions);
                    }

                    for (const auto& core :
                         receiver_pool.get_available_cores(device_resources.core_workload_, sender.link_id)) {
                        core_counts[core]++;
                    }
                }

                std::optional<std::pair<CoreCoord, uint32_t>> uniform_payload_receiver = std::nullopt;
                std::optional<std::pair<CoreCoord, uint32_t>> uniform_atomic_receiver = std::nullopt;
                std::optional<CoreCoord> selected_core = std::nullopt;

                // Take the first core (in CoreCoord order) available on every destination, using the lowest address
                // free on all of them. Addresses are only examined for candidate cores, and only for the kinds this
                // ntype needs, rather than histogramming every free address of every core up front.
                for (const auto& [core, device_count] : core_counts) {
                    if (device_count != dst_node_ids.size()) {
                        continue;
                    }

                    std::optional<uint32_t> payload_addr = std::nullopt;
                    if (needs_payload) {
                        std::map<uint32_t, uint32_t> payload_address_histogram;
                        for (auto* device_resources : dst_device_resources) {
                            auto& core_resources =
                                device_resources->get_or_create_core_resources(core, CoreType::RECEIVER);
                            for (auto addr : core_resources.get_available_payload_chunks()) {
                                payload_address_histogram[addr]++;
                            }
                        }
                        for (const auto& [addr, addr_device_count] : payload_address_histogram) {
                            if (addr_device_count == dst_node_ids.size()) {
                                payload_addr = addr;
                                break;
                            }
                        }
                        if (!payload_addr.has_value()) {
                            continue;
                        }
                    }

                    std::optional<uint32_t> atomic_addr = std::nullopt;
                    if (needs_atomic) {
                        // Free atomic counters on a core are the contiguous run [next free, region end), so the
                        // lowest counter free on every device is the highest of the per-device first-free ones.
                        std::optional<uint32_t> first_common;
                        bool all_have_counters = true;
                        for (auto* device_resources : dst_device_resources) {
                            auto& core_resources =
                                device_resources->get_or_create_core_resources(core, CoreType::RECEIVER);
                            const auto counters = core_resources.get_available_atomic_counters();
                            if (counters.empty()) {
                                all_have_counters = false;
                                break;
                            }
                            first_common = std::max(first_common.value_or(0), counters.front());
                        }
                        if (!all_have_counters) {
                            continue;
                        }
                        atomic_addr = first_common;
                    }

                    selected_core = core;
                    if (payload_addr.has_value()) {
                        uniform_payload_receiver = std::make_pair(core, payload_addr.value());
                    }
                    if (atomic_addr.has_value()) {
                        uniform_atomic_receiver = std::make_pair(core, atomic_addr.value());
                    }
                    break;  // Found a suitable core, stop searching
                }

                if (!selected_core.has_value() || (needs_payload && !uniform_payload_receiver.has_value()) ||
                    (needs_atomic && !uniform_atomic_receiver.has_value())) {
                    TT_THROW(
                        "Could not find uniform core and memory addresses for multicast pattern with ntype {}.",
                        noc_send_type_label(pattern.ntype.value()));
                }

                dest.core = selected_core.value();
                if (uniform_payload_receiver.has_value()) {
                    dest.target_address = uniform_payload_receiver->second;
                }
                if (uniform_atomic_receiver.has_value()) {
                    dest.atomic_inc_address = uniform_atomic_receiver->second;
                }

                // Reserve resources on all destination devices
                for (const auto& node_id : dst_node_ids) {
                    auto& device_resources = get_or_create_device_resources(node_id);
                    device_resources.reserve_receiver_core(dest.core, sender.link_id, receiver_partitions);
                    auto& core_resources =
                        device_resources.get_or_create_core_resources(dest.core.value(), CoreType::RECEIVER);
                    // Reserve payload chunk if needed
                    if (needs_payload) {
                        core_resources.reserve_payload_chunk(dest.target_address.value());
                    }
                    // Reserve atomic counter if needed
                    if (needs_atomic) {
                        core_resources.reserve_atomic_counter(dest.atomic_inc_address.value());
                    }
                }

                num_receivers_for_credits = static_cast<uint32_t>(dst_node_ids.size());
            } else if (dest.device.has_value()) {  // process dest devices directly
                auto& device_resources = get_or_create_device_resources(dest.device.value());

                // Lazy init if needed
                if (!device_resources.core_pools_[RECEIVER_TYPE_IDX].initialized) {
                    device_resources.initialize_receiver_pool(receiver_partitions);
                }

                if (!dest.core.has_value()) {
                    dest.core =
                        device_resources.reserve_receiver_core(std::nullopt, sender.link_id, receiver_partitions);
                } else {
                    device_resources.reserve_receiver_core(dest.core, sender.link_id, receiver_partitions);
                }

                auto& core_resources =
                    device_resources.get_or_create_core_resources(dest.core.value(), CoreType::RECEIVER);

                bool allocate_write_address = pattern.ntype->type != NocSendType::NOC_UNICAST_ATOMIC_INC;
                bool allocate_atomic_inc_address = pattern_needs_atomic_inc(pattern);

                if (allocate_write_address) {
                    TT_FATAL(
                        pattern.size.value() <= core_resources.payload_chunk_size,
                        "Requested payload size {} exceeds the per-worker buffer chunk size of {}",
                        pattern.size.value(),
                        core_resources.payload_chunk_size);
                    if (!core_resources.has_available_payload_chunk()) {
                        TT_THROW(
                            "No payload buffer chunk available on device {} core {} for allocation.",
                            dest.device.value(),
                            dest.core.value());
                    }
                    dest.target_address = core_resources.allocate_payload_chunk();
                }
                if (allocate_atomic_inc_address) {
                    dest.atomic_inc_address = core_resources.allocate_atomic_counter();
                }

                num_receivers_for_credits = 1;  // Direct-device = single receiver
            }

            if (enable_flow_control_) {
                // Allocate credit chunk from sender core's L1
                auto& sender_device_resources = get_or_create_device_resources(sender.device);
                uint32_t credit_chunk_base = sender_device_resources.allocate_credit_chunk(
                    sender.core.value(), num_receivers_for_credits, sender_memory_map_);

                // Calculate credit configuration using simple policy
                uint32_t buffer_capacity_bytes = policies_.default_payload_chunk_size;
                uint32_t packet_size_bytes = pattern.size.value();
                uint32_t num_packets = pattern.num_packets.value();
                auto [initial_credits, batch_size] =
                    calculate_credit_config(buffer_capacity_bytes, packet_size_bytes, num_packets);

                // Populate sender credit info directly in pattern
                pattern.sender_credit_info = SenderCreditInfo{
                    .expected_receiver_count = num_receivers_for_credits,
                    .credit_reception_address_base = credit_chunk_base,
                    .initial_credits = initial_credits};
                pattern.credit_return_batch_size = batch_size;
            }
        }
    }
}

inline void GlobalAllocator::reset() {
    // Reset credit allocators before clearing device resources
    for (auto& [node_id, device_resources_ptr] : all_device_resources_) {
        device_resources_ptr->reset_credit_allocators();
    }

    all_device_resources_.clear();
    worker_grid_size_ = std::nullopt;
    enable_flow_control_ = false;
}

inline std::vector<CoreCoord> GlobalAllocator::get_pristine_cores_for_device(const FabricNodeId& node_id) const {
    auto it = all_device_resources_.find(node_id);
    if (it == all_device_resources_.end()) {
        // Device not found in allocator (no workers allocated on this device)
        // Return empty vector - no cores available for mux allocation
        return {};
    }
    // Collect remaining pristine cores from all pools (after allocation is complete)
    return it->second->collect_remaining_pristine_cores();
}

}  // namespace tt::tt_fabric::fabric_tests
