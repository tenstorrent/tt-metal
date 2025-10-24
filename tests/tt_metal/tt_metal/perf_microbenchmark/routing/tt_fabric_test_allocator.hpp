// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include <tt-metalium/host_api.hpp>
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common_types.hpp"
#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_memory_map.hpp"

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

    std::vector<CoreCoord> get_available_cores(const std::unordered_map<CoreCoord, uint32_t>& core_workload) const {
        std::vector<CoreCoord> available;
        available.reserve(active_pool.size());
        for (const auto& core : active_pool) {
            auto it = core_workload.find(core);
            if (it == core_workload.end() || it->second < policy.max_configs_per_core) {
                available.push_back(core);
            }
        }
        return available;
    }

    const CoreAllocationConfig& policy;
    std::vector<CoreCoord> active_pool;
    size_t next_pool_idx = 0;
    bool initialized = false;
};

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

    void initialize_receiver_pool();
    CoreCoord reserve_sync_core();
    CoreCoord reserve_sender_core(const std::optional<CoreCoord>& specified_core);
    CoreCoord reserve_receiver_core(const std::optional<CoreCoord>& specified_core);
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

    // Credit allocation for senders on this device
    std::optional<DynamicMemoryRegion> credit_allocator_;

    // Collect remaining pristine cores from all pools (for mux allocation)
    std::vector<CoreCoord> collect_remaining_pristine_cores() const;

    // Initialize credit allocator (lazy)
    void initialize_credit_allocator(const SenderMemoryMap& sender_memory_map);

    // Allocate credit chunk from this device's L1 credit region
    // Returns the base address of the allocated chunk
    // Throws if allocation would exceed region bounds
    uint32_t allocate_credit_chunk(uint32_t num_receivers, const SenderMemoryMap& sender_memory_map);

    // Reset credit allocator
    void reset_credit_allocator();

private:
    void reserve_core_internal(const CoreCoord& core, CoreType core_type);
    CoreCoord find_next_available_core(CorePool& pool);
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

inline void TestDeviceResources::initialize_receiver_pool() {
    CorePool& receiver_pool = core_pools_[RECEIVER_TYPE_IDX];
    if (!receiver_pool.initialized) {
        // All cores not used for senders are available for receivers.
        receiver_pool.active_pool = pristine_cores_;
        std::sort(receiver_pool.active_pool.begin(), receiver_pool.active_pool.end());
        pristine_cores_.clear();
        receiver_pool.initialized = true;
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

inline CoreCoord TestDeviceResources::reserve_receiver_core(const std::optional<CoreCoord>& specified_core) {
    if (!core_pools_[RECEIVER_TYPE_IDX].initialized) {
        initialize_receiver_pool();
    }

    if (specified_core.has_value()) {
        reserve_core_internal(specified_core.value(), CoreType::RECEIVER);
        return specified_core.value();
    }

    CorePool& pool = core_pools_[RECEIVER_TYPE_IDX];
    CoreCoord core = find_next_available_core(pool);
    reserve_core_internal(core, CoreType::RECEIVER);
    return core;
}

inline CoreResources& TestDeviceResources::get_or_create_core_resources(const CoreCoord& core, CoreType core_type) {
    if (core_resources_.find(core) == core_resources_.end()) {
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
    uint32_t refill_count;
    if (pool.active_pool.empty()) {
        refill_count = pool.policy.initial_pool_size;
    } else {
        refill_count = pool.policy.pool_refill_size;
    }

    for (uint32_t i = 0; i < refill_count; ++i) {
        if (pristine_cores_.empty()) {
            // This is not an error if we are just refilling. But if the pool is empty, it is.
            if (pool.active_pool.empty()) {
                TT_THROW("No more pristine cores available to create a new active pool.");
            }
            break;
        }
        pool.active_pool.push_back(pristine_cores_.back());
        pristine_cores_.pop_back();
    }
    std::sort(pool.active_pool.begin(), pool.active_pool.end());
}

inline void TestDeviceResources::initialize_credit_allocator(const SenderMemoryMap& sender_memory_map) {
    if (!credit_allocator_.has_value()) {
        credit_allocator_.emplace(
            sender_memory_map.get_credit_addresses_base(),
            sender_memory_map.get_credit_addresses_size(),
            SenderMemoryMap::CREDIT_ADDRESS_STRIDE);
    }
}

inline uint32_t TestDeviceResources::allocate_credit_chunk(
    uint32_t num_receivers, const SenderMemoryMap& sender_memory_map) {
    // Lazy initialization
    if (!credit_allocator_.has_value()) {
        initialize_credit_allocator(sender_memory_map);
    }

    // Allocate from the per-device allocator
    uint32_t chunk_base = credit_allocator_->allocate_chunk(num_receivers);

    log_debug(
        tt::LogTest,
        "Device {} allocated credit chunk: base={:#x}, num_receivers={}",
        node_id_,
        chunk_base,
        num_receivers);

    return chunk_base;
}

inline void TestDeviceResources::reset_credit_allocator() {
    if (credit_allocator_.has_value()) {
        credit_allocator_->reset();
        log_debug(tt::LogTest, "Reset credit allocator for device {}", node_id_);
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

inline CoreCoord TestDeviceResources::find_next_available_core(CorePool& pool) {
    while (true) {
        // Search the current pool for an available core.
        for (size_t i = 0; i < pool.active_pool.size(); ++i) {
            size_t idx_to_check = (pool.next_pool_idx + i) % pool.active_pool.size();
            const CoreCoord& core = pool.active_pool[idx_to_check];
            auto it = core_workload_.find(core);

            if (it == core_workload_.end() || it->second < pool.policy.max_configs_per_core) {
                // Found an available core. Update index for next search and return.
                if (pool.policy.policy == CoreAllocationPolicy::ExhaustFirst) {
                    // For ExhaustFirst, keep pointing to this core until it's full.
                    if (it == core_workload_.end() || (it->second + 1) < pool.policy.max_configs_per_core) {
                        pool.next_pool_idx = idx_to_check;
                    } else {
                        // Core will be full after this allocation, so move to the next.
                        pool.next_pool_idx = (idx_to_check + 1) % pool.active_pool.size();
                    }
                } else {  // RoundRobin
                    pool.next_pool_idx = (idx_to_check + 1) % pool.active_pool.size();
                }
                return core;
            }
        }

        // If we are here, the pool is exhausted. Try to refill it.
        size_t old_pool_size = pool.active_pool.size();
        refill_pool(pool);

        // If the pool size didn't change after refill, we're out of cores for good.
        if (pool.active_pool.size() == old_pool_size) {
            TT_THROW("No available core found and the pool could not be refilled. All cores are at max workload.");
        }
        // Point the search to the start of the newly added cores to avoid re-scanning the exhausted ones.
        pool.next_pool_idx = old_pool_size;
    }
}

inline void TestDeviceResources::reserve_core_internal(const CoreCoord& core, CoreType core_type) {
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

    // If core is pristine, it must be added to the active pool first.
    auto pristine_it = std::find(pristine_cores_.begin(), pristine_cores_.end(), core);
    if (pristine_it != pristine_cores_.end()) {
        pristine_cores_.erase(pristine_it);
        // Put it in the active pool so it's not "lost" from tracking.
        pool.active_pool.push_back(core);
        std::sort(pool.active_pool.begin(), pool.active_pool.end());
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
 * Calculate credit configuration (100% initial, 20% batch)
 * Simple policy: Give sender full buffer capacity, receiver returns in 20% batches
 */
inline static std::pair<uint32_t, uint32_t> calculate_credit_config(
    uint32_t buffer_capacity_bytes, uint32_t packet_size_bytes, uint32_t num_packets) {
    uint32_t buffer_capacity_packets = buffer_capacity_bytes / packet_size_bytes;
    uint32_t initial_credits = std::min(buffer_capacity_packets, num_packets);
    uint32_t batch_size = std::max(1u, initial_credits / 4);
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

    // Set whether to use dynamic allocation policies
    void set_use_dynamic_policies(bool use_dynamic) { use_dynamic_policies_ = use_dynamic; }

private:
    TestDeviceResources& get_or_create_device_resources(const FabricNodeId& node_id);

    // Compute dynamic allocation policies based on test configuration
    AllocatorPolicies compute_dynamic_policies(const TestConfig& config);

    const IDeviceInfoProvider& device_info_provider_;
    const IRouteManager& route_manager_;
    AllocatorPolicies policies_;
    const SenderMemoryMap& sender_memory_map_;
    const ReceiverMemoryMap& receiver_memory_map_;
    std::optional<CoreCoord> worker_grid_size_;
    std::unordered_map<FabricNodeId, std::unique_ptr<TestDeviceResources>> all_device_resources_;
    bool enable_flow_control_ = false;  // Set during allocate_resources, used during device creation
    bool use_dynamic_policies_ = true;  // Default to dynamic policy computation
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

// Constants for dynamic policy computation
namespace {
constexpr uint32_t MIN_BUFFER_SIZE_BYTES = 16 * 1024;                                            // 16KB
constexpr uint32_t USABLE_L1_SIZE_BYTES = 1024 * 1024;                                           // 1MB
constexpr uint32_t MAX_CONFIGS_PER_CORE_CEILING = USABLE_L1_SIZE_BYTES / MIN_BUFFER_SIZE_BYTES;  // 64
constexpr uint32_t SAFETY_MARGIN_CORES = 2;
constexpr uint32_t DEFAULT_MIN_CONFIGS_PER_CORE = 1;
constexpr uint32_t NUM_DIRECTIONS = 4;  // N, S, E, W
}  // namespace

inline AllocatorPolicies GlobalAllocator::compute_dynamic_policies(const TestConfig& config) {
    // 1. Query system parameters
    CoreCoord worker_grid = device_info_provider_.get_worker_grid_size();
    uint32_t total_worker_cores = worker_grid.x * worker_grid.y;

    // 2. Determine mux cores per device (if flow control enabled)
    uint32_t mux_cores_per_device = 0;
    if (config.enable_flow_control) {
        // Determine num_links from test config
        uint32_t max_link_id = 0;
        for (const auto& sender : config.senders) {
            max_link_id = std::max(max_link_id, sender.link_id);
        }
        uint32_t num_links = max_link_id + 1;  // link_id is 0-indexed

        // Per device max: 4 directions × num_links
        mux_cores_per_device = NUM_DIRECTIONS * num_links;
    }

    // 3. Build per-device receiver load histogram
    std::unordered_map<FabricNodeId, uint32_t> receiver_load_per_device;

    for (const auto& sender : config.senders) {
        for (const auto& pattern : sender.patterns) {
            const auto& dest = pattern.destination.value();

            if (dest.hops.has_value()) {
                auto dst_node_ids = route_manager_.get_dst_node_ids_from_hops(
                    sender.device,
                    const_cast<std::unordered_map<RoutingDirection, uint32_t>&>(dest.hops.value()),
                    pattern.ftype.value());
                for (const auto& dst_id : dst_node_ids) {
                    receiver_load_per_device[dst_id]++;
                }
            } else if (dest.device.has_value()) {
                receiver_load_per_device[dest.device.value()]++;
            }
        }
    }

    // 4. Per-device analysis - find worst case
    uint32_t max_configs_per_core_needed = DEFAULT_MIN_CONFIGS_PER_CORE;
    std::optional<FabricNodeId> worst_case_device;  // Use optional since we might have no receivers
    uint32_t worst_case_reserved = 0;
    uint32_t worst_case_receivers = 0;

    for (const auto& [device_id, num_receivers] : receiver_load_per_device) {
        // Count reserved cores on this device
        uint32_t sender_cores_on_device = 0;
        for (const auto& sender : config.senders) {
            if (sender.device == device_id) {
                sender_cores_on_device++;
            }
        }

        bool has_sync = false;
        for (const auto& sync : config.global_sync_configs) {
            if (sync.device == device_id) {
                has_sync = true;
                break;
            }
        }

        uint32_t reserved_cores =
            (has_sync ? 1 : 0) + sender_cores_on_device + mux_cores_per_device + SAFETY_MARGIN_CORES;

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
                mux_cores_per_device,
                SAFETY_MARGIN_CORES,
                num_receivers);
            TT_FATAL(false, "Infeasible allocation configuration");
        }

        uint32_t available_for_receivers = total_worker_cores - reserved_cores;

        // Feasibility check 2: Insufficient cores for minimum buffer size
        uint32_t min_cores_needed = (num_receivers + MAX_CONFIGS_PER_CORE_CEILING - 1) / MAX_CONFIGS_PER_CORE_CEILING;

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

        // Compute required configs per core
        uint32_t required = (num_receivers + available_for_receivers - 1) / available_for_receivers;

        if (required > max_configs_per_core_needed) {
            max_configs_per_core_needed = required;
            worst_case_device = device_id;
            worst_case_reserved = reserved_cores;
            worst_case_receivers = num_receivers;
        }
    }

    // 5. Apply bounds
    uint32_t max_configs_per_core = std::min(max_configs_per_core_needed, MAX_CONFIGS_PER_CORE_CEILING);
    uint32_t payload_chunk_size = USABLE_L1_SIZE_BYTES / max_configs_per_core;

    // 6. Validate packet sizes
    uint32_t max_packet_size = 0;
    for (const auto& sender : config.senders) {
        for (const auto& pattern : sender.patterns) {
            if (pattern.size.has_value()) {
                max_packet_size = std::max(max_packet_size, pattern.size.value());
            }
        }
    }

    if (max_packet_size > payload_chunk_size) {
        log_fatal(
            tt::LogTest,
            "Test configuration is INVALID!\n"
            "  Max packet size: {} bytes\n"
            "  Computed buffer size: {} bytes\n"
            "  The packet size exceeds buffer capacity even with optimal allocation.\n"
            "  Fix: Reduce packet size to <= {} bytes.",
            max_packet_size,
            payload_chunk_size,
            payload_chunk_size);
        TT_FATAL(false, "Invalid test configuration: packet size exceeds buffer capacity");
    }

    // 7. Log computed policy
    if (worst_case_device.has_value()) {
        log_info(tt::LogTest, "");
        log_info(tt::LogTest, "[Dynamic Allocation Policy - Worst Case Device]");
        log_info(
            tt::LogTest,
            "  Device: [mesh={}, chip={}]",
            worst_case_device.value().mesh_id,
            worst_case_device.value().chip_id);
        log_info(tt::LogTest, "  Reserved cores: {} (of {} total)", worst_case_reserved, total_worker_cores);
        log_info(tt::LogTest, "  Available for receivers: {}", total_worker_cores - worst_case_reserved);
        log_info(tt::LogTest, "  Receiver configs: {}", worst_case_receivers);
        log_info(tt::LogTest, "  Computed: max_configs_per_core = {}", max_configs_per_core);
        log_info(
            tt::LogTest,
            "  Computed: default_payload_chunk_size = {} bytes ({} KB)",
            payload_chunk_size,
            payload_chunk_size / 1024);
        log_info(tt::LogTest, "");
    }

    // 8. Build and return policies
    AllocatorPolicies computed_policies = policies_;
    computed_policies.receiver_config.max_configs_per_core = max_configs_per_core;
    computed_policies.default_payload_chunk_size = payload_chunk_size;

    return computed_policies;
}

inline void GlobalAllocator::allocate_resources(TestConfig& test_config) {
    // Determine effective policies
    AllocatorPolicies effective_policies = policies_;

    if (use_dynamic_policies_) {
        // Compute dynamic policies based on test configuration
        effective_policies = compute_dynamic_policies(test_config);

        log_debug(tt::LogTest, "[Using Dynamic Allocation Policy]");
        log_debug(tt::LogTest, "  Test: {}", test_config.parametrized_name);
        log_debug(tt::LogTest, "  max_configs_per_core: {}", effective_policies.receiver_config.max_configs_per_core);
        log_debug(
            tt::LogTest, "  default_payload_chunk_size: {} KB", effective_policies.default_payload_chunk_size / 1024);
        log_debug(tt::LogTest, "");
    } else {
        log_debug(tt::LogTest, "[Using Explicit YAML Allocation Policy]");
        log_debug(tt::LogTest, "  Test: {}", test_config.parametrized_name);
        log_debug(tt::LogTest, "  max_configs_per_core: {}", effective_policies.receiver_config.max_configs_per_core);
        log_debug(
            tt::LogTest, "  default_payload_chunk_size: {} KB", effective_policies.default_payload_chunk_size / 1024);
        log_debug(tt::LogTest, "");
    }

    // Temporarily use effective policies for this allocation
    AllocatorPolicies original_policies = policies_;
    policies_ = effective_policies;

    // Store flow control flag for use during device creation
    enable_flow_control_ = test_config.enable_flow_control;

    // PASS 0: Reserve sync cores for synchronization
    for (auto& sync_sender : test_config.global_sync_configs) {
        auto& device_resources = get_or_create_device_resources(sync_sender.device);
        sync_sender.core = device_resources.reserve_sync_core();
    }

    // PASS 1: Reserve all specified sender cores first. This establishes the pool of
    // cores that are *not* available for receivers.
    for (const auto& sender : test_config.senders) {
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
                ((pattern.ntype.value() != NocSendType::NOC_UNICAST_ATOMIC_INC &&
                  pattern.ntype.value() != NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) ||
                 dest.atomic_inc_address.has_value())) {
                // Fully specified, just book-keep.
                auto& device_resources = get_or_create_device_resources(dest.device.value());
                device_resources.reserve_receiver_core(dest.core);
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

                // Use histogram analysis to find uniform receiver core and memory address
                std::map<CoreCoord, uint32_t> core_counts;
                std::map<CoreCoord, std::map<uint32_t, uint32_t>> payload_memory_histograms;
                std::map<CoreCoord, std::map<uint32_t, uint32_t>> atomic_memory_histograms;

                // Build histograms for all destination devices
                for (const auto& device_id : dst_node_ids) {
                    auto& device_resources = get_or_create_device_resources(device_id);

                    const auto& receiver_pool = device_resources.core_pools_[RECEIVER_TYPE_IDX];
                    if (!receiver_pool.initialized) {
                        device_resources.initialize_receiver_pool();
                    }

                    const auto available_cores = receiver_pool.get_available_cores(device_resources.core_workload_);
                    for (const auto& core : available_cores) {
                        core_counts[core]++;
                        auto& core_resources = device_resources.get_or_create_core_resources(core, CoreType::RECEIVER);
                        // Build histogram for payload addresses
                        if (core_resources.has_available_payload_chunk()) {
                            const auto& available_chunks = core_resources.get_available_payload_chunks();
                            for (auto addr : available_chunks) {
                                payload_memory_histograms[core][addr]++;
                            }
                        }
                        // Build histogram for atomic counter addresses
                        for (auto addr : core_resources.get_available_atomic_counters()) {
                            atomic_memory_histograms[core][addr]++;
                        }
                    }
                }

                std::optional<std::pair<CoreCoord, uint32_t>> uniform_payload_receiver = std::nullopt;
                std::optional<std::pair<CoreCoord, uint32_t>> uniform_atomic_receiver = std::nullopt;
                std::optional<CoreCoord> selected_core = std::nullopt;

                // Validate we found all required uniform addresses
                bool needs_payload =
                    (pattern.ntype.value() == NocSendType::NOC_UNICAST_WRITE ||
                     pattern.ntype.value() == NocSendType::NOC_UNICAST_SCATTER_WRITE ||
                     pattern.ntype.value() == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC);
                bool needs_atomic =
                    (pattern.ntype.value() == NocSendType::NOC_UNICAST_ATOMIC_INC ||
                     pattern.ntype.value() == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC);

                // Find a core that satisfies ALL requirements (both payload and atomic if both needed)
                for (const auto& [core, device_count] : core_counts) {
                    // skip the core if it doesnt match the expected device count
                    if (device_count != dst_node_ids.size()) {
                        continue;
                    }

                    bool core_has_all_required_addresses = true;
                    std::optional<uint32_t> payload_addr = std::nullopt;
                    std::optional<uint32_t> atomic_addr = std::nullopt;

                    // Check if this core has uniform payload address if needed
                    if (needs_payload) {
                        const auto& payload_address_histogram = payload_memory_histograms[core];
                        for (const auto& [addr, addr_device_count] : payload_address_histogram) {
                            if (addr_device_count == dst_node_ids.size()) {
                                payload_addr = addr;
                                break;
                            }
                        }
                        if (!payload_addr.has_value()) {
                            core_has_all_required_addresses = false;
                        }
                    }

                    // Check if this core has uniform atomic address if needed
                    if (needs_atomic && core_has_all_required_addresses) {
                        const auto& atomic_address_histogram = atomic_memory_histograms[core];
                        for (const auto& [addr, addr_device_count] : atomic_address_histogram) {
                            if (addr_device_count == dst_node_ids.size()) {
                                atomic_addr = addr;
                                break;
                            }
                        }
                        if (!atomic_addr.has_value()) {
                            core_has_all_required_addresses = false;
                        }
                    }

                    // If this core has all required addresses, use it
                    if (core_has_all_required_addresses) {
                        selected_core = core;
                        if (payload_addr.has_value()) {
                            uniform_payload_receiver = std::make_pair(core, payload_addr.value());
                        }
                        if (atomic_addr.has_value()) {
                            uniform_atomic_receiver = std::make_pair(core, atomic_addr.value());
                        }
                        break;  // Found a suitable core, stop searching
                    }
                }

                if (!selected_core.has_value() || (needs_payload && !uniform_payload_receiver.has_value()) ||
                    (needs_atomic && !uniform_atomic_receiver.has_value())) {
                    TT_THROW(
                        "Could not find uniform core and memory addresses for multicast pattern with ntype {}.",
                        static_cast<int>(pattern.ntype.value()));
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
                    device_resources.reserve_receiver_core(dest.core);
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

                if (!dest.core.has_value()) {
                    dest.core = device_resources.reserve_receiver_core(std::nullopt);
                } else {
                    device_resources.reserve_receiver_core(dest.core);
                }

                auto& core_resources =
                    device_resources.get_or_create_core_resources(dest.core.value(), CoreType::RECEIVER);

                bool allocate_write_address = true;
                bool allocate_atomic_inc_address = true;
                if (pattern.ntype.value() == NocSendType::NOC_UNICAST_WRITE ||
                    pattern.ntype.value() == NocSendType::NOC_UNICAST_SCATTER_WRITE) {
                    allocate_atomic_inc_address = false;
                } else if (pattern.ntype.value() == NocSendType::NOC_UNICAST_ATOMIC_INC) {
                    allocate_write_address = false;
                }

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
                // Allocate credit chunk from sender device's L1
                auto& sender_device_resources = get_or_create_device_resources(sender.device);
                uint32_t credit_chunk_base =
                    sender_device_resources.allocate_credit_chunk(num_receivers_for_credits, sender_memory_map_);

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

    // Restore original policies for next test
    policies_ = original_policies;
}

inline void GlobalAllocator::reset() {
    // Reset credit allocators before clearing device resources
    for (auto& [node_id, device_resources_ptr] : all_device_resources_) {
        device_resources_ptr->reset_credit_allocator();
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
