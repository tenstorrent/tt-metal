// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <array>
#include <map>
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
        atomic_region_(atomic_region) {
        this->next_atomic_addr_ = this->atomic_region_.start;
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
            "Payload chunk_size must be positive and a multiple of alignment");

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
        CoreAllocationConfig policy = core_pools_[static_cast<size_t>(core_type)].policy;

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

private:
    TestDeviceResources& get_or_create_device_resources(const FabricNodeId& node_id);

    const IDeviceInfoProvider& device_info_provider_;
    const IRouteManager& route_manager_;
    AllocatorPolicies policies_;
    const SenderMemoryMap& sender_memory_map_;
    const ReceiverMemoryMap& receiver_memory_map_;
    std::optional<CoreCoord> worker_grid_size_;
    std::unordered_map<FabricNodeId, std::unique_ptr<TestDeviceResources>> all_device_resources_;
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
            policies_.default_payload_chunk_size.value_or(detail::DEFAULT_PAYLOAD_CHUNK_SIZE_BYTES),
            policies_.sender_config,
            policies_.receiver_config,
            receiver_memory_map_.payload_chunks,
            receiver_memory_map_.atomic_counters));
    return *inserted_it->second;
}

inline void GlobalAllocator::allocate_resources(TestConfig& test_config) {
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

            if (pattern.ftype.value() == ChipSendType::CHIP_MULTICAST) {
                std::vector<FabricNodeId> dst_node_ids;
                if (dest.hops.has_value()) {
                    dst_node_ids = route_manager_.get_dst_node_ids_from_hops(
                        sender.device, dest.hops.value(), pattern.ftype.value());
                } else {
                    TT_FATAL(dest.device.has_value(), "Multicast destination requires hops");
                    dst_node_ids.push_back(dest.device.value());
                }

                uint32_t chunk_size =
                    policies_.default_payload_chunk_size.value_or(detail::DEFAULT_PAYLOAD_CHUNK_SIZE_BYTES);
                TT_FATAL(
                    pattern.size.value() <= chunk_size,
                    "Requested payload size {} exceeds the per-worker buffer chunk size of {}",
                    pattern.size.value(),
                    chunk_size);

                // Use histogram analysis to find uniform receiver core and memory address
                std::map<CoreCoord, uint32_t> core_counts;
                std::map<CoreCoord, std::map<uint32_t, uint32_t>> memory_histograms;

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
                        if (core_resources.has_available_payload_chunk()) {
                            const auto& available_chunks = core_resources.get_available_payload_chunks();
                            for (auto addr : available_chunks) {
                                memory_histograms[core][addr]++;
                            }
                        }
                    }
                }

                std::optional<CoreCoord> best_core = std::nullopt;
                uint32_t max_count = 0;
                for (const auto& [core, count] : core_counts) {
                    if (count > max_count) {
                        max_count = count;
                        best_core = core;
                    }
                }

                std::optional<std::pair<CoreCoord, uint32_t>> uniform_receiver = std::nullopt;
                if (best_core.has_value()) {
                    const auto& address_histogram = memory_histograms[best_core.value()];
                    for (const auto& [addr, count] : address_histogram) {
                        if (count == dst_node_ids.size()) {
                            uniform_receiver = std::make_pair(best_core.value(), addr);
                            break;
                        }
                    }
                }

                if (!uniform_receiver.has_value()) {
                    TT_THROW("Could not find a uniform core and memory address for multicast pattern.");
                }

                dest.core = uniform_receiver->first;
                if (pattern.ntype.value() == NocSendType::NOC_UNICAST_WRITE) {
                    dest.target_address = uniform_receiver->second;
                } else {
                    TT_THROW("Multicast atomic/fused atomic not supported yet");
                }

                // Reserve resources on all destination devices
                for (const auto& node_id : dst_node_ids) {
                    auto& device_resources = get_or_create_device_resources(node_id);
                    device_resources.reserve_receiver_core(dest.core);
                    auto& core_resources =
                        device_resources.get_or_create_core_resources(dest.core.value(), CoreType::RECEIVER);
                    if (pattern.ntype.value() == NocSendType::NOC_UNICAST_ATOMIC_INC ||
                        pattern.ntype.value() == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
                        // TODO: need to allocate from a separate pool?
                        TT_THROW("Multicast atomic/fused atomic not supported yet");
                    } else {
                        core_resources.reserve_payload_chunk(dest.target_address.value());
                    }
                }
            } else {  // Unicast
                TT_FATAL(dest.device.has_value(), "Unicast destination requires a device ID.");
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
                if (pattern.ntype.value() == NocSendType::NOC_UNICAST_WRITE) {
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
            }
        }
    }
}

inline void GlobalAllocator::reset() {
    all_device_resources_.clear();
    worker_grid_size_ = std::nullopt;
}

}  // namespace tt::tt_fabric::fabric_tests
