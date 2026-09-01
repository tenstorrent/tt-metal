// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/dispatch/vector_aligned.hpp"
#include <tt_stl/span.hpp>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "allocator.hpp"
#include "hal_types.hpp"
#include "sub_device.hpp"
#include "sub_device_types.hpp"
#include <impl/context/context_types.hpp>
#include <tt-metalium/mesh_trace_id.hpp>

namespace tt::tt_metal::distributed {
struct MeshTraceBuffer;
}
namespace tt::tt_metal {

class IDevice;

class SubDeviceManager {
public:
    // Constructor used for the default/global device
    SubDeviceManager(
        IDevice* device, std::unique_ptr<AllocatorImpl>&& global_allocator, ttsl::Span<const SubDevice> sub_devices);
    // Constructor used for regular sub-devices
    SubDeviceManager(ttsl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size, IDevice* device);

    SubDeviceManager(const SubDeviceManager& other) = delete;
    SubDeviceManager& operator=(const SubDeviceManager& other) = delete;

    SubDeviceManager(SubDeviceManager&& other) noexcept = default;
    SubDeviceManager& operator=(SubDeviceManager&& other) noexcept = default;

    ~SubDeviceManager();

    SubDeviceManagerId id() const;

    const std::vector<SubDeviceId>& get_sub_device_ids() const;
    const SubDevice& sub_device(SubDeviceId sub_device_id) const;

    const vector_aligned<uint32_t>& noc_mcast_unicast_data() const;
    bool has_noc_mcast_txns(SubDeviceId sub_device_id) const;
    uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const;
    uint8_t noc_unicast_data_start_index(SubDeviceId sub_device_id) const;

    const std::vector<std::pair<CoreRangeSet, uint32_t>>& get_core_go_message_mapping() const;

    const std::unique_ptr<AllocatorImpl>& allocator(SubDeviceId sub_device_id) const;
    std::unique_ptr<AllocatorImpl>& sub_device_allocator(SubDeviceId sub_device_id);
    const std::vector<std::unique_ptr<AllocatorImpl>>& allocators() const { return sub_device_allocators_; }

    std::shared_ptr<distributed::MeshTraceBuffer>& create_trace(const distributed::MeshTraceId& trace_id);
    void release_trace(const distributed::MeshTraceId& trace_id);
    std::shared_ptr<distributed::MeshTraceBuffer> get_trace(const distributed::MeshTraceId& trace_id);
    DeviceAddr get_max_trace_high_water_mark() const;

    uint8_t num_sub_devices() const;
    bool has_allocations() const;
    DeviceAddr local_l1_size() const;
    DeviceAddr global_l1_bottom_reservation_size() const;

    const std::vector<SubDeviceId>& get_sub_device_stall_group() const;
    void set_sub_device_stall_group(ttsl::Span<const SubDeviceId> sub_device_ids);
    void reset_sub_device_stall_group();

private:
    void validate_sub_devices() const;
    uint8_t get_sub_device_index(SubDeviceId sub_device_id) const;
    void populate_sub_device_ids();
    void populate_num_cores();
    void populate_sub_allocators();
    void populate_noc_data();
    // Release this manager's persistent-L1 seals so those cores may take new
    // PrefetcherPipe (or other arena) allocations. See PersistentL1Arena for the
    // per-core L1 layout. No-op if this manager never placed a local L1 slab.
    void unseal_persistent_l1();

    static std::atomic<uint64_t> next_sub_device_manager_id_;

    SubDeviceManagerId id_;

    // TODO: We have a max number of sub-devices, so we can use a fixed size array
    std::vector<SubDevice> sub_devices_;
    std::vector<SubDeviceId> sub_device_ids_;
    std::vector<SubDeviceId> sub_device_stall_group_;
    IDevice* device_;
    tt::tt_metal::ContextId context_id_;

    DeviceAddr local_l1_size_;
    DeviceAddr global_l1_bottom_reservation_size_ = 0;
    std::vector<std::unique_ptr<AllocatorImpl>> sub_device_allocators_;
    // Cores this manager sealed in populate_sub_allocators; unseal_persistent_l1
    // walks this list. Empty when local_l1_size_ == 0.
    std::vector<CoreRangeSet> persistent_l1_seals_;

    std::array<uint32_t, NumHalProgrammableCoreTypes> num_cores_{};

    vector_aligned<uint32_t> noc_mcast_unicast_data_;
    std::vector<bool> has_noc_mcast_txns_;
    std::vector<uint8_t> num_noc_unicast_txns_;
    std::vector<uint8_t> noc_unicast_data_start_index_;

    std::vector<std::pair<CoreRangeSet, uint32_t>> core_go_message_mapping_;

    std::unordered_map<distributed::MeshTraceId, std::shared_ptr<distributed::MeshTraceBuffer>> trace_buffer_pool_;

    // TODO #15944: Temporary until migration to actual fabric is complete
    std::optional<SubDeviceId> fabric_sub_device_id_ = std::nullopt;
};

}  // namespace tt::tt_metal
