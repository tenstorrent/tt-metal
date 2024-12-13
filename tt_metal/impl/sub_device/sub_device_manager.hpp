// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/dispatch/memcpy.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/impl/sub_device/sub_device.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "tt_metal/tt_stl/span.hpp"

namespace tt::tt_metal {

class LaunchMessageRingBufferState;
class TraceBuffer;

inline namespace v0 {
class Device;
}  // namespace v0

namespace detail {
class SubDeviceManager {
public:
    static constexpr uint32_t MAX_NUM_SUB_DEVICES = 16;
    static_assert(
        MAX_NUM_SUB_DEVICES <= std::numeric_limits<SubDeviceId::Id>::max(),
        "MAX_NUM_SUB_DEVICES must be less than or equal to the max value of SubDeviceId::Id");
    // Constructor used for the default/global device
    SubDeviceManager(Device* device, std::unique_ptr<Allocator>&& global_allocator);
    // Constructor used for regular sub-devices
    SubDeviceManager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size, Device* device);

    SubDeviceManager(const SubDeviceManager& other) = delete;
    SubDeviceManager& operator=(const SubDeviceManager& other) = delete;

    SubDeviceManager(SubDeviceManager&& other) noexcept = default;
    SubDeviceManager& operator=(SubDeviceManager&& other) noexcept = default;

    ~SubDeviceManager();

    const std::vector<SubDeviceId>& get_sub_device_ids() const;
    const SubDevice& sub_device(SubDeviceId sub_device_id) const;

    const vector_memcpy_aligned<uint32_t>& noc_mcast_unicast_data() const;
    uint8_t num_noc_mcast_txns(SubDeviceId sub_device_id) const;
    uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const;
    uint8_t noc_mcast_data_start_index(SubDeviceId sub_device_id) const;
    uint8_t noc_unicast_data_start_index(SubDeviceId sub_device_id) const;

    const std::unique_ptr<Allocator>& get_initialized_allocator(SubDeviceId sub_device_id) const;
    std::unique_ptr<Allocator>& sub_device_allocator(SubDeviceId sub_device_id);

    std::shared_ptr<TraceBuffer>& create_trace(uint32_t tid);
    void release_trace(uint32_t tid);
    std::shared_ptr<TraceBuffer> get_trace(uint32_t tid);

    void reset_worker_launch_message_buffer_state();
    LaunchMessageRingBufferState& get_worker_launch_message_buffer_state(SubDeviceId sub_device_id);

    uint8_t num_sub_devices() const;
    bool has_allocations() const;
    DeviceAddr local_l1_size() const;

    // #TODO #15944: Temporary until migration to actual fabric is complete
    void set_fabric_sub_device_id(SubDeviceId sub_device_id);
    std::optional<SubDeviceId> fabric_sub_device_id() const;

private:
    void validate_sub_devices() const;
    uint8_t get_sub_device_index(SubDeviceId sub_device_id) const;
    void populate_sub_device_ids();
    void populate_num_cores();
    void populate_sub_allocators();
    void populate_noc_data();
    void populate_worker_launch_message_buffer_state();

    // TODO: We have a max number of sub-devices, so we can use a fixed size array
    std::vector<SubDevice> sub_devices_;
    std::vector<SubDeviceId> sub_device_ids_;
    Device* device_;

    DeviceAddr local_l1_size_;
    std::vector<std::unique_ptr<Allocator>> sub_device_allocators_;

    std::array<uint32_t, NumHalProgrammableCoreTypes> num_cores_{};

    // mcast txn data followed by unicast txn data
    vector_memcpy_aligned<uint32_t> noc_mcast_unicast_data_;
    std::vector<uint8_t> num_noc_mcast_txns_;
    std::vector<uint8_t> num_noc_unicast_txns_;
    std::vector<uint8_t> noc_mcast_data_start_index_;
    std::vector<uint8_t> noc_unicast_data_start_index_;

    std::unordered_map<uint32_t, std::shared_ptr<TraceBuffer>> trace_buffer_pool_;

    std::vector<LaunchMessageRingBufferState> worker_launch_message_buffer_state_;

    // TODO #15944: Temporary until migration to actual fabric is complete
    std::optional<SubDeviceId> fabric_sub_device_id_ = std::nullopt;
};

}  // namespace detail

}  // namespace tt::tt_metal
