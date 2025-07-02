// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

class IDevice;

class Allocator;
class SubDeviceManager;

// This class is used by Device-like objects for tracking the SubDeviceManagers set up by the user
class SubDeviceManagerTracker {
public:
    // TODO: Potentially move the global allocator creation into here instead of from the device
    // This creates the SubDeviceManagerTracker with a default SubDeviceManager that has the entire grid as a sub-device
    SubDeviceManagerTracker(
        IDevice* device, std::unique_ptr<Allocator>&& global_allocator, tt::stl::Span<const SubDevice> sub_devices);

    SubDeviceManagerTracker(const SubDeviceManagerTracker& other) = delete;
    SubDeviceManagerTracker& operator=(const SubDeviceManagerTracker& other) = delete;

    SubDeviceManagerTracker(SubDeviceManagerTracker&& other) noexcept = default;
    SubDeviceManagerTracker& operator=(SubDeviceManagerTracker&& other) noexcept = default;

    ~SubDeviceManagerTracker();

    SubDeviceManagerId create_sub_device_manager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);

    void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id);

    void clear_loaded_sub_device_manager();

    void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id);

    SubDeviceManager* get_active_sub_device_manager() const;

    SubDeviceManager* get_default_sub_device_manager() const;

    // Used for caching program state by manager and buffers to check that the required manager is still active
    SubDeviceManagerId get_active_sub_device_manager_id() const;

    // Currently only used by program's determine_sub_device_ids algorithm to avoid running the search algorithm in the
    // default case to not affect performance
    SubDeviceManagerId get_default_sub_device_manager_id() const;

    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) const;

private:
    void reset_sub_device_state(const std::unique_ptr<SubDeviceManager>& sub_device_manager);

    IDevice* device_ = nullptr;

    std::unordered_map<SubDeviceManagerId, std::unique_ptr<SubDeviceManager>> sub_device_managers_;
    SubDeviceManager* active_sub_device_manager_ = nullptr;
    SubDeviceManager* default_sub_device_manager_ = nullptr;
};

}  // namespace tt::tt_metal
