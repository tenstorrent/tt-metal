// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include <buffer_types.hpp>
#include <command_queue.hpp>
#include <device.hpp>
#include <sub_device.hpp>
#include <sub_device_types.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "core_coord.hpp"
#include "hal_types.hpp"
#include "mesh_command_queue.hpp"
#include "mesh_device.hpp"
#include <tt_stl/strong_type.hpp>
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"

namespace tt::tt_metal {

SubDeviceManagerTracker::SubDeviceManagerTracker(
    IDevice* device, std::unique_ptr<Allocator>&& global_allocator, tt::stl::Span<const SubDevice> sub_devices) :
    device_(device) {
    auto sub_device_manager = std::make_unique<SubDeviceManager>(device, std::move(global_allocator), sub_devices);
    default_sub_device_manager_ = sub_device_manager.get();
    active_sub_device_manager_ = default_sub_device_manager_;
    sub_device_managers_.insert_or_assign(sub_device_manager->id(), std::move(sub_device_manager));
}

SubDeviceManagerTracker::~SubDeviceManagerTracker() {
    active_sub_device_manager_ = nullptr;
    for (auto sub_device_manager = sub_device_managers_.begin(); sub_device_manager != sub_device_managers_.end();) {
        this->remove_sub_device_manager((sub_device_manager++)->first);
    }
    default_sub_device_manager_ = nullptr;
}

SubDeviceManagerId SubDeviceManagerTracker::create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    auto sub_device_manager = std::make_unique<SubDeviceManager>(sub_devices, local_l1_size, device_);
    auto sub_device_manager_id = sub_device_manager->id();
    sub_device_managers_.insert_or_assign(sub_device_manager_id, std::move(sub_device_manager));
    return sub_device_manager_id;
}

std::tuple<SubDeviceManagerId, SubDeviceId> SubDeviceManagerTracker::create_sub_device_manager_with_fabric(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    auto fabric_sub_device = SubDevice(std::array{
        CoreRangeSet(),
        default_sub_device_manager_->sub_device(SubDeviceId{0}).cores(HalProgrammableCoreType::ACTIVE_ETH)});
    auto new_sub_devices = std::vector<SubDevice>(sub_devices.begin(), sub_devices.end());
    new_sub_devices.push_back(fabric_sub_device);
    auto fabric_sub_device_id = SubDeviceId{static_cast<uint32_t>(new_sub_devices.size() - 1)};
    auto sub_device_manager_id = this->create_sub_device_manager(new_sub_devices, local_l1_size);
    return {sub_device_manager_id, fabric_sub_device_id};
}

void SubDeviceManagerTracker::reset_sub_device_state(const std::unique_ptr<SubDeviceManager>& sub_device_manager) {
    auto num_sub_devices = sub_device_manager->num_sub_devices();
    // Dynamic resolution of device types is unclean and poor design. This will be cleaned up
    // when MeshCommandQueue + CommandQueue are unified under the same API
    if (dynamic_cast<distributed::MeshDevice*>(device_)) {
        // Multi CQ support for MeshDevice is not currently available
        distributed::MeshDevice* mesh_device = dynamic_cast<distributed::MeshDevice*>(device_);
        mesh_device->mesh_command_queue().reset_worker_state(
            true, num_sub_devices, sub_device_manager->noc_mcast_unicast_data());
    } else {
        for (uint8_t cq_id = 0; cq_id < device_->num_hw_cqs(); ++cq_id) {
            auto& hw_cq = device_->command_queue(cq_id);
            // Only need to reset launch messages once, so reset on cq 0
            hw_cq.reset_worker_state(cq_id == 0, num_sub_devices, sub_device_manager->noc_mcast_unicast_data());
        }
    }
    sub_device_manager->reset_sub_device_stall_group();
}

void SubDeviceManagerTracker::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    TT_FATAL(!device_->using_slow_dispatch(), "Using sub device managers is unsupported with slow dispatch");
    if (active_sub_device_manager_->id() == sub_device_manager_id) {
        return;
    }
    if (active_sub_device_manager_->id() != default_sub_device_manager_->id()) {
        TT_FATAL(
            !active_sub_device_manager_->has_allocations(),
            "Cannot switch sub device managers while sub devices still have local allocations");
    }
    auto sub_device_manager = sub_device_managers_.find(sub_device_manager_id);
    TT_FATAL(sub_device_manager != sub_device_managers_.end(), "Sub device manager does not exist");
    this->reset_sub_device_state(sub_device_manager->second);
    const auto& default_allocator = default_sub_device_manager_->allocator(SubDeviceId{0});
    default_allocator->reset_allocator_size(BufferType::L1);
    // Shrink the global allocator size to make room for sub-device allocators
    auto local_l1_size = sub_device_manager->second->local_l1_size();
    default_allocator->shrink_allocator_size(BufferType::L1, local_l1_size, /*bottom_up=*/true);
    active_sub_device_manager_ = sub_device_manager->second.get();
}

void SubDeviceManagerTracker::clear_loaded_sub_device_manager() {
    this->load_sub_device_manager(default_sub_device_manager_->id());
}

void SubDeviceManagerTracker::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    if (active_sub_device_manager_ != nullptr) {
        TT_FATAL(
            sub_device_manager_id != active_sub_device_manager_->id(),
            "Cannot remove active sub device manager {}",
            sub_device_manager_id);
        TT_FATAL(
            sub_device_manager_id != default_sub_device_manager_->id(),
            "Cannot remove default sub device manager {}",
            sub_device_manager_id);
    }
    auto sub_device_manager = sub_device_managers_.find(sub_device_manager_id);
    TT_FATAL(sub_device_manager != sub_device_managers_.end(), "Sub device manager does not exist");
    sub_device_managers_.erase(sub_device_manager);
}

SubDeviceManager* SubDeviceManagerTracker::get_active_sub_device_manager() const { return active_sub_device_manager_; }

SubDeviceManager* SubDeviceManagerTracker::get_default_sub_device_manager() const {
    return default_sub_device_manager_;
}

SubDeviceManagerId SubDeviceManagerTracker::get_active_sub_device_manager_id() const {
    return active_sub_device_manager_->id();
}

SubDeviceManagerId SubDeviceManagerTracker::get_default_sub_device_manager_id() const {
    return default_sub_device_manager_->id();
}

std::optional<DeviceAddr> SubDeviceManagerTracker::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    constexpr uint32_t global_bank_id = 0;
    DeviceAddr lowest_addr = std::numeric_limits<DeviceAddr>::max();
    // Global bank id needs to look up a bank from the compute grid (not the storage grid)
    // Since banks are lockstep in an allocator it doesn't matter if the actual core matches or not
    const auto& global_allocator = default_sub_device_manager_->allocator(SubDeviceId{0});
    auto found_addr = global_allocator->get_lowest_occupied_l1_address(global_bank_id);
    if (found_addr.has_value()) {
        lowest_addr = std::min(lowest_addr, *found_addr);
    }
    // If no sub device ids are specified, check all sub_device ids
    if (sub_device_ids.empty() && default_sub_device_manager_ != active_sub_device_manager_) {
        static_assert(
            std::is_reference_v<
                std::invoke_result_t<decltype(&SubDeviceManager::get_sub_device_ids), SubDeviceManager>>,
            "Getting a span from get_sub_device_ids requires it to be a reference");
        sub_device_ids = tt::stl::Span<const SubDeviceId>(active_sub_device_manager_->get_sub_device_ids());
    }
    for (const auto& sub_device_id : sub_device_ids) {
        const auto& allocator = this->get_active_sub_device_manager()->sub_device_allocator(sub_device_id);
        if (allocator) {
            // Having an allocator means there are Tensix cores in this sub-device
            const auto& cores =
                this->get_active_sub_device_manager()->sub_device(sub_device_id).cores(HalProgrammableCoreType::TENSIX);
            auto bank_id = allocator->get_bank_ids_from_logical_core(BufferType::L1, cores.ranges()[0].start_coord)[0];
            found_addr = allocator->get_lowest_occupied_l1_address(bank_id);
            if (found_addr.has_value()) {
                lowest_addr = std::min(lowest_addr, *found_addr);
            }
        }
    }
    return lowest_addr == std::numeric_limits<DeviceAddr>::max() ? std::nullopt : std::make_optional(lowest_addr);
}

}  // namespace tt::tt_metal
