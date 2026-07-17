// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <allocator.hpp>
#include <buffer_types.hpp>
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

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "mesh_device.hpp"
#include "tt_metal/distributed/mesh_command_queue_base.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include <tt_stl/strong_type.hpp>
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"

namespace tt::tt_metal {

SubDeviceManagerTracker::SubDeviceManagerTracker(
    IDevice* device, std::unique_ptr<AllocatorImpl>&& global_allocator, ttsl::Span<const SubDevice> sub_devices) :
    device_(device) {
    TT_FATAL(device_ != nullptr, "SubDeviceManagerTracker requires a valid device");
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
    ttsl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    auto sub_device_manager = std::make_unique<SubDeviceManager>(sub_devices, local_l1_size, device_);
    auto sub_device_manager_id = sub_device_manager->id();
    sub_device_managers_.insert_or_assign(sub_device_manager_id, std::move(sub_device_manager));
    return sub_device_manager_id;
}

void SubDeviceManagerTracker::reset_sub_device_state(const std::unique_ptr<SubDeviceManager>& sub_device_manager) {
    auto num_sub_devices = sub_device_manager->num_sub_devices();
    std::vector<uint32_t> workers_per_sub_device;
    workers_per_sub_device.reserve(num_sub_devices);
    for (uint8_t i = 0; i < num_sub_devices; ++i) {
        const auto sub_device_id = SubDeviceId{i};
        const auto& sub_device = sub_device_manager->sub_device(sub_device_id);
        workers_per_sub_device.push_back(
            sub_device.cores(HalProgrammableCoreType::TENSIX).num_cores() +
            sub_device.cores(HalProgrammableCoreType::ACTIVE_ETH).num_cores());
    }
    // Dynamic resolution of device types is unclean and poor design. This will be cleaned up
    // when MeshCommandQueue + HWCommandQueue are unified under the same API
    if (dynamic_cast<distributed::MeshDevice*>(device_)) {
        // Multi CQ support for MeshDevice is not currently available
        distributed::MeshDevice* mesh_device = dynamic_cast<distributed::MeshDevice*>(device_);
        for (uint8_t cq_id = 0; cq_id < mesh_device->num_hw_cqs(); ++cq_id) {
            mesh_device->impl().mesh_command_queue_base(cq_id).reset_worker_state(
                cq_id == 0,
                num_sub_devices,
                sub_device_manager->noc_mcast_unicast_data(),
                sub_device_manager->get_core_go_message_mapping(),
                ttsl::Span<const uint32_t>(workers_per_sub_device.data(), workers_per_sub_device.size()));
        }
    } else {
        TT_FATAL(false, "Sub device managers are unsupported with non-mesh devices");
    }
    sub_device_manager->reset_sub_device_stall_group();
}

void SubDeviceManagerTracker::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    TT_FATAL(
        tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch(),
        "Using sub device managers is unsupported with slow dispatch");
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

DeviceAddr SubDeviceManagerTracker::get_max_trace_high_water_mark() const {
    DeviceAddr max_high_water_mark = 0;
    for (const auto& entry : sub_device_managers_) {
        max_high_water_mark = std::max(max_high_water_mark, entry.second->get_max_trace_high_water_mark());
    }
    return max_high_water_mark;
}

std::optional<DeviceAddr> SubDeviceManagerTracker::get_min_trace_buffer_address() const {
    std::optional<DeviceAddr> min_trace_buffer_address;
    for (const auto& entry : sub_device_managers_) {
        const auto manager_min_address = entry.second->get_min_trace_buffer_address();
        if (manager_min_address.has_value()) {
            const DeviceAddr& address = manager_min_address.value();
            min_trace_buffer_address = std::min(min_trace_buffer_address.value_or(address), address);
        }
    }
    return min_trace_buffer_address;
}

SubDeviceManagerId SubDeviceManagerTracker::get_active_sub_device_manager_id() const {
    return active_sub_device_manager_->id();
}

SubDeviceManagerId SubDeviceManagerTracker::get_default_sub_device_manager_id() const {
    return default_sub_device_manager_->id();
}

std::optional<DeviceAddr> SubDeviceManagerTracker::lowest_occupied_compute_l1_address(
    ttsl::Span<const SubDeviceId> sub_device_ids) const {
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
        sub_device_ids = ttsl::Span<const SubDeviceId>(active_sub_device_manager_->get_sub_device_ids());
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
