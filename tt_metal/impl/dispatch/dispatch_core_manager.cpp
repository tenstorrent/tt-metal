// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_core_manager.hpp"

#include <algorithm>
#include <functional>
#include <list>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "core_descriptor.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

const tt_cxy_pair& dispatch_core_manager::prefetcher_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.prefetcher.has_value()) {
        return assignment.prefetcher.value();
    }
    // Issue queue interface is on the MMIO device
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    CoreCoord issue_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
    assignment.prefetcher = tt_cxy_pair(mmio_device_id, issue_queue_coord.x, issue_queue_coord.y);
    log_dispatch_assignment("Prefetcher", assignment.prefetcher.value(), device_id, channel, cq_id);
    return assignment.prefetcher.value();
}

bool dispatch_core_manager::is_prefetcher_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.prefetcher.has_value();
}

const tt_cxy_pair& dispatch_core_manager::prefetcher_d_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.prefetcher_d.has_value()) {
        return assignment.prefetcher_d.value();
    }
    CoreCoord prefetch_d_coord = this->get_next_available_dispatch_core(device_id);
    assignment.prefetcher_d = tt_cxy_pair(device_id, prefetch_d_coord.x, prefetch_d_coord.y);
    log_dispatch_assignment("Prefetcher D", assignment.prefetcher_d.value(), device_id, channel, cq_id);
    return assignment.prefetcher_d.value();
}

bool dispatch_core_manager::is_prefetcher_d_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.prefetcher_d.has_value();
}

const tt_cxy_pair& dispatch_core_manager::completion_queue_writer_core(
    ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.completion_queue_writer.has_value()) {
        return assignment.completion_queue_writer.value();
    }
    // Completion queue interface is on the MMIO device
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    CoreCoord completion_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
    assignment.completion_queue_writer =
        tt_cxy_pair(mmio_device_id, completion_queue_coord.x, completion_queue_coord.y);
    TT_ASSERT(
        not assignment.dispatcher.has_value(),
        "Command dispatcher core {} must match completion queue interface core for MMIO device {}",
        assignment.dispatcher.value().str(),
        device_id);
    assignment.dispatcher = assignment.completion_queue_writer;
    log_dispatch_assignment(
        "Completion Queue Writer", assignment.completion_queue_writer.value(), device_id, channel, cq_id);
    return assignment.completion_queue_writer.value();
}

bool dispatch_core_manager::is_completion_queue_writer_core_allocated(
    ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.completion_queue_writer.has_value();
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    return this->dispatcher_core_locked(device_id, channel, cq_id);
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_core_locked(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher.has_value()) {
        return assignment.dispatcher.value();
    }
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    CoreCoord dispatcher_coord = this->get_next_available_dispatch_core(mmio_device_id);
    assignment.dispatcher = tt_cxy_pair(mmio_device_id, dispatcher_coord.x, dispatcher_coord.y);
    TT_ASSERT(
        not assignment.completion_queue_writer.has_value(),
        "Command dispatcher core must match completion queue interface core for MMIO device {}",
        device_id);
    assignment.completion_queue_writer = assignment.dispatcher;
    log_dispatch_assignment("Dispatcher", assignment.dispatcher.value(), device_id, channel, cq_id);
    return assignment.dispatcher.value();
}

bool dispatch_core_manager::is_dispatcher_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.dispatcher.has_value();
}

bool dispatch_core_manager::is_dispatcher_s_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.dispatcher_s.has_value();
}

bool dispatch_core_manager::is_dispatcher_d_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.dispatcher_d.has_value();
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_d_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    return this->dispatcher_d_core_locked(device_id, channel, cq_id);
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_d_core_locked(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher_d.has_value()) {
        return assignment.dispatcher_d.value();
    }
    CoreCoord dispatcher_d_coord = this->get_next_available_dispatch_core(device_id);
    assignment.dispatcher_d = tt_cxy_pair(device_id, dispatcher_d_coord.x, dispatcher_d_coord.y);
    log_dispatch_assignment("Dispatcher D", assignment.dispatcher_d.value(), device_id, channel, cq_id);
    return assignment.dispatcher_d.value();
}

const tt_cxy_pair& dispatch_core_manager::fabric_mux_core(
    ChipId device_id, uint16_t channel, uint8_t cq_id, int tunnel) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (!assignment.fabric_mux.contains(tunnel)) {
        CoreCoord coord = this->get_next_available_dispatch_core(device_id);
        assignment.fabric_mux[tunnel] = tt_cxy_pair(device_id, coord.x, coord.y);
        log_dispatch_assignment("FabricMux", assignment.fabric_mux[tunnel], device_id, channel, cq_id);
    }
    return assignment.fabric_mux[tunnel];
}

bool dispatch_core_manager::is_fabric_mux_core_allocated(
    ChipId device_id, uint16_t channel, uint8_t cq_id, int tunnel) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.fabric_mux.contains(tunnel);
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_s_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher_s.has_value()) {
        return assignment.dispatcher_s.value();
    }
    CoreCoord dispatcher_s_coord;
    if (this->get_dispatch_core_type() == CoreType::WORKER) {
        ChipId mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
        if (mmio_device_id == device_id) {
            // dispatch_s is on the same tensix core as dispatch_hd
            dispatcher_s_coord = this->dispatcher_core_locked(device_id, channel, cq_id);
        } else {
            // dispatch_s is on the same tensix as dispatch_d
            dispatcher_s_coord = this->dispatcher_d_core_locked(device_id, channel, cq_id);
        }
    } else {
        dispatcher_s_coord = this->get_next_available_dispatch_core(device_id);
    }
    assignment.dispatcher_s = tt_cxy_pair(device_id, dispatcher_s_coord.x, dispatcher_s_coord.y);
    log_dispatch_assignment("Dispatcher S", assignment.dispatcher_s.value(), device_id, channel, cq_id);
    return assignment.dispatcher_s.value();
}

CoreType dispatch_core_manager::get_dispatch_core_type() {
    return get_core_type_from_config(this->dispatch_core_config_);
}

DispatchCoreConfig dispatch_core_manager::get_dispatch_core_config() { return this->dispatch_core_config_; }

void dispatch_core_manager::add_dispatch_core_to_device(ChipId device_id, const CoreCoord& core) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    this->add_dispatch_core_to_device_locked(device_id, core);
}

void dispatch_core_manager::add_dispatch_core_to_device_locked(ChipId device_id, const CoreCoord& core) {
    // TODO: remove this API, we should read the core descriptor once, should not have backdoors like this to add cores
    auto& dispatch_cores = available_dispatch_cores_by_device.at(device_id);
    if (std::find(dispatch_cores.begin(), dispatch_cores.end(), core) == dispatch_cores.end()) {
        dispatch_cores.push_back(core);
    }
}

std::vector<CoreCoord> dispatch_core_manager::get_all_logical_dispatch_cores(ChipId device_id) {
    return tt::get_logical_dispatch_cores(device_id, MAX_NUM_HW_CQS, this->dispatch_core_config_);
}

// private methods

dispatch_core_manager::dispatch_core_manager(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) {
    this->reset_dispatch_core_manager(dispatch_core_config, num_hw_cqs);
}

void dispatch_core_manager::reset_dispatch_core_manager(
    const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) {
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    this->dispatch_core_assignments.clear();
    this->available_dispatch_cores_by_device.clear();
    this->dispatch_core_config_ = dispatch_core_config;
    for (ChipId device_id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
        std::list<CoreCoord>& logical_dispatch_cores = this->available_dispatch_cores_by_device[device_id];
        for (const CoreCoord& logical_dispatch_core :
             tt::get_logical_dispatch_cores(device_id, MAX_NUM_HW_CQS, dispatch_core_config)) {
            logical_dispatch_cores.push_back(logical_dispatch_core);
        }

        this->num_hw_cqs = num_hw_cqs;

        // When running Multiple CQs using Ethernet Dispatch, we may need more dispatch cores than those allocated in
        // the core descriptor (ex: 2 CQs on N300 need 10 dispatch cores and the core descriptor only allocates 6).
        // Infer the remaining dispatch cores from the idle eth core list (this is device dependent).
        if (get_core_type_from_config(dispatch_core_config) == CoreType::ETH) {
            for (const auto& idle_eth_core :
                 tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(device_id)) {
                add_dispatch_core_to_device_locked(device_id, idle_eth_core);
            }
        }
    }
}

CoreCoord dispatch_core_manager::get_next_available_dispatch_core(ChipId device_id) {
    if (!this->available_dispatch_cores_by_device.contains(device_id)) {
        TT_THROW("Invalid device ID to assign dispatch cores {}", device_id);
    }
    if (this->available_dispatch_cores_by_device.at(device_id).empty()) {
        TT_THROW(
            "No more available dispatch cores on device {} to assign. Expand dispatch cores specified in core "
            "descriptor YAML",
            device_id);
    }
    CoreCoord avail_dispatch_core = this->available_dispatch_cores_by_device.at(device_id).front();
    this->available_dispatch_cores_by_device.at(device_id).pop_front();
    return avail_dispatch_core;
}

void dispatch_core_manager::log_dispatch_assignment(
    [[maybe_unused]] std::string name,
    [[maybe_unused]] tt_cxy_pair& cxy,
    [[maybe_unused]] ChipId device_id,
    [[maybe_unused]] uint16_t channel,
    [[maybe_unused]] uint8_t cq_id,
    [[maybe_unused]] bool force_ethernet) {
    log_debug(
        tt::LogMetal,
        "Allocated {} Core: {}({}) for Device {} Channel {} CQ ID {}",
        name,
        cxy.str(),
        tt::tt_metal::MetalContext::instance()
            .get_cluster()
            .get_virtual_coordinate_from_logical_coordinates(
                cxy, force_ethernet ? CoreType::ETH : get_dispatch_core_type())
            .str(),
        device_id,
        channel,
        cq_id);
}

}  // namespace tt::tt_metal
