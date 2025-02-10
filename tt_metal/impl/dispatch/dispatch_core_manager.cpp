// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_core_manager.hpp"

#include "core_descriptor.hpp"
#include "core_coord.hpp"
#include <list>
#include "dispatch_core_common.hpp"

#include "tt_cluster.hpp"

namespace tt::tt_metal {

dispatch_core_manager* dispatch_core_manager::_inst = nullptr;

void dispatch_core_manager::initialize(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) noexcept {
    log_debug(tt::LogMetal, "DevicePool initialize");
    if (_inst == nullptr) {
        static dispatch_core_manager dispatch_core_manager(dispatch_core_config, num_hw_cqs);
        _inst = &dispatch_core_manager;
    } else if (_inst->dispatch_core_config_by_device[0] != dispatch_core_config or num_hw_cqs != _inst->num_hw_cqs) {
        _inst->reset_dispatch_core_manager(dispatch_core_config, num_hw_cqs);
    }
}

dispatch_core_manager& dispatch_core_manager::instance() {
    TT_ASSERT(dispatch_core_manager::_inst != nullptr, "Trying to get dispatch_core_manager without initializing it");
    return *dispatch_core_manager::_inst;
}

const tt_cxy_pair& dispatch_core_manager::prefetcher_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.prefetcher.has_value()) {
        return assignment.prefetcher.value();
    }
    // Issue queue interface is on the MMIO device
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    CoreCoord issue_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
    assignment.prefetcher = tt_cxy_pair(mmio_device_id, issue_queue_coord.x, issue_queue_coord.y);
    log_dispatch_assignment("Prefetcher", assignment.prefetcher.value(), device_id, channel, cq_id);
    return assignment.prefetcher.value();
}

bool dispatch_core_manager::is_prefetcher_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.prefetcher.has_value()) {
        return true;
    }
    return false;
}

const tt_cxy_pair& dispatch_core_manager::prefetcher_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.prefetcher_d.has_value()) {
        return assignment.prefetcher_d.value();
    }
    CoreCoord prefetch_d_coord = this->get_next_available_dispatch_core(device_id);
    assignment.prefetcher_d = tt_cxy_pair(device_id, prefetch_d_coord.x, prefetch_d_coord.y);
    log_dispatch_assignment("Prefetcher D", assignment.prefetcher_d.value(), device_id, channel, cq_id);
    return assignment.prefetcher_d.value();
}

bool dispatch_core_manager::is_prefetcher_d_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.prefetcher_d.has_value()) {
        return true;
    }
    return false;
}

const tt_cxy_pair& dispatch_core_manager::mux_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.mux.has_value()) {
        return assignment.mux.value();
    }
    // Mux interface is on the MMIO device
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    CoreCoord mux_coord = this->get_next_available_dispatch_core(mmio_device_id);
    assignment.mux = tt_cxy_pair(mmio_device_id, mux_coord.x, mux_coord.y);
    log_dispatch_assignment("Mux", assignment.mux.value(), device_id, channel, cq_id);
    return assignment.mux.value();
}

bool dispatch_core_manager::is_mux_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.mux.has_value()) {
        return true;
    }
    return false;
}

const tt_cxy_pair& dispatch_core_manager::mux_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.mux_d.has_value()) {
        return assignment.mux_d.value();
    }
    // mux_d is on remote device
    CoreCoord mux_d_coord = this->get_next_available_dispatch_core(device_id);
    assignment.mux_d = tt_cxy_pair(device_id, mux_d_coord.x, mux_d_coord.y);
    log_dispatch_assignment("Mux D", assignment.mux_d.value(), device_id, channel, cq_id);
    return assignment.mux_d.value();
}

const tt_cxy_pair& dispatch_core_manager::demux_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.demux.has_value()) {
        return assignment.demux.value();
    }
    // demux interface is on the MMIO device
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    CoreCoord demux_coord = this->get_next_available_dispatch_core(mmio_device_id);
    assignment.demux = tt_cxy_pair(mmio_device_id, demux_coord.x, demux_coord.y);
    log_dispatch_assignment("Demux", assignment.demux.value(), device_id, channel, cq_id);
    return assignment.demux.value();
}

bool dispatch_core_manager::is_demux_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.demux.has_value()) {
        return true;
    }
    return false;
}

const tt_cxy_pair& dispatch_core_manager::demux_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.demux_d.has_value()) {
        return assignment.demux_d.value();
    }
    // demux_d is on remote device
    CoreCoord demux_d_coord = this->get_next_available_dispatch_core(device_id);
    assignment.demux_d = tt_cxy_pair(device_id, demux_d_coord.x, demux_d_coord.y);
    log_dispatch_assignment("Demux D", assignment.demux_d.value(), device_id, channel, cq_id);
    return assignment.demux_d.value();
}

const tt_cxy_pair& dispatch_core_manager::tunneler_core(
    chip_id_t upstream_device_id, chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.tunneler.has_value()) {
        return assignment.tunneler.value();
    }

    auto [us_core, ds_core] =
        tt::Cluster::instance().get_eth_tunnel_core(upstream_device_id, device_id, EthRouterMode::BI_DIR_TUNNELING);

    assignment.tunneler = us_core;
    assignment.tunneler_d = ds_core;

    log_dispatch_assignment("Tunneler Remote", assignment.tunneler.value(), device_id, channel, cq_id, true);
    log_dispatch_assignment("Tunneler Local", assignment.tunneler_d.value(), device_id, channel, cq_id, true);
    return assignment.tunneler.value();
}

const tt_cxy_pair& dispatch_core_manager::us_tunneler_core_local(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.tunneler_d.has_value()) {
        return assignment.tunneler_d.value();
    }
    TT_ASSERT(false, "Device {} has no allocation for Local Upstream Tunneler Core.", device_id);
    assignment.tunneler_d = tt_cxy_pair(0, 0, 0);
    return assignment.tunneler_d.value();
}

const tt_cxy_pair& dispatch_core_manager::completion_queue_writer_core(
    chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.completion_queue_writer.has_value()) {
        return assignment.completion_queue_writer.value();
    }
    // Completion queue interface is on the MMIO device
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
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
    chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.completion_queue_writer.has_value()) {
        return true;
    }
    return false;
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher.has_value()) {
        return assignment.dispatcher.value();
    }
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
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

bool dispatch_core_manager::is_dispatcher_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher.has_value()) {
        return true;
    }
    return false;
}

bool dispatch_core_manager::is_dispatcher_s_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.dispatcher_s.has_value();
}

bool dispatch_core_manager::is_dispatcher_d_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.dispatcher_d.has_value();
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher_d.has_value()) {
        return assignment.dispatcher_d.value();
    }
    CoreCoord dispatcher_d_coord = this->get_next_available_dispatch_core(device_id);
    assignment.dispatcher_d = tt_cxy_pair(device_id, dispatcher_d_coord.x, dispatcher_d_coord.y);
    log_dispatch_assignment("Dispatcher D", assignment.dispatcher_d.value(), device_id, channel, cq_id);
    return assignment.dispatcher_d.value();
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_s_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher_s.has_value()) {
        return assignment.dispatcher_s.value();
    }
    CoreCoord dispatcher_s_coord;
    if (this->get_dispatch_core_type(device_id) == CoreType::WORKER) {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        if (mmio_device_id == device_id) {
            // dispatch_s is on the same tensix core as dispatch_hd
            dispatcher_s_coord = this->dispatcher_core(device_id, channel, cq_id);
        } else {
            // dispatch_s is on the same tensix as dispatch_d
            dispatcher_s_coord = this->dispatcher_d_core(device_id, channel, cq_id);
        }
    } else {
        dispatcher_s_coord = this->get_next_available_dispatch_core(device_id);
    }
    assignment.dispatcher_s = tt_cxy_pair(device_id, dispatcher_s_coord.x, dispatcher_s_coord.y);
    log_dispatch_assignment("Dispatcher S", assignment.dispatcher_s.value(), device_id, channel, cq_id);
    return assignment.dispatcher_s.value();
}

CoreType dispatch_core_manager::get_dispatch_core_type(chip_id_t device_id) {
    return this->dispatch_core_config_by_device[device_id].get_core_type();
}

DispatchCoreConfig dispatch_core_manager::get_dispatch_core_config(chip_id_t device_id) {
    return this->dispatch_core_config_by_device[device_id];
}

void dispatch_core_manager::add_dispatch_core_to_device(chip_id_t device_id, const CoreCoord& core) {
    // TODO: remove this API, we should read the core descriptor once, should not have backdoors like this to add cores
    auto& dispatch_cores = available_dispatch_cores_by_device.at(device_id);
    if (std::find(dispatch_cores.begin(), dispatch_cores.end(), core) == dispatch_cores.end()) {
        dispatch_cores.push_back(core);
    }
}

std::vector<CoreCoord> dispatch_core_manager::get_all_logical_dispatch_cores(chip_id_t device_id) {
    return tt::get_logical_dispatch_cores(device_id, MAX_NUM_HW_CQS, this->dispatch_core_config_by_device[device_id]);
}

// private methods

dispatch_core_manager::dispatch_core_manager(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) {
    this->reset_dispatch_core_manager(dispatch_core_config, num_hw_cqs);
}

void dispatch_core_manager::reset_dispatch_core_manager(
    const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) {
    this->dispatch_core_assignments.clear();
    this->available_dispatch_cores_by_device.clear();
    this->dispatch_core_config_by_device.clear();
    for (chip_id_t device_id = 0; device_id < tt::Cluster::instance().number_of_devices(); device_id++) {
        std::list<CoreCoord>& logical_dispatch_cores = this->available_dispatch_cores_by_device[device_id];
        for (const CoreCoord& logical_dispatch_core :
             tt::get_logical_dispatch_cores(device_id, MAX_NUM_HW_CQS, dispatch_core_config)) {
            logical_dispatch_cores.push_back(logical_dispatch_core);
        }

        this->dispatch_core_config_by_device[device_id] = dispatch_core_config;
        this->num_hw_cqs = num_hw_cqs;
    }
}

CoreCoord dispatch_core_manager::get_next_available_dispatch_core(chip_id_t device_id) {
    if (this->available_dispatch_cores_by_device.find(device_id) == this->available_dispatch_cores_by_device.end()) {
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
    std::string name, tt_cxy_pair& cxy, chip_id_t device_id, uint16_t channel, uint8_t cq_id, bool force_ethernet) {
    log_debug(
        tt::LogMetal,
        "Allocated {} Core: {}({}) for Device {} Channel {} CQ ID {}",
        name,
        cxy.str(),
        tt::Cluster::instance()
            .get_virtual_coordinate_from_logical_coordinates(
                cxy, force_ethernet ? CoreType::ETH : get_dispatch_core_type(cxy.chip))
            .str(),
        device_id,
        channel,
        cq_id);
}

}  // namespace tt::tt_metal
