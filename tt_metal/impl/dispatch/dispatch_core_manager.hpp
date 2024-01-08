// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_descriptor.hpp"

namespace tt::tt_metal {

// Dispatch core manager APIs track which cores are assigned to which dispatch functionality

// A command queue is split into an issue queue and completion queue and needs two cores to interface with each queue.
// One core dispatches commands to worker cores on the device

// std::optional is used to determine whether core has been assigned
// tt_cxy_pair is used over CoreCoord to denote location because remote device command queue interface cores are on the associated MMIO device
struct dispatch_core_types_t {
    std::optional<tt_cxy_pair> issue_queue_interface = std::nullopt;  // Pulls commands from the issue queue for a given command queue on a device
    std::optional<tt_cxy_pair> completion_queue_interface = std::nullopt; // Pushes to completion queue for a given command queue on a device
    std::optional<tt_cxy_pair> command_dispatcher = std::nullopt; // Relays work to worker cores on device that command is targeting. Currently for MMIO devices, command_dispatcher == completion_queue_interface
    // TODO (abhullar): consider renaming these when supporting GH #3953 and #3954
    std::optional<tt_cxy_pair> remote_processor = std::nullopt;    // Receives fast dispatch commands from ethernet router and relays to command_dispatcher
    std::optional<tt_cxy_pair> remote_signaller = std::nullopt;    // Transmits data from command_dispatcher to ethernet router to send fast dispatch results off chip
};

inline CoreCoord get_physical_dispatch_coord(const tt_cxy_pair &logical_location) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(logical_location.chip);
    CoreCoord worker_core({
            .x = static_cast<size_t>(soc_desc.worker_log_to_physical_routing_x.at(logical_location.x)),
            .y = static_cast<size_t>(soc_desc.worker_log_to_physical_routing_y.at(logical_location.y)),
    });
    return worker_core;
}

class dispatch_core_manager {
   public:
    dispatch_core_manager &operator=(const dispatch_core_manager &) = delete;
    dispatch_core_manager &operator=(dispatch_core_manager &&other) noexcept = delete;
    dispatch_core_manager(const dispatch_core_manager &) = delete;
    dispatch_core_manager(dispatch_core_manager &&other) noexcept = delete;

    static dispatch_core_manager &get() {
        static dispatch_core_manager inst;
        return inst;
    }

    const tt_cxy_pair &issue_queue_interface_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_types_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.issue_queue_interface.has_value()) {
            return assignment.issue_queue_interface.value();
        }
        // Issue queue interface is on the MMIO device
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord issue_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.issue_queue_interface = tt_cxy_pair(mmio_device_id, issue_queue_coord.x, issue_queue_coord.y);
        std::cout << "For device " << device_id << " channel " << channel << " CQ " << std::to_string(cq_id) << " issue queue interface location: " << assignment.issue_queue_interface.value().str() << std::endl;
        return assignment.issue_queue_interface.value();
    }

    const tt_cxy_pair &completion_queue_interface_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_types_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.completion_queue_interface.has_value()) {
            return assignment.completion_queue_interface.value();
        }
        // Completion queue interface is on the MMIO device
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord completion_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.completion_queue_interface = tt_cxy_pair(mmio_device_id, completion_queue_coord.x, completion_queue_coord.y);
        std::cout << "For device " << device_id << " channel " << channel << " CQ " << std::to_string(cq_id) << " completion queue interface location: " << assignment.completion_queue_interface.value().str() << std::endl;
        if (mmio_device_id == device_id) {
            // For MMIO devices completion queue core is same as command dispatcher core
            TT_ASSERT(not assignment.command_dispatcher.has_value(), "Command dispatcher core {} must match completion queue interface core for MMIO device {}", assignment.command_dispatcher.value().str(), device_id);
            assignment.command_dispatcher = assignment.completion_queue_interface;
        }
        return assignment.completion_queue_interface.value();
    }

    const tt_cxy_pair &command_dispatcher_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_types_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.command_dispatcher.has_value()) {
            return assignment.command_dispatcher.value();
        }
        CoreCoord command_dispatcher_coord = this->get_next_available_dispatch_core(device_id);
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        assignment.command_dispatcher = tt_cxy_pair(device_id, command_dispatcher_coord.x, command_dispatcher_coord.y);
        std::cout << "For device " << device_id << " channel " << channel << " CQ " << std::to_string(cq_id) << " command dispatcher location: " << assignment.command_dispatcher.value().str() << std::endl;
        if (mmio_device_id == device_id) {
            // For MMIO devices completion queue core is same as command dispatcher core
            TT_ASSERT(not assignment.completion_queue_interface.has_value(), "Command dispatcher core must match completion queue interface core for MMIO device {}", device_id);
            assignment.completion_queue_interface = assignment.command_dispatcher;
        }
        return assignment.command_dispatcher.value();
    }

    const std::optional<tt_cxy_pair> &remote_processor_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        TT_THROW("Do not currently support programming remote processor dispatch core. See https://github.com/tenstorrent-metal/tt-metal/issues/3953");
        return this->dispatch_core_assignments[device_id][channel][cq_id].remote_processor;
    }

    const std::optional<tt_cxy_pair> &remote_signaller_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        TT_THROW("Do not currently support programming remote signaller dispatch core. See https://github.com/tenstorrent-metal/tt-metal/issues/3954");
        return this->dispatch_core_assignments[device_id][channel][cq_id].remote_signaller;
    }

   private:
    dispatch_core_manager() {
        for (chip_id_t device_id = 0; device_id < tt::Cluster::instance().number_of_devices(); device_id++) {
            std::list<CoreCoord> &logical_dispatch_cores = this->available_dispatch_cores_by_device[device_id];
            for (const CoreCoord &logical_dispatch_core : tt::get_logical_dispatch_cores(device_id, 1)) {   // TODO: DO NOT HARDCODE NUMBER OF CQS HERE
                logical_dispatch_cores.push_back(logical_dispatch_core);
            }
        }
    }

    CoreCoord get_next_available_dispatch_core(chip_id_t device_id) {
        if (this->available_dispatch_cores_by_device.find(device_id) == this->available_dispatch_cores_by_device.end()) {
            TT_THROW("Invalid device ID to assign dispatch cores {}", device_id);
        }
        if (this->available_dispatch_cores_by_device.at(device_id).empty()) {
            TT_THROW("No more available dispatch cores on device {} to assign. Expand dispatch cores specified in core descriptor YAML", device_id);
        }
        CoreCoord avail_dispatch_core = this->available_dispatch_cores_by_device.at(device_id).front();
        this->available_dispatch_cores_by_device.at(device_id).pop_front();
        return avail_dispatch_core;
    }

    // {device ID : {channel (hugepage) : {cq_id : dispatch assignment}}}
    // Each device has an assigned hugepage at a specific channel that holds (up to 2) hardware command queues (represented by cq_id)
    std::unordered_map<chip_id_t, std::unordered_map<uint16_t, std::unordered_map<uint8_t, dispatch_core_types_t>>> dispatch_core_assignments;
    std::unordered_map<chip_id_t, std::list<CoreCoord>> available_dispatch_cores_by_device;
};


}   // namespace tt::tt_metal
