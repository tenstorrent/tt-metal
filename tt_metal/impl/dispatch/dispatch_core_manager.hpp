// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_descriptor.hpp"

namespace tt::tt_metal {

// Dispatch core manager APIs track which cores are assigned to which dispatch functionality

// A command queue is split into an issue queue and completion queue
//  Host enqueues commands and data to be sent to device into the issue queue, and device reads from the issue queue.
//  command_queue_producer and remote_issue_queue_reader kernels read commands targetting the MMIO or remote device respectively from the issue queue
//  Device writes data into the completion queue for host to read back
//  command_queue_consumer and remote_completion_queue_writer (to be added) kernels write into the completion queue for MMIO or remote device respectively
//  Currently two cores are used to interface with each command queue region, marked as `issue_queue_reader` and `completion_queue_writer` below
// One core dispatches commands to worker cores on the device `command_dispatcher`
// The `remote_x` cores are used for remote fast dispatch and receive / transmit fast dispatch packets from ethernet cores

// std::optional is used to determine whether core has been assigned
// tt_cxy_pair is used over CoreCoord to denote location because remote device command queue interface cores are on the associated MMIO device
struct dispatch_core_types_t {
    std::optional<tt_cxy_pair> issue_queue_reader = std::nullopt;  // Pulls commands from the issue queue for a given command queue on a device
    std::optional<tt_cxy_pair> completion_queue_writer = std::nullopt; // Pushes to completion queue for a given command queue on a device
    std::optional<tt_cxy_pair> command_dispatcher = std::nullopt; // Relays work to worker cores on device that command is targeting. Currently for MMIO devices, command_dispatcher == completion_queue_writer
    std::optional<tt_cxy_pair> remote_push_and_pull = std::nullopt;    // Receives fast dispatch commands from DST ethernet router, relays programs to command_dispatcher, and returns data through SRC router
};

class dispatch_core_manager {
   public:
    dispatch_core_manager &operator=(const dispatch_core_manager &) = delete;
    dispatch_core_manager &operator=(dispatch_core_manager &&other) noexcept = delete;
    dispatch_core_manager(const dispatch_core_manager &) = delete;
    dispatch_core_manager(dispatch_core_manager &&other) noexcept = delete;

    // Ugly to accept num HW CQs here but it is needed to pull the correct number of initially available dispatch cores for assignment
    static dispatch_core_manager &get(uint8_t num_hw_cqs) {
        static dispatch_core_manager inst = dispatch_core_manager(num_hw_cqs);
        return inst;
    }

    /// @brief Gets the location of the kernel desginated to read from the issue queue region from a particular command queue
    ///         Each command queue has an issue queue where host enqueues commands. This core relays to the dispatcher core to interpret and launch
    ///         For remote devices, this core is located on the associated MMIO device since it can access sysmem (location of command queue)
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the issue queue interface
    const tt_cxy_pair &issue_queue_reader_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_types_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.issue_queue_reader.has_value()) {
            return assignment.issue_queue_reader.value();
        }
        // Issue queue interface is on the MMIO device
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord issue_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.issue_queue_reader = tt_cxy_pair(mmio_device_id, issue_queue_coord.x, issue_queue_coord.y);

        if (tt::Cluster::instance().arch() == tt::ARCH::WORMHOLE_B0) {
            if (mmio_device_id == device_id) {
                // For MMIO devices completion queue core is same as command dispatcher core
                TT_ASSERT(not assignment.completion_queue_writer.has_value(), "Completion queue writer core must match issue queue reader core for MMIO device {}", device_id);
                assignment.completion_queue_writer = assignment.issue_queue_reader;
            }
        }
        return assignment.issue_queue_reader.value();
    }

    /// @brief Gets the location of the kernel desginated to write to the completion queue region for a particular command queue
    ///         Each command queue has one completion queue
    ///         For MMIO devices this core is the same as the issue queue reader core core because one kernel is responisble for interpreting + relaying commands and writing to completion queue
    ///         For remote devices, this core is located on the associated MMIO device since it can access sysmem (location of command queue)
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the completion queue interface
    const tt_cxy_pair &completion_queue_writer_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_types_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.completion_queue_writer.has_value()) {
            return assignment.completion_queue_writer.value();
        }
        // Completion queue interface is on the MMIO device
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord completion_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.completion_queue_writer = tt_cxy_pair(mmio_device_id, completion_queue_coord.x, completion_queue_coord.y);
        if (mmio_device_id == device_id) {
            if (tt::Cluster::instance().arch() == tt::ARCH::WORMHOLE_B0) {
                // For MMIO devices with FD v1.3 completion queue core is same as issue queue core
                TT_ASSERT(not assignment.issue_queue_reader.has_value(), "Issue queue reader core {} must match completion queue interface core for MMIO device {}", assignment.issue_queue_reader.value().str(), device_id);
                assignment.issue_queue_reader = assignment.completion_queue_writer;
            } else if (tt::Cluster::instance().arch() == tt::ARCH::GRAYSKULL) {
                TT_ASSERT(not assignment.command_dispatcher.has_value(), "Command dispatcher core {} must match completion queue interface core for MMIO device {}", assignment.command_dispatcher.value().str(), device_id);
                assignment.command_dispatcher = assignment.completion_queue_writer;
            }
        }
        return assignment.completion_queue_writer.value();
    }

    /// @brief Gets the location of the kernel designated to relay fast dispatch commands to worker cores from a particular command queue
    /// @param device_id ID of the device that should be running the command
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the dispatcher core
    const tt_cxy_pair &command_dispatcher_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_types_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.command_dispatcher.has_value()) {
            return assignment.command_dispatcher.value();
        }
        CoreCoord command_dispatcher_coord = this->get_next_available_dispatch_core(device_id);
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        assignment.command_dispatcher = tt_cxy_pair(device_id, command_dispatcher_coord.x, command_dispatcher_coord.y);
        if (tt::Cluster::instance().arch() == tt::ARCH::GRAYSKULL) {
            TT_ASSERT(not assignment.completion_queue_writer.has_value(), "Command dispatcher core must match completion queue interface core for MMIO device {}", device_id);
            assignment.completion_queue_writer = assignment.command_dispatcher;
        }
        return assignment.command_dispatcher.value();
    }

    /// @brief Gets the location of the kernel that receives fast dispatch packets from an ethernet router core and transfers programs to the `command_dispatcher_core`
    ///         This core is also responsible for relaying data (completion events and chip-local buffers) from remote chip back to MMIO chip
    ///         This core is only programmed when targeting fast dispatch on the remote chip and its location is on the same chip that runs the fast dispatch commands
    /// @param device_id ID of the remote device that will run fast dispatch commands
    /// @param channel assigned to the command queue where commands targeting the device are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the remote processor core
    const tt_cxy_pair &remote_push_and_pull_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        TT_ASSERT(tt::Cluster::instance().get_associated_mmio_device(device_id) != device_id, "Remote processor core should only be used on remote devices running fast dispatch, not for MMIO device {}", device_id);
        dispatch_core_types_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.remote_push_and_pull.has_value()) {
            return assignment.remote_push_and_pull.value();
        }
        CoreCoord remote_push_and_pull_coord = this->get_next_available_dispatch_core(device_id);
        assignment.remote_push_and_pull = tt_cxy_pair(device_id, remote_push_and_pull_coord.x, remote_push_and_pull_coord.y);
        return assignment.remote_push_and_pull.value();
    }

    CoreType get_dispatch_core_type(chip_id_t device_id) {
        return this->dispatch_core_type_by_device[device_id];
    }

   private:
    /// @brief dispatch_core_manager constructor initializes a list of cores per device that are designated for any dispatch functionality
    ///         This list contains dispatch cores that have not been assigned to a particular dispatch function
    /// @param num_hw_cqs is used to get the correct collection of dispatch cores for a particular device
    dispatch_core_manager(uint8_t num_hw_cqs) {
        for (chip_id_t device_id = 0; device_id < tt::Cluster::instance().number_of_devices(); device_id++) {
            std::list<CoreCoord> &logical_dispatch_cores = this->available_dispatch_cores_by_device[device_id];
            for (const CoreCoord &logical_dispatch_core : tt::get_logical_dispatch_cores(device_id, num_hw_cqs)) {
                logical_dispatch_cores.push_back(logical_dispatch_core);
            }
            this->dispatch_core_type_by_device[device_id] = tt::get_dispatch_core_type(device_id, num_hw_cqs);
        }
    }

    /// @brief getting any
    /// @param device_id
    /// @return
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
    std::unordered_map<chip_id_t, CoreType> dispatch_core_type_by_device;
};


}   // namespace tt::tt_metal
