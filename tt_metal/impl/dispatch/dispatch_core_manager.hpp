// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_descriptor.hpp"
#include "tt_metal/common/core_coord.h"
#include <list>

namespace tt::tt_metal {

// Dispatch core manager APIs track which cores are assigned to which dispatch functionality

// A command queue is split into an issue queue and completion queue
//  Host enqueues commands and data to be sent to device into the issue queue, and device reads from the issue queue.
//  prefetcher kernels read commands targetting the MMIO or remote device respectively from the issue queue
//  Device writes data into the completion queue for host to read back
//  command_queue_consumer and remote_completion_queue_writer (to be added) kernels write into the completion queue for MMIO or remote device respectively
//  Currently two cores are used to interface with each command queue region, marked as `prefetcher` and `completion_queue_writer` below
// One core dispatches commands to worker cores on the device `dispatcher`
// The `remote_x` cores are used for remote fast dispatch and receive / transmit fast dispatch packets from ethernet cores

enum DispatchWorkerType : uint32_t {
    PREFETCH = 0,
    PREFETCH_D = 1,
    DISPATCH = 2,
    DISPATCH_D = 3,
    MUX = 4,
    MUX_D = 5,
    DEMUX = 6,
    DEMUX_D = 7,
    US_TUNNELER_LOCAL = 8,
    US_TUNNELER_REMOTE = 9,
    DS_TUNNELER_LOCAL = 10,
    DS_TUNNELER_REMOTE = 11,
    COUNT = 12
};

enum DispatchCoreType: uint32_t {
    WORKER = 0,
    ETH = 1,
};


struct dispatch_worker_build_settings_t{
    std::string kernel_file;
    std::vector<uint32_t> compile_args;
    std::vector<tt_cxy_pair> upstream_cores;
    std::vector<tt_cxy_pair> downstream_cores;
    tt_cxy_pair worker_physical_core;
    tt_cxy_pair eth_partner_physical_core;
    CoreType dispatch_core_type;
    uint32_t command_queue_start_addr;
    uint32_t issue_queue_start_addr;
    uint32_t issue_queue_size;
    uint32_t completion_queue_start_addr;
    uint32_t completion_queue_size;
    std::vector<uint32_t> semaphores;
    uint32_t producer_semaphore_id;
    uint32_t consumer_semaphore_id;
    uint32_t cb_start_address;
    uint32_t cb_size_bytes;
    uint32_t cb_log_page_size;
    uint32_t cb_pages;
    uint32_t tunnel_stop;
    uint32_t num_compute_cores;
    uint32_t vc_count;
};

// std::optional is used to determine whether core has been assigned
// tt_cxy_pair is used over CoreCoord to denote location because remote device command queue interface cores are on the associated MMIO device
struct dispatch_core_placement_t {
    std::optional<tt_cxy_pair> prefetcher = std::nullopt;  // Pulls commands from the issue queue for a given command queue on a device
    std::optional<tt_cxy_pair> completion_queue_writer = std::nullopt; // Pushes to completion queue for a given command queue on a device
    std::optional<tt_cxy_pair> dispatcher = std::nullopt; // Relays work to worker cores on device that command is targeting. Currently for MMIO devices, dispatcher == completion_queue_writer
    std::optional<tt_cxy_pair> mux = std::nullopt; // Mux
    std::optional<tt_cxy_pair> demux = std::nullopt; // Demux
    std::optional<tt_cxy_pair> tunneler = std::nullopt; // ethernet tunneler
    std::optional<tt_cxy_pair> prefetcher_d = std::nullopt;
    std::optional<tt_cxy_pair> dispatcher_d = std::nullopt;
    std::optional<tt_cxy_pair> mux_d = std::nullopt; // Mux
    std::optional<tt_cxy_pair> demux_d = std::nullopt; // Demux
    std::optional<tt_cxy_pair> tunneler_d = std::nullopt; // ethernet tunneler
};

class dispatch_core_manager {
   public:
    dispatch_core_manager &operator=(const dispatch_core_manager &) = delete;
    dispatch_core_manager &operator=(dispatch_core_manager &&other) noexcept = delete;
    dispatch_core_manager(const dispatch_core_manager &) = delete;
    dispatch_core_manager(dispatch_core_manager &&other) noexcept = delete;


    //TODO: this should probably be in command_queue_interface.hpp, but it's here for now due to circular dependency
    static constexpr uint8_t MAX_NUM_HW_CQS = 2;
    static void initialize(DispatchCoreType dispatch_core_type) noexcept {
        log_debug(tt::LogMetal, "DevicePool initialize");
        const std::unordered_map<DispatchCoreType, CoreType> dispatch_core_type_map = {
            {DispatchCoreType::WORKER, CoreType::WORKER},
            {DispatchCoreType::ETH, CoreType::ETH}
        };
        auto internal_core_type = dispatch_core_type_map.at(dispatch_core_type);
        if (_inst == nullptr) {
            static dispatch_core_manager dispatch_core_manager(internal_core_type);
            _inst = &dispatch_core_manager;
        } else if (_inst->dispatch_core_type_by_device[0] != internal_core_type) {
            _inst->reset_dispatch_core_manager(internal_core_type);
        }
    }

    static dispatch_core_manager &instance() {
        TT_ASSERT(_inst != nullptr, "Trying to get dispatch_core_manager without initializing it");
        return *_inst;
    }

    /// @brief Gets the location of the kernel desginated to read from the issue queue region from a particular command queue
    ///         Each command queue has an issue queue where host enqueues commands. This core relays to the dispatcher core to interpret and launch
    ///         For remote devices, this core is located on the associated MMIO device since it can access sysmem (location of command queue)
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the issue queue interface
    const tt_cxy_pair &prefetcher_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.prefetcher.has_value()) {
            return assignment.prefetcher.value();
        }
        // Issue queue interface is on the MMIO device
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord issue_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.prefetcher = tt_cxy_pair(mmio_device_id, issue_queue_coord.x, issue_queue_coord.y);
        log_debug(tt::LogMetal, "Allocated Prefetch Core: {} for Device {}", assignment.prefetcher.value().str(), device_id);
        return assignment.prefetcher.value();
    }

    bool is_prefetcher_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.prefetcher.has_value()) {
            return true;
        }
        return false;
    }

    /// @brief Gets the location of the kernel desginated to interface with prefetcher kernel running on mmio device.
    ///         Prefetcher kernel on mmio device relays commands to prefetcher_d running on remote device.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the issue queue interface
    const tt_cxy_pair &prefetcher_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.prefetcher_d.has_value()) {
            return assignment.prefetcher_d.value();
        }
        CoreCoord prefetch_d_coord = this->get_next_available_dispatch_core(device_id);
        assignment.prefetcher_d = tt_cxy_pair(device_id, prefetch_d_coord.x, prefetch_d_coord.y);
        log_debug(tt::LogMetal, "Allocated Prefetch D Core: {} for Device {}", assignment.prefetcher_d.value().str(), device_id);
        return assignment.prefetcher_d.value();
    }

    bool is_prefetcher_d_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.prefetcher_d.has_value()) {
            return true;
        }
        return false;
    }

    /// @brief Gets the location of the kernel desginated for multiplexing issue queue traffic to tunneler.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the mux core
    const tt_cxy_pair &mux_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.mux.has_value()) {
            return assignment.mux.value();
        }
        // Mux interface is on the MMIO device
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord mux_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.mux = tt_cxy_pair(mmio_device_id, mux_coord.x, mux_coord.y);
        log_debug(tt::LogMetal, "Allocated Mux Core: {} for Device {}", assignment.mux.value().str(), device_id);
        return assignment.mux.value();
    }

    bool is_mux_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.mux.has_value()) {
            return true;
        }
        return false;
    }

    /// @brief Gets the location of the kernel desginated for multiplexing traffic back towards mmio chip.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the mux_d core

    const tt_cxy_pair &mux_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.mux_d.has_value()) {
            return assignment.mux_d.value();
        }
        // mux_d is on remote device
        CoreCoord mux_d_coord = this->get_next_available_dispatch_core(device_id);
        assignment.mux_d = tt_cxy_pair(device_id, mux_d_coord.x, mux_d_coord.y);
        log_debug(tt::LogMetal, "Allocated Mux D Core: {} for Device {}", assignment.mux_d.value().str(), device_id);
        return assignment.mux_d.value();
    }

    /// @brief Gets the location of the kernel desginated for demultiplexing traffic to completion queues.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the mux core
    const tt_cxy_pair &demux_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.demux.has_value()) {
            return assignment.demux.value();
        }
        // demux interface is on the MMIO device
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord demux_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.demux = tt_cxy_pair(mmio_device_id, demux_coord.x, demux_coord.y);
        log_debug(tt::LogMetal, "Allocated Demux Core: {} for Device {}", assignment.demux.value().str(), device_id);
        return assignment.demux.value();
    }

    bool is_demux_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.demux.has_value()) {
            return true;
        }
        return false;
    }

    /// @brief Gets the location of the kernel desginated for demultiplexing traffic on remote chip.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the demux_d core
    const tt_cxy_pair &demux_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.demux_d.has_value()) {
            return assignment.demux_d.value();
        }
        // demux_d is on remote device
        CoreCoord demux_d_coord = this->get_next_available_dispatch_core(device_id);
        assignment.demux_d = tt_cxy_pair(device_id, demux_d_coord.x, demux_d_coord.y);
        log_debug(tt::LogMetal, "Allocated Demux D Core: {} for Device {}", assignment.demux_d.value().str(), device_id);
        return assignment.demux_d.value();
    }

    /// @brief Gets the location of the kernel desginated for tunneling over ethernet.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the ethernet tunnel core
    const tt_cxy_pair &tunneler_core(chip_id_t upstream_device_id, chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.tunneler.has_value()) {
            return assignment.tunneler.value();
        }

        auto[us_core, ds_core] = tt::Cluster::instance().get_eth_tunnel_core(upstream_device_id, device_id, EthRouterMode::BI_DIR_TUNNELING);

        assignment.tunneler = us_core;
        assignment.tunneler_d = ds_core;

        log_debug(tt::LogMetal, "Allocated Tunneler Core: {} for Device {}", us_core.str(), device_id);
        return assignment.tunneler.value();
    }

    const tt_cxy_pair &us_tunneler_core_local(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.tunneler_d.has_value()) {
            return assignment.tunneler_d.value();
        }
        TT_ASSERT(false, "Device {} has no allocation for Local Upstream Tunneler Core.", device_id);
        assignment.tunneler_d = tt_cxy_pair(0, 0, 0);
        return assignment.tunneler_d.value();
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
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.completion_queue_writer.has_value()) {
            return assignment.completion_queue_writer.value();
        }
        // Completion queue interface is on the MMIO device
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord completion_queue_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.completion_queue_writer = tt_cxy_pair(mmio_device_id, completion_queue_coord.x, completion_queue_coord.y);
        TT_ASSERT(not assignment.dispatcher.has_value(), "Command dispatcher core {} must match completion queue interface core for MMIO device {}", assignment.dispatcher.value().str(), device_id);
        assignment.dispatcher = assignment.completion_queue_writer;
        log_debug(tt::LogMetal, "Allocated Completion Queue Writer Core: {} for Device {}", assignment.completion_queue_writer.value().str(), device_id);
        return assignment.completion_queue_writer.value();
    }

    bool is_completion_queue_writer_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.completion_queue_writer.has_value()) {
            return true;
        }
        return false;
    }

    /// @brief Gets the location of the kernel designated to relay fast dispatch commands to worker cores from a particular command queue
    /// @param device_id ID of the device that should be running the command
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the dispatcher core
    const tt_cxy_pair &dispatcher_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.dispatcher.has_value()) {
            return assignment.dispatcher.value();
        }
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        CoreCoord dispatcher_coord = this->get_next_available_dispatch_core(mmio_device_id);
        assignment.dispatcher = tt_cxy_pair(mmio_device_id, dispatcher_coord.x, dispatcher_coord.y);
        TT_ASSERT(not assignment.completion_queue_writer.has_value(), "Command dispatcher core must match completion queue interface core for MMIO device {}", device_id);
        assignment.completion_queue_writer = assignment.dispatcher;
        log_debug(tt::LogMetal, "Allocated Dispatcher Core: {} for Device {}", assignment.dispatcher.value().str(), device_id);
        return assignment.dispatcher.value();
    }

    bool is_dispatcher_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.dispatcher.has_value()) {
            return true;
        }
        return false;
    }

    /// @brief Gets the location of the kernel designated to relay fast dispatch commands to worker cores from a particular command queue
    /// @param device_id ID of the device that should be running the command
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the dispatcher_d core
    const tt_cxy_pair &dispatcher_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id) {
        dispatch_core_placement_t &assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
        if (assignment.dispatcher_d.has_value()) {
            return assignment.dispatcher_d.value();
        }
        CoreCoord dispatcher_d_coord = this->get_next_available_dispatch_core(device_id);
        assignment.dispatcher_d = tt_cxy_pair(device_id, dispatcher_d_coord.x, dispatcher_d_coord.y);
        log_debug(tt::LogMetal, "Allocated Dispatcher D Core: {} for Device {}", assignment.dispatcher_d.value().str(), device_id);
        return assignment.dispatcher_d.value();
    }

    CoreType get_dispatch_core_type(chip_id_t device_id) {
        return this->dispatch_core_type_by_device[device_id];
    }

    void add_dispatch_core_to_device(chip_id_t device_id, const CoreCoord& core) {
        // TODO: remove this API, we should read the core descriptor once, should not have backdoors like this to add cores
        auto& dispatch_cores = available_dispatch_cores_by_device.at(device_id);
        if (std::find(dispatch_cores.begin(), dispatch_cores.end(), core) == dispatch_cores.end()) {
            dispatch_cores.push_back(core);
        }
    }

   private:
    /// @brief dispatch_core_manager constructor initializes a list of cores per device that are designated for any dispatch functionality
    ///         This list contains dispatch cores that have not been assigned to a particular dispatch function
    /// @param num_hw_cqs is used to get the correct collection of dispatch cores for a particular device
    /// @param dispatch_core_type specfies the core type that is designated for dispatch functionality
    dispatch_core_manager(CoreType dispatch_core_type) {
        this->reset_dispatch_core_manager(dispatch_core_type);
    }


    /// @brief reset_dispatch_core_manager initializes vector of cores per device for dispatch kernels
    /// @param dispatch_core_type specfies the core type for dispatch kernels
    void reset_dispatch_core_manager(CoreType dispatch_core_type) {
        this->dispatch_core_assignments.clear();
        this->available_dispatch_cores_by_device.clear();
        this->dispatch_core_type_by_device.clear();
        for (chip_id_t device_id = 0; device_id < tt::Cluster::instance().number_of_devices(); device_id++) {
            std::list<CoreCoord> &logical_dispatch_cores = this->available_dispatch_cores_by_device[device_id];
            for (const CoreCoord &logical_dispatch_core :
                 tt::get_logical_dispatch_cores(device_id, MAX_NUM_HW_CQS, dispatch_core_type)) {
                logical_dispatch_cores.push_back(logical_dispatch_core);
            }

            this->dispatch_core_type_by_device[device_id] = dispatch_core_type;
        }
    }

    /// @brief getting any available dispatch core for a device
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
    std::unordered_map<chip_id_t, std::unordered_map<uint16_t, std::unordered_map<uint8_t, dispatch_core_placement_t>>> dispatch_core_assignments;
    std::unordered_map<chip_id_t, std::list<CoreCoord>> available_dispatch_cores_by_device;
    std::unordered_map<chip_id_t, CoreType> dispatch_core_type_by_device;  //TODO: dispatch_core_type_by_device should probably be for all devices, not per device
    static dispatch_core_manager *_inst;

};


}   // namespace tt::tt_metal
