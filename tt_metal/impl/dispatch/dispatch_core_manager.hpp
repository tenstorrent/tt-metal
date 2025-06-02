// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/core_descriptor.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt::tt_metal {

// Dispatch core manager APIs track which cores are assigned to which dispatch functionality

// A command queue is split into an issue queue and completion queue
//  Host enqueues commands and data to be sent to device into the issue queue, and device reads from the issue queue.
//  prefetcher kernels read commands targetting the MMIO or remote device respectively from the issue queue
//  Device writes data into the completion queue for host to read back
//  command_queue_consumer and remote_completion_queue_writer (to be added) kernels write into the completion queue for
//  MMIO or remote device respectively Currently two cores are used to interface with each command queue region, marked
//  as `prefetcher` and `completion_queue_writer` below
// One core dispatches commands to worker cores on the device `dispatcher`
// The `remote_x` cores are used for remote fast dispatch and receive / transmit fast dispatch packets from ethernet
// cores

// std::optional is used to determine whether core has been assigned
// tt_cxy_pair is used over CoreCoord to denote location because remote device command queue interface cores are on the
// associated MMIO device
struct dispatch_core_placement_t {
    std::optional<tt_cxy_pair> prefetcher =
        std::nullopt;  // Pulls commands from the issue queue for a given command queue on a device
    std::optional<tt_cxy_pair> completion_queue_writer =
        std::nullopt;  // Pushes to completion queue for a given command queue on a device
    std::optional<tt_cxy_pair> dispatcher =
        std::nullopt;  // Relays work to worker cores on device that command is targeting. Currently for MMIO devices,
                       // dispatcher == completion_queue_writer
    std::optional<tt_cxy_pair> mux = std::nullopt;       // Mux
    std::optional<tt_cxy_pair> demux = std::nullopt;     // Demux
    std::optional<tt_cxy_pair> tunneler = std::nullopt;  // ethernet tunneler
    std::optional<tt_cxy_pair> prefetcher_d = std::nullopt;
    std::optional<tt_cxy_pair> dispatcher_d = std::nullopt;
    std::optional<tt_cxy_pair> dispatcher_s = std::nullopt;
    std::optional<tt_cxy_pair> mux_d = std::nullopt;       // Mux
    std::optional<tt_cxy_pair> demux_d = std::nullopt;     // Demux
    std::optional<tt_cxy_pair> tunneler_d = std::nullopt;  // ethernet tunneler
    std::unordered_map<int, tt_cxy_pair> fabric_mux;       // Fabric Mux indexed by tunnel / link index
};

class dispatch_core_manager {
public:
    dispatch_core_manager& operator=(const dispatch_core_manager&) = delete;
    dispatch_core_manager& operator=(dispatch_core_manager&& other) noexcept = delete;
    dispatch_core_manager(const dispatch_core_manager&) = delete;
    dispatch_core_manager(dispatch_core_manager&& other) noexcept = delete;

    /// @brief dispatch_core_manager constructor initializes a list of cores per device that are designated for any
    /// dispatch functionality
    ///         This list contains dispatch cores that have not been assigned to a particular dispatch function
    /// @param num_hw_cqs is used to get the correct collection of dispatch cores for a particular device
    /// @param dispatch_core_config specfies the core type that is designated for dispatch functionality
    dispatch_core_manager(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs);

    // TODO: this should probably be in command_queue_interface.hpp, but it's here for now due to circular dependency
    static constexpr uint8_t MAX_NUM_HW_CQS = 2;

    /// @brief Gets the location of the kernel desginated to read from the issue queue region from a particular command
    /// queue
    ///         Each command queue has an issue queue where host enqueues commands. This core relays to the dispatcher
    ///         core to interpret and launch For remote devices, this core is located on the associated MMIO device
    ///         since it can access sysmem (location of command queue)
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the issue queue interface
    const tt_cxy_pair& prefetcher_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    bool is_prefetcher_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel desginated to interface with prefetcher kernel running on mmio device.
    ///         Prefetcher kernel on mmio device relays commands to prefetcher_d running on remote device.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the issue queue interface
    const tt_cxy_pair& prefetcher_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    bool is_prefetcher_d_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel desginated for multiplexing issue queue traffic to tunneler.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the mux core
    const tt_cxy_pair& mux_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    bool is_mux_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel desginated for multiplexing traffic back towards mmio chip.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the mux_d core

    const tt_cxy_pair& mux_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel desginated for demultiplexing traffic to completion queues.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the mux core
    const tt_cxy_pair& demux_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    bool is_demux_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel desginated for demultiplexing traffic on remote chip.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the demux_d core
    const tt_cxy_pair& demux_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel desginated for tunneling over ethernet.
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the ethernet tunnel core
    const tt_cxy_pair& tunneler_core(
        chip_id_t upstream_device_id, chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    const tt_cxy_pair& us_tunneler_core_local(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel desginated to write to the completion queue region for a particular
    /// command queue
    ///         Each command queue has one completion queue
    ///         For MMIO devices this core is the same as the issue queue reader core core because one kernel is
    ///         responisble for interpreting + relaying commands and writing to completion queue For remote devices,
    ///         this core is located on the associated MMIO device since it can access sysmem (location of command
    ///         queue)
    /// @param device_id ID of the device that a fast dispatch command targets
    /// @param channel assigned to the command queue
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the completion queue interface
    const tt_cxy_pair& completion_queue_writer_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    bool is_completion_queue_writer_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel designated to relay fast dispatch commands to worker cores from a
    /// particular command queue
    /// @param device_id ID of the device that should be running the command
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the dispatcher core
    const tt_cxy_pair& dispatcher_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    bool is_dispatcher_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    bool is_dispatcher_s_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    bool is_dispatcher_d_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel designated to relay fast dispatch commands to worker cores from a
    /// particular command queue
    /// @param device_id ID of the device that should be running the command
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the dispatcher_d core
    const tt_cxy_pair& dispatcher_d_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    const tt_cxy_pair& dispatcher_s_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id);

    /// @brief Gets the location of the kernel designated to relay fast dispatch commands to worker cores from a
    /// particular command queue
    /// @param device_id ID of the device that this core is on
    /// @param channel assigned to the command queue where commands are enqueued
    /// @param cq_id ID of the command queue within the channel
    /// @param tunnel ID of the tunnel which this fabric mux will send data through
    /// @return tt_cxy_pair logical location (chip + core coordinate) of the fabric mux core
    const tt_cxy_pair& fabric_mux_core(chip_id_t device_id, uint16_t channel, uint8_t cq_id, int tunnel);

    bool is_fabric_mux_core_allocated(chip_id_t device_id, uint16_t channel, uint8_t cq_id, int tunnel);

    CoreType get_dispatch_core_type();

    DispatchCoreConfig get_dispatch_core_config();

    uint8_t get_num_hw_cqs() { return this->num_hw_cqs; }

    // TODO: remove this API, we should read the core descriptor once, should not have backdoors like this to add cores
    void add_dispatch_core_to_device(chip_id_t device_id, const CoreCoord& core);

    std::vector<CoreCoord> get_all_logical_dispatch_cores(chip_id_t device_id);

private:
    /// @brief reset_dispatch_core_manager initializes vector of cores per device for dispatch kernels
    /// @param dispatch_core_config specfies the core type for dispatch kernels
    void reset_dispatch_core_manager(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs);

    /// @brief getting any available dispatch core for a device
    /// @param device_id
    /// @return
    CoreCoord get_next_available_dispatch_core(chip_id_t device_id);

    void log_dispatch_assignment(
        std::string name,
        tt_cxy_pair& cxy,
        chip_id_t device_id,
        uint16_t channel,
        uint8_t cq_id,
        bool force_ethernet = false);

    // {device ID : {channel (hugepage) : {cq_id : dispatch assignment}}}
    // Each device has an assigned hugepage at a specific channel that holds (up to 2) hardware command queues
    // (represented by cq_id)
    std::unordered_map<chip_id_t, std::unordered_map<uint16_t, std::unordered_map<uint8_t, dispatch_core_placement_t>>>
        dispatch_core_assignments;
    std::unordered_map<chip_id_t, std::list<CoreCoord>> available_dispatch_cores_by_device;
    DispatchCoreConfig dispatch_core_config_;
    uint8_t num_hw_cqs;
    static dispatch_core_manager* _inst;
};

}  // namespace tt::tt_metal
