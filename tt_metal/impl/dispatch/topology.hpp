// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <device.hpp>
#include <stdint.h>
#include <memory>
#include <set>
#include <vector>

#include "core_coord.hpp"
#include "data_types.hpp"
#include "tt-metalium/program.hpp"
#include "tt_metal/impl/dispatch/kernel_config/fd_kernel.hpp"

namespace tt::tt_metal {

class IDevice;
enum DispatchWorkerType : uint32_t;

// NOC ID used by dispatch kernels to communicate with downstream cores. This parameter
// is required when setting up Command Queue objects on host.
constexpr NOC k_dispatch_downstream_noc = NOC::NOC_0;

struct DispatchKernelNode {
    int id;
    chip_id_t device_id;             // Device that this kernel is located on
    chip_id_t servicing_device_id;   // Remote device that this kernel services, used for kernels on MMIO
    uint8_t cq_id;                   // CQ this kernel implements
    DispatchWorkerType kernel_type;  // Type of dispatch kernel this is
    std::vector<int> upstream_ids;   // Upstream dispatch kernels
    std::vector<int> downstream_ids;  // Downstream dispatch kernels
    noc_selection_t noc_selection;    // NOC selection
    int tunnel_index{-1};             // Tunnel index
};

// Create FD kernels for all given device ids. Creates all objects, but need to call create_and_compile_cq_program() use
// a created Device to fill out the settings. First version automatically generates the topology based on devices, num
// cqs, and detected board. Second version uses the topology passed in.
void populate_fd_kernels(const std::vector<IDevice*>& devices, uint32_t num_hw_cqs);
void populate_fd_kernels(const std::set<chip_id_t>& device_ids, uint32_t num_hw_cqs);
void populate_fd_kernels(const std::vector<DispatchKernelNode>& nodes);

// Populate the static arguments for a device.
// Prerequisites: Must call populate_fd_kernels
void populate_cq_static_args(IDevice* device);

// Fill out all settings for FD kernels on the given device, and add them to a Program and return it.
// Prerequisites: Must call populate_cq_static_args
std::unique_ptr<tt::tt_metal::Program> create_and_compile_cq_program(tt::tt_metal::IDevice* device);

// Perform additional configuration (writing to specific L1 addresses, etc.) for FD kernels on this device.
void configure_dispatch_cores(tt::tt_metal::IDevice* device);

// Compile fabric kernels needed to support scaleout systems.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device);

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
void configure_fabric_cores(tt::tt_metal::IDevice* device);

// Return the virtual dispatch cores running on a given device
const std::unordered_set<CoreCoord>& get_virtual_dispatch_cores(chip_id_t dev_id);

// Return the virtual cores used for dispatch routing/tunneling on a given device
const std::unordered_set<CoreCoord>& get_virtual_dispatch_routing_cores(chip_id_t dev_id);

// Return the list of termination targets that were registered for this device
const std::vector<tt::tt_metal::TerminationInfo>& get_registered_termination_cores(chip_id_t dev_id);

}  // namespace tt::tt_metal
