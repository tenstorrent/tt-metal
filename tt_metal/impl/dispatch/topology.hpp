// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <device.hpp>
#include <memory>
#include "dispatch/kernel_config/fd_kernel.hpp"

namespace tt::tt_metal::dispatch {

using FDTopologyGraph = std::vector<std::shared_ptr<FDKernel>>;

// Create FD kernels for all given device ids. Creates all objects, but need to call create_and_compile_cq_program() use
// a created Device to fill out the settings.
FDTopologyGraph populate_fd_kernels(const std::set<chip_id_t>& device_ids, uint32_t num_hw_cqs);

// Fill out all settings for FD kernels on the given device, and add them to a Program and return it.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_cq_program(
    tt::tt_metal::IDevice* device, FDTopologyGraph& graph);

// Performa additional configuration (writing to specific L1 addresses, etc.) for FD kernels on this device.
void configure_dispatch_cores(tt::tt_metal::IDevice* device, FDTopologyGraph& graph);

}  // namespace tt::tt_metal::dispatch
