// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal {

class Hal;

struct JitDeviceConfig {
    const Hal* hal = nullptr;
    tt::ARCH arch = tt::ARCH::Invalid;

    size_t num_dram_banks = 0;
    size_t num_l1_banks = 0;
    CoreCoord pcie_core{0, 0};

    uint32_t harvesting_mask = 0;
    DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
    DispatchCoreAxis dispatch_core_axis = DispatchCoreAxis::ROW;
    bool coordinate_virtualization_enabled = false;

    uint32_t dispatch_message_addr = 0;
    uint32_t max_cbs = 0;
    uint8_t num_hw_cqs = 0;

    bool routing_fw_enabled = false;

    // Pre-computed in the factory so that JitBuildEnv::init can consume it without
    // calling get_profiler_dram_bank_size_per_risc_bytes(), which has a side-effect
    // of mutating rtoptions (set_profiler_program_support_count). The build module
    // must only observe const RunTimeOptions; any mutation belongs in the factory or
    // in the profiler subsystem itself.
    uint32_t profiler_dram_bank_size_per_risc_bytes = 0;
};

JitDeviceConfig create_jit_device_config(ChipId device_id, uint8_t num_hw_cqs);

std::map<std::string, std::string> initialize_device_kernel_defines(const JitDeviceConfig& config);

uint64_t compute_build_key(const JitDeviceConfig& config, const llrt::RunTimeOptions& rtoptions);

// TODO: Add a factory method to create JitDeviceConfig from a YAML profile

}  // namespace tt::tt_metal
