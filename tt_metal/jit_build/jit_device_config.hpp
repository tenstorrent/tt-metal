// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {

class Hal;

// Device-specific configuration snapshot consumed by the JIT build system.
//
// Captures the hardware topology, dispatch layout, and memory parameters that
// influence kernel compilation. Instances are intended to be created once per
// device via `create_jit_device_config` and then treated as read-only; the
// build pipeline uses these values to produce compiler defines and to compute
// a cache key that uniquely identifies a build configuration.
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

// Construct a JitDeviceConfig by querying a live device through MetalContext.
//
// All topology and memory parameters (bank counts, PCIe core location,
// harvesting mask, dispatch layout, etc.) are read from the cluster and SoC
// descriptor for the given `device_id`; nothing is inferred from static
// configuration files alone. Because of this, the device must be accessible at
// call time.
JitDeviceConfig create_jit_device_config(ChipId device_id, uint8_t num_hw_cqs);

// TODO: Add a factory method to create JitDeviceConfig from a YAML profile

}  // namespace tt::tt_metal
