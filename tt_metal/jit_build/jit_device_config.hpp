// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "impl/context/context_types.hpp"

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

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

    // True when the device is a DM-only Quasar configuration (e.g. 9x4 sim) that has
    // no TRISC compute hardware.  Causes QUASAR_DM_ONLY to be defined for all firmware
    // and kernel JIT compilations so that dm.cc can skip TRISC register accesses that
    // do not exist on such devices.
    bool quasar_dm_only = false;

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
//
// `context_id` selects which MetalContext instance to query. It MUST match the
// context that owns `device_id`. Passing the wrong id (including the default id
// when `device_id` is owned by a non-default context) does NOT fail loudly:
// MetalContext::instance(context_id) resolves to whichever context happens to
// occupy that slot, and the no-arg fallback returns any existing context if the
// requested slot is empty. The returned JitDeviceConfig is then a snapshot of
// the wrong cluster's arch / topology / dispatch layout, and downstream queries
// against `device_id` silently misbehave (mismatched harvesting masks, dispatch
// cores in the wrong place, kernel build cache hits across clusters). Callers
// must thread through the same ContextId they used to create the device.
JitDeviceConfig create_jit_device_config(ChipId device_id, uint8_t num_hw_cqs, ContextId context_id);

// enumerate_jit_device_configs walks `core_descriptor_path` YAML; `soc_descriptor_path` must be the
// matching SoC descriptor YAML used to derive base DRAM bank count (dram_views section).
void enumerate_jit_device_configs(
    tt::ARCH arch,
    const std::string& core_descriptor_path,
    const std::string& soc_descriptor_path,
    const std::function<void(const JitDeviceConfig&)>& callback);

// Iterate every JitDeviceConfig that is officially supported for ahead-of-time
// (offline) compilation, covering both firmware precompile and offline kernel
// compile. Resolves core/SoC descriptor paths under the rtoptions root and
// expands each (arch, core_descriptor, soc_descriptor) tuple via
// enumerate_jit_device_configs. The supported-products table itself is an
// implementation detail of jit_device_config.cpp.
void enumerate_offline_compile_device_configs(
    const tt::llrt::RunTimeOptions& rtoptions, const std::function<void(const JitDeviceConfig&)>& callback);

}  // namespace tt::tt_metal
