// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal::experimental::quasar {
struct QuasarDataMovementConfig;
}  // namespace tt::tt_metal::experimental::quasar

struct metal_SocDescriptor;

namespace tt::tt_metal {
class IDevice;
class MetalEnvImpl;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::detail {

// Returns synthetic logical dispatch-engine cores CoreCoord(index, 0) from the ordered soc `dispatch:` list.
std::vector<CoreCoord> get_quasar_soc_dispatch_engine_logical_cores(const metal_SocDescriptor& soc_desc);

// Fail fast dispatch init when Quasar has no usable dispatch cores for the active path.
void validate_quasar_dispatch_cores_for_fd(
    tt::tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    uint8_t num_hw_cqs,
    const tt_metal::DispatchCoreConfig& dispatch_core_config);

// Arch-gated dispatch core type (Quasar: DISPATCH vs interim Tensix WORKER; WH/BH: from DispatchCoreConfig).
CoreType resolve_dispatch_core_type(
    tt::ARCH arch,
    const tt_metal::DispatchCoreConfig& dispatch_core_config,
    const metal_SocDescriptor& soc_desc,
    bool use_quasar_tensix_dispatch_cores);

// Explicit DM pinning for dispatch-engine cq kernels (SD + FD).
KernelHandle CreateDispatchEngineKernel(
    Program& program,
    const std::string& file_name,
    const CoreCoord& core,
    DataMovementProcessor dm_processor,
    const experimental::quasar::QuasarDataMovementConfig& config);

// SD cq-kernel test helpers (test_prefetcher / test_dispatcher).
CoreType resolve_sd_cq_kernel_core_type(const tt::tt_metal::IDevice* device);
CoreCoord dispatch_engine_core(const tt::tt_metal::IDevice* device, uint32_t index);
CoreCoord dispatch_engine_virtual_core(const tt::tt_metal::IDevice* device, uint32_t index);
CoreCoord sd_cq_prefetch_core(const tt::tt_metal::IDevice* device);
CoreCoord sd_cq_dispatch_core(const tt::tt_metal::IDevice* device);
CoreCoord sd_cq_virtual_core(const tt::tt_metal::IDevice* device, const CoreCoord& logical_core);
bool sd_cq_kernel_tests_should_skip(const tt::tt_metal::IDevice* device);
uint32_t fd_core_type_define_value(const tt::tt_metal::IDevice* device);
DataMovementProcessor prefetch_dm_processor();
DataMovementProcessor dispatch_dm_processor();

}  // namespace tt::tt_metal::detail
