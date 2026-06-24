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
class MetalEnvImpl;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::internal {

// Returns synthetic logical dispatch-engine cores CoreCoord(index, 0) from the ordered soc `dispatch:` list.
std::vector<CoreCoord> get_quasar_soc_dispatch_engine_logical_cores(const metal_SocDescriptor& soc_desc);

// Resolve the Quasar dispatch core pool (env override checked first).
const std::vector<CoreCoord>& get_quasar_dispatch_cores(
    tt::tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    uint8_t num_hw_cqs,
    const tt_metal::DispatchCoreConfig& dispatch_core_config);

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

CoreType resolve_dispatch_core_type(
    tt::tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    const tt_metal::DispatchCoreConfig& dispatch_core_config);

// Explicit DM pinning for dispatch-engine cq kernels (SD + FD).
KernelHandle CreateDispatchEngineKernel(
    Program& program,
    const std::string& file_name,
    const CoreCoord& core,
    DataMovementProcessor dm_processor,
    const experimental::quasar::QuasarDataMovementConfig& config);

}  // namespace tt::tt_metal::internal
