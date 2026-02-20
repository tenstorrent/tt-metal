// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/dispatch_core_common.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>  // CoreType
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal {

enum DispatchWorkerType : uint32_t {
    PREFETCH = 0,
    PREFETCH_HD = 1,
    PREFETCH_H = 2,
    PREFETCH_D = 3,
    DISPATCH = 4,
    DISPATCH_HD = 5,
    DISPATCH_H = 6,
    DISPATCH_D = 7,
    DISPATCH_S = 8,
    FABRIC_MUX = 17,          // Downstream from MMIO to remote mux. Tunnel index is required.
    RETURN_FABRIC_MUX = 18,   // Upstream from remote to MMIO mux. Tunnel index will be determined from the device id.
    PREFETCH_HD_WRITER = 19,  // Writer stub that runs on NCRISC of the same core as PREFETCH_HD (on BRISC).
    COUNT,
};

CoreType get_core_type_from_config(const DispatchCoreConfig& config);

// Resolve the dispatch core axis from a DispatchCoreConfig without depending on MetalContext.
// Uses the config's explicit axis if set; otherwise falls back to arch-based resolution.
// TODO: https://github.com/tenstorrent/tt-metal/issues/39974
DispatchCoreAxis resolve_dispatch_core_axis(
    const DispatchCoreConfig& config, tt::ARCH arch, tt_fabric::FabricTensixConfig fabric_tensix_config);

// Helper functions to get the dispatch core config/type
DispatchCoreConfig get_dispatch_core_config();

}  // namespace tt::tt_metal
