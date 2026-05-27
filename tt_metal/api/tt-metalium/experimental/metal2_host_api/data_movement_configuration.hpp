// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/kernel_types.hpp>  // For DataMovementProcessor, NOC, etc.

namespace tt::tt_metal::experimental::metal2_host_api {

struct DataMovementConfiguration {
    // Data movement resource configuration for DM kernels.
    // NOTE: The DM configuration is different for Gen1 and Gen2.

    struct Gen1DataMovementConfig {
        tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
        tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    };
    std::optional<Gen1DataMovementConfig> gen1_data_movement_config = std::nullopt;

    struct Gen2DataMovementConfig {
        // Opt-out of DFB implicit sync (on a per-DFB basis)
        //  - Implicit sync enables streamlined kernel-side syntax, but triggers ISR handling.
        //  - Use this control to revert to legacy explicit sync APIs (for specific bound DFBs).
        //  - This feature is mainly for debug purposes, or for backwards-compatible code style.
        // Any bound DFB not listed here will use implicit sync by default.
        std::vector<DFBSpecName> disable_implicit_sync_for;
    };
    std::optional<Gen2DataMovementConfig> gen2_data_movement_config = std::nullopt;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
