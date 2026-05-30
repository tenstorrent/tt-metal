// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/kernel_types.hpp>  // For DataMovementProcessor, NOC, etc.

namespace tt::tt_metal::experimental::metal2_host_api {

// ============================================================================
//  KernelDMConfig
// ============================================================================
//
// The KernelDMConfig describes the configuration of the hardware resources
// controlled by a data movement kernel. ("DM" is the abbreviation for Data
// Movement, used throughout the API).
//
// The DM configuration differs between Gen1 architectures (Wormhole, Blackhole)
// and Gen2 architectures (Quasar and derivatives).
//
// The KernelDMConfig struct exposes both a Gen1Config and a Gen2Config variant.
// The runtime will dynamically select the appropriate variant based on the
// target architecture. For architecture-agnostic host code, you may specify
// both variants in the same KernelDMConfig.
//
// If the variant for the target architecture is not supplied:
//  - Gen 1 will trigger an error (config is mandatory for now)
//  - Gen 2 will use default settings
//
// ============================================================================

struct KernelDMConfig {
    struct Gen1Config {
        tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
        tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    };
    std::optional<Gen1Config> gen1_config = std::nullopt;

    struct Gen2Config {
        // Opt-out of DFB implicit sync (on a per-DFB basis)
        //  - Implicit sync enables streamlined kernel-side syntax, but triggers ISR handling.
        //  - Use this control to revert to legacy explicit sync APIs (for specific bound DFBs).
        //  - This feature is mainly for debug purposes, or for backwards-compatible code style.
        // Any bound DFB not listed here will use implicit sync by default.
        std::vector<DFBSpecName> disable_implicit_sync_for;
    };
    std::optional<Gen2Config> gen2_config = std::nullopt;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
