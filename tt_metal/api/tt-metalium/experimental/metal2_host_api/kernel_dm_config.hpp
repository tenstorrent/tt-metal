// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/kernel_types.hpp>  // For DataMovementProcessor, NOC, etc.

namespace tt::tt_metal::experimental::metal2_host_api {

// KernelDMConfig: Data Movement (DM) resource configuration for DM kernels.
// "DM" is the canonical Tenstorrent abbreviation for Data Movement, used freely
// throughout the API surface for type names; method names paired with sibling
// predicates (e.g., is_data_movement_kernel) use the full form.
//
// The DM configuration differs between Gen1 (Wormhole, Blackhole) and Gen2 (Quasar
// and derivatives), so the struct exposes a Gen1Config and a Gen2Config variant —
// the kernel author populates the one matching their target architecture.
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
