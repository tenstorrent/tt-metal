// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/group.hpp>
#include <tt-metalium/kernel_types.hpp>  // For DataMovementProcessor, NOC, etc.

namespace tt::tt_metal::experimental {

// ============================================================================
//  DataMovementHardwareConfig
// ============================================================================
//
// Describes the hardware resources controlled by a data movement ("DM") kernel.
//
// The DM hardware differs between Gen1 architectures (Wormhole, Blackhole) and
// Gen2 architectures (Quasar and derivatives), so this config carries a separate
// Gen1Config and Gen2Config. The runtime selects the one matching the target
// architecture, so architecture-agnostic host code may populate both.
//
// Gen1 DM config (i.e., which RISC and NOC a kernel uses) is performance-critical,
// but the common case is handled for you: set the `role` hint to READER or WRITER
// and the runtime fills in the conventional processor/NOC/NOC-mode. Power users
// who want to override that convention can provide an explicit Gen1Config
// (alongside RoleHint::UNSPECIFIED).
//
// Gen2 has a unified NOC and its DM kernel core selection is fully automated.
// The gen2_config controls only whether the ISR-based DFB implicit sync is enabled.
// This config is optional: when absent, the runtime uses default settings.
//
// ============================================================================

struct DataMovementHardwareConfig {
    // Declares what the DM kernel does.
    //  - If specified, the runtime will automatically fill in an omitted Gen1Config
    //    with the standard optimal Gen1 DM config for a READER or WRITER kernel.
    //  - If an explicit Gen1Config is provided, set the role hint to UNSPECIFIED.
    //  - For Gen2, the role hint is ignored / informational.
    enum class RoleHint { READER, WRITER, UNSPECIFIED };
    RoleHint role = RoleHint::UNSPECIFIED;

    struct Gen1Config {
        tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
        tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    };
    // For a Gen1 DM kernel, specifying the Gen1Config is optional.
    // If unspecified, the runtime infers the settings from the role hint (READER/WRITER).
    // (A Gen1 DM kernel must provide either a role hint or an explicit Gen1Config.)
    std::optional<Gen1Config> gen1_config = std::nullopt;

    struct Gen2Config {
        // Opt-out of DFB implicit sync (on a per-DFB basis)
        //  - Implicit sync enables streamlined kernel-side syntax, but triggers ISR handling.
        //  - Use this control to revert to legacy explicit sync APIs (for specific bound DFBs).
        //  - This feature is mainly for debug purposes, or for backwards-compatible code style.
        // Any bound DFB not listed here will use implicit sync by default.
        Group<DFBSpecName> disable_implicit_sync_for;
    };
    // For a Gen2 DM kernel, providing the Gen2Config is optional.
    std::optional<Gen2Config> gen2_config = std::nullopt;
};

// Convenience alias for the nested role enum
using DataMovementRoleHint = DataMovementHardwareConfig::RoleHint;

}  // namespace tt::tt_metal::experimental
