// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
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
// Gen1 placement (which RISC and NOC a kernel uses) is performance-critical, but
// the common case is handled for you: set the `role` hint to READER or WRITER and
// the runtime fills in the conventional processor/NOC/NOC-mode. Power users who
// need to override that convention instead set an explicit Gen1Config (alongside
// RoleHint::UNSPECIFIED). On Gen1, exactly one of these two paths is required.
//
// Gen2 has a unified NOC, and DM kernel placement is fully automated. The
// gen2_config controls only whether the ISR-based DFB implicit sync is enabled.
// This config is optional: when absent, the runtime uses default settings.
//
// ============================================================================

struct DataMovementHardwareConfig {
    // Declares what the DM kernel does, so the runtime can fill in the Gen1 hardware
    // knobs (processor, NOC) for you using the standard reader/writer convention.
    //   - READER / WRITER: the runtime fills in the Gen1 config; leave gen1_config unset.
    //   - UNSPECIFIED:     you supply an explicit gen1_config yourself (power-user path).
    // Only has teeth on Gen1 (WH/BH), where RISC/NOC choice is performance-critical;
    // on Gen2 it is informational (the hardware is indifferent to DM core / NOC).
    enum class RoleHint { READER, WRITER, UNSPECIFIED };
    RoleHint role = RoleHint::UNSPECIFIED;

    struct Gen1Config {
        tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
        tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    };
    // Explicit Gen1 placement override. Leave unset when using a READER/WRITER role
    // (the runtime fills it in); set it together with RoleHint::UNSPECIFIED to take
    // manual control. On Gen1, one of {role hint, explicit gen1_config} is required.
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

}  // namespace tt::tt_metal::experimental
