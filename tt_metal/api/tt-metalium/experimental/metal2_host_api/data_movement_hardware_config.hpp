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
// but the common case is handled for you: build the Gen1Config with
// Gen1Config::create_from_role(READER/WRITER), which fills in the conventional
// processor/NOC/NOC-mode. Power users who want to override that convention construct
// a Gen1Config directly.
//
// Gen2 has a unified NOC and its DM kernel core selection is fully automated.
// The gen2_config controls only whether the ISR-based DFB implicit sync is enabled.
// This config is optional: when absent, the runtime uses default settings.
//
// ============================================================================

struct DataMovementHardwareConfig {
    // Names the common DM kernel roles, used only as an argument to
    // Gen1Config::create_from_role() to build the conventional Gen1 placement.
    enum class RoleHint { READER, WRITER };

    struct Gen1Config {
        tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
        tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;

        // Build the conventional Gen1 placement for a READER/WRITER kernel
        // (mirrors the legacy Reader/WriterDataMovementConfig convention):
        //   READER -> NCRISC (RISCV_1) on NOC_0;  WRITER -> BRISC (RISCV_0) on NOC_1
        // Power users who need a non-conventional placement construct a Gen1Config directly.
        static Gen1Config create_from_role(RoleHint role);
    };
    // For a Gen1 DM kernel (Wormhole / Blackhole), specify the Gen1Config — either built via
    // Gen1Config::create_from_role(READER/WRITER) for the common case, or constructed directly.
    std::optional<Gen1Config> gen1_config = std::nullopt;

    struct Gen2Config {
        // Opt-out of DFB implicit sync (on a per-DFB basis)
        //  - Implicit sync enables streamlined kernel-side syntax, but triggers ISR handling.
        //  - Use this control to revert to legacy explicit sync APIs (for specific bound DFBs).
        //  - This feature is mainly for debug purposes, or for backwards-compatible code style.
        // Any bound DFB not listed here will use implicit sync by default.
        Group<DFBSpecName> disable_dfb_implicit_sync_for;

        // Opt out of DFB implicit sync for ALL the DFBs this kernel binds.
        // (The per-kernel hammer; equivalent to listing every bound DFB above.)
        bool disable_dfb_implicit_sync_for_all = false;
    };
    // For a Gen2 DM kernel, providing the Gen2Config is optional.
    std::optional<Gen2Config> gen2_config = std::nullopt;
};

// Convenience alias for the nested role enum
using DataMovementRoleHint = DataMovementHardwareConfig::RoleHint;

}  // namespace tt::tt_metal::experimental
