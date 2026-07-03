// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
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
// Gen2 architectures (Quasar and derivatives). A DM kernel targets exactly one
// generation, so the config is a variant holding either a DataMovementGen1Config
// or a DataMovementGen2Config. The alternative MUST match the target architecture:
// there is no implicit cross-generation substitution, and a mismatch is a hard
// error (validated at program build), not silently defaulted. Architecture-agnostic
// host code selects the matching alternative from the device arch at build time.
//
// Gen1 DM config (i.e., which RISC and NOC a kernel uses) is performance-critical,
// but the common case is handled for you: build the DataMovementGen1Config with
// create_reader_gen1_datamovement_config() / create_writer_gen1_datamovement_config(),
// which fill in the conventional processor/NOC/NOC-mode. Power users who want to override
// that convention construct a DataMovementGen1Config directly.
//
// Gen2 has a unified NOC and its DM kernel core selection is fully automated.
// The DataMovementGen2Config controls only whether the ISR-based DFB implicit
// sync is enabled.
//
// ============================================================================

struct DataMovementGen1Config {
    tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
    tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
};

// Build the conventional Gen1 placement for a reader / writer DM kernel.
DataMovementGen1Config create_reader_gen1_datamovement_config();
DataMovementGen1Config create_writer_gen1_datamovement_config();

struct DataMovementGen2Config {
    // Opt-out of DFB implicit sync (on a per-DFB basis)
    //  - Implicit sync enables streamlined kernel-side syntax, but triggers ISR handling.
    //  - Use this control to revert to legacy explicit sync APIs (for specific bound DFBs).
    //  - This feature is mainly for debug purposes, or for backwards-compatible code style.
    // Any bound DFB not listed here will use implicit sync by default.
    Group<DFBSpecName> disable_implicit_sync_for;
    // TODO -- rename to disable_dfb_implicit_sync_for (need to update existing users)

    // Opt out of DFB implicit sync for ALL the DFBs this kernel binds.
    // (The per-kernel hammer; equivalent to listing every bound DFB above.)
    bool disable_dfb_implicit_sync_for_all = false;
};

// A DM kernel's hardware config holds exactly one generation's config.
using DataMovementHardwareConfig = std::variant<DataMovementGen1Config, DataMovementGen2Config>;

}  // namespace tt::tt_metal::experimental
