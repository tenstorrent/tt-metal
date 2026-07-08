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
// The DM hardware differs between Tenstorrent accelerators:
//  - Gen1 architectures (Wormhole, Blackhole) and
//  - Gen2 architectures (Quasar and derivatives).
//
// The DataMovementHardwareConfig is therefore generation-specific.
// You must provide the variant that matches your kernel's target architecture.
//
// ============================================================================

// The Gen1 data movement hardware config specifies which RISC-V core and which
// NOC a kernel uses. This selection is performance-critical.
//
// The common case is handled for you by role-specific factory functions:
//  - For a DM kernel that reads from DRAM: CreateReaderGen1DataMovementConfig()
//  - For a DM kernel that writes to DRAM:  CreateWriterGen1DataMovementConfig()
//
// Power users can override these conventions by constructing a
// DataMovementGen1Config directly.
//
struct DataMovementGen1Config {
    // The RISC-V core that runs this DM kernel (RISCV_0 or RISCV_1)
    // Each DM kernel on a node must be assigned to a unique RISC-V core
    tt::tt_metal::DataMovementProcessor processor;

    // The physical NOC that this DM kernel uses (NOC_0 or NOC_1)
    tt::tt_metal::NOC noc;

    // NOC ownership model. Leave as DM_DEDICATED_NOC unless you specifically
    // need both DM cores to share a single NOC (e.g. to keep the other NOC
    // free for fabric/CCL traffic). Dynamic mode adds cross-core coordination
    // overhead and must be set identically on both DM kernels on a node.
    tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
};

// Factory helper:
// Default config for a reader DM kernel (i.e. that reads from DRAM)
inline DataMovementGen1Config CreateReaderGen1DataMovementConfig() noexcept {
    return DataMovementGen1Config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::NOC_0,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
}

// Factory helper:
// Default config for a writer DM kernel (i.e. that writes to DRAM)
inline DataMovementGen1Config CreateWriterGen1DataMovementConfig() noexcept {
    return DataMovementGen1Config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_1,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
}

// Gen2 architectures have a unified NOC and fully automated DM kernel core selection.
// The DataMovementGen2Config controls whether the DFB implicit sync feature for DM
// kernels is enabled or disabled.
//
struct DataMovementGen2Config {
    // Opt-out of DFB implicit sync (on a per-DFB basis)
    //  - Implicit sync enables streamlined kernel-side syntax, but triggers ISR handling.
    //  - Use this control to revert to legacy explicit sync APIs (for specific bound DFBs).
    //  - Opting out is mainly for debug purposes, or for backwards-compatible code style.
    // Any bound DFB not listed here will use implicit sync by default.
    Group<DFBSpecName> disable_dfb_implicit_sync_for;

    // Opt out of DFB implicit sync for ALL the DFBs this kernel binds.
    // (The per-kernel hammer; equivalent to listing every bound DFB above.)
    bool disable_dfb_implicit_sync_for_all = false;
};

// A DM kernel's hardware config holds exactly one generation's config.
using DataMovementHardwareConfig = std::variant<DataMovementGen1Config, DataMovementGen2Config>;

}  // namespace tt::tt_metal::experimental
