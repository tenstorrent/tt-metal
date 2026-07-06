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
// generation, it MUST match the target architecture the program is dispatched onto.
// This invariant is checked at program build and will result in a hard error if not upheld.
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
    // Defaults for processor and noc are intentionally omitted as they're perf sensitive to their use case.
    // Please use factory functions `create_reader/write_gen1_datamovement_config` if you would like to create a
    // default DataMovementGen1Config.
    tt::tt_metal::DataMovementProcessor processor;
    tt::tt_metal::NOC noc;
    tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
};

// Build the conventional Gen1 placement for a reader / writer DM kernel, mirroring the legacy
// Reader/WriterDataMovementConfig convention (see kernel_types.cpp):
//   reader -> NCRISC (RISCV_1) on NOC_0;  writer -> BRISC (RISCV_0) on NOC_1
// NOC mode is always DM_DEDICATED_NOC; DM_DYNAMIC_NOC is a power-user knob reached only by
// constructing a DataMovementGen1Config directly.
inline DataMovementGen1Config create_reader_gen1_datamovement_config() noexcept {
    return DataMovementGen1Config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(tt::ARCH::WORMHOLE_B0),
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
}

inline DataMovementGen1Config create_writer_gen1_datamovement_config() noexcept {
    return DataMovementGen1Config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(tt::ARCH::WORMHOLE_B0),
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
}

struct DataMovementGen2Config {
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

// A DM kernel's hardware config holds exactly one generation's config.
using DataMovementHardwareConfig = std::variant<DataMovementGen1Config, DataMovementGen2Config>;

}  // namespace tt::tt_metal::experimental
