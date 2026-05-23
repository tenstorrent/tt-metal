// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/kernel_types.hpp>  // For DataMovementProcessor, NOC, etc.

namespace tt::tt_metal::experimental {

// Hardware generations referenced in this API:
//   Gen1 — Wormhole (WH), Blackhole (BH)
//   Gen2 — Quasar and derivative architectures

// Data movement hardware resource configuration for DM kernels
struct DataMovementConfiguration {
    // The DM configuration differs between hardware generations.
    //  - Gen1 requires explicit processor/NOC selection.
    //  - Gen2 has no user-facing configuration yet (placeholder only).

    struct Gen1DM {
        tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
        tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    };
    std::optional<Gen1DM> gen1 = std::nullopt;

    struct Gen2DM {
        // Currently, no configuration is needed for Gen2!
        // The empty struct is here as a placeholder.
        // (It is NOT required to provide an empty Gen2DM config for Gen2 architectures.)
    };
    std::optional<Gen2DM> gen2 = std::nullopt;
};

}  // namespace tt::tt_metal::experimental
