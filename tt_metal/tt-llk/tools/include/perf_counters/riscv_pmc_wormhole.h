// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/riscv_pmc_types.h"

// Wormhole: the RISC-V cores implement no performance counter CSRs, so there are no events to select.
// Include through perf_counters/riscv_pmc_inventory.h.

namespace llk::perf::riscv
{

inline constexpr std::array<Entry, 0> baby_events = {};

} // namespace llk::perf::riscv
