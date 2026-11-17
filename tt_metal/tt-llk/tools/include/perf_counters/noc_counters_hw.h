// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/noc_counters_inventory.h"

namespace llk::perf::noc
{

// A plain load of the tile's own NIU register; any core of the tile can read it.
inline std::uint32_t read(std::uint32_t noc, std::uint32_t reg)
{
    return *reinterpret_cast<volatile std::uint32_t*>(NIU_STATUS_BASE + noc * NIU_NOC_STRIDE + reg * 4);
}

} // namespace llk::perf::noc
