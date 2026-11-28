// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/router_counters_inventory.h"

namespace llk::perf::router
{

// Plain load of the tile's own router register; on Blackhole it returns the low word.
inline std::uint32_t read(std::uint32_t noc, std::uint32_t bank_offset, std::uint32_t vc)
{
    return *reinterpret_cast<volatile std::uint32_t*>(ROUTER_REGS_BASE + noc * ROUTER_NOC_STRIDE + bank_offset + vc * ROUTER_VC_STRIDE);
}

} // namespace llk::perf::router
