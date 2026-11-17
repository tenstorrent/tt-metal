// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "perf_counters/eth_counters_types.h"

// Quasar: the ethernet controller register map is not known, so the tables are empty and the layout has zeros.
// Include through perf_counters/eth_counters_inventory.h.

namespace llk::perf::eth::quasar
{

inline constexpr Layout layout = {0u, 0u, 0u, 0u, 0u, 0u, 0u};

inline constexpr std::array<QueueEntry, 0> tx_queue_counters = {};

inline constexpr std::array<QueueEntry, 0> rx_queue_counters = {};

inline constexpr std::array<L1Entry, 0> l1_counters = {};

} // namespace llk::perf::eth::quasar
