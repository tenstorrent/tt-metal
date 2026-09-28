// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Device-safe constants for cq_prefetch kernel. This header must not pull in
// host-only deps (e.g. fmt, core_coordinates) so it can be included from
// firmware (brisc) builds.
namespace tt::tt_metal {

struct PrefetchConstants {
    static constexpr uint32_t PREFETCH_MAX_OUTSTANDING_PCIE_READS = 4U;
    // With snooped fetch queue entries, the host flips this bit each lap around the ring so the prefetcher
    // can tell a new entry from the previous lap's without clearing consumed slots.
    static constexpr uint32_t PREFETCH_Q_PHASE_BIT = 30U;
};
static_assert(
    (PrefetchConstants::PREFETCH_MAX_OUTSTANDING_PCIE_READS &
     (PrefetchConstants::PREFETCH_MAX_OUTSTANDING_PCIE_READS - 1U)) == 0U);

}  // namespace tt::tt_metal
