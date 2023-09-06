/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include "hostdevcommon/common_runtime_address_map.h"

struct TLB_OFFSETS {
    uint32_t local_offset;
    uint32_t x_end;
    uint32_t y_end;
    uint32_t x_start;
    uint32_t y_start;
    uint32_t noc_sel;
    uint32_t mcast;
    uint32_t ordering;
    uint32_t linked;
    uint32_t static_vc;
    uint32_t static_vc_end;
};

struct TLB_DATA {
    uint64_t local_offset = 0;
    uint64_t x_end = 0;
    uint64_t y_end = 0;
    uint64_t x_start = 0;
    uint64_t y_start = 0;
    uint64_t noc_sel = 0;
    uint64_t mcast = 0;
    uint64_t ordering = 0;
    uint64_t linked = 0;
    uint64_t static_vc = 0;

    // Orderings
    static constexpr uint64_t Relaxed = 0;
    static constexpr uint64_t Strict  = 1;
    static constexpr uint64_t Posted  = 2;

    bool check(const TLB_OFFSETS offset);
    std::optional<uint64_t> apply_offset(const TLB_OFFSETS offset);
};

std::string TensixSoftResetOptionsToString(TensixSoftResetOptions value);
