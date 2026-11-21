// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "perf_counters/dram_counters_types.h"

// Blackhole DRAM channel tile counters. Include through perf_counters/dram_counters_inventory.h.
// The NIU slave counters are the status registers 0x33 to 0x3B of the tile NIU: offset = 0x200 + 4 * index.

namespace llk::perf::dram::blackhole
{

inline constexpr Layout layout = {0xFFB20000ull, 0x10000u, 0u, 1u, 0u, 64u, 2u};

inline constexpr std::array<Entry, 6> niu_counters = {{
    {DramCounter::NIU_SLV_RD_REQ_RECEIVED, 0x2D4},
    {DramCounter::NIU_SLV_RD_DATA_WORD_SENT, 0x2CC},
    {DramCounter::NIU_SLV_NONPOSTED_WR_REQ_RECEIVED, 0x2E8},
    {DramCounter::NIU_SLV_POSTED_WR_REQ_RECEIVED, 0x2EC},
    {DramCounter::NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED, 0x2E0},
    {DramCounter::NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED, 0x2E4},
}};

// GDDR6 controller behind the tiles of one channel. AXI ports are numbered 1 to 3 as in the controller's own map.
inline constexpr ControllerLayout controller_layout = {1ull << 40, 0xFC104A30u, 0xFC104A34u, 0xFC200004u, 0x7u, 0x7u << 16};

inline constexpr std::array<ControllerEntry, 42> controller_counters = {{
    {"RD_CMD", 0xFC10AC08},
    {"RD_AP_CMD", 0xFC10AC0C},
    {"WR_CMD", 0xFC10AC00},
    {"WR_AP_CMD", 0xFC10AC04},
    {"ACT_CMD", 0xFC10AC14},
    {"PRECHARGE_CMD", 0xFC10AC18},
    {"PRECHARGE_ALL_CMD", 0xFC10AC1C},
    {"REFRESH_CMD", 0xFC10AC10},
    {"POWER_DOWN_CYCLES", 0xFC10AC20},
    {"SELF_REFRESH_CYCLES", 0xFC10AC24},
    {"RD_TO_WR_SWITCH", 0xFC10AC28},
    {"REORDER_AGE_LIMIT", 0xFC10AC2C},
    {"RMW_CYCLE", 0xFC10AC30},
    {"QUEUE_SUM_LO", 0xFC10ACA0},
    {"QUEUE_SUM_HI", 0xFC10ACA4},
    {"AXI1_CLOCKS", 0xFC10AC34},
    {"AXI2_CLOCKS", 0xFC10AC38},
    {"AXI3_CLOCKS", 0xFC10AC3C},
    {"AXI1_AW", 0xFC10AC40},
    {"AXI2_AW", 0xFC10AC44},
    {"AXI3_AW", 0xFC10AC48},
    {"AXI1_AW_STALL", 0xFC10AC4C},
    {"AXI2_AW_STALL", 0xFC10AC50},
    {"AXI3_AW_STALL", 0xFC10AC54},
    {"AXI1_AR", 0xFC10AC58},
    {"AXI2_AR", 0xFC10AC5C},
    {"AXI3_AR", 0xFC10AC60},
    {"AXI1_AR_STALL", 0xFC10AC64},
    {"AXI2_AR_STALL", 0xFC10AC68},
    {"AXI3_AR_STALL", 0xFC10AC6C},
    {"AXI1_W", 0xFC10AC70},
    {"AXI2_W", 0xFC10AC74},
    {"AXI3_W", 0xFC10AC78},
    {"AXI1_W_STALL", 0xFC10AC7C},
    {"AXI2_W_STALL", 0xFC10AC80},
    {"AXI3_W_STALL", 0xFC10AC84},
    {"AXI1_R", 0xFC10AC88},
    {"AXI2_R", 0xFC10AC8C},
    {"AXI3_R", 0xFC10AC90},
    {"AXI1_R_STALL", 0xFC10AC94},
    {"AXI2_R_STALL", 0xFC10AC98},
    {"AXI3_R_STALL", 0xFC10AC9C},
}};

} // namespace llk::perf::dram::blackhole
