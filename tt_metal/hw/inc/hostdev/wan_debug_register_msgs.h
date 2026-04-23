// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared host + ERISC: WAN / Ethernet debug snapshot struct, and Blackhole DRAM ring layout
// (must match tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp unreserved-base accounting).

#pragma once

#include <cstdint>
#include "wan_debug_register_addresses.h"

static_assert(WAN_DEBUG_REG_ADDR_VERSION == 0x0102, "WAN debug register address version mismatch, expected 0x0102");

/// Trailing snapshot pad (zeroed by fabric ERISC); sized so 50 MiB / sizeof(WANDebugRegisterState) is even.
/// Payload growth shrinks this pad so total `sizeof` stays 256 B / 16-aligned.
inline constexpr std::uint32_t WAN_DEBUG_REGISTER_STATE_RESERVED_PAD_U32 = 8;

struct WANDebugRegisterState {
    /// Monotonic snapshot id per write (host/tools use for ring ordering; 0 = never written in DRAM).
    uint32_t sequence_number;
    uint32_t version_number;  // Used to help track which version of the test this data come from.
    /// Lower 16 bits: absolute logical X [7:0], Y [15:8] (see get_absolute_logical_x/y in dataflow_api.h).
    uint32_t eth_noc_coordinates;
    /// Obfuscated WAN snapshot dwords; READ_REG mapping is `WAN_DEBUG_REGISTER_*` in wan_debug_register_addresses.h.
    uint32_t wan_debug_register_0;
    uint32_t wan_debug_register_1;
    uint32_t wan_debug_register_3;
    /// `WAN_DEBUG_REGISTER_SET_9_BASE` + lane * `WAN_DEBUG_REGISTER_SET_9_OFFSET` (see addresses header).
    uint32_t wan_debug_register_set_9[WAN_DEBUG_REGISTER_SET_9_NUM];
    /// `WAN_DEBUG_REGISTER_SET_10_BASE` + i * `WAN_DEBUG_REGISTER_SET_10_OFFSET` (byte stride between SET_10 words).
    uint32_t wan_debug_register_set_10[WAN_DEBUG_REGISTER_SET_10_NUM];
    /// L1-backed fields sourced from `eth_live_status_t` (see blackhole/eth_fw_api.h).
    /// Lo/hi 32-bit halves of the 64-bit counters; loaded via a single volatile struct copy.
    uint32_t corr_cw_lo;
    uint32_t corr_cw_hi;
    uint32_t uncorr_cw_lo;
    uint32_t uncorr_cw_hi;
    uint32_t txq0_resend_cnt_lo;
    uint32_t txq0_resend_cnt_hi;
    uint32_t txq1_resend_cnt_lo;
    uint32_t txq1_resend_cnt_hi;
    uint32_t txq2_resend_cnt_lo;
    uint32_t txq2_resend_cnt_hi;
    /// Trailing dwords: 16 B NOC DRAM slot alignment, and even `WAN_DEBUG_DRAM_NUM_ENTRIES` for half-ring split
    /// (50 MiB / sizeof(slot) must be even; 208 B would yield an odd entry count).
    uint32_t reserved_pad_u32_for_16byte_stride[WAN_DEBUG_REGISTER_STATE_RESERVED_PAD_U32]{};
};

static_assert(sizeof(WANDebugRegisterState) == 256, "WANDebugRegisterState size drives DRAM/L1 layout");
static_assert(sizeof(WANDebugRegisterState) % 16 == 0, "WAN DRAM ring slot stride must be 16-byte aligned");
static_assert(alignof(WANDebugRegisterState) == 4, "WANDebugRegisterState alignment");
static_assert(
    sizeof(WANDebugRegisterState::reserved_pad_u32_for_16byte_stride) ==
        WAN_DEBUG_REGISTER_STATE_RESERVED_PAD_U32 * sizeof(uint32_t),
    "WAN_DEBUG_REGISTER_STATE_RESERVED_PAD_U32 must match reserved_pad_u32_for_16byte_stride[]");

// Matches Blackhole noc_parameters.h DRAM_ALIGNMENT (max of NOC DRAM read/write alignments).
inline constexpr std::uint32_t WAN_DEBUG_DRAM_ALIGNMENT_BYTES = 64;

inline constexpr std::uint32_t WAN_DEBUG_DRAM_TARGET_BYTES = 50U * 1024U * 1024U;
inline constexpr std::uint32_t WAN_DEBUG_DRAM_NUM_ENTRIES =
    WAN_DEBUG_DRAM_TARGET_BYTES / static_cast<std::uint32_t>(sizeof(WANDebugRegisterState));
inline constexpr std::uint32_t WAN_DEBUG_DRAM_RESERVED_SIZE =
    ((WAN_DEBUG_DRAM_NUM_ENTRIES * static_cast<std::uint32_t>(sizeof(WANDebugRegisterState)) +
      WAN_DEBUG_DRAM_ALIGNMENT_BYTES - 1) /
     WAN_DEBUG_DRAM_ALIGNMENT_BYTES) *
    WAN_DEBUG_DRAM_ALIGNMENT_BYTES;
// Slot count per ERISC implied by reserved DRAM size (host passes this as the CT arg on Blackhole).
inline constexpr std::uint32_t WAN_DEBUG_DRAM_RESERVED_SLOTS_PER_HALF = WAN_DEBUG_DRAM_NUM_ENTRIES / 2;

static_assert(WAN_DEBUG_DRAM_NUM_ENTRIES >= 2, "WAN debug DRAM ring needs at least two entries for half splits");
static_assert(WAN_DEBUG_DRAM_NUM_ENTRIES % 2 == 0, "WAN debug DRAM entry count must be even");
