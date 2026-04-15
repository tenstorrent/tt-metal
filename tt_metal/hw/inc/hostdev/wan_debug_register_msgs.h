// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared host + ERISC: WAN / Ethernet debug snapshot struct, and Blackhole DRAM ring layout
// (must match tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp unreserved-base accounting).

#pragma once

#include <cstdint>
#include "wan_debug_register_addresses.h"

struct WANDebugRegisterState {
    /// Monotonic snapshot id per write (host/tools use for ring ordering; 0 = never written in DRAM).
    uint32_t sequence_number;
    uint32_t version_number;  // Used to help track which version of the test this data come from.
    /* wan_csr_*: full 32-bit READ_REG(WAN_DEBUG_REGISTER_A..D); see wan_debug_register_addresses.h for mapping. */
    /// Lower 16 bits: absolute logical X [7:0], Y [15:8] (see get_absolute_logical_x/y in dataflow_api.h).
    uint32_t eth_noc_coordinates;
    /// WAN snapshot payload (maps to WAN_DEBUG_REGISTER_A..D / fabric READ_REG order).
    uint32_t wan_csr_a;
    uint32_t wan_csr_b;
    uint32_t wan_csr_c;
    uint32_t wan_csr_d;
    /// Explicit tail padding (was implicit) plus dwords to reach 32 B so `ring_index * sizeof` stays 16-byte aligned
    /// for Blackhole NOC DRAM writes (NOC_DRAM_WRITE_ALIGNMENT_BYTES == 16). Zero on each ERISC snapshot write.
    uint32_t reserved_pad_u32_0 = 0;
};

static_assert(
    sizeof(WANDebugRegisterState) == 32, "WANDebugRegisterState size drives DRAM/L1 layout (NOC-aligned stride)");
static_assert(alignof(WANDebugRegisterState) == 4, "WANDebugRegisterState alignment");

// Matches Blackhole noc_parameters.h DRAM_ALIGNMENT (max of NOC DRAM read/write alignments).
inline constexpr std::uint32_t WAN_DEBUG_DRAM_ALIGNMENT_BYTES = 64;

inline constexpr std::uint32_t WAN_DEBUG_DRAM_TARGET_BYTES = 10U * 1024U * 1024U;
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
