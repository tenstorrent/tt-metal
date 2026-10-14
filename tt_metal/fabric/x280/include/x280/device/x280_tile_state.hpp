// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What a host can observe about one L2CPU tile, and how to render it.
//
// These are values, not handles: nothing here touches hardware, UMD, or Metal.
// They live in the pure target so a snapshot can be constructed and checked
// without a device.

#pragma once

#include "x280/device/x280_state.hpp"
#include "x280_hart_protocol.h"

#include <array>
#include <compare>
#include <cstdint>
#include <optional>
#include <string>

namespace x280 {

struct HartState {
    uint32_t hart_id = 0;
    bool observation_valid = false;
    std::string invalid_reason;
    x280_hart_control_t control{};
    x280_hart_state_t state{};
    x280_hart_trap_t trap{};
};

static_assert(
    kHartsPerTile == X280_HART_COUNT, "HartState arrays must match the firmware protocol's hardware-hart count");

std::string describe(const HartState& hart);

struct TileKey {
    int chip_id = -1;
    int tile_index = -1;

    auto operator<=>(const TileKey&) const = default;
};

struct TileIdentity {
    int chip_id = -1;

    // Hardware L2CPU number: reset bit 4+N, harts 4N..4N+3, telemetry
    // L2CPUCLK<N>. Not the order UMD stores its cores in.
    int hw_index = -1;
    uint32_t noc_x = 0;
    uint32_t noc_y = 0;
    uint32_t num_harts = kHartsPerTile;

    auto operator<=>(const TileIdentity&) const = default;
};

inline TileKey tile_key(const TileIdentity& id) { return {id.chip_id, id.hw_index}; }

struct TileState {
    TileIdentity id;

    // False when the tile could not be observed usably. Every decoded field
    // below is meaningless in that case, and the component refuses to act on
    // it.
    //
    // Two causes, deliberately collapsed into one signal: a raw all-ones read,
    // and a transport error such as an MMIO timeout. An unclocked tile can
    // produce either — the access takes tens of milliseconds, so whether the
    // all-ones value comes back or a per-operation budget expires first is a
    // property of the driver, not of the tile. Neither may be masked into an
    // apparently valid field value.
    bool observation_valid = false;

    // Why the observation is unusable. Empty when observation_valid is true.
    std::string invalid_reason;

    bool reset_released = false;

    uint32_t raw_wayenable = 0;  // pre-mask, as read
    uint64_t raw_ecc_marker = 0;
    uint64_t raw_ecc_extent = 0;
    LimEccState ecc = LimEccState::Invalid;
    // LimEccClassification::primed_bytes: set only when ecc is Primed and the
    // marker records an extent.
    std::optional<uint64_t> ecc_primed_bytes;
    uint64_t lim_backed_bytes = 0;

    std::array<HartState, kHartsPerTile> harts{};

    // All four versioned per-hart records are committed IDLE with their
    // distinct heartbeat markers, and the tile is out of reset.
    bool resident_idle_live = false;

    // Reported by ARC telemetry, so it reflects what the hardware is doing
    // rather than what was last requested. Empty when unavailable.
    std::optional<uint32_t> pll_mhz;
};

std::string describe(const TileState& state);

}  // namespace x280
