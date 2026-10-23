// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Per-tile ECC observation and priming primitives. X280Runtime owns the
// chip-wide sequence, safe PLL setup, and reset-required reporting.

#include "x280_tile_impl.hpp"

#include "x280_format.hpp"
#include "x280_hw_addrs.hpp"

#include "x280_address_map.h"

namespace x280::detail {

TileImpl::LimEccObservation TileImpl::read_lim_ecc_unlocked() const {
    LimEccObservation observation;
    const auto tile = core();
    observation.raw_wayenable = transport().read_reg32(id().chip_id, tile, kL3CacheWayEnable);
    if (is_all_ones(observation.raw_wayenable)) {
        observation.state = LimEccState::Invalid;
        return observation;
    }
    if (lim_backed_bytes(observation.raw_wayenable) == 0) {
        observation.state = LimEccState::LimUnavailable;
        return observation;
    }
    observation.raw_marker = transport().read_u64(id().chip_id, tile, X280_ECC_PRIME_MARKER_ADDR);
    observation.raw_extent = transport().read_u64(id().chip_id, tile, X280_ECC_PRIME_EXTENT_ADDR);
    const auto ecc = classify_lim_ecc(observation.raw_wayenable, observation.raw_marker, observation.raw_extent);
    observation.state = ecc.state;
    observation.primed_bytes = ecc.primed_bytes;
    return observation;
}

void TileImpl::prime_lim_ecc_unlocked(bool& wayenable_committed) {
    const auto tile = core();
    // 1. Commit every way to cache. LIM is gone from here until an ASIC reset.
    //
    //    Flagged before the write goes out, not after: a write that throws may
    //    still have landed, and the caller must treat an attempted commit as a
    //    commit.
    wayenable_committed = true;
    transport().write_reg32(id().chip_id, tile, kL3CacheWayEnable, 0xF);

    // Verify the one-way step actually landed. A silently dropped write would
    // otherwise send the caller off to reset the board and come back to a tile
    // that is still unprimed, with nothing to explain the loop.
    const uint32_t committed = transport().read_reg32(id().chip_id, tile, kL3CacheWayEnable);
    if (is_all_ones(committed)) {
        fail("WayEnable read all ones while priming; tile unclocked or transport broken");
    }
    if ((committed & 0xFu) != 0xFu) {
        fail("WayEnable did not commit while priming: " + format_readback_values(0xF, committed & 0xFu));
    }

    // 2. Pin all masters to the top way, so the walk below lands in the one way
    //    that reappears at LIM base once WayEnable returns to 0.
    for (uint32_t master = 0; master < kL3CacheNumMasters; master++) {
        transport().write_reg32(
            id().chip_id, tile, kL3CacheWayMaskBase + master * kL3WayMaskStride, kL3WayMaskPinWay15);
    }

    // 3. Walk that way one cache line at a time through the cached Zero Device
    //    alias. Each 4-byte write misses, so the cache controller fetches the
    //    line from the Zero Device — safe zeros, not uninitialised SRAM —
    //    merges the write into it, and commits data and ECC together into the
    //    pinned way.
    for (uint32_t offset = 0; offset < kLimEccPrimeBytes; offset += kLimEccPrimeLineSize) {
        transport().write_reg32(id().chip_id, tile, kZeroDeviceBase + offset, 0);
    }

    // 4. Seed the marker and, beside it, the extent step 3 walked. Their
    //    contents survive the ASIC reset alongside the rest of the primed
    //    SRAM, which is how the next boot knows.
    //
    //    Not verified by readback, unlike the WayEnable commit above: LIM is
    //    gone at this point, so the marker cannot be read at its LIM address
    //    until the reset brings Way 15 back. Steps 2 and 3 are likewise
    //    unverifiable — a Zero Device read returns zero whatever was written.
    transport().write_reg32(
        id().chip_id,
        tile,
        kZeroDeviceBase + X280_ECC_PRIME_MARKER_OFFSET,
        static_cast<uint32_t>(X280_ECC_PRIME_MARKER_MAGIC & 0xFFFFFFFFull));
    transport().write_reg32(
        id().chip_id,
        tile,
        kZeroDeviceBase + X280_ECC_PRIME_MARKER_OFFSET + 4,
        static_cast<uint32_t>(X280_ECC_PRIME_MARKER_MAGIC >> 32));
    transport().write_reg32(id().chip_id, tile, kZeroDeviceBase + X280_ECC_PRIME_EXTENT_OFFSET, kLimEccPrimeBytes);
    transport().write_reg32(id().chip_id, tile, kZeroDeviceBase + X280_ECC_PRIME_EXTENT_OFFSET + 4, 0);
}

void TileImpl::set_lim_shape_unlocked(uint32_t target_wayenable, bool& commit_attempted) {
    if (target_wayenable > 0xFu) {
        fail("target WayEnable " + std::to_string(target_wayenable) + " exceeds the four-bit hardware field");
    }

    const auto tile = core();
    const uint32_t raw_current = transport().read_reg32(id().chip_id, tile, kL3CacheWayEnable);
    if (is_all_ones(raw_current)) {
        fail("WayEnable read all ones while applying LIM shape; tile unclocked or transport broken");
    }

    const uint32_t current = raw_current & 0xFu;
    if (current > target_wayenable) {
        fail(
            "current WayEnable " + std::to_string(current) + " exceeds target " + std::to_string(target_wayenable) +
            "; increasing LIM requires an ASIC reset");
    }
    if (current == target_wayenable) {
        return;
    }

    // The caller has established that every L2CPU on the chip is held in
    // reset. Flag the one-way attempt before issuing it: a throwing write may
    // still have increased WayEnable.
    commit_attempted = true;
    transport().write_reg32(id().chip_id, tile, kL3CacheWayEnable, target_wayenable);
    const uint32_t observed = transport().read_reg32(id().chip_id, tile, kL3CacheWayEnable);
    if (is_all_ones(observed)) {
        fail("WayEnable read all ones after applying LIM shape");
    }
    if ((observed & 0xFu) != target_wayenable) {
        fail(
            "WayEnable did not reach the requested LIM shape: " +
            format_readback_values(target_wayenable, observed & 0xFu));
    }
}

}  // namespace x280::detail
