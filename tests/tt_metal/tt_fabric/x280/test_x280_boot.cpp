// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/experimental/x280/x280_boot.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

namespace {

using tt::tt_metal::experimental::x280::BootRegion;

// Per-region sentinel constants. Must stay in sync with the same values in
// tt_metal/impl/experimental/x280/x280_boot.cpp (kLimProfile / kDramProfile).
// The test doesn't have access to the internal `BootProfile` table (it lives
// in an anonymous namespace inside the library), so we mirror the two
// constants we need here.
struct SentinelLayout {
    uint64_t addr;
    uint64_t value;
    const char* name;
};

constexpr SentinelLayout kLimSentinel{
    0x08100000ULL,
    0xDEADBEEFCAFEBABEULL,
    "LIM",
};

constexpr SentinelLayout kDramSentinel{
    // DDR_BASE + IDLE_SENTINEL_OFF (x280/src/idle.c).
    0x400030000000ULL + 0x5D8000ULL,
    // IDLE_SENTINEL_VAL (x280/src/idle.c).
    0xCAFEF00D00000099ULL,
    "DRAM",
};

const SentinelLayout& sentinel_for(BootRegion region) {
    return region == BootRegion::Lim ? kLimSentinel : kDramSentinel;
}

// Physical L2CPU index (0..3) -> NOC0 (x, y) coords. Must mirror
// `kL2CpuTilesByPhysicalIndex` in tt_metal/impl/experimental/x280/x280_boot.cpp;
// the test needs the same mapping to know which sentinel address corresponds
// to which physical L2CPU bit when interpreting `TT_METAL_TEST_L2CPU_MASK`.
constexpr std::array<std::pair<uint32_t, uint32_t>, 4> kL2CpuTilesByPhysicalIndex = {{
    {8, 3},  // physical 0
    {8, 9},  // physical 1
    {8, 5},  // physical 2
    {8, 7},  // physical 3
}};

// Reads the 8-byte sentinel at `sentinel_addr` on the given L2CPU tile and
// returns it as a single u64. Tile is a NOC0 coord — get_cores(L2CPU, NOC0)
// already filters out harvested tiles, so callers may iterate it directly.
//
// Uses read_core (memory-NOC) instead of read_reg (uc_tlb AXI-register)
// because: (a) the DRAM sentinel address 0x400030000000 isn't in the AXI
// register range at all and uc_tlb can't reach it; and (b) per the
// production x280_boot.cpp comments, read_reg silently misroutes L2CPU-tile
// accesses on multi-chip hosts for any chip != 0. read_core works for both
// regions and on all chips.
uint64_t read_sentinel(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& /*mesh*/,
    tt::ChipId chip_id,
    const tt::umd::CoreCoord& l2cpu_tile,
    uint64_t sentinel_addr) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    uint64_t value = 0;
    cluster.read_core(
        &value, sizeof(value), tt_cxy_pair(chip_id, CoreCoord{l2cpu_tile.x, l2cpu_tile.y}), sentinel_addr);
    return value;
}

}  // namespace

// X280 boot verification: the test opens a MeshDevice (which brings up the
// Cluster + ARC), explicitly invokes the experimental X280 boot utility for
// the requested region (LIM or DRAM), then verifies that every selected,
// non-harvested L2CPU wrote its region-specific sentinel value. tt-metal
// stages both firmware-lim-idle.bin and firmware-idle.bin under
// runtime/hw/firmware/blackhole/x280/.
// Skips on non-Blackhole.
TEST(X280Boot, SentinelMatchesAfterInit) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "X280 boot only applies to Blackhole; skipping on " << tt::arch_to_str(cluster.arch());
    }

    const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();

    // Allow the user to pin the test to a specific chip via TT_METAL_TEST_CHIP_ID
    // (e.g. on a multi-chip host). Defaults to the first enumerated chip.
    tt::ChipId chip_id = *cluster.all_chip_ids().begin();
    if (const char* env_chip = std::getenv("TT_METAL_TEST_CHIP_ID"); env_chip != nullptr && *env_chip != '\0') {
        const tt::ChipId requested = static_cast<tt::ChipId>(std::stoi(env_chip));
        ASSERT_TRUE(cluster.all_chip_ids().contains(requested))
            << "TT_METAL_TEST_CHIP_ID=" << env_chip << " is not in the set of enumerated chips";
        chip_id = requested;
    }

    // Allow the user to select which physical L2CPUs to boot via
    // TT_METAL_TEST_L2CPU_MASK (4-bit mask; bit i selects L2CPU i). Accepts
    // hex (0x...), octal (0...), or decimal — std::stoi with base=0. Defaults
    // to 0xF (boot every enabled L2CPU).
    std::uint8_t l2cpu_mask = tt::tt_metal::experimental::x280::kAllL2CpuMask;
    if (const char* env_mask = std::getenv("TT_METAL_TEST_L2CPU_MASK"); env_mask != nullptr && *env_mask != '\0') {
        int parsed = 0;
        try {
            parsed = std::stoi(env_mask, nullptr, 0);
        } catch (const std::exception& e) {
            FAIL() << "TT_METAL_TEST_L2CPU_MASK='" << env_mask << "' failed to parse as an integer: " << e.what();
        }
        ASSERT_GE(parsed, 0) << "TT_METAL_TEST_L2CPU_MASK=" << env_mask << " is negative";
        ASSERT_LE(parsed, 0xF) << "TT_METAL_TEST_L2CPU_MASK=" << env_mask
                               << " has bits set above bit 3; only 0x0..0xF are valid (4 L2CPUs per chip)";
        l2cpu_mask = static_cast<std::uint8_t>(parsed);
    }

    // Allow the user to select which firmware load region via
    // TT_METAL_TEST_BOOT_REGION = "lim" | "dram" (case-insensitive).
    // Defaults to LIM (matches boot_idle()'s default and current CI usage).
    BootRegion region = BootRegion::Lim;
    if (const char* env_region = std::getenv("TT_METAL_TEST_BOOT_REGION");
        env_region != nullptr && *env_region != '\0') {
        std::string s = env_region;
        std::transform(
            s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "lim") {
            region = BootRegion::Lim;
        } else if (s == "dram") {
            region = BootRegion::Dram;
        } else {
            FAIL() << "TT_METAL_TEST_BOOT_REGION must be 'lim' or 'dram', got '" << env_region << "'";
        }
    }
    const SentinelLayout& sentinel = sentinel_for(region);

    log_info(tt::LogTest, "X280Boot using chip_id={} l2cpu_mask=0x{:x} region={}", chip_id, l2cpu_mask, sentinel.name);

    // Triggers MetalContext::initialize() (Cluster + UMD device-open, ARC
    // power state -> BUSY). No X280 work happens here — the runtime is
    // X280-agnostic after the experimental/x280 extraction.
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(
        chip_id, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);

    // Explicit X280 boot. Throws on sentinel timeout (with full diagnostic
    // log line); returns the number of L2CPUs that successfully booted.
    const std::size_t booted = tt::tt_metal::experimental::x280::boot_idle(chip_id, region, l2cpu_mask);
    log_info(
        tt::LogTest,
        "X280 boot_idle returned {} for chip {} (region={} mask=0x{:x})",
        booted,
        chip_id,
        sentinel.name,
        l2cpu_mask);

    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    auto enabled_l2cpus = soc_desc.get_cores(tt::CoreType::L2CPU, tt::CoordSystem::NOC0);
    ASSERT_GE(enabled_l2cpus.size(), 1u) << "Blackhole chip should expose >=1 L2CPU";

    // Derive the physical-index set of L2CPUs that the mask actually picked
    // out of the *enabled* set (mask is in physical-index space; not every
    // physical L2CPU is necessarily enabled on a harvested chip). This is
    // what boot_idle internally booted, so it's also the set whose sentinel
    // we should verify.
    std::uint8_t enabled_phys_mask = 0;
    for (const auto& tile : enabled_l2cpus) {
        for (std::size_t i = 0; i < kL2CpuTilesByPhysicalIndex.size(); ++i) {
            if (tile.x == kL2CpuTilesByPhysicalIndex[i].first && tile.y == kL2CpuTilesByPhysicalIndex[i].second) {
                enabled_phys_mask |= static_cast<std::uint8_t>(1u << i);
                break;
            }
        }
    }
    const std::uint8_t selected_mask = static_cast<std::uint8_t>(enabled_phys_mask & (l2cpu_mask & 0x0Fu));

    std::size_t expected_booted = 0;
    for (std::size_t i = 0; i < kL2CpuTilesByPhysicalIndex.size(); ++i) {
        if (selected_mask & (1u << i)) {
            ++expected_booted;
        }
    }
    EXPECT_EQ(booted, expected_booted) << "boot_idle reported " << booted << " booted L2CPUs, expected "
                                       << expected_booted << " (enabled_phys_mask=0x" << std::hex << +enabled_phys_mask
                                       << " l2cpu_mask=0x" << +l2cpu_mask << ")";

    // Verify the region-specific sentinel on every L2CPU that boot_idle
    // should have booted. Skipped L2CPUs (not selected by the mask, or
    // harvested) are not touched — neither sentinel-zeroed nor reset-
    // released — so we must not read them here.
    std::size_t passed = 0;
    for (std::size_t i = 0; i < kL2CpuTilesByPhysicalIndex.size(); ++i) {
        if (!(selected_mask & (1u << i))) {
            continue;
        }
        // Look up the umd::CoreCoord from enabled_l2cpus so we hand
        // read_sentinel the exact object type it expects.
        const auto& [tile_x, tile_y] = kL2CpuTilesByPhysicalIndex[i];
        const tt::umd::CoreCoord* tile_ptr = nullptr;
        for (const auto& tile : enabled_l2cpus) {
            if (tile.x == tile_x && tile.y == tile_y) {
                tile_ptr = &tile;
                break;
            }
        }
        ASSERT_NE(tile_ptr, nullptr) << "Internal test bug: physical L2CPU " << i << " (" << tile_x << "," << tile_y
                                     << ") is in selected_mask but missing from enabled_l2cpus";

        const uint64_t value = read_sentinel(mesh_device, chip_id, *tile_ptr, sentinel.addr);
        EXPECT_EQ(value, sentinel.value) << "Device " << chip_id << " L2CPU phys=" << i << " noc0=(" << tile_x << ","
                                         << tile_y << ") " << sentinel.name << " sentinel @ 0x" << std::hex
                                         << sentinel.addr << " mismatch: expected 0x" << sentinel.value << " got 0x"
                                         << value;
        if (value == sentinel.value) {
            ++passed;
        }
    }
    log_info(
        tt::LogTest,
        "Device {} X280 {} sentinel verified on {}/{} L2CPUs (mask=0x{:x})",
        chip_id,
        sentinel.name,
        passed,
        expected_booted,
        l2cpu_mask);

    ASSERT_TRUE(mesh_device->close());
}
