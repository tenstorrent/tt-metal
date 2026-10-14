// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/session/session.hpp -- the Metal + X280Runtime + PLL fixture the
// host cases that open an X280 runtime reproduce.
//
// A characterization case's SetUp is nearly identical across those
// cases:
//   1) if the Metal watcher is enabled, skip;
//   2) create the case's MeshDevice: a unit mesh on chip 0 unless it names
//      its chips;
//   3) open an X280Runtime with LimEccPrimingPolicy::RequirePrepared;
//   4) pick the first enabled tile;
//   5) boot resident idle on every chip whose L2CPUs are all held in reset,
//      then require the tile to be in live resident idle;
//   6) pin the chip-global X280 PLL to `X280_PLL_MHZ` (default 1750 MHz) on
//      every chip with a tile, and confirm the ARC telemetry echoes it.
//
// This header owns exactly that fixture. It does not carry any suite-
// specific state; per-case regions, L1 buffers, and args blocks are the
// caller's business.

#ifndef X280_HARNESS_SESSION_HPP
#define X280_HARNESS_SESSION_HPP

#include "x280/device/x280_runtime.hpp"
#include "harness/artifacts/artifacts.hpp"
#include "harness/targets/x280_tile.hpp"
#include "harness/metal_context.hpp"

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace x280::harness::session {

namespace ttmd = tt::tt_metal::distributed;

/**
 * One canonical read of X280_PLL_MHZ: 1750 when unset. Throws
 * std::invalid_argument naming the value when it is outside the
 * characterization set {200, 800, 1000, 1750}, and env_u32's
 * std::runtime_error when it is not a decimal.
 */
inline uint32_t x280_pll_mhz_from_env() {
    const uint32_t v = artifacts::env_u32("X280_PLL_MHZ", 1750u);
    switch (v) {
        case 200u:
        case 800u:
        case 1000u:
        case 1750u: return v;
        default: break;
    }
    throw std::invalid_argument(
        "X280_PLL_MHZ=" + std::to_string(v) + " is not a supported X280 PLL frequency; use 200, 800, 1000 or 1750");
}

// A configuration for `open_or_skip`, so every case configures the same
// set of switches by name rather than by argument position.
struct OpenConfig {
    // The case's X280 image: the variable that overrides it and its file in
    // the build's image directory (artifacts::image_path). Empty for cases
    // that ship no X280 image (transport_bw tensix->x280). `open_or_skip`
    // does not read them.
    const char* image_env_var = "";
    const char* image_file = "";

    // The case's Metal kernel source directory; suites resolve their kernel
    // paths and JIT include paths from it. `open_or_skip` does not read it.
    std::filesystem::path kernel_dir;

    // Required executables always use the prepared MEM-03 profile; the rest
    // run at the reset shape. WayEnable cannot decrease without an ASIC reset,
    // so a chip provisioned for the profile refuses the rest.
    bool prepared_suite_requested = false;

    // Selected component memory layout profile. Default is Legacy.
    x280::LimLayout lim_layout = x280::LimLayout::Legacy;

    // Physical chip ids of the mesh, opened as one 1 x N row in this order.
    // Empty opens the unit mesh on chip 0.
    std::vector<int> mesh_device_ids;
};

// The living fixture. `open_or_skip` populates every field; TearDown
// resets them in the order the underlying types require (tile pointer is
// non-owning, runtime is unique_ptr, mesh is shared_ptr).
struct Fixture {
    std::shared_ptr<ttmd::MeshDevice> mesh;
    std::unique_ptr<x280::X280Runtime> runtime;
    x280::X280Tile* tile = nullptr;
    x280::TileId tile_id{};
    /** Each chip's first tile in live resident idle, in tile_ids() order:
     * the tile whose PLL was set and confirmed on that chip. `tile_id` is
     * chip 0's. */
    std::vector<x280::TileId> chip_tiles;
    /** The tiles this open booted from reset-held, in tile_ids() order; empty
     * when every chip already had a tile out of reset. */
    std::vector<x280::TileId> booted;

    uint32_t pll_mhz = 0;

    /** The session's tile on `chip`, or nullptr when the chip has none. */
    const x280::TileId* tile_for_chip(int chip) const {
        for (const auto& id : chip_tiles) {
            if (id.chip_id == chip) {
                return &id;
            }
        }
        return nullptr;
    }

    targets::X280TileHandle handle() const { return targets::X280TileHandle{runtime.get(), tile, tile_id}; }

    // `id` must name a live resident-idle member of runtime->tile_ids().
    targets::X280TileHandle handle_for(const x280::TileId& id) const {
        return targets::X280TileHandle{runtime.get(), &runtime->tile(id), id};
    }

    // TearDown-time reset; safe to call on a partially-opened fixture.
    void close() {
        tile = nullptr;
        runtime.reset();
        mesh.reset();
    }
};

/**
 * Boots resident idle on every chip of `ids` whose enabled L2CPUs are all held
 * in reset: one X280Runtime::boot_tiles batch per chip, with every enabled tile
 * of that chip, because boot_tiles admits a batch only while all of them are
 * held and only one per chip-reset lifetime. A chip with any tile out of reset
 * is left as it is. The image is artifacts::image_path("X280_IDLE_FW_IMAGE",
 * "firmware-lim-idle.bin"). Reset release is one-way for the chip-reset
 * lifetime and leaves PLL4 at the safe release frequency.
 *
 * Appends each tile of a batch that came up ready to `booted`. Records against
 * the running test: a skip when a tile needs an ASIC reset first, a fatal
 * failure for any other refusal. The caller checks IsSkipped() and
 * HasFatalFailure().
 */
inline void boot_held_chips(
    x280::X280Runtime& runtime, const std::vector<x280::TileId>& ids, std::vector<x280::TileId>& booted) {
    std::vector<int> chips;
    for (const auto& id : ids) {
        if (std::find(chips.begin(), chips.end(), id.chip_id) == chips.end()) {
            chips.push_back(id.chip_id);
        }
    }
    for (const int chip : chips) {
        x280::MultiTileBootRequest request;
        bool all_held = true;
        for (const auto& id : ids) {
            if (id.chip_id != chip) {
                continue;
            }
            const auto state = runtime.tile(id).inspect();
            ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
            all_held = all_held && !state.reset_released;
            request.tiles.push_back(id);
        }
        if (!all_held) {
            continue;
        }
        request.idle_image_path = artifacts::image_path("X280_IDLE_FW_IMAGE", "firmware-lim-idle.bin");
        request.allow_reset_release = true;
        const auto result = runtime.boot_tiles(request);
        if (result.ready()) {
            booted.insert(booted.end(), request.tiles.begin(), request.tiles.end());
            continue;
        }
        for (const auto& report : result.tiles) {
            if (x280::reset_required(report.result.outcome)) {
                GTEST_SKIP() << "resident idle cannot boot on chip " << chip
                             << " until an ASIC reset (tt-smi -r <device>): " << report.result.diagnostic;
            }
        }
        FAIL() << "resident idle boot failed on chip " << chip << ": " << result.diagnostic;
    }
}

// The whole SetUp pattern in one call. Emits GTEST_SKIP() / ASSERT_* into
// the calling test case exactly the way the pre-harness code did. Returns
// void because GTEST_SKIP() expands to `return void_expr;` and cannot sit
// in a bool-returning helper. Callers inspect ::testing::Test::IsSkipped()
// / HasFatalFailure() if they need to short-circuit follow-up work; the
// GTest dispatcher already does that check at the top of TEST_P bodies.
inline void open_or_skip(Fixture& f, const OpenConfig& cfg) {
    if (x280::harness::watcher_enabled()) {
        GTEST_SKIP() << "Metal watcher admits no L2CPU coordinate; run with watcher disabled";
    }

    if (cfg.mesh_device_ids.empty()) {
        f.mesh = ttmd::MeshDevice::create_unit_mesh(0);
    } else {
        f.mesh = ttmd::MeshDevice::create(ttmd::MeshDeviceConfig(
            ttmd::MeshShape(1u, static_cast<uint32_t>(cfg.mesh_device_ids.size())),
            std::nullopt,
            std::vector<tt::ChipId>(cfg.mesh_device_ids.begin(), cfg.mesh_device_ids.end())));
    }
    x280::RuntimeOpenConfig runtime_config{
        .priming_policy = x280::LimEccPrimingPolicy::RequirePrepared,
        .lim_layout = cfg.lim_layout,
    };
    if (cfg.prepared_suite_requested || cfg.lim_layout == x280::LimLayout::PacketInjection) {
        runtime_config.lim_shape = x280::LimShape{.lim_ways = x280::kPreparedLimWays};
        runtime_config.shape_change_policy = x280::LimShapeChangePolicy::RequireExact;
    }
    auto opened = x280::X280Runtime::open(f.mesh, runtime_config);
    if (!opened.runtime) {
        std::string why = opened.diagnostic;
        if (opened.requires_reset()) {
            why += "; reset the device (tt-smi -r <device>)";
        }
        if (opened.requires_preparation() || opened.requires_reset()) {
            why +=
                "; an unprimed device is primed once by X280_ALLOW_LIM_ECC_PRIME=1 x280_prime_test, then "
                "tt-smi -r <device>";
        }
        GTEST_SKIP() << "prepared X280 runtime unavailable: " << why;
    }
    f.runtime = std::make_unique<x280::X280Runtime>(std::move(*opened.runtime));

    const auto ids = f.runtime->tile_ids();
    if (ids.empty()) {
        GTEST_SKIP() << "no enabled X280 tiles";
    }
    boot_held_chips(*f.runtime, ids, f.booted);
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
        return;
    }
    f.tile_id = ids.front();
    f.tile = &f.runtime->tile(f.tile_id);

    // Check liveness before moving the chip-global PLL; see the PLL4 ownership
    // contract in x280/device/x280_runtime.hpp.
    const auto state = f.tile->inspect();
    ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
    if (!state.resident_idle_live) {
        GTEST_SKIP() << "requires a tile in live resident idle; chip " << f.tile_id.chip_id
                     << " is partly out of reset, so none of its L2CPUs can boot until an ASIC reset "
                        "(tt-smi -r <device>)";
    }

    // Every tile out of reset must be in resident idle; reset-held tiles are
    // not participants.
    for (const auto& other_id : ids) {
        if (other_id == f.tile_id) {
            continue;
        }
        const auto other = f.runtime->tile(other_id).inspect();
        ASSERT_TRUE(other.observation_valid) << other.invalid_reason;
        if (other.reset_released && !other.resident_idle_live) {
            GTEST_SKIP() << "L2CPU " << other_id.hw_index << " is out of reset without a live resident idle";
        }
    }

    f.pll_mhz = x280_pll_mhz_from_env();

    // Loud failure rather than a silent mis-measurement: a wedged CMFW
    // or a stray device opener surfaces here.
    // One live resident-idle tile per chip; ids.front() is chip 0's, checked
    // above. A chip with tiles but none in resident idle cannot participate.
    for (const auto& id : ids) {
        if (f.tile_for_chip(id.chip_id) == nullptr && f.runtime->tile(id).inspect().resident_idle_live) {
            f.chip_tiles.push_back(id);
        }
    }
    for (const auto& id : ids) {
        if (f.tile_for_chip(id.chip_id) == nullptr) {
            GTEST_SKIP() << "chip " << id.chip_id << " has no L2CPU in live resident idle";
        }
    }
    for (const auto& first : f.chip_tiles) {
        ASSERT_NO_THROW(f.runtime->set_pll(first.chip_id, f.pll_mhz, std::chrono::milliseconds{4000}))
            << "could not establish X280 PLL " << f.pll_mhz << " MHz on chip " << first.chip_id;
        const auto pll_confirm = f.runtime->tile(first).inspect();
        ASSERT_TRUE(pll_confirm.observation_valid) << pll_confirm.invalid_reason;
        ASSERT_TRUE(pll_confirm.pll_mhz.has_value())
            << "ARC L2CPUCLK telemetry unavailable after set_pll on chip " << first.chip_id;
        ASSERT_EQ(*pll_confirm.pll_mhz, f.pll_mhz)
            << "ARC telemetry reports " << *pll_confirm.pll_mhz << " MHz on chip " << first.chip_id
            << " after asking for " << f.pll_mhz;
    }
}

}  // namespace x280::harness::session

#endif  // X280_HARNESS_SESSION_HPP
