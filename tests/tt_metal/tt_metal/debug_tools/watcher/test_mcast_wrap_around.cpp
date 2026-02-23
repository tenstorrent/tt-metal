// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side validation tests for wrap-around multicast coordinates.
//
// Verifies that Watcher correctly validates wrap-around multicast coordinates for
// NoC torus architectures (WH/BH). On these architectures, multicasts can have
// end < start in a dimension, allowing wrapping around grid edges.
//
// These tests verify the validation logic accepts all wrap-around coordinate patterns.
// Complementary device-side tests verify actual hardware behavior.

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include "debug_tools_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Helper to perform host noc_multicast_write with given coordinates.
// Uses valid L1 address and small payload to only exercise the sanitize check.
void DoHostMcastWrite(ChipId chip_id, CoreCoord core_start, CoreCoord core_end, bool expect_throw) {
    const uint64_t l1_addr =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    uint32_t payload = 0xdeadbeef;
    const uint32_t sz = 4;

    if (expect_throw) {
        // Before fix: Watcher should reject wrap-around coordinates
        EXPECT_THROW(
            {
                MetalContext::instance().get_cluster().noc_multicast_write(
                    &payload, sz, chip_id, core_start, core_end, l1_addr);
            },
            std::runtime_error);
    } else {
        // After fix: Watcher should accept wrap-around coordinates
        EXPECT_NO_THROW(MetalContext::instance().get_cluster().noc_multicast_write(
            &payload, sz, chip_id, core_start, core_end, l1_addr));
    }
}

}  // namespace

// Test Y-wrap down: start at bottom row, end at top row (y_end < y_start)
TEST_F(MeshWatcherFixture, HostMcastWrapAroundY_Down) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->logical_grid_size();

    if (grid.y < 2) {
        GTEST_SKIP() << "Need grid.y >= 2 for Y-wrap test";
    }

    // Wrap down: start at bottom (high Y), end at top (low Y)
    // Convert logical to NOC coordinates
    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(0, grid.y - 1));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(0, 0));

    log_info(LogTest, "Testing Y-wrap down: start={}, end={}", core_start.str(), core_end.str());

    // After fix: Watcher should accept wrap-around for torus architectures (WH/BH)
    DoHostMcastWrite(chip_id, core_start, core_end, /*expect_throw=*/false);
}

// Test Y-wrap up: start at top row, end at bottom row (y_end > y_start, wraps up)
TEST_F(MeshWatcherFixture, HostMcastWrapAroundY_Up) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->logical_grid_size();

    if (grid.y < 2) {
        GTEST_SKIP() << "Need grid.y >= 2 for Y-wrap test";
    }

    // Wrap up: start at top (low Y), end at bottom (high Y)
    // This is the "normal" direction but tests the opposite wrap
    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(0, 0));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(0, grid.y - 1));

    log_info(LogTest, "Testing Y-wrap up: start={}, end={}", core_start.str(), core_end.str());

    // This should NOT throw even before fix (start < end), but include for completeness
    DoHostMcastWrite(chip_id, core_start, core_end, /*expect_throw=*/false);
}

// Test X-wrap right: start at right edge, end at left edge (x_end < x_start)
TEST_F(MeshWatcherFixture, HostMcastWrapAroundX_Right) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 for X-wrap test";
    }

    // Wrap right: start at right edge (high X), end at left edge (low X)
    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, 0));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(0, 0));

    log_info(LogTest, "Testing X-wrap right: start={}, end={}", core_start.str(), core_end.str());

    // After fix: Watcher should accept wrap-around for torus architectures (WH/BH)
    DoHostMcastWrite(chip_id, core_start, core_end, /*expect_throw=*/false);
}

// Test X-wrap left: start at left edge, end at right edge (x_end > x_start, wraps left)
TEST_F(MeshWatcherFixture, HostMcastWrapAroundX_Left) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 for X-wrap test";
    }

    // Wrap left: start at left edge (low X), end at right edge (high X)
    // This is the "normal" direction but tests the opposite wrap
    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(0, 0));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, 0));

    log_info(LogTest, "Testing X-wrap left: start={}, end={}", core_start.str(), core_end.str());

    // This should NOT throw even before fix (start < end), but include for completeness
    DoHostMcastWrite(chip_id, core_start, core_end, /*expect_throw=*/false);
}

// Test XY-wrap diagonal: both dimensions wrap (bottom-right to top-left)
TEST_F(MeshWatcherFixture, HostMcastWrapAroundXY_DownRight) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 2 || grid.y < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 and grid.y >= 2 for XY-wrap test";
    }

    // Wrap both: start at bottom-right, end at top-left
    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, grid.y - 1));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(0, 0));

    log_info(LogTest, "Testing XY-wrap down-right: start={}, end={}", core_start.str(), core_end.str());

    // After fix: Watcher should accept wrap-around for torus architectures (WH/BH)
    DoHostMcastWrite(chip_id, core_start, core_end, /*expect_throw=*/false);
}

// Test XY-wrap opposite diagonal: both dimensions wrap opposite direction (top-left to bottom-right)
TEST_F(MeshWatcherFixture, HostMcastWrapAroundXY_UpLeft) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 2 || grid.y < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 and grid.y >= 2 for XY-wrap test";
    }

    // Wrap both opposite: start at top-left, end at bottom-right
    // This is the "normal" direction but tests the opposite wrap
    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(0, 0));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, grid.y - 1));

    log_info(LogTest, "Testing XY-wrap up-left: start={}, end={}", core_start.str(), core_end.str());

    // This should NOT throw even before fix (start < end), but include for completeness
    DoHostMcastWrite(chip_id, core_start, core_end, /*expect_throw=*/false);
}
