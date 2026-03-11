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
void DoHostMcastWrite(ChipId chip_id, CoreCoord core_start, CoreCoord core_end) {
    const uint64_t l1_addr =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    uint32_t payload = 0xdeadbeef;
    const uint32_t sz = 4;

    EXPECT_NO_THROW(MetalContext::instance().get_cluster().noc_multicast_write(
        &payload, sz, chip_id, core_start, core_end, l1_addr));
}

}  // namespace

TEST_F(MeshWatcherFixture, HostMcastWrapAroundY_Down) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.y < 2) {
        GTEST_SKIP() << "Need grid.y >= 2 for Y-wrap test";
    }

    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(0, grid.y - 1));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(0, 0));

    log_info(LogTest, "Testing Y-wrap down: start={}, end={}", core_start.str(), core_end.str());

    DoHostMcastWrite(chip_id, core_start, core_end);
}

TEST_F(MeshWatcherFixture, HostMcastWrapAroundY_Up) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.y < 2) {
        GTEST_SKIP() << "Need grid.y >= 2 for Y-wrap test";
    }

    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(0, 0));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(0, grid.y - 1));

    log_info(LogTest, "Testing Y-wrap up: start={}, end={}", core_start.str(), core_end.str());

    DoHostMcastWrite(chip_id, core_start, core_end);
}

TEST_F(MeshWatcherFixture, HostMcastWrapAroundX_Right) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.x < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 for X-wrap test";
    }

    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, 0));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(0, 0));

    log_info(LogTest, "Testing X-wrap right: start={}, end={}", core_start.str(), core_end.str());

    DoHostMcastWrite(chip_id, core_start, core_end);
}

TEST_F(MeshWatcherFixture, HostMcastWrapAroundX_Left) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.x < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 for X-wrap test";
    }

    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(0, 0));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, 0));

    log_info(LogTest, "Testing X-wrap left: start={}, end={}", core_start.str(), core_end.str());

    DoHostMcastWrite(chip_id, core_start, core_end);
}

TEST_F(MeshWatcherFixture, HostMcastWrapAroundXY_DownRight) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.x < 2 || grid.y < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 and grid.y >= 2 for XY-wrap test";
    }

    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, grid.y - 1));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(0, 0));

    log_info(LogTest, "Testing XY-wrap down-right: start={}, end={}", core_start.str(), core_end.str());

    DoHostMcastWrite(chip_id, core_start, core_end);
}

TEST_F(MeshWatcherFixture, HostMcastWrapAroundXY_UpLeft) {
    auto* device = this->devices_[0]->get_devices()[0];
    ChipId chip_id = device->id();
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.x < 2 || grid.y < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 and grid.y >= 2 for XY-wrap test";
    }

    CoreCoord core_start = device->worker_core_from_logical_core(CoreCoord(0, 0));
    CoreCoord core_end = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, grid.y - 1));

    log_info(LogTest, "Testing XY-wrap up-left: start={}, end={}", core_start.str(), core_end.str());

    DoHostMcastWrite(chip_id, core_start, core_end);
}
