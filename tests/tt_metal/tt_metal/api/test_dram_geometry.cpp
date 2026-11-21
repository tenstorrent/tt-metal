// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The DRAM / worker geometry queries (tt-metalium/experimental/dram_geometry.hpp).

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/dram_geometry.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal {
namespace {

using experimental::DramBankTile;

// Every bank has a NOC0 and a NOC1 endpoint tile (the NOC0 one first), the tiles' coordinates agree with the
// device's, and worker NOC0 positions are distinct.
TEST_F(UnitMeshFastDispatchFixture, DramGeometry) {
    if (devices_.empty()) {
        GTEST_SKIP();
    }
    auto& mesh = *devices_.front();
    if (mesh.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole DRAM tile geometry";
    }
    const distributed::MeshCoordinate coord{0, 0};
    const uint32_t banks = mesh.dram_grid_size().x;
    ASSERT_GT(banks, 0u);
    for (uint32_t bank = 0; bank < banks; ++bank) {
        const std::vector<DramBankTile> tiles = experimental::GetDramBankTiles(mesh, coord, bank);
        ASSERT_FALSE(tiles.empty());
        uint32_t endpoints = 0;
        for (uint32_t i = 0; i < tiles.size(); ++i) {
            const DramBankTile& t = tiles[i];
            EXPECT_EQ(t.logical_core, CoreCoord(bank, i));
            EXPECT_EQ(t.virtual_core, mesh.virtual_core_from_logical_core(t.logical_core, CoreType::DRAM));
            endpoints |= t.endpoint_noc_mask;
        }
        EXPECT_TRUE(tiles[0].endpoint_noc_mask & 1u) << "bank " << bank;
        EXPECT_EQ(endpoints, 3u) << "bank " << bank;
    }
    const CoreCoord a = experimental::GetWorkerNoc0Core(mesh, coord, CoreCoord(0, 0));
    const CoreCoord b = experimental::GetWorkerNoc0Core(mesh, coord, CoreCoord(1, 0));
    EXPECT_NE(a, b);
}

}  // namespace
}  // namespace tt::tt_metal
