// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <set>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "device_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"

namespace tt::tt_metal {

class DramSubchannelHelperFixture : public BlackholeSingleCardFixture {};

TEST_F(DramSubchannelHelperFixture, PicksUnreservedSubchannelPerBank) {
    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());

    const uint32_t num_banks = soc_desc.get_num_dram_views();
    const uint32_t num_subchannels = soc_desc.get_grid_size(tt::CoreType::DRAM).y;
    ASSERT_GT(num_banks, 0u);
    ASSERT_GT(num_subchannels, 1u);

    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        std::set<std::pair<size_t, size_t>> reserved;
        for (const auto& c : soc_desc.dram_view_worker_cores.at(bank)) {
            reserved.emplace(c.x, c.y);
        }
        for (const auto& c : soc_desc.dram_view_eth_cores.at(bank)) {
            reserved.emplace(c.x, c.y);
        }

        const size_t channel = soc_desc.get_channel_for_dram_view(static_cast<int>(bank));
        uint32_t expected_free = num_subchannels;
        for (uint32_t sub = 0; sub < num_subchannels; ++sub) {
            tt::umd::CoreCoord coord = soc_desc.get_dram_core_for_channel(
                static_cast<int>(channel), static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
            if (!reserved.contains({coord.x, coord.y})) {
                expected_free = sub;
                break;
            }
        }
        ASSERT_LT(expected_free, num_subchannels) << "Test setup error: no free subchannel for bank " << bank;

        const CoreCoord expected_logical =
            soc_desc.get_logical_dram_core_for_subchannel(static_cast<int>(bank), static_cast<int>(expected_free));
        const CoreCoord picked_logical = mesh_device->impl().pick_unused_dram_logical_core(bank);
        EXPECT_EQ(picked_logical, expected_logical) << "Mismatch for bank " << bank;

        tt::umd::CoreCoord picked_coord = soc_desc.get_dram_core_for_channel(
            static_cast<int>(channel), static_cast<int>(expected_free), tt::CoordSystem::TRANSLATED);
        EXPECT_FALSE(reserved.contains({picked_coord.x, picked_coord.y}))
            << "Picked logical core for bank " << bank << " collides with a worker/eth endpoint";
    }
}

TEST_F(DramSubchannelHelperFixture, RejectsOutOfRangeBank) {
    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const uint32_t num_banks = soc_desc.get_num_dram_views();
    EXPECT_ANY_THROW(mesh_device->impl().pick_unused_dram_logical_core(num_banks));
}

}  // namespace tt::tt_metal
