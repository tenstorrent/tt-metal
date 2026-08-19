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
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device->get_device_ids()[0]);

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

        // Logical/compacted channel — get_dram_core_for_channel indexes the compacted DRAM grid,
        // so on a harvested board this must match what pick_unused_dram_logical_core uses (passing
        // the raw physical channel here indexes the wrong core and the harvested grid throws).
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
        const CoreCoord picked_logical =
            mesh_device->impl().pick_unused_dram_logical_core(mesh_device->get_devices()[0], bank);
        EXPECT_EQ(picked_logical, expected_logical) << "Mismatch for bank " << bank;

        tt::umd::CoreCoord picked_coord = soc_desc.get_dram_core_for_channel(
            static_cast<int>(channel), static_cast<int>(expected_free), tt::CoordSystem::TRANSLATED);
        EXPECT_FALSE(reserved.contains({picked_coord.x, picked_coord.y}))
            << "Picked logical core for bank " << bank << " collides with a worker/eth endpoint";
    }
}

// A DRAM sender mapping is resolved against one device and then reused for the whole mesh, which
// only holds while a logical y names the same endpoint role on every bank and every device. That is
// the property to hold the descriptor to -- not the order it happens to build the table in: y=0 the
// syseng-owned NOC0 worker endpoint, y=1 the NOC1 worker endpoint, and the free subchannels a
// sender may actually run on above both. dram_sender_logical_cores enforces the same invariant at
// runtime.
TEST_F(DramSubchannelHelperFixture, LogicalSubchannelOrderFollowsEndpointRole) {
    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());

    const uint32_t num_banks = soc_desc.get_num_dram_views();
    const uint32_t num_subchannels = soc_desc.get_grid_size(tt::CoreType::DRAM).y;
    ASSERT_GT(num_banks, 0u);
    ASSERT_GT(num_subchannels, 2u) << "a bank needs a subchannel beyond its two worker endpoints to host a sender";

    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        SCOPED_TRACE(fmt::format("bank {}", bank));
        const CoreCoord noc0_endpoint = soc_desc.get_preferred_worker_core_for_dram_view(static_cast<int>(bank), 0);
        const CoreCoord noc1_endpoint = soc_desc.get_preferred_worker_core_for_dram_view(static_cast<int>(bank), 1);
        const auto& endpoints = soc_desc.dram_bank_endpoint_coords.at(bank);

        // The table renames this bank's subchannels; it must not drop, duplicate, or invent one.
        const size_t channel = soc_desc.get_channel_for_dram_view(static_cast<int>(bank));
        std::set<std::pair<size_t, size_t>> subchannel_coords;
        for (uint32_t sub = 0; sub < num_subchannels; ++sub) {
            const tt::umd::CoreCoord coord = soc_desc.get_dram_core_for_channel(
                static_cast<int>(channel), static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
            subchannel_coords.emplace(coord.x, coord.y);
        }
        std::set<std::pair<size_t, size_t>> table_coords;
        for (const CoreCoord& coord : endpoints) {
            table_coords.emplace(coord.x, coord.y);
        }
        ASSERT_EQ(endpoints.size(), num_subchannels);
        EXPECT_EQ(table_coords, subchannel_coords) << "the table is not a permutation of the bank's subchannels";

        // Roles sit at fixed logical y. The NOC0/NOC1 split is what keeps them there: a descriptor
        // naming one subchannel for both NOCs would let the free and NOC1 roles trade places
        // between devices, and a mesh-wide sender mapping would then drive the wrong DRISC core.
        ASSERT_NE(noc1_endpoint, noc0_endpoint) << "bank's NOC0 and NOC1 worker endpoints must be distinct";
        EXPECT_EQ(endpoints.at(0), noc0_endpoint) << "logical y=0 must be the NOC0 worker endpoint";
        EXPECT_EQ(endpoints.at(1), noc1_endpoint) << "logical y=1 must be the NOC1 worker endpoint";

        // A sender only ever runs on a subchannel above the endpoints, on every bank alike.
        const CoreCoord free_logical = mesh_device->impl().pick_unused_dram_logical_core(device, bank);
        EXPECT_EQ(free_logical.x, bank);
        EXPECT_GE(free_logical.y, 2u) << "the free subchannel collides with a worker endpoint role";

        // The two sender roles the prefetcher provisions are exactly [free, NOC1 endpoint].
        const std::vector<CoreCoord> senders = mesh_device->impl().dram_sender_logical_cores(device, bank);
        EXPECT_EQ(senders, (std::vector<CoreCoord>{free_logical, CoreCoord(bank, 1)}));
    }
}

// get_metal_dram_cores(LOGICAL) must name the same cores as get_metal_dram_cores(TRANSLATED), which is
// the set firmware init and watcher's mailbox init write to. Resolving the logical coords back through
// the same path a caller uses is what catches a coordinate-space mismatch: UMD's logical DRAM coord is
// {channel, raw subchannel} while Metal's is {dram_view, dram_bank_endpoint_coords index}, and the
// table orders each view's NOC0 worker endpoint first, so returning the UMD coord resolved onto the
// syseng-owned NOC0 endpoint for every view whose worker_endpoint[0] is not subchannel 0.
TEST_F(DramSubchannelHelperFixture, MetalDramCoresLogicalResolvesToTranslatedSet) {
    auto mesh_device = devices_[0];
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device->get_device_ids()[0]);
    const auto& cluster = MetalContext::instance().get_cluster();

    const auto translated_cores = soc_desc.get_metal_dram_cores(tt::CoordSystem::TRANSLATED);
    const auto logical_cores = soc_desc.get_metal_dram_cores(tt::CoordSystem::LOGICAL);
    ASSERT_FALSE(translated_cores.empty());
    ASSERT_EQ(logical_cores.size(), translated_cores.size());

    // Every DRAM view's NOC0 worker endpoint is syseng-owned on Blackhole and runs no DRISC firmware,
    // so no returned core may land on one.
    std::set<std::pair<size_t, size_t>> noc0_endpoints;
    for (uint32_t view = 0; view < soc_desc.get_num_dram_views(); ++view) {
        const auto& noc0_endpoint = soc_desc.dram_view_worker_cores.at(view).at(0);
        noc0_endpoints.emplace(noc0_endpoint.x, noc0_endpoint.y);
    }

    std::set<std::pair<size_t, size_t>> expected;
    for (const auto& c : translated_cores) {
        expected.emplace(c.x, c.y);
        EXPECT_FALSE(noc0_endpoints.contains({c.x, c.y}))
            << "TRANSLATED core (" << c.x << ", " << c.y << ") is a NOC0 worker endpoint";
    }

    std::set<std::pair<size_t, size_t>> resolved;
    for (const auto& logical_core : logical_cores) {
        // The conversion watcher and any other logical-coord consumer goes through.
        const CoreCoord virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
            mesh_device->get_device_ids()[0], logical_core, CoreType::DRAM);
        EXPECT_FALSE(noc0_endpoints.contains({virtual_core.x, virtual_core.y}))
            << "LOGICAL core " << logical_core.str() << " resolved to NOC0 worker endpoint (" << virtual_core.x << ", "
            << virtual_core.y << ")";
        resolved.emplace(virtual_core.x, virtual_core.y);
    }

    EXPECT_EQ(resolved, expected) << "LOGICAL and TRANSLATED requests named different DRAM cores";
    // No two logical coords may collapse onto one core, or a core would go unvisited.
    EXPECT_EQ(resolved.size(), logical_cores.size()) << "LOGICAL DRAM coords are not distinct after resolution";
}

TEST_F(DramSubchannelHelperFixture, RejectsOutOfRangeBank) {
    auto mesh_device = devices_[0];
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device->get_device_ids()[0]);
    const uint32_t num_banks = soc_desc.get_num_dram_views();
    EXPECT_ANY_THROW(mesh_device->impl().pick_unused_dram_logical_core(mesh_device->get_devices()[0], num_banks));
}

}  // namespace tt::tt_metal
