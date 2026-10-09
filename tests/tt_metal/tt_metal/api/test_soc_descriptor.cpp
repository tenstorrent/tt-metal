// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <string>
#include <unordered_set>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/coordinates/coordinate_manager.hpp>
#include <umd/device/soc_arch_descriptor.hpp>
#include <umd/device/types/arch.hpp>
#include "common/tt_backend_api_types.hpp"
#include <llrt/rtoptions.hpp>
#include <llrt/tt_cluster.hpp>
#include <filesystem>
#include <memory>
#include <tuple>
#include <vector>

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::soc_desc {
std::unordered_set<int> get_harvested_rows(ChipId device_id) {
    uint32_t harvested_rows_mask = tt::umd::CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).arch,
        tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id));
    std::unordered_set<int> harvested_rows;
    int row_coordinate = 0;
    int tmp = harvested_rows_mask;
    std::string delim;
    std::string harvested_row_str;
    while (tmp) {
        if (tmp & 1) {
            harvested_rows.insert(row_coordinate);
            harvested_row_str += delim + std::to_string(row_coordinate);
            delim = ", ";
        }
        tmp = tmp >> 1;
        row_coordinate++;
    }
    log_info(
        LogTest,
        "Device {} has {} harvested rows. Physical harvested row coordinates are: {}",
        device_id,
        harvested_rows.size(),
        harvested_row_str);
    return harvested_rows;
}

// A Blackhole SOC descriptor built from the checked-in YAML, with NOC translation on as on silicon.
metal_SocDescriptor make_blackhole_soc_desc(size_t dram_harvesting_mask) {
    // Silicon with all 14 ETH channels always harvests two of them, one of 4-6 and one of 7-9 (UMD
    // rejects any other count). Which two doesn't move any DRAM core.
    constexpr size_t eth_harvesting_mask = (1u << 6) | (1u << 9);
    const tt::llrt::RunTimeOptions rtoptions;
    const std::string soc_yaml =
        (std::filesystem::path(rtoptions.get_root_dir()) / "tt_metal/soc_descriptors/blackhole_140_arch.yaml").string();
    tt::umd::SocDescriptor umd_soc(
        std::make_shared<tt::umd::SocArchDescriptor>(soc_yaml),
        {.noc_translation_enabled = true,
         .harvesting_masks = {.dram_harvesting_mask = dram_harvesting_mask, .eth_harvesting_mask = eth_harvesting_mask},
         .board_type = tt::BoardType::P100});
    umd_soc.device_descriptor_file_path = soc_yaml;
    return metal_SocDescriptor(umd_soc, tt::BoardType::P100);
}

// Every translated core a DRAM view names: per view, its worker and eth endpoints on each NOC, then
// the cores its logical DRAM coords resolve to.
std::vector<tt::tt_metal::CoreCoord> dram_view_cores(const metal_SocDescriptor& soc) {
    std::vector<tt::tt_metal::CoreCoord> cores;
    for (int view = 0; view < static_cast<int>(soc.get_num_dram_views()); ++view) {
        for (uint8_t noc = 0; noc < 2; ++noc) {
            cores.push_back(soc.get_preferred_worker_core_for_dram_view(view, noc));
            cores.push_back(soc.get_preferred_eth_core_for_dram_view(view, noc));
        }
        const auto& bank_cores = soc.dram_bank_endpoint_coords.at(view);
        cores.insert(cores.end(), bank_cores.begin(), bank_cores.end());
    }
    return cores;
}
}  // namespace unit_tests::basic::soc_desc

namespace tt::tt_metal {

// This test ensures that no logical core maps to a harvested row
TEST_F(AnyDispatchMeshDeviceFixture, TensixValidateLogicalToPhysicalCoreCoordHostMapping) {
    for (const auto& mesh_device : this->devices_) {
        const auto device_id = mesh_device->get_device_ids()[0];
        uint32_t harvested_rows_mask =
            tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id);
        const metal_SocDescriptor& soc_desc =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
        log_info(LogTest, "Device {} harvesting mask {}", device_id, harvested_rows_mask);
        std::unordered_set<int> harvested_rows = unit_tests::basic::soc_desc::get_harvested_rows(device_id);
        auto tensix_harvest_axis = tt::tt_metal::MetalContext::instance().hal().get_tensix_harvest_axis();

        CoreCoord logical_grid_size = mesh_device->logical_grid_size();
        for (int x = 0; x < logical_grid_size.x; x++) {
            for (int y = 0; y < logical_grid_size.y; y++) {
                CoreCoord logical_core_coord(x, y);
                CoreCoord physical_core_coord = soc_desc.get_physical_tensix_core_from_logical(logical_core_coord);
                EXPECT_TRUE(!harvested_rows.contains(
                    tensix_harvest_axis == HalTensixHarvestAxis::ROW ? physical_core_coord.y : physical_core_coord.x));
            }
        }
    }
}

// Without harvesting, view k is channel k with the endpoints dram_views gives it: NOC0 on subchannel 0
// for channels 1-3 and 2 for the rest, NOC1 on subchannel 1, both on that channel's translated column.
TEST(DramViewMapping, BlackholeUnharvestedViewsFollowDramViews) {
    const metal_SocDescriptor soc = unit_tests::basic::soc_desc::make_blackhole_soc_desc(0);
    const uint32_t num_channels = soc.get_grid_size(CoreType::DRAM).x;
    ASSERT_EQ(soc.get_num_dram_views(), num_channels);
    for (uint32_t channel = 0; channel < num_channels; ++channel) {
        EXPECT_EQ(soc.get_channel_for_dram_view(channel), channel);
        const auto first = soc.get_dram_core_for_channel(channel, 0, CoordSystem::TRANSLATED);
        const uint32_t noc0_subchannel = (channel >= 1 && channel <= 3) ? 0 : 2;
        EXPECT_EQ(
            soc.get_preferred_worker_core_for_dram_view(channel, 0), CoreCoord(first.x, first.y + noc0_subchannel))
            << "channel " << channel;
        EXPECT_EQ(soc.get_preferred_worker_core_for_dram_view(channel, 1), CoreCoord(first.x, first.y + 1))
            << "channel " << channel;
    }
}

// With one DRAM channel harvested, the views follow harvested_dram_views in translated order, so
// view k names the same translated cores whichever channel is harvested.
TEST(DramViewMapping, BlackholeHarvestedViewsNameTheSameCoresWhicheverChannelIsHarvested) {
    const uint32_t num_channels = unit_tests::basic::soc_desc::make_blackhole_soc_desc(0).get_num_dram_views();
    const metal_SocDescriptor reference_soc = unit_tests::basic::soc_desc::make_blackhole_soc_desc(0b1);
    ASSERT_EQ(reference_soc.get_num_dram_views(), num_channels - 1);
    for (uint32_t view = 1; view < reference_soc.get_num_dram_views(); ++view) {
        const CoreCoord prev = reference_soc.get_preferred_worker_core_for_dram_view(view - 1, 0);
        const CoreCoord cur = reference_soc.get_preferred_worker_core_for_dram_view(view, 0);
        EXPECT_LT(std::tie(prev.x, prev.y), std::tie(cur.x, cur.y)) << "views out of translated order at " << view;
    }

    const auto reference = unit_tests::basic::soc_desc::dram_view_cores(reference_soc);
    for (uint32_t harvested = 1; harvested < num_channels; ++harvested) {
        EXPECT_EQ(
            unit_tests::basic::soc_desc::dram_view_cores(
                unit_tests::basic::soc_desc::make_blackhole_soc_desc(1u << harvested)),
            reference)
            << "harvested channel " << harvested;
    }
}

}  // namespace tt::tt_metal
