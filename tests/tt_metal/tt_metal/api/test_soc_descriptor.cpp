// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::soc_desc {
std::unordered_set<int> get_harvested_rows(chip_id_t device_id) {
    uint32_t harvested_rows_mask = CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        tt::Cluster::instance().get_soc_desc(device_id).arch, tt::Cluster::instance().get_harvesting_mask(device_id));
    std::unordered_set<int> harvested_rows;
    int row_coordinate = 0;
    int tmp = harvested_rows_mask;
    string delim = "";
    string harvested_row_str;
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
}  // namespace unit_tests::basic::soc_desc

namespace tt::tt_metal {

// This test ensures that no logical core maps to a harvested row
TEST(SOC, TensixValidateLogicalToPhysicalCoreCoordHostMapping) {
    size_t num_devices = tt_metal::GetNumAvailableDevices();
    ASSERT_TRUE(num_devices > 0);
    tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    num_devices = (arch == tt::ARCH::GRAYSKULL) ? 1 : num_devices;
    for (int device_id = 0; device_id < num_devices; device_id++) {
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);
        uint32_t harvested_rows_mask = tt::Cluster::instance().get_harvesting_mask(device_id);
        const metal_SocDescriptor& soc_desc = tt::Cluster::instance().get_soc_desc(device_id);
        log_info(LogTest, "Device {} harvesting mask {}", device_id, harvested_rows_mask);
        std::unordered_set<int> harvested_rows = unit_tests::basic::soc_desc::get_harvested_rows(device_id);

        CoreCoord logical_grid_size = device->logical_grid_size();
        for (int x = 0; x < logical_grid_size.x; x++) {
            for (int y = 0; y < logical_grid_size.y; y++) {
                CoreCoord logical_core_coord(x, y);
                CoreCoord physical_core_coord = soc_desc.get_physical_tensix_core_from_logical(logical_core_coord);
                ASSERT_TRUE(harvested_rows.find(physical_core_coord.y) == harvested_rows.end());
            }
        }

        tt_metal::CloseDevice(device);
    }
}

}  // namespace tt::tt_metal
