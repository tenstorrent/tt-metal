// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/host_api.hpp>
#include <string>
#include <unordered_set>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "umd/device/coordinate_manager.h"
#include "umd/device/types/arch.h"
#include <tt-metalium/utils.hpp>

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::soc_desc {
std::unordered_set<int> get_harvested_rows(chip_id_t device_id) {
    uint32_t harvested_rows_mask = CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).arch,
        tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id));
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
        uint32_t harvested_rows_mask =
            tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id);
        const metal_SocDescriptor& soc_desc =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
        log_info(LogTest, "Device {} harvesting mask {}", device_id, harvested_rows_mask);
        std::unordered_set<int> harvested_rows = unit_tests::basic::soc_desc::get_harvested_rows(device_id);
        auto tensix_harvest_axis = tt::tt_metal::MetalContext::instance().hal().get_tensix_harvest_axis();

        CoreCoord logical_grid_size = device->logical_grid_size();
        for (int x = 0; x < logical_grid_size.x; x++) {
            for (int y = 0; y < logical_grid_size.y; y++) {
                CoreCoord logical_core_coord(x, y);
                CoreCoord physical_core_coord = soc_desc.get_physical_tensix_core_from_logical(logical_core_coord);
                EXPECT_TRUE(
                    harvested_rows.find(
                        tensix_harvest_axis == HalTensixHarvestAxis::ROW
                            ? physical_core_coord.y
                            : physical_core_coord.x) == harvested_rows.end());
            }
        }

        tt_metal::CloseDevice(device);
    }
}

}  // namespace tt::tt_metal
