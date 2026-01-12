// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include "llrt/metal_soc_descriptor.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/coordinates/coordinate_manager.hpp>
#include <umd/device/types/arch.hpp>
#include "common/tt_backend_api_types.hpp"
#include <llrt/tt_cluster.hpp>

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
}  // namespace unit_tests::basic::soc_desc

namespace tt::tt_metal {

// This test ensures that no logical core maps to a harvested row
TEST(SOC, TensixValidateLogicalToPhysicalCoreCoordHostMapping) {
    size_t num_devices = tt_metal::GetNumAvailableDevices();
    ASSERT_TRUE(num_devices > 0);
    tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    num_devices = (arch == tt::ARCH::GRAYSKULL) ? 1 : num_devices;
    std::vector<int> devices_to_open;
    for (int device_id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
        devices_to_open.push_back(device_id);
    }
    auto devices = detail::CreateDevices(devices_to_open);
    for (int device_id = 0; device_id < num_devices; device_id++) {
        tt_metal::IDevice* device = devices[device_id];
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
                EXPECT_TRUE(!harvested_rows.contains(
                    tensix_harvest_axis == HalTensixHarvestAxis::ROW ? physical_core_coord.y : physical_core_coord.x));
            }
        }
    }

    tt::tt_metal::detail::CloseDevices(devices);
}

}  // namespace tt::tt_metal
