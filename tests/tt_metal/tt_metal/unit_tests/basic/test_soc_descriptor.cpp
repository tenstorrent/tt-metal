// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "basic_fixture.hpp"
#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::soc_desc {
    std::unordered_set<int> get_harvested_rows(chip_id_t device_id) {
        uint32_t harvested_rows_mask = tt::Cluster::instance().get_harvested_rows(device_id);
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
        log_info(LogTest, "Device {} has {} harvested rows. Physical harvested row coordinates are: {}", device_id, harvested_rows.size(), harvested_row_str);
        return harvested_rows;
    }
}


// This test ensures that no logical core maps to a harvested row
TEST_F(BasicFixture, ValidateLogicalToPhysicalCoreCoordHostMapping) {
    size_t num_devices = tt_metal::GetNumAvailableDevices();
    ASSERT_TRUE(num_devices > 0);
    tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    num_devices = (arch == tt::ARCH::GRAYSKULL) ? 1 : num_devices;
    for (int device_id = 0; device_id < num_devices; device_id++) {
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        uint32_t harvested_rows_mask = tt::Cluster::instance().get_harvested_rows(device_id);
        log_info(LogTest, "Device {} harvesting mask {}", device_id, harvested_rows_mask);
        std::unordered_set<int> harvested_rows = unit_tests::basic::soc_desc::get_harvested_rows(device_id);

        CoreCoord logical_grid_size = device->logical_grid_size();
        for (int x = 0; x < logical_grid_size.x; x++) {
            for (int y = 0; y < logical_grid_size.y; y++) {
                CoreCoord logical_core_coord(x, y);
                CoreCoord physical_core_coord = device->worker_core_from_logical_core(logical_core_coord);
                ASSERT_TRUE(harvested_rows.find(physical_core_coord.y) == harvested_rows.end());
            }
        }

        tt_metal::CloseDevice(device);
    }
}

TEST_F(DeviceFixture, ValidateMetalSocDescriptors) {
    for (chip_id_t device_id = 0; device_id < this->num_devices_; device_id++) {
        const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(device_id);

        EXPECT_EQ(soc_desc.physical_cores.size(), soc_desc.cores.size());
        EXPECT_EQ(soc_desc.physical_workers.size(), soc_desc.workers.size());
        EXPECT_EQ(soc_desc.physical_harvested_workers.size(), soc_desc.harvested_workers.size());

        // Ensure that only tensix workers are remapped to virtual coordinates
        for (const auto &[physical_core, core_desc] : soc_desc.physical_cores) {
            if (core_desc.type != CoreType::WORKER and core_desc.type != CoreType::HARVESTED) {
                ASSERT_TRUE(soc_desc.cores.find(physical_core) != soc_desc.cores.end());
                EXPECT_EQ(core_desc.type, soc_desc.cores.at(physical_core).type);
                tt_cxy_pair physical_chip_core(device_id, physical_core);
                tt_cxy_pair umd_chip_core = soc_desc.convert_to_umd_coordinates(physical_chip_core);
                EXPECT_EQ(physical_chip_core, umd_chip_core);
            }
        }

        // Ensure the correct cores are marked as harvested
        std::unordered_set<int> harvested_rows = unit_tests::basic::soc_desc::get_harvested_rows(device_id);
        for (const CoreCoord &physical_harvested_core : soc_desc.physical_harvested_workers) {
            ASSERT_TRUE(harvested_rows.find(physical_harvested_core.y) != harvested_rows.end());
            tt_cxy_pair physical_chip_core(device_id, physical_harvested_core);
            tt_cxy_pair umd_chip_core = soc_desc.convert_to_umd_coordinates(physical_chip_core);
            CoreCoord umd_harvested_core(umd_chip_core.x, umd_chip_core.y);
            bool found_harvested_core = std::find(soc_desc.harvested_workers.begin(), soc_desc.harvested_workers.end(), umd_harvested_core) != soc_desc.harvested_workers.end();
            ASSERT_TRUE(found_harvested_core);
            EXPECT_EQ(soc_desc.physical_cores.at(physical_harvested_core).type, CoreType::HARVESTED);
            EXPECT_EQ(soc_desc.cores.at(umd_harvested_core).type, CoreType::HARVESTED);
        }

        // Ensure the correct cores are marked as worker cores
        for (const CoreCoord &physical_worker_core : soc_desc.physical_workers) {
            ASSERT_TRUE(harvested_rows.find(physical_worker_core.y) == harvested_rows.end());
            tt_cxy_pair physical_chip_core(device_id, physical_worker_core);
            tt_cxy_pair umd_chip_core = soc_desc.convert_to_umd_coordinates(physical_chip_core);
            CoreCoord umd_worker_core(umd_chip_core.x, umd_chip_core.y);
            bool found_worker_core = std::find(soc_desc.workers.begin(), soc_desc.workers.end(), umd_worker_core) != soc_desc.workers.end();
            ASSERT_TRUE(found_worker_core);
            EXPECT_EQ(soc_desc.physical_cores.at(physical_worker_core).type, CoreType::WORKER);
            EXPECT_EQ(soc_desc.cores.at(umd_worker_core).type, CoreType::WORKER);
        }
    }
}
