// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "basic_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::test_noc {

const uint32_t init_value = 0x1234B33F;

uint32_t read_reg (Device* device, CoreCoord logical_node, uint32_t reg_addr) {
    // Read and return reg value form reading
    uint32_t reg_data = unit_tests::basic::test_noc::init_value;
    tt_metal::detail::ReadRegFromDevice(device, logical_node, reg_addr, reg_data);
    return reg_data;
}

void read_translation_table (Device* device, CoreCoord logical_node, std::vector<unsigned int>& x_remap, std::vector<unsigned int>& y_remap) {
#ifdef NOC_X_ID_TRANSLATE_TABLE_0
    std::vector<uint32_t> x_reg_addrs = {
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_0),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_1),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_2),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_3)
    };
    std::vector<uint32_t> y_reg_addrs = {
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_0),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_1),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_2),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_3)
    };
    x_remap.clear();
    for (const auto& reg_addr : x_reg_addrs) {
        auto regval = read_reg(device, logical_node, reg_addr);
        for (int i = 0; i < 8; i++) {
            x_remap.push_back(regval & 0xF);
            regval = regval >> 4;
        }
    }
    y_remap.clear();
    for (const auto& reg_addr : y_reg_addrs) {
        auto regval = read_reg(device, logical_node, reg_addr);
        ASSERT_NE(regval, init_value); // Need to make sure we read in valid reg
        for (int i = 0; i < 8; i++) {
            y_remap.push_back(regval & 0xF);
            regval = regval >> 4;
        }
    }
#else
    // If the translation tables are not defined, we should skip :)
    std::vector<uint32_t> x_reg_addrs = {};
    std::vector<uint32_t> y_reg_addrs = {};
#endif
}

}  // namespace unit_tests::basic::device



TEST_F(BasicFixture, VerifyNocNodeIDs) {
    auto arch = tt::get_arch_from_string(get_env_arch_name());
    tt::tt_metal::Device* device;
    const unsigned int device_id = 0;
    device = tt::tt_metal::CreateDevice(device_id);
    // Ping all the Noc Nodes
    auto logical_grid_size = device->logical_grid_size();
    for (size_t y = 0; y < logical_grid_size.y; y++) {
        for (size_t x = 0; x < logical_grid_size.x; x++) {
            auto worker_core = device->worker_core_from_logical_core(CoreCoord(x, y));
            // Read register from specific node
            uint32_t node_id_regval;
            node_id_regval = unit_tests::basic::test_noc::read_reg(device, CoreCoord(x, y), NOC_NODE_ID);
            ASSERT_NE(node_id_regval, unit_tests::basic::test_noc::init_value); // Need to make sure we read in valid reg
            // Check it matches software translated xy
            uint32_t my_x = node_id_regval & NOC_NODE_ID_MASK;
            uint32_t my_y = (node_id_regval >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
            EXPECT_EQ(my_x, worker_core.x);
            EXPECT_EQ(my_y, worker_core.y);
        }
    }
    ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
}
TEST_F(BasicFixture, VerifyNocIdentityTranslationTable) {
    auto arch = tt::get_arch_from_string(get_env_arch_name());
#ifndef NOC_X_ID_TRANSLATE_TABLE_0
    // If the translation tables are not defined, we should skip :)
    GTEST_SKIP();
#endif
    tt::tt_metal::Device* device;
    const unsigned int device_id = 0;
    device = tt::tt_metal::CreateDevice(device_id);
    // Ping all the registers for NOC
    auto logical_grid_size = device->logical_grid_size();
    for (size_t y = 0; y < logical_grid_size.y; y++) {
        for (size_t x = 0; x < logical_grid_size.x; x++) {
            std::vector<unsigned int> x_remap = {};
            std::vector<unsigned int> y_remap = {};
            unit_tests::basic::test_noc::read_translation_table(device, CoreCoord(x, y), x_remap, y_remap);
            bool core_has_translation_error = false;
            // bottom 16 values are not remapped --> identity
            for (int x=0; x<16; x++) {
                EXPECT_EQ (x, x_remap[x]);
                core_has_translation_error |= (x != x_remap[x]);
            }
            // bottom 16 values are not remapped --> identity
            for (int y=0; y<16; y++) {
                EXPECT_EQ ( y, y_remap[y]);
                core_has_translation_error |= (y != y_remap[y]);
            }
            ASSERT_FALSE(core_has_translation_error);
        }
    }
    ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
}
