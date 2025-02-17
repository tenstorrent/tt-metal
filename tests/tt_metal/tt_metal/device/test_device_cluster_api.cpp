// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>

#include "multi_device_fixture.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::multichip::cluster {

// Run this on Nebula X2 only, validate etherent core apis are correct
// Known connectivity: chip 0 (x=9, y=6) <--> chip 1 (x=9, y=0)
//                     chip 0 (x=1, y=6) <--> chip 1 (x=1, y=0)
TEST_F(N300DeviceFixture, EthValidateEthernetConnectivity) {
    const auto& device_0 = this->devices_.at(0);
    const auto& device_1 = this->devices_.at(1);

    // Check active and inactive core counts
    const auto& device_0_active_eth_cores = device_0->get_active_ethernet_cores();
    const auto& device_1_active_eth_cores = device_1->get_active_ethernet_cores();

    ASSERT_TRUE(device_0_active_eth_cores.size() == 2);
    ASSERT_TRUE(device_1_active_eth_cores.size() == 2);
    // mmio device (0) has 2 ports (8, 9) reserved for umd non-mmio access.
    // mmio device (0) port 15 is reserved for syseng tools.
    ASSERT_TRUE(device_0->get_inactive_ethernet_cores().size() == 13);
    ASSERT_TRUE(device_1->get_inactive_ethernet_cores().size() == 14);

    for (const auto& core : device_0_active_eth_cores) {
        std::tuple<chip_id_t, CoreCoord> core_on_chip_1 = device_0->get_connected_ethernet_core(core);
        ASSERT_TRUE(std::get<0>(core_on_chip_1) == 1);
        ASSERT_TRUE(device_1_active_eth_cores.find(std::get<1>(core_on_chip_1)) != device_1_active_eth_cores.end());
    }
    for (const auto& core : device_1_active_eth_cores) {
        std::tuple<chip_id_t, CoreCoord> core_on_chip_0 = device_1->get_connected_ethernet_core(core);
        ASSERT_TRUE(std::get<0>(core_on_chip_0) == 0);
        ASSERT_TRUE(device_0_active_eth_cores.find(std::get<1>(core_on_chip_0)) != device_0_active_eth_cores.end());
    }

    // Check conversion to noc coords
    std::vector<CoreCoord> chip_0_eth_noc_coords_expected = {CoreCoord(25, 17), CoreCoord(18, 17)};

    std::vector<CoreCoord> chip_0_eth_logical_coords;
    std::copy(
        device_0_active_eth_cores.begin(),
        device_0_active_eth_cores.end(),
        std::back_inserter(chip_0_eth_logical_coords));
    std::vector<CoreCoord> chip_0_eth_noc_coords_returned =
        device_0->ethernet_cores_from_logical_cores(chip_0_eth_logical_coords);

    std::sort(chip_0_eth_noc_coords_expected.begin(), chip_0_eth_noc_coords_expected.end());
    std::sort(chip_0_eth_noc_coords_returned.begin(), chip_0_eth_noc_coords_returned.end());
    ASSERT_TRUE(chip_0_eth_noc_coords_returned == chip_0_eth_noc_coords_expected);

    std::vector<CoreCoord> chip_1_eth_noc_coords_expected = {CoreCoord(25, 16), CoreCoord(18, 16)};

    std::vector<CoreCoord> chip_1_eth_logical_coords;
    std::copy(
        device_1_active_eth_cores.begin(),
        device_1_active_eth_cores.end(),
        std::back_inserter(chip_1_eth_logical_coords));
    std::vector<CoreCoord> chip_1_eth_noc_coords_returned =
        device_1->ethernet_cores_from_logical_cores(chip_1_eth_logical_coords);

    std::sort(chip_1_eth_noc_coords_expected.begin(), chip_1_eth_noc_coords_expected.end());
    std::sort(chip_1_eth_noc_coords_returned.begin(), chip_1_eth_noc_coords_returned.end());
    ASSERT_TRUE(chip_1_eth_noc_coords_returned == chip_1_eth_noc_coords_expected);
}

TEST_F(N300DeviceFixture, EthInvalidLogicalEthernetCore) {
    const auto& device_0 = this->devices_.at(0);
    EXPECT_ANY_THROW(device_0->ethernet_core_from_logical_core(CoreCoord(1, 0)));
    EXPECT_ANY_THROW(device_0->ethernet_core_from_logical_core(CoreCoord(0, 16)));
}

TEST_F(N300DeviceFixture, EthValidateAllEthernetCoreMapping) {
    static std::map<CoreCoord, CoreCoord> expected_mapping_logical_to_physical = {
        {CoreCoord(0, 0), CoreCoord(25, 16)},
        {CoreCoord(0, 1), CoreCoord(18, 16)},
        {CoreCoord(0, 2), CoreCoord(24, 16)},
        {CoreCoord(0, 3), CoreCoord(19, 16)},
        {CoreCoord(0, 4), CoreCoord(23, 16)},
        {CoreCoord(0, 5), CoreCoord(20, 16)},
        {CoreCoord(0, 6), CoreCoord(22, 16)},
        {CoreCoord(0, 7), CoreCoord(21, 16)},
        {CoreCoord(0, 8), CoreCoord(25, 17)},
        {CoreCoord(0, 9), CoreCoord(18, 17)},
        {CoreCoord(0, 10), CoreCoord(24, 17)},
        {CoreCoord(0, 11), CoreCoord(19, 17)},
        {CoreCoord(0, 12), CoreCoord(23, 17)},
        {CoreCoord(0, 13), CoreCoord(20, 17)},
        {CoreCoord(0, 14), CoreCoord(22, 17)},
        {CoreCoord(0, 15), CoreCoord(21, 17)},
    };
    const auto& device_0 = this->devices_.at(0);
    for (const auto& logical_core : device_0->ethernet_cores()) {
        ASSERT_TRUE(
            device_0->ethernet_core_from_logical_core(logical_core) ==
            expected_mapping_logical_to_physical.at(logical_core));
    }
}

TEST_F(N300DeviceFixture, EthValidatePhysicalCoreConversion) {
    static std::map<CoreCoord, CoreCoord> expected_mapping_logical_to_physical = {
        {CoreCoord(0, 0), CoreCoord(25, 16)},
        {CoreCoord(0, 1), CoreCoord(18, 16)},
        {CoreCoord(0, 2), CoreCoord(24, 16)},
        {CoreCoord(0, 3), CoreCoord(19, 16)},
        {CoreCoord(0, 4), CoreCoord(23, 16)},
        {CoreCoord(0, 5), CoreCoord(20, 16)},
        {CoreCoord(0, 6), CoreCoord(22, 16)},
        {CoreCoord(0, 7), CoreCoord(21, 16)},
        {CoreCoord(0, 8), CoreCoord(25, 17)},
        {CoreCoord(0, 9), CoreCoord(18, 17)},
        {CoreCoord(0, 10), CoreCoord(24, 17)},
        {CoreCoord(0, 11), CoreCoord(19, 17)},
        {CoreCoord(0, 12), CoreCoord(23, 17)},
        {CoreCoord(0, 13), CoreCoord(20, 17)},
        {CoreCoord(0, 14), CoreCoord(22, 17)},
        {CoreCoord(0, 15), CoreCoord(21, 17)},
    };
    const auto& device_0 = this->devices_.at(0);
    for (const auto& logical_core : device_0->ethernet_cores()) {
        ASSERT_TRUE(
            device_0->virtual_core_from_logical_core(logical_core, CoreType::ETH) ==
            expected_mapping_logical_to_physical.at(logical_core));
    }
    // Check an invalid core type
    EXPECT_ANY_THROW(device_0->virtual_core_from_logical_core(CoreCoord(0, 0), CoreType::PCIE));
}

TEST_F(N300DeviceFixture, ActiveEthValidateEthernetSockets) {
    const auto& device_0 = this->devices_.at(0);
    const auto& device_1 = this->devices_.at(1);

    std::vector<CoreCoord> device_0_sockets = device_0->get_ethernet_sockets(1);
    std::vector<CoreCoord> device_1_sockets = device_1->get_ethernet_sockets(0);

    ASSERT_TRUE(device_0_sockets.size() == 2);
    ASSERT_TRUE(device_1_sockets.size() == 2);
    ASSERT_TRUE(
        device_0->get_connected_ethernet_core(device_0_sockets.at(0)) ==
        std::make_tuple(device_1->id(), device_1_sockets.at(0)));
    ASSERT_TRUE(
        device_0->get_connected_ethernet_core(device_0_sockets.at(1)) ==
        std::make_tuple(device_1->id(), device_1_sockets.at(1)));
    EXPECT_ANY_THROW(device_0->get_ethernet_sockets(2));
}
}  // namespace unit_tests::multichip::cluster
