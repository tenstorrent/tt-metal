// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "n300_device_fixture.hpp"
#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;

bool is_galaxy_device(const chip_id_t device_id)
{
    return tt::Cluster::instance().get_board_type(device_id) == GALAXY;
}

bool is_n150_device(const chip_id_t device_id)
{
    return tt::Cluster::instance().get_board_type(device_id) == N150;
}

namespace unit_tests::multichip::cluster {

// Run this on Nebula X2 only, validate etherent core apis are correct
// Known connectivity: chip 0 (x=9, y=6) <--> chip 1 (x=9, y=0)
//                     chip 0 (x=1, y=6) <--> chip 1 (x=1, y=0)
TEST_F(N300DeviceFixture, ValidateEthernetConnectivity) {
    const auto& device_0 = this->devices_.at(0);
    const auto& device_1 = this->devices_.at(1);

    // Check active and inactive core counts
    const auto& device_0_active_eth_cores = device_0->get_active_ethernet_cores();
    const auto& device_1_active_eth_cores = device_1->get_active_ethernet_cores();

    ASSERT_TRUE(device_0_active_eth_cores.size() == 2);
    ASSERT_TRUE(device_1_active_eth_cores.size() == 2);
    //mmio device (0) has 2 ports (8, 9) reserved for umd non-mmio access.
    //mmio device (0) port 15 is reserved for syseng tools.
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
    std::vector<CoreCoord> chip_0_eth_noc_coords_expected = {CoreCoord(9, 6), CoreCoord(1, 6)};

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

    std::vector<CoreCoord> chip_1_eth_noc_coords_expected = {CoreCoord(9, 0), CoreCoord(1, 0)};

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

TEST_F(N300DeviceFixture, InvalidLogicalEthernetCore) {
    const auto& device_0 = this->devices_.at(0);
    EXPECT_ANY_THROW(device_0->ethernet_core_from_logical_core(CoreCoord(1, 0)));
    EXPECT_ANY_THROW(device_0->ethernet_core_from_logical_core(CoreCoord(0, 16)));
}

TEST_F(N300DeviceFixture, ValidateAllEthernetCoreMapping) {
    static std::map<CoreCoord, CoreCoord> expected_mapping_logical_to_physical = {
        {CoreCoord(0, 0), CoreCoord(9, 0)},
        {CoreCoord(0, 1), CoreCoord(1, 0)},
        {CoreCoord(0, 2), CoreCoord(8, 0)},
        {CoreCoord(0, 3), CoreCoord(2, 0)},
        {CoreCoord(0, 4), CoreCoord(7, 0)},
        {CoreCoord(0, 5), CoreCoord(3, 0)},
        {CoreCoord(0, 6), CoreCoord(6, 0)},
        {CoreCoord(0, 7), CoreCoord(4, 0)},
        {CoreCoord(0, 8), CoreCoord(9, 6)},
        {CoreCoord(0, 9), CoreCoord(1, 6)},
        {CoreCoord(0, 10), CoreCoord(8, 6)},
        {CoreCoord(0, 11), CoreCoord(2, 6)},
        {CoreCoord(0, 12), CoreCoord(7, 6)},
        {CoreCoord(0, 13), CoreCoord(3, 6)},
        {CoreCoord(0, 14), CoreCoord(6, 6)},
        {CoreCoord(0, 15), CoreCoord(4, 6)},
    };
    const auto& device_0 = this->devices_.at(0);
    for (const auto& logical_core : device_0->ethernet_cores()) {
        ASSERT_TRUE(
            device_0->ethernet_core_from_logical_core(logical_core) ==
            expected_mapping_logical_to_physical.at(logical_core));
    }
}

TEST_F(N300DeviceFixture, ValidatePhysicalCoreConversion) {
    static std::map<CoreCoord, CoreCoord> expected_mapping_logical_to_physical = {
        {CoreCoord(0, 0), CoreCoord(9, 0)},
        {CoreCoord(0, 1), CoreCoord(1, 0)},
        {CoreCoord(0, 2), CoreCoord(8, 0)},
        {CoreCoord(0, 3), CoreCoord(2, 0)},
        {CoreCoord(0, 4), CoreCoord(7, 0)},
        {CoreCoord(0, 5), CoreCoord(3, 0)},
        {CoreCoord(0, 6), CoreCoord(6, 0)},
        {CoreCoord(0, 7), CoreCoord(4, 0)},
        {CoreCoord(0, 8), CoreCoord(9, 6)},
        {CoreCoord(0, 9), CoreCoord(1, 6)},
        {CoreCoord(0, 10), CoreCoord(8, 6)},
        {CoreCoord(0, 11), CoreCoord(2, 6)},
        {CoreCoord(0, 12), CoreCoord(7, 6)},
        {CoreCoord(0, 13), CoreCoord(3, 6)},
        {CoreCoord(0, 14), CoreCoord(6, 6)},
        {CoreCoord(0, 15), CoreCoord(4, 6)},
    };
    const auto& device_0 = this->devices_.at(0);
    for (const auto& logical_core : device_0->ethernet_cores()) {
        ASSERT_TRUE(
            device_0->physical_core_from_logical_core(logical_core, CoreType::ETH) ==
            expected_mapping_logical_to_physical.at(logical_core));
    }
    // Check an invalid core type
    EXPECT_ANY_THROW(device_0->physical_core_from_logical_core(CoreCoord(0, 0), CoreType::PCIE));
}

TEST_F(N300DeviceFixture, ValidateEthernetSockets) {
    const auto& device_0 = this->devices_.at(0);
    const auto& device_1 = this->devices_.at(1);

    std::vector<CoreCoord> device_0_sockets = device_0->get_ethernet_sockets(1);
    std::vector<CoreCoord> device_1_sockets = device_1->get_ethernet_sockets(0);

    ASSERT_TRUE(device_0_sockets.size() == 2);
    ASSERT_TRUE(device_1_sockets.size() == 2);
    ASSERT_TRUE(
        device_0->get_connected_ethernet_core(device_0_sockets.at(0)) == std::make_tuple(device_1->id(), device_1_sockets.at(0)));
    ASSERT_TRUE(
        device_0->get_connected_ethernet_core(device_0_sockets.at(1)) == std::make_tuple(device_1->id(), device_1_sockets.at(1)));
    EXPECT_ANY_THROW(device_0->get_ethernet_sockets(2));
}

// Validate that every pair of adjacent galaxy chips has 4 links between them
TEST_F(GalaxyFixture, ValidateNumLinksBetweenAdjacentGalaxyChips) {
    for (Device* device : this->devices_)
    {
        const chip_id_t device_id = device->id();
        if (is_galaxy_device(device_id))
        {
            const std::unordered_set<chip_id_t>& connected_device_ids = tt::Cluster::instance().get_ethernet_connected_device_ids(device_id);
            for (const chip_id_t connected_device_id : connected_device_ids)
            {
                if (!is_galaxy_device(connected_device_id))
                {
                    continue;
                }
                const std::vector<CoreCoord>& ethernet_sockets = tt::Cluster::instance().get_ethernet_sockets(device_id, connected_device_id);
                ASSERT_TRUE(ethernet_sockets.size() == 4);
            }
        }
    }
}

// Validate that each MMIO chip links to two separate Galaxy chips,
// and that each Galaxy chip links to at most one MMIO chip
TEST_F(GalaxyFixture, ValidateLinksBetweenMMIOAndGalaxyChips) {
    for (Device* device : this->devices_)
    {
        const chip_id_t device_id = device->id();
        const std::unordered_set<chip_id_t>& connected_device_ids = tt::Cluster::instance().get_ethernet_connected_device_ids(device_id);
        if (is_galaxy_device(device_id))
        {
            uint32_t num_links_to_mmio_chips = 0;
            for (const chip_id_t connected_device_id : connected_device_ids)
            {
                if (is_galaxy_device(connected_device_id))
                {
                    continue;
                }
                num_links_to_mmio_chips++;
            }
            ASSERT_TRUE(num_links_to_mmio_chips <= 1);
        }
        else
        {
            bool do_both_links_go_to_separate_devices = connected_device_ids.size() == 2;
            ASSERT_TRUE(do_both_links_go_to_separate_devices);

            bool do_both_links_go_to_galaxy_devices = true;
            for (const chip_id_t connected_device_id : connected_device_ids)
            {
                do_both_links_go_to_galaxy_devices = do_both_links_go_to_galaxy_devices && is_galaxy_device(connected_device_id);
            }
            ASSERT_TRUE(do_both_links_go_to_galaxy_devices);
        }
    }
}

// Validate that all galaxy chips are unharvested
TEST_F(GalaxyFixture, ValidateAllGalaxyChipsAreUnharvested) {
    for (Device* device : this->devices_)
    {
        const chip_id_t device_id = device->id();
        if (is_galaxy_device(device_id))
        {
            const uint32_t harvest_mask = tt::Cluster::instance().get_harvested_rows(device_id);
            ASSERT_TRUE(harvest_mask == 0);
        }
    }
}

// Validate that all MMIO chips have a single row harvested
TEST_F(GalaxyFixture, ValidateAllMMIOChipsHaveSingleRowHarvested) {
    for (Device* device : this->devices_)
    {
        const chip_id_t device_id = device->id();
        if (!is_galaxy_device(device_id))
        {
            uint32_t num_rows_harvested = 0;
            uint32_t harvest_mask = tt::Cluster::instance().get_harvested_rows(device_id);
            while (harvest_mask)
            {
                if (harvest_mask & 1)
                {
                    num_rows_harvested++;
                }
                harvest_mask = harvest_mask >> 1;
            }
            ASSERT_TRUE(num_rows_harvested == 1);
        }
    }
}

TEST_F(TGFixture, ValidateNumMMIOChips) {
    const size_t num_mmio_chips = tt::Cluster::instance().number_of_pci_devices();
    ASSERT_TRUE(num_mmio_chips == 4);
}

TEST_F(TGFixture, ValidateNumGalaxyChips) {
    const size_t num_galaxy_chips = tt::Cluster::instance().number_of_user_devices();
    ASSERT_TRUE(num_galaxy_chips == 32);
}

// Validate that there are 4 N150 chips and 32 Galaxy chips
TEST_F(TGFixture, ValidateChipBoardTypes) {
    uint32_t num_n150_chips = 0;
    uint32_t num_galaxy_chips = 0;
    for (Device* device : this->devices_)
    {
        const chip_id_t device_id = device->id();
        if (is_galaxy_device(device_id))
        {
            num_galaxy_chips++;
        }
        else if (is_n150_device(device_id))
        {
            num_n150_chips++;
        }
    }
    ASSERT_TRUE(num_galaxy_chips == 32);
    ASSERT_TRUE(num_n150_chips == 4);
}

TEST_F(TGGFixture, ValidateNumMMIOChips) {
    const size_t num_mmio_chips = tt::Cluster::instance().number_of_pci_devices();
    ASSERT_TRUE(num_mmio_chips == 8);
}

TEST_F(TGGFixture, ValidateNumGalaxyChips) {
    const size_t num_galaxy_chips = tt::Cluster::instance().number_of_user_devices();
    ASSERT_TRUE(num_galaxy_chips == 64);
}

// Validate that there are 8 N150 chips and 64 Galaxy chips
TEST_F(TGGFixture, ValidateChipBoardTypes) {
    uint32_t num_n150_chips = 0;
    uint32_t num_galaxy_chips = 0;
    for (Device* device : this->devices_)
    {
        const chip_id_t device_id = device->id();
        if (is_galaxy_device(device_id))
        {
            num_galaxy_chips++;
        }
        else if (is_n150_device(device_id))
        {
            num_n150_chips++;
        }
    }
    ASSERT_TRUE(num_galaxy_chips == 64);
    ASSERT_TRUE(num_n150_chips == 8);
}
}  // namespace unit_tests::multichip::cluster
