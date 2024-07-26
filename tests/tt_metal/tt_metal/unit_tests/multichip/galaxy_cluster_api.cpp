// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;

bool is_galaxy_device(const chip_id_t device_id)
{
    return tt::Cluster::instance().get_board_type(device_id) == BoardType::GALAXY;
}

bool is_n150_device(const chip_id_t device_id)
{
    return tt::Cluster::instance().get_board_type(device_id) == BoardType::N150;
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
            const bool do_both_links_go_to_separate_devices = connected_device_ids.size() == 2;
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
