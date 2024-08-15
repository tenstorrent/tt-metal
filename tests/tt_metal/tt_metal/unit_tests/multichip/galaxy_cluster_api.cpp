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

/*
Returns the ids of the devices that are connected via ethernet links.

tt::Cluster::get_ethernet_connected_device_ids() does not return device ids
for devices that are only connected via ethernet links reserved for FD tunneling.
tt::Cluster::get_ethernet_connected_device_ids() should be updated to allow these
device ids to be returned as an option. Once tt::Cluster::get_ethernet_connected_device_ids()
is updated with this option, we can call that over this function.
*/
std::unordered_set<chip_id_t> get_ethernet_connected_device_ids(const chip_id_t device_id)
{
    std::unordered_set<chip_id_t> connected_device_ids;
    const std::unordered_set<CoreCoord>& active_ethernet_cores = tt::Cluster::instance().get_active_ethernet_cores(device_id);
    for (const CoreCoord& ethernet_core : active_ethernet_cores)
    {
        const auto& [connected_device_id, _] = tt::Cluster::instance().get_connected_ethernet_core({device_id, ethernet_core});
        connected_device_ids.insert(connected_device_id);
    }
    return connected_device_ids;
}

// Validate that every pair of adjacent galaxy chips has 4 links between them
// The reason that this test is in TGFixture instead of GalaxyFixture is
// because there are 2 links between adjacent Galaxy chips that are on different
// shelves and 4 links between adjacent Galaxy chips that are on the same
// shelf, and currently tt::Cluster does not expose a way of determining
// which shelf a particular Galaxy chip is on.
TEST_F(TGFixture, ValidateNumLinksBetweenAdjacentGalaxyChips) {
    for (Device* device : this->devices_)
    {
        const chip_id_t device_id = device->id();
        if (is_galaxy_device(device_id))
        {
            std::unordered_map<chip_id_t, uint32_t> connected_devices_to_num_links_found;
            const std::unordered_set<CoreCoord>& active_ethernet_cores = tt::Cluster::instance().get_active_ethernet_cores(device_id);
            for (const CoreCoord& ethernet_core : active_ethernet_cores)
            {
                const auto& [connected_device_id, _] = tt::Cluster::instance().get_connected_ethernet_core({device_id, ethernet_core});

                if (!is_galaxy_device(connected_device_id)) {
                    continue;
                }

                if (!connected_devices_to_num_links_found.contains(connected_device_id))
                {
                    connected_devices_to_num_links_found[connected_device_id] = 0;
                }
                connected_devices_to_num_links_found[connected_device_id]++;
            }

            for (const auto& [connected_device_id, num_links_found] : connected_devices_to_num_links_found)
            {
                ASSERT_TRUE(num_links_found == 4) << "Detected " << num_links_found << " links between chip " << device_id
                                                  << " and chip " << connected_device_id << std::endl;
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
        const std::unordered_set<chip_id_t>& connected_device_ids = get_ethernet_connected_device_ids(device_id);
        if (is_galaxy_device(device_id))
        {
            uint32_t num_mmio_chips_that_curr_chip_is_linked_to = 0;
            for (const chip_id_t connected_device_id : connected_device_ids)
            {
                if (is_galaxy_device(connected_device_id))
                {
                    continue;
                }
                num_mmio_chips_that_curr_chip_is_linked_to++;
            }
            ASSERT_TRUE(num_mmio_chips_that_curr_chip_is_linked_to <= 1)
                << "Detected " << num_mmio_chips_that_curr_chip_is_linked_to << " MMIO chips that chip " << device_id
                << " is linked to" << std::endl;
        } else {
            const uint32_t num_chips_that_curr_chip_is_linked_to = connected_device_ids.size();
            const bool do_both_links_go_to_separate_devices = num_chips_that_curr_chip_is_linked_to == 2;
            ASSERT_TRUE(do_both_links_go_to_separate_devices)
                << "Detected " << num_chips_that_curr_chip_is_linked_to << " chips that chip " << device_id
                << " is linked to" << std::endl;

            bool do_both_links_go_to_galaxy_devices = true;
            for (const chip_id_t connected_device_id : connected_device_ids)
            {
                do_both_links_go_to_galaxy_devices = do_both_links_go_to_galaxy_devices && is_galaxy_device(connected_device_id);
            }
            ASSERT_TRUE(do_both_links_go_to_galaxy_devices)
                << "Detected links from chip " << device_id << " to chip " << *(connected_device_ids.begin())
                << " and chip " << *(++connected_device_ids.begin()) << std::endl;
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
            ASSERT_TRUE(harvest_mask == 0) << "Harvest mask for chip " << device_id << ": " << harvest_mask << std::endl;
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
            ASSERT_TRUE(num_rows_harvested == 1) << "Detected " << num_rows_harvested << " harvested rows for chip " << device_id << std::endl;
        }
    }
}

TEST_F(TGFixture, ValidateNumMMIOChips) {
    const size_t num_mmio_chips = tt::Cluster::instance().number_of_pci_devices();
    ASSERT_TRUE(num_mmio_chips == 4) << "Detected " << num_mmio_chips << " MMIO chips" << std::endl;
}

TEST_F(TGFixture, ValidateNumGalaxyChips) {
    const size_t num_galaxy_chips = tt::Cluster::instance().number_of_user_devices();
    ASSERT_TRUE(num_galaxy_chips == 32) << "Detected " << num_galaxy_chips << " Galaxy chips" << std::endl;
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
    ASSERT_TRUE(num_galaxy_chips == 32) << "Detected " << num_galaxy_chips << " Galaxy chips" << std::endl;
    ASSERT_TRUE(num_n150_chips == 4) << "Detected " << num_n150_chips << " N150 chips" << std::endl;
}

TEST_F(TGGFixture, ValidateNumMMIOChips) {
    const size_t num_mmio_chips = tt::Cluster::instance().number_of_pci_devices();
    ASSERT_TRUE(num_mmio_chips == 8) << "Detected " << num_mmio_chips << " MMIO chips" << std::endl;
}

TEST_F(TGGFixture, ValidateNumGalaxyChips) {
    const size_t num_galaxy_chips = tt::Cluster::instance().number_of_user_devices();
    ASSERT_TRUE(num_galaxy_chips == 64) << "Detected " << num_galaxy_chips << " Galaxy chips" << std::endl;
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
    ASSERT_TRUE(num_galaxy_chips == 64) << "Detected " << num_galaxy_chips << " Galaxy chips" << std::endl;
    ASSERT_TRUE(num_n150_chips == 8) << "Detected " << num_n150_chips << " N150 chips" << std::endl;
}
