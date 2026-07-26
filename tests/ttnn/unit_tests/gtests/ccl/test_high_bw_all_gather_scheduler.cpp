// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/experimental/deepseek_prefill/high_bw_all_gather/device/high_bw_all_gather_scheduler.hpp"

namespace scheduler = ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather::scheduler;

namespace {

struct ScheduleCase {
    uint32_t num_pages;
    uint32_t num_links;
    uint32_t workers_per_direction;
    uint32_t num_banks;
};

constexpr std::array schedule_cases = {
    ScheduleCase{1003, 2, 4, 7},   // harvested Blackhole
    ScheduleCase{1024, 2, 4, 8},   // full Blackhole
    ScheduleCase{1013, 2, 6, 11},  // harvested Wormhole
    ScheduleCase{1031, 2, 6, 12},  // full Wormhole
    ScheduleCase{997, 1, 12, 12},  // one-link Wormhole
    ScheduleCase{509, 4, 2, 7},    // harvested Blackhole with four links
};

}  // namespace

TEST(HighBwAllGatherScheduler, BankOwnedSlicesCoverEveryPageExactlyOnce) {
    for (const auto& test : schedule_cases) {
        ASSERT_TRUE(scheduler::can_partition_workers_by_bank(
            test.num_pages, test.num_links, test.workers_per_direction, test.num_banks));

        std::vector<uint32_t> page_owners(test.num_pages, 0);
        std::vector<uint32_t> bank_owners(test.num_banks, 0);
        std::vector<uint32_t> pages_per_link(test.num_links, 0);

        for (uint32_t link = 0; link < test.num_links; ++link) {
            for (uint32_t worker = 0; worker < test.workers_per_direction; ++worker) {
                const auto slice = scheduler::derive_bank_owned_slice(
                    test.num_pages, test.num_links, test.workers_per_direction, test.num_banks, link, worker);
                ASSERT_LT(slice.bank, test.num_banks);
                ASSERT_EQ(slice.input_page_start % test.num_banks, slice.bank);
                ++bank_owners[slice.bank];
                pages_per_link[link] += slice.page_count;

                for (uint32_t i = 0; i < slice.page_count; ++i) {
                    const uint32_t page = slice.input_page_start + i * test.num_banks;
                    ASSERT_LT(page, test.num_pages);
                    ++page_owners[page];
                }
            }
        }

        EXPECT_TRUE(std::all_of(page_owners.begin(), page_owners.end(), [](uint32_t owners) { return owners == 1; }));
        EXPECT_TRUE(std::all_of(bank_owners.begin(), bank_owners.end(), [](uint32_t owners) { return owners >= 1; }));

        const auto [min_pages, max_pages] = std::minmax_element(pages_per_link.begin(), pages_per_link.end());
        const uint32_t largest_bank = (test.num_pages + test.num_banks - 1) / test.num_banks;
        EXPECT_LE(*max_pages - *min_pages, largest_bank);
    }
}

TEST(HighBwAllGatherScheduler, EligibilityRequiresEnoughWorkersAndPages) {
    EXPECT_FALSE(scheduler::can_partition_workers_by_bank(1024, 2, 2, 7));
    EXPECT_TRUE(scheduler::can_partition_workers_by_bank(1024, 2, 4, 7));
    EXPECT_FALSE(scheduler::can_partition_workers_by_bank(1024, 2, 4, 11));
    EXPECT_TRUE(scheduler::can_partition_workers_by_bank(1024, 2, 6, 11));
    EXPECT_FALSE(scheduler::can_partition_workers_by_bank(7, 2, 4, 7));
    EXPECT_FALSE(scheduler::can_partition_workers_by_bank(1024, 0, 4, 7));
}

TEST(HighBwAllGatherScheduler, BankOwnedSliceRejectsInvalidInputs) {
    EXPECT_ANY_THROW(scheduler::derive_bank_owned_slice(1024, 0, 4, 8, 0, 0));
    EXPECT_ANY_THROW(scheduler::derive_bank_owned_slice(1024, 2, 0, 8, 0, 0));
    EXPECT_ANY_THROW(scheduler::derive_bank_owned_slice(1024, 2, 4, 0, 0, 0));
    EXPECT_ANY_THROW(scheduler::derive_bank_owned_slice(1024, 2, 4, 8, 2, 0));
    EXPECT_ANY_THROW(scheduler::derive_bank_owned_slice(1024, 2, 4, 8, 0, 4));
    EXPECT_ANY_THROW(scheduler::derive_bank_owned_slice(1024, 2, 2, 8, 0, 0));
}

TEST(HighBwAllGatherScheduler, PreservesExistingEightBankLinkMapping) {
    constexpr std::array expected_banks = {
        std::array<uint32_t, 8>{0, 0, 2, 2, 4, 4, 6, 6},
        std::array<uint32_t, 8>{1, 1, 3, 3, 5, 5, 7, 7},
    };
    for (uint32_t link = 0; link < expected_banks.size(); ++link) {
        for (uint32_t worker = 0; worker < expected_banks[link].size(); ++worker) {
            const auto slice = scheduler::derive_bank_owned_slice(1024, 2, 8, 8, link, worker);
            EXPECT_EQ(slice.bank, expected_banks[link][worker]);
        }
    }
}

TEST(HighBwAllGatherScheduler, WorkerCountCoversFullAndHarvestedDevices) {
    EXPECT_EQ(scheduler::workers_per_direction_to_cover_banks(2, 7), 4);
    EXPECT_EQ(scheduler::workers_per_direction_to_cover_banks(2, 8), 4);
    EXPECT_EQ(scheduler::workers_per_direction_to_cover_banks(2, 11), 6);
    EXPECT_EQ(scheduler::workers_per_direction_to_cover_banks(2, 12), 6);
    EXPECT_EQ(scheduler::workers_per_direction_to_cover_banks(1, 12), 12);
    EXPECT_EQ(scheduler::workers_per_direction_to_cover_banks(4, 7), 2);
}

TEST(HighBwAllGatherScheduler, WorkerCountAccountsForMuxCores) {
    // Two directions, each containing W workers and one mux, on every link.
    EXPECT_TRUE(scheduler::worker_count_fits(6, 2, 28));
    EXPECT_FALSE(scheduler::worker_count_fits(6, 2, 27));
    EXPECT_TRUE(scheduler::worker_count_fits(12, 1, 26));
    EXPECT_FALSE(scheduler::worker_count_fits(12, 1, 25));
}

TEST(HighBwAllGatherScheduler, RestrictedGridUsesMeasuredWorkerTiers) {
    constexpr std::array blackhole_tiers{8u, 4u, 2u, 1u};
    constexpr std::array harvested_wormhole_tiers{8u, 6u, 2u, 1u};

    EXPECT_EQ(scheduler::select_fitting_worker_count(8, 2, 35, blackhole_tiers), 4);
    EXPECT_EQ(scheduler::select_fitting_worker_count(4, 2, 19, blackhole_tiers), 2);
    EXPECT_EQ(scheduler::select_fitting_worker_count(8, 2, 28, harvested_wormhole_tiers), 6);
    EXPECT_EQ(scheduler::select_fitting_worker_count(6, 2, 11, harvested_wormhole_tiers), 1);
}
