// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "noc/att/att.h"
#include "noc/att/configs/grendel_qsr1_att_config.h"
#include "noc/att/configs/quasar_aether_2x3_att_config.h"

namespace {

template <std::size_t N>
void expect_valid_unique_selectors(const std::uint8_t (&selectors)[N], const noc_att::Window& window) {
    const std::uint32_t selector_limit = std::uint32_t{1} << window.endpoint_size;
    std::vector<bool> seen(selector_limit, false);
    for (const std::uint8_t selector : selectors) {
        ASSERT_LT(selector, selector_limit);
        EXPECT_FALSE(seen[selector]);
        seen[selector] = true;
    }
}

template <std::size_t N>
std::uint32_t count_matching_windows(const noc_att::Window (&windows)[N], std::uint64_t address) {
    std::uint32_t count = 0;
    for (const auto& window : windows) {
        count += noc_att::matches(window, address);
    }
    return count;
}

TEST(QuasarAtt, WindowValidationAndAddressBounds) {
    constexpr noc_att::Window window{
        .compare = 0x10000000000ull,
        .mask_bits = 30,
        .endpoint_shift = 24,
        .endpoint_size = 6,
        .endpoint_table_offset = 128,
        .translate_address = true,
    };
    constexpr noc_att::Window misaligned_compare{
        .compare = window.compare | 1,
        .mask_bits = window.mask_bits,
        .endpoint_shift = window.endpoint_shift,
        .endpoint_size = window.endpoint_size,
        .endpoint_table_offset = window.endpoint_table_offset,
        .translate_address = window.translate_address,
    };
    constexpr noc_att::Window selector_outside_mask{
        .compare = window.compare,
        .mask_bits = 29,
        .endpoint_shift = window.endpoint_shift,
        .endpoint_size = window.endpoint_size,
        .endpoint_table_offset = window.endpoint_table_offset,
        .translate_address = window.translate_address,
    };
    constexpr noc_att::Window no_selector_window{
        .compare = 0x100000,
        .mask_bits = 20,
        .endpoint_shift = 0,
        .endpoint_size = 0,
        .endpoint_table_offset = 256,
        .translate_address = false,
    };

    EXPECT_TRUE(noc_att::valid(window));
    EXPECT_FALSE(noc_att::valid(misaligned_compare));
    EXPECT_FALSE(noc_att::valid(selector_outside_mask));
    EXPECT_EQ(noc_att::local_address_limit(window), std::uint64_t{1} << 24);

    constexpr std::uint64_t address = noc_att::make_address(window, 31, 0xabcde);
    EXPECT_EQ(address, window.compare | (std::uint64_t{31} << 24) | 0xabcde);
    EXPECT_TRUE(noc_att::matches(window, address));
    EXPECT_EQ(
        noc_att::replace_local_address(window, address, 0x1234), window.compare | (std::uint64_t{31} << 24) | 0x1234);

    constexpr std::uint64_t no_selector_address = noc_att::make_local_address(no_selector_window, 0xabcde);
    EXPECT_EQ(noc_att::replace_local_address(no_selector_window, no_selector_address, 0x1234), 0x101234u);
}

TEST(QuasarAtt, GrendelMapEncodesWorkerDramAndTileSelectors) {
    namespace config = grendel_qsr1_att_config;

    EXPECT_EQ(std::size(config::ATT_WORKER_SELECTORS), config::ATT_WORKER_GRID_X * config::ATT_WORKER_GRID_Y);
    EXPECT_EQ(std::size(config::ATT_TILE_SELECTORS), config::ATT_TILE_GRID_X * config::ATT_TILE_GRID_Y);
    expect_valid_unique_selectors(config::ATT_WORKER_SELECTORS, config::WORKER_WINDOW);
    expect_valid_unique_selectors(config::ATT_DRAM_SELECTORS, config::DRAM_WINDOW);
    expect_valid_unique_selectors(config::ATT_TILE_SELECTORS, config::TILE_WINDOW);

    constexpr std::uint64_t worker_address =
        noc_att::make_address(config::WORKER_WINDOW, config::ATT_WORKER_SELECTORS[3 * 8 + 7], 0xabcde);
    constexpr std::uint64_t dram_address =
        noc_att::make_address(config::DRAM_WINDOW, config::ATT_DRAM_SELECTORS[7], 0x12345678);
    constexpr std::uint64_t tile_address =
        noc_att::make_address(config::TILE_WINDOW, config::ATT_TILE_SELECTORS[5 * 10 + 9], 0x1234567);

    EXPECT_EQ(worker_address, 0x1001f0abcdeull);
    EXPECT_EQ(dram_address, 0x1002612345678ull);
    EXPECT_EQ(tile_address, 0x19c1234567ull);
    EXPECT_EQ(count_matching_windows(config::WINDOWS, worker_address), 1u);
    EXPECT_EQ(count_matching_windows(config::WINDOWS, dram_address), 1u);
    EXPECT_EQ(count_matching_windows(config::WINDOWS, tile_address), 1u);
}

TEST(QuasarAtt, AetherMapPreservesDescriptorAliases) {
    namespace config = quasar_aether_2x3_att_config;

    EXPECT_EQ(std::size(config::ATT_WORKER_SELECTORS), config::ATT_WORKER_GRID_X * config::ATT_WORKER_GRID_Y);
    EXPECT_EQ(std::size(config::ATT_TILE_SELECTORS), config::ATT_TILE_GRID_X * config::ATT_TILE_GRID_Y);
    expect_valid_unique_selectors(config::ATT_WORKER_SELECTORS, config::WORKER_WINDOW);
    expect_valid_unique_selectors(config::ATT_DRAM_SELECTORS, config::DRAM_WINDOW);
    expect_valid_unique_selectors(config::ATT_TILE_SELECTORS, config::TILE_WINDOW);

    EXPECT_EQ(config::ATT_TILE_SELECTORS[2 * 2], 5u);
    EXPECT_EQ(config::ATT_TILE_SELECTORS[2 * 2 + 1], 4u);

    constexpr std::uint64_t local_address = noc_att::make_local_address(config::LOCAL_WINDOW, 0x1234);
    constexpr std::uint64_t worker_address =
        noc_att::make_address(config::WORKER_WINDOW, config::ATT_WORKER_SELECTORS[1], 0x1234);
    constexpr std::uint64_t dram_address =
        noc_att::make_address(config::DRAM_WINDOW, config::ATT_DRAM_SELECTORS[1], 0x1234);

    EXPECT_EQ(local_address, 0x1234u);
    EXPECT_EQ(worker_address, 0x1004001234ull);
    EXPECT_EQ(dram_address, 0x100c001234ull);
    EXPECT_EQ(count_matching_windows(config::WINDOWS, local_address), 1u);
    EXPECT_EQ(count_matching_windows(config::WINDOWS, worker_address), 1u);
    EXPECT_EQ(count_matching_windows(config::WINDOWS, dram_address), 1u);
}

}  // namespace
