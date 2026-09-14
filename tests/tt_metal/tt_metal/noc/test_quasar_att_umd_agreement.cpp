// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only cross-check that tt-metal and UMD resolve a core to the same flat ATT address.
//
// Both sides carry their own transcription of the qsr.s1 map: tt-metal's kernels build operands
// from theirs, UMD's host accesses are resolved through its own. Nothing else compares them, and a
// disagreement means a kernel and the host reach different cores through the same coordinate. No
// device is opened.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include "internal/tt-2xx/quasar/noc/att/att_address.h"
#include "internal/tt-2xx/quasar/noc/att/configs/grendel_qsr1_att_config.h"
#include "umd/device/coordinates/att/att_resolver.hpp"
#include "umd/device/coordinates/att/configs/grendel_qsr1_att_map.hpp"
#include "umd/device/types/core_coordinates.hpp"
#include "umd/device/types/xy_pair.hpp"

namespace {

constexpr const noc_att::MapData& METAL_MAP = grendel_qsr1_att_config::MAP;
const tt::umd::att::MapData& UMD_MAP = tt::umd::att::GRENDEL_QSR1_MAP;

// An offset small enough to sit inside every window's slot, so a mismatch is the window or the
// selector rather than a rejected transfer.
constexpr std::uint64_t TEST_OFFSET = 0x1000;
constexpr std::uint64_t TEST_SIZE = 4;

// A packed endpoint word carries POR package coordinates; UMD is driven in the soc descriptor's
// frame, which the map offsets from it.
std::uint32_t descriptor_x(std::uint16_t endpoint_word) {
    return (endpoint_word & 0x3f) - UMD_MAP.package_offset_x;
}

std::uint32_t descriptor_y(std::uint16_t endpoint_word) {
    return (endpoint_word >> 6) - UMD_MAP.package_offset_y;
}

bool is_worker_endpoint(std::uint16_t endpoint_word) {
    for (std::uint32_t selector = 0; selector < METAL_MAP.worker_endpoint_words.size(); ++selector) {
        if (METAL_MAP.worker_endpoint_words[selector] == endpoint_word) {
            return true;
        }
    }
    return false;
}

}  // namespace

// tt-metal addresses a worker in the frame kernels receive, which is the frame UMD's soc descriptor
// uses, so both sides take the same coordinate here.
TEST(QuasarAttUmdAgreement, WorkerAddressesMatch) {
    const tt::umd::att::Resolver umd_resolver(UMD_MAP);

    for (std::uint32_t y = METAL_MAP.worker_origin_y; y < METAL_MAP.worker_origin_y + METAL_MAP.worker_grid_y; ++y) {
        for (std::uint32_t x = METAL_MAP.worker_origin_x; x < METAL_MAP.worker_origin_x + METAL_MAP.worker_grid_x;
             ++x) {
            SCOPED_TRACE("worker (" + std::to_string(x) + ", " + std::to_string(y) + ")");

            const std::optional<noc_att::NocAddress> metal_address =
                noc_att::Address::worker(x, y, TEST_OFFSET).encode<METAL_MAP>(TEST_SIZE);
            ASSERT_TRUE(metal_address.has_value());

            EXPECT_EQ(
                umd_resolver.resolve(tt_xy_pair(x, y), tt::CoreType::TENSIX, TEST_OFFSET, TEST_SIZE), *metal_address);
        }
    }
}

// The perimeter is where the two transcriptions can most easily disagree: those slots follow the POR
// endpoint table rather than the mesh geometry. tt-metal has no forward lookup by coordinate for
// this window, so its inverse lookup is the equivalent path.
TEST(QuasarAttUmdAgreement, PerimeterConfigAddressesMatch) {
    const tt::umd::att::Resolver umd_resolver(UMD_MAP);
    std::uint32_t compared = 0;

    for (std::uint32_t selector = 0; selector < METAL_MAP.full_tile_endpoint_words.size(); ++selector) {
        const std::uint16_t endpoint_word = METAL_MAP.full_tile_endpoint_words[selector];

        // A NEO tile appears in both tables, and tt-metal's inverse lookup answers with the worker
        // window for those, so it cannot speak for their config aperture.
        if (is_worker_endpoint(endpoint_word)) {
            continue;
        }

        const std::uint32_t noc_x = endpoint_word & 0x3f;
        const std::uint32_t noc_y = endpoint_word >> 6;
        SCOPED_TRACE("package core (" + std::to_string(noc_x) + ", " + std::to_string(noc_y) + ")");

        const noc_att::ResolvedTile tile = noc_att::resolve_current(METAL_MAP, noc_x, noc_y);
        ASSERT_TRUE(tile.valid);
        ASSERT_EQ(tile.window, noc_att::WindowClass::FullTile);
        const noc_att::NocAddress metal_address =
            noc_att::map_window(METAL_MAP, tile.window).make_address(tile.selector, TEST_OFFSET);

        const tt_xy_pair core(descriptor_x(endpoint_word), descriptor_y(endpoint_word));
        EXPECT_EQ(
            umd_resolver.resolve(core, tt::CoreType::ROUTER_ONLY, TEST_OFFSET, TEST_SIZE), metal_address);
        ++compared;
    }

    EXPECT_EQ(compared, METAL_MAP.full_tile_endpoint_words.size() - METAL_MAP.worker_endpoint_words.size());
}

// Both sides offset the descriptor frame to the package frame the endpoint words are in. The
// kernels' inverse lookups (packed bank words, CQ coordinate words, is_local_bank) apply tt-metal's
// copy of that offset, so it must be the one UMD uses for the same map.
TEST(QuasarAttUmdAgreement, FrameOffsetsMatch) {
    EXPECT_EQ(METAL_MAP.node_id_offset_x, UMD_MAP.package_offset_x);
    EXPECT_EQ(METAL_MAP.node_id_offset_y, UMD_MAP.package_offset_y);
}

// A dispatch entry is keyed by the descriptor coordinate the host publishes. Its selector must be
// the row UMD's table holds for that tile in the same window, and the operand must be the one UMD
// builds for a host access to the same core.
TEST(QuasarAttUmdAgreement, DispatchEntriesMatchUmd) {
    const tt::umd::att::Resolver umd_resolver(UMD_MAP);
    ASSERT_FALSE(METAL_MAP.dispatch_entries.empty());

    for (std::uint32_t i = 0; i < METAL_MAP.dispatch_entries.size(); ++i) {
        const noc_att::MapData::DispatchEntry& entry = METAL_MAP.dispatch_entries[i];
        SCOPED_TRACE("dispatch (" + std::to_string(entry.x) + ", " + std::to_string(entry.y) + ")");

        const bool tensix_hosted = entry.window == noc_att::WindowClass::Worker;
        ASSERT_TRUE(tensix_hosted || entry.window == noc_att::WindowClass::FullTile);
        const tt::umd::att::Table<std::uint16_t>& umd_words = UMD_MAP.endpoint_words[static_cast<std::size_t>(
            tensix_hosted ? tt::umd::att::WindowClass::Worker : tt::umd::att::WindowClass::FullTile)];
        ASSERT_LT(entry.selector, umd_words.size());
        const std::uint16_t package_word = static_cast<std::uint16_t>(
            ((entry.y + UMD_MAP.package_offset_y) << 6) | (entry.x + UMD_MAP.package_offset_x));
        EXPECT_EQ(umd_words[entry.selector], package_word);

        const std::optional<noc_att::NocAddress> metal_address =
            noc_att::Address::dispatch(entry.x, entry.y, TEST_OFFSET).encode<METAL_MAP>(TEST_SIZE);
        ASSERT_TRUE(metal_address.has_value());
        // UMD picks the window from the core type: a Tensix-hosted dispatch tile is a worker, a DE
        // tile is reached through its config aperture like every non-Tensix core.
        const tt::CoreType core_type = tensix_hosted ? tt::CoreType::TENSIX : tt::CoreType::DISPATCH;
        EXPECT_EQ(
            umd_resolver.resolve(tt_xy_pair(entry.x, entry.y), core_type, TEST_OFFSET, TEST_SIZE), *metal_address);
    }
}

// UMD resolves DRAM through the GDDR window by the channel tile's descriptor coordinate; tt-metal by
// logical bank (one interleaved bank per channel, so bank N is channel N). Both must land on the
// same selector for every one of the four boot-programmed channels.
TEST(QuasarAttUmdAgreement, DramAddressesMatch) {
    const tt::umd::att::Resolver umd_resolver(UMD_MAP);
    const tt::umd::att::Table<std::uint16_t>& umd_dram_words =
        UMD_MAP.endpoint_words[static_cast<std::size_t>(tt::umd::att::WindowClass::Dram)];
    ASSERT_EQ(METAL_MAP.dram_selectors.size(), 4u);

    for (std::uint32_t bank = 0; bank < METAL_MAP.dram_selectors.size(); ++bank) {
        SCOPED_TRACE("dram bank " + std::to_string(bank));
        const std::uint32_t selector = METAL_MAP.dram_selectors[bank];
        ASSERT_LT(selector, umd_dram_words.size());
        const std::uint16_t package_word = umd_dram_words[selector];
        ASSERT_NE(package_word, tt::umd::att::ENDPOINT_UNPOPULATED);

        const std::optional<noc_att::NocAddress> metal_address =
            noc_att::Address::dram(bank, TEST_OFFSET).encode<METAL_MAP>(TEST_SIZE);
        ASSERT_TRUE(metal_address.has_value());

        const tt_xy_pair channel_core(descriptor_x(package_word), descriptor_y(package_word));
        EXPECT_EQ(umd_resolver.resolve(channel_core, tt::CoreType::DRAM, TEST_OFFSET, TEST_SIZE), *metal_address);
    }
}
