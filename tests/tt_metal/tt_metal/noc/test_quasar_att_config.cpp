// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only checks of the Quasar ATT window math and the transcribed product
// map configurations. Every window field is asserted against its source dump:
// the QSR1 values against the grendelemulation qmk_att_dump.md mask table, the
// aether values against firmware/datamover/perf_testing_lib/aether_utils.h.
// No device is opened.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>

#include "internal/tt-2xx/quasar/noc/att/att.h"
#include "internal/tt-2xx/quasar/noc/att/configs/grendel_qsr1_att_config.h"
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_att_config.h"

namespace {

using noc_att::Window;

// A window claims [compare, compare + 2^mask_bits).
constexpr bool windows_disjoint(const Window& a, const Window& b) {
    const std::uint64_t a_end = a.compare + noc_att::low_mask(a.mask_bits) + 1;
    const std::uint64_t b_end = b.compare + noc_att::low_mask(b.mask_bits) + 1;
    return a_end <= b.compare || b_end <= a.compare;
}

template <typename T, std::size_t N>
bool all_unique(const T (&values)[N]) {
    std::set<T> seen(std::begin(values), std::end(values));
    return seen.size() == N;
}

// ---------------------------------------------------------------------------
// Window math (att.h) against hand-computed values
// ---------------------------------------------------------------------------

TEST(QuasarAttWindowMath, MakeAddressComposesBaseSelectorAndOffset) {
    constexpr Window w{0x10000000000ull, 30, 24, 6, 128, true};
    static_assert(w.make_address(0, 0) == 0x10000000000ull);
    static_assert(w.make_address(31, 0x1234) == 0x1001F001234ull);
    static_assert(w.selector(0x1001F001234ull) == 31);
    static_assert(w.local_address(0x1001F001234ull) == 0x1234);
    static_assert(w.endpoint_index(0x1001F001234ull) == 128 + 31);
}

TEST(QuasarAttWindowMath, LimitsFollowSelectorPlacement) {
    constexpr Window with_selector{0x10000000000ull, 30, 24, 6, 128, true};
    // The local address ends where the selector starts.
    static_assert(with_selector.local_address_limit() == (1ull << 24));
    static_assert(with_selector.selector_limit() == 64);

    constexpr Window no_selector{0x100000ull, 20, 0, 0, 256, false};
    // A selector-free window dedicates all ignored comparison bits to the local address.
    static_assert(no_selector.local_address_limit() == (1ull << 20));
    static_assert(no_selector.selector_limit() == 1);
    static_assert(no_selector.selector(0x1ABCDEull) == 0);
}

TEST(QuasarAttWindowMath, TransferSupportedChecksTheCompleteTransfer) {
    constexpr Window w{0x10000000000ull, 30, 24, 6, 128, true};
    constexpr std::uint64_t limit = 1ull << 24;
    static_assert(w.transfer_supported(0, 1));
    static_assert(w.transfer_supported(limit - 4, 4));
    static_assert(!w.transfer_supported(limit - 4, 5));  // crosses the window
    static_assert(!w.transfer_supported(limit, 1));      // starts outside
    static_assert(!w.transfer_supported(0, 0));          // zero-size
}

TEST(QuasarAttWindowMath, MatchesComparesAboveTheWildcardBits) {
    constexpr Window w{0x1800000000ull, 33, 27, 6, 256, true};
    static_assert(w.matches(0x1800000000ull));
    static_assert(w.matches(0x1800000000ull + noc_att::low_mask(33)));
    static_assert(!w.matches(0x1800000000ull + (1ull << 33)));
    static_assert(!w.matches(0x0));
}

// ---------------------------------------------------------------------------
// QSR1 boot map: every field against the QMK dump
// ---------------------------------------------------------------------------

TEST(QuasarAttQsr1Config, WorkerWindowMatchesDumpSlot4) {
    constexpr const Window& w = grendel_qsr1_att_config::WORKER_WINDOW;
    EXPECT_EQ(w.compare, 0x10000000000ull);
    EXPECT_EQ(w.mask_bits, 30);
    EXPECT_EQ(w.endpoint_shift, 24);
    EXPECT_EQ(w.endpoint_size, 6);
    EXPECT_EQ(w.endpoint_table_offset, 128);
    EXPECT_TRUE(w.translate_address);
    // 16 MiB per NEO.
    EXPECT_EQ(w.local_address_limit(), 16ull * 1024 * 1024);
}

TEST(QuasarAttQsr1Config, DramWindowMatchesDumpSlot5) {
    constexpr const Window& w = grendel_qsr1_att_config::DRAM_WINDOW;
    EXPECT_EQ(w.compare, 0x1000000000000ull);
    EXPECT_EQ(w.mask_bits, 38);
    EXPECT_EQ(w.endpoint_shift, 33);
    EXPECT_EQ(w.endpoint_size, 5);
    EXPECT_EQ(w.endpoint_table_offset, 96);
    EXPECT_FALSE(w.translate_address);
}

TEST(QuasarAttQsr1Config, LoopbackScratchWindowMatchesDumpSlot13) {
    constexpr const Window& w = grendel_qsr1_att_config::LOOPBACK_SCRATCH_WINDOW;
    EXPECT_EQ(w.compare, 0x100000ull);
    EXPECT_EQ(w.mask_bits, 20);
    EXPECT_EQ(w.endpoint_shift, 0);
    EXPECT_EQ(w.endpoint_size, 0);
    EXPECT_EQ(w.endpoint_table_offset, 256);
    EXPECT_FALSE(w.translate_address);
    // The 1 MiB pass-through aperture [0x100000, 0x200000).
    EXPECT_EQ(w.local_address_limit(), 0x100000ull);
}

TEST(QuasarAttQsr1Config, TileWindowMatchesDumpSlot14) {
    constexpr const Window& w = grendel_qsr1_att_config::TILE_WINDOW;
    EXPECT_EQ(w.compare, 0x1800000000ull);
    EXPECT_EQ(w.mask_bits, 33);
    EXPECT_EQ(w.endpoint_shift, 27);
    EXPECT_EQ(w.endpoint_size, 6);
    EXPECT_EQ(w.endpoint_table_offset, 256);
    EXPECT_TRUE(w.translate_address);
    // 128 MiB per tile.
    EXPECT_EQ(w.local_address_limit(), 128ull * 1024 * 1024);
}

TEST(QuasarAttQsr1Config, LocalWindowIsTheTileWindowAtSelectorZero) {
    static_assert(grendel_qsr1_att_config::LOCAL_WINDOW_BASE == 0x1800000000ull);
    static_assert(
        grendel_qsr1_att_config::LOCAL_WINDOW.make_address(0, 0) == grendel_qsr1_att_config::LOCAL_WINDOW_BASE);
    // The base has no bits inside the local-address or selector fields, so
    // base | local_address is exactly make_address(selector 0, local_address).
    constexpr const Window& w = grendel_qsr1_att_config::LOCAL_WINDOW;
    static_assert((grendel_qsr1_att_config::LOCAL_WINDOW_BASE & noc_att::low_mask(w.endpoint_shift)) == 0);
    static_assert(w.selector(grendel_qsr1_att_config::LOCAL_WINDOW_BASE) == 0);
}

TEST(QuasarAttQsr1Config, SelectorTablesAreCompleteAndUnique) {
    constexpr std::size_t worker_count =
        grendel_qsr1_att_config::ATT_WORKER_GRID_X * grendel_qsr1_att_config::ATT_WORKER_GRID_Y;
    static_assert(sizeof(grendel_qsr1_att_config::ATT_WORKER_SELECTORS) / sizeof(std::uint8_t) == worker_count);
    static_assert(sizeof(grendel_qsr1_att_config::ATT_WORKER_ENDPOINT_WORDS) / sizeof(std::uint16_t) == worker_count);
    // 60 populated full-tile endpoints (256..315).
    static_assert(sizeof(grendel_qsr1_att_config::ATT_FULL_TILE_ENDPOINT_WORDS) / sizeof(std::uint16_t) == 60);

    EXPECT_TRUE(all_unique(grendel_qsr1_att_config::ATT_WORKER_SELECTORS));
    EXPECT_TRUE(all_unique(grendel_qsr1_att_config::ATT_WORKER_ENDPOINT_WORDS));
    EXPECT_TRUE(all_unique(grendel_qsr1_att_config::ATT_FULL_TILE_ENDPOINT_WORDS));

    for (std::uint8_t selector : grendel_qsr1_att_config::ATT_WORKER_SELECTORS) {
        EXPECT_TRUE(grendel_qsr1_att_config::WORKER_WINDOW.selector_supported(selector));
    }
    // The worker endpoints are the first 32 rows of the full-tile table: the
    // same physical tiles reachable through either window.
    for (std::size_t i = 0; i < worker_count; ++i) {
        EXPECT_EQ(
            grendel_qsr1_att_config::ATT_WORKER_ENDPOINT_WORDS[i],
            grendel_qsr1_att_config::ATT_FULL_TILE_ENDPOINT_WORDS[i]);
    }
}

TEST(QuasarAttQsr1Config, WindowsAreDisjoint) {
    constexpr const Window* windows[] = {
        &grendel_qsr1_att_config::LOOPBACK_SCRATCH_WINDOW,
        &grendel_qsr1_att_config::WORKER_WINDOW,
        &grendel_qsr1_att_config::DRAM_WINDOW,
        &grendel_qsr1_att_config::TILE_WINDOW,
    };
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            EXPECT_TRUE(windows_disjoint(*windows[i], *windows[j])) << "windows " << i << " and " << j << " overlap";
        }
    }
}

// ---------------------------------------------------------------------------
// Aether 2x3 bring-up map: every field against aether_utils.h
// ---------------------------------------------------------------------------

TEST(QuasarAttAetherConfig, LocalWindowHasQsr1Slot14Geometry) {
    constexpr const Window& w = quasar_aether_2x3_att_config::LOCAL_WINDOW;
    EXPECT_EQ(w.compare, 0x1800000000ull);
    EXPECT_EQ(w.mask_bits, 33);
    EXPECT_EQ(w.endpoint_shift, 26);
    EXPECT_EQ(w.endpoint_size, 6);
    EXPECT_EQ(w.endpoint_table_offset, 0);
    EXPECT_TRUE(w.translate_address);
    // base | local matches the window at selector 0 (the patched self entry)
    // with the local address intact; a RAW L1 address matches no window.
    static_assert(quasar_aether_2x3_att_config::LOCAL_WINDOW.matches(0x1800040000ull));
    static_assert(quasar_aether_2x3_att_config::LOCAL_WINDOW.selector(0x1800040000ull) == 0);
    static_assert(quasar_aether_2x3_att_config::LOCAL_WINDOW.local_address(0x1800040000ull) == 0x40000);
    static_assert(!quasar_aether_2x3_att_config::LOCAL_WINDOW.matches(0x40000ull));
}

TEST(QuasarAttAetherConfig, RemoteWindowMatchesSlot1Narrowed) {
    constexpr const Window& w = quasar_aether_2x3_att_config::REMOTE_WINDOW;
    EXPECT_EQ(w.compare, 0x1000000000ull);
    EXPECT_EQ(w.mask_bits, 32);
    EXPECT_EQ(w.endpoint_shift, 26);
    EXPECT_EQ(w.endpoint_size, 6);
    EXPECT_EQ(w.endpoint_table_offset, 1);
    EXPECT_TRUE(w.translate_address);
}

TEST(QuasarAttAetherConfig, LocalWindowBaseIsTheQsr1Base) {
    static_assert(quasar_aether_2x3_att_config::LOCAL_WINDOW_BASE == 0x1800000000ull);
    static_assert(quasar_aether_2x3_att_config::LOCAL_WINDOW.make_address(0, 0x123456ull) == 0x1800123456ull);
}

TEST(QuasarAttAetherConfig, SelectorTablesMatchAetherUtils) {
    EXPECT_TRUE(all_unique(quasar_aether_2x3_att_config::ATT_TILE_SELECTORS));
    EXPECT_TRUE(all_unique(quasar_aether_2x3_att_config::ATT_FULL_TILE_ENDPOINT_WORDS));
    static_assert(
        sizeof(quasar_aether_2x3_att_config::ATT_TILE_SELECTORS) / sizeof(std::uint8_t) ==
        quasar_aether_2x3_att_config::ATT_TILE_GRID_X * quasar_aether_2x3_att_config::ATT_TILE_GRID_Y);

    // aether_utils.h: workers (0,1),(1,1) at selectors 0,1; DRAM (0,0),(1,0) at 2,3.
    EXPECT_EQ(quasar_aether_2x3_att_config::ATT_WORKER_SELECTORS[0], 0);
    EXPECT_EQ(quasar_aether_2x3_att_config::ATT_WORKER_SELECTORS[1], 1);
    EXPECT_EQ(quasar_aether_2x3_att_config::ATT_LOGICAL_DRAM_SELECTORS[0], 2);
    EXPECT_EQ(quasar_aether_2x3_att_config::ATT_LOGICAL_DRAM_SELECTORS[1], 3);
    // Endpoint words encode (y << 6) | x.
    EXPECT_EQ(quasar_aether_2x3_att_config::ATT_WORKER_ENDPOINT_WORDS[0], (1 << 6) | 0);
    EXPECT_EQ(quasar_aether_2x3_att_config::ATT_WORKER_ENDPOINT_WORDS[1], (1 << 6) | 1);
}

TEST(QuasarAttAetherConfig, WindowsAreDisjoint) {
    EXPECT_TRUE(
        windows_disjoint(quasar_aether_2x3_att_config::LOCAL_WINDOW, quasar_aether_2x3_att_config::REMOTE_WINDOW));
}

}  // namespace
