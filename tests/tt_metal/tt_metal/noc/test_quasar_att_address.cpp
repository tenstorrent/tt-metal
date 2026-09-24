// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only golden tests of the typed ATT resolution layer: exact encoded
// addresses for every address kind on both product maps, host-frame (packed
// word) resolution through the map's frame offset, self-detection, rejection
// of out-of-map identities, and the worker multicast rectangle. No device is
// opened.

#include <gtest/gtest.h>

#include <cstdint>

#include "internal/tt-2xx/quasar/noc/att/att_address.h"
#include "internal/tt-2xx/quasar/noc/att/configs/grendel_qsr1_att_config.h"
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_att_config.h"

namespace {

using noc_att::Address;
using noc_att::ResolvedTile;
using noc_att::WindowClass;

constexpr const noc_att::MapData& QSR1 = grendel_qsr1_att_config::MAP;
constexpr const noc_att::MapData& AETHER = quasar_aether_2x3_att_config::MAP;

// ---------------------------------------------------------------------------
// QSR1 boot map goldens
// ---------------------------------------------------------------------------

TEST(QuasarAttAddressQsr1, LocalEncodesThroughTheConfigWindow) {
    // Local L1 0x300000 -> boot-patched ep256 self window: 0x18_0000_0000 | 0x300000.
    constexpr auto r = Address::local(0x300000).encode<QSR1>();
    static_assert(r.has_value());
    static_assert(*r == 0x1800300000ull);
    // The full 4 MiB of L1 is reachable; the 128 MiB window never relocates.
    static_assert(Address::local(0x3FFFFC).encode<QSR1>(4).has_value());
    static_assert(*Address::local(0).encode<QSR1>() == noc_att::local_window_base(QSR1));
}

TEST(QuasarAttAddressQsr1, WorkerEncodesThroughTheWorkerWindow) {
    // Worker (9,5): logical (7,3) -> selector 31 -> 0x100_0000_0000 | 31<<24 | offset.
    constexpr auto r = Address::worker(9, 5, 0x1234).encode<QSR1>();
    static_assert(r.has_value());
    static_assert(*r == 0x1001F001234ull);
    // The frame origin (2,2) is selector 0.
    static_assert(*Address::worker(2, 2, 0).encode<QSR1>() == 0x10000000000ull);
}

TEST(QuasarAttAddressQsr1, LoopbackScratchIsTheAbsoluteAperture) {
    static_assert(*Address::loopback_scratch(0x100000).encode<QSR1>() == 0x100000);
    static_assert(Address::loopback_scratch(0x1FFFFC).encode<QSR1>(4).has_value());
    // Below, above, and crossing the [0x100000, 0x200000) aperture are rejected.
    static_assert(!Address::loopback_scratch(0xFFFFF).encode<QSR1>().has_value());
    static_assert(!Address::loopback_scratch(0x200000).encode<QSR1>().has_value());
    static_assert(!Address::loopback_scratch(0x1FFFFC).encode<QSR1>(8).has_value());
}

TEST(QuasarAttAddressQsr1, OutOfMapIdentitiesAreRejected) {
    // Outside the 8x4 worker grid at frame origin (2,2).
    static_assert(!Address::worker(1, 2, 0).encode<QSR1>().has_value());
    static_assert(!Address::worker(2, 1, 0).encode<QSR1>().has_value());
    static_assert(!Address::worker(10, 2, 0).encode<QSR1>().has_value());
    static_assert(!Address::worker(2, 6, 0).encode<QSR1>().has_value());
    // Logical DRAM is bound to the four boot-programmed channels only.
    static_assert(!Address::dram(4, 0).encode<QSR1>().has_value());
    // Dispatch keys are descriptor-frame coordinates of the listed tiles: an
    // unlisted tile and the LIVE-frame coordinate of a DE tile both miss.
    static_assert(!Address::dispatch(0, 0, 0).encode<QSR1>().has_value());
    static_assert(!Address::dispatch(12, 8, 0).encode<QSR1>().has_value());
}

TEST(QuasarAttAddressQsr1, DramEncodesThroughTheDramWindow) {
    // Logical bank N -> selector N -> 0x1_0000_0000_0000 | N<<33 | offset (pass-through, no rebase).
    static_assert(*Address::dram(0, 0x1000).encode<QSR1>() == 0x1000000001000ull);
    static_assert(*Address::dram(1, 0x1000).encode<QSR1>() == (0x1000000000000ull | (1ull << 33) | 0x1000));
    static_assert(*Address::dram(3, 0).encode<QSR1>() == (0x1000000000000ull | (3ull << 33)));
    // The 8 GiB slot ends where the selector field starts.
    static_assert(Address::dram(0, (1ull << 33) - 4).encode<QSR1>(4).has_value());
    static_assert(!Address::dram(0, (1ull << 33) - 4).encode<QSR1>(8).has_value());
}

TEST(QuasarAttAddressQsr1, DispatchEncodesThroughItsTileWindow) {
    // DE tiles (descriptor frame) -> full-tile selectors 56..58: 0x18_0000_0000 | sel<<27 | offset.
    static_assert(*Address::dispatch(10, 6, 0).encode<QSR1>() == 0x19c0000000ull);
    static_assert(*Address::dispatch(10, 6, 0x40).encode<QSR1>() == (0x1800000000ull | (56ull << 27) | 0x40));
    static_assert(*Address::dispatch(10, 1, 0).encode<QSR1>() == (0x1800000000ull | (57ull << 27)));
    static_assert(*Address::dispatch(1, 6, 0).encode<QSR1>() == (0x1800000000ull | (58ull << 27)));
    // The interim Tensix dispatch tiles are worker selectors: the same operand Address::worker builds.
    static_assert(*Address::dispatch(9, 5, 0x1234).encode<QSR1>() == *Address::worker(9, 5, 0x1234).encode<QSR1>());
    static_assert(*Address::dispatch(9, 5, 0).encode<QSR1>() == 0x1001F000000ull);
}

TEST(QuasarAttAddressQsr1, HostFrameCoordinatesResolveThroughTheOffset) {
    // Host-packed words are descriptor-frame; the inverse tables are live-frame
    // (+2,+2). Worker 2-2 -> word 0x104 -> selector 0; 9-5 -> 0x1cb -> 31.
    static_assert(QSR1.node_id_offset_x == 2 && QSR1.node_id_offset_y == 2);
    constexpr ResolvedTile origin = noc_att::resolve_host_coordinate(QSR1, 2, 2);
    static_assert(origin.valid && origin.window == WindowClass::Worker && origin.selector == 0);
    constexpr ResolvedTile corner = noc_att::resolve_host_coordinate(QSR1, 9, 5);
    static_assert(corner.valid && corner.window == WindowClass::Worker && corner.selector == 31);
    // A mid-grid worker: host coordinate (2,5) is worker selector 24, and the forward path agrees.
    constexpr ResolvedTile row3 = noc_att::resolve_host_coordinate(QSR1, 2, 5);
    static_assert(row3.valid && row3.selector == 24);
    static_assert(noc_att::resolve(QSR1, Address::worker(2, 5, 0)).selector == 24);
    static_assert(
        noc_att::map_window(QSR1, row3.window).make_address(row3.selector, 0x4da00) ==
        *Address::worker(2, 5, 0x4da00).encode<QSR1>());
    // A DE tile resolves through the full-tile table to its dispatch selector.
    constexpr ResolvedTile de = noc_att::resolve_host_coordinate(QSR1, 10, 6);
    static_assert(de.valid && de.window == WindowClass::FullTile && de.selector == 56);
    // The live-frame coordinate of the origin is not a host coordinate...
    static_assert(!noc_att::resolve_host_coordinate(QSR1, 0, 0).valid);
    // ...while the DRAM channel tiles ([4-7], [8-7], [7-0], [3-0] -> live 0x246,
    // 0x24a, 0x089, 0x085) resolve through the DRAM endpoint words to the DRAM
    // window at the channel's selector (lane A), so a host coordinate naming a
    // DRAM tile (CQ write_linear to DRAM-sharded buffers) reaches the Mimir.
    constexpr auto dram0 = noc_att::resolve_host_coordinate(QSR1, 4, 7);
    static_assert(dram0.valid && dram0.window == WindowClass::Dram && dram0.selector == 0);
    constexpr auto dram1 = noc_att::resolve_host_coordinate(QSR1, 8, 7);
    static_assert(dram1.valid && dram1.window == WindowClass::Dram && dram1.selector == 1);
    constexpr auto dram2 = noc_att::resolve_host_coordinate(QSR1, 7, 0);
    static_assert(dram2.valid && dram2.window == WindowClass::Dram && dram2.selector == 2);
    constexpr auto dram3 = noc_att::resolve_host_coordinate(QSR1, 3, 0);
    static_assert(dram3.valid && dram3.window == WindowClass::Dram && dram3.selector == 3);
    // Lane B (second core of each channel pair) reaches the same channel via selector 16+N.
    constexpr auto dram0b = noc_att::resolve_host_coordinate(QSR1, 5, 7);
    static_assert(dram0b.valid && dram0b.window == WindowClass::Dram && dram0b.selector == 16);
    // Self-detection by host coordinate: tile 2-2 latches NOC_NODE_ID (4,4).
    static_assert(noc_att::host_coordinate_is_current(QSR1, 2, 2, 4, 4));
    static_assert(!noc_att::host_coordinate_is_current(QSR1, 4, 4, 4, 4));
    static_assert(!noc_att::host_coordinate_is_current(QSR1, 2, 2, 2, 2));
}

TEST(QuasarAttAddressQsr1, TransfersAreValidatedAgainstTheWindow) {
    constexpr std::uint64_t worker_limit = 16ull * 1024 * 1024;
    static_assert(Address::worker(2, 2, worker_limit - 4).encode<QSR1>(4).has_value());
    static_assert(!Address::worker(2, 2, worker_limit - 4).encode<QSR1>(8).has_value());  // crosses
    static_assert(!Address::worker(2, 2, 0).encode<QSR1>(0).has_value());                 // zero-size
    // The same check works on a finished operand.
    static_assert(noc_att::transfer_supported<QSR1>(0x1001F001234ull, 4));
    static_assert(!noc_att::transfer_supported<QSR1>(0x1001F001234ull, worker_limit));
    static_assert(!noc_att::transfer_supported<QSR1>(0xDEAD, 4));  // matches no window
}

TEST(QuasarAttAddressQsr1, SelfDetectionCoversAllSelfRoutes) {
    // (4,4) in the hardware frame is worker endpoint word 0x104 -> selector 0
    // of the worker window.
    constexpr ResolvedTile current = noc_att::resolve_current(QSR1, 4, 4);
    static_assert(current.valid);
    static_assert(current.window == WindowClass::Worker);
    static_assert(current.selector == 0);

    // The constant Local identity is self regardless of coordinates.
    static_assert(noc_att::is_self_address<QSR1>(0x1800300000ull, noc_att::INVALID_TILE));
    // This initiator's own worker-window address is self; a different selector is not.
    static_assert(noc_att::is_self_address<QSR1>(0x10000001234ull, current));
    static_assert(!noc_att::is_self_address<QSR1>(0x1001F001234ull, current));
    // The loopback scratch aperture is always local.
    static_assert(noc_att::is_self_address<QSR1>(0x100040ull, noc_att::INVALID_TILE));
}

TEST(QuasarAttAddressQsr1, ExtractLocalAddressInvertsEncode) {
    constexpr auto worker = noc_att::extract_local_address<QSR1>(0x1001F001234ull);
    static_assert(worker.has_value());
    static_assert(*worker == 0x1234);
    constexpr auto local = noc_att::extract_local_address<QSR1>(0x1800300000ull);
    static_assert(local.has_value());
    static_assert(*local == 0x300000);
    static_assert(!noc_att::extract_local_address<QSR1>(0xDEAD).has_value());
}

TEST(QuasarAttAddressQsr1, ScratchExtractionPreservesAbsoluteAddressesAndBounds) {
    for (std::uint64_t address : {0x100000ull, 0x100040ull, 0x1FFFFCull}) {
        const auto encoded = Address::loopback_scratch(address).encode<QSR1>(4);
        ASSERT_TRUE(encoded.has_value());
        EXPECT_EQ(*encoded, address);
        EXPECT_TRUE(noc_att::is_self_address<QSR1>(*encoded, noc_att::INVALID_TILE));
        EXPECT_EQ(noc_att::extract_local_address<QSR1>(*encoded), address);
        EXPECT_TRUE(noc_att::transfer_supported<QSR1>(*encoded, 4));
    }
    static_assert(!noc_att::transfer_supported<QSR1>(0x1FFFFCull, 8));
    static_assert(!Address::loopback_scratch(0x100000).encode<QSR1>(UINT64_MAX).has_value());
    static_assert(!Address::loopback_scratch(0x100000).encode<QSR1>(0).has_value());
}

TEST(QuasarAttAddressQsr1, FullTileAliasesMatchEveryInitiatorsPhysicalIdentity) {
    // Endpoint 256 is patched to the initiator; all other full-tile rows are
    // fixed. Workers must recognize both their worker and full-tile aliases.
    for (std::uint16_t current_word : grendel_qsr1_att_config::ATT_FULL_TILE_ENDPOINT_WORDS) {
        const auto current = noc_att::resolve_current(QSR1, current_word & 63, current_word >> 6);
        ASSERT_TRUE(current.valid);
        for (std::uint32_t selector = 0; selector < QSR1.full_tile_endpoint_words.size(); ++selector) {
            const auto address = grendel_qsr1_att_config::TILE_WINDOW.make_address(selector, 0x40);
            const bool expected_self = selector == 0 || QSR1.full_tile_endpoint_words[selector] == current_word;
            EXPECT_EQ(noc_att::is_self_address<QSR1>(address, current), expected_self)
                << "initiator " << current_word << ", full-tile selector " << selector;
            EXPECT_EQ(noc_att::is_self_address<QSR1>(address, noc_att::INVALID_TILE), selector == 0);
        }
        for (std::uint32_t selector = 0; selector < QSR1.worker_endpoint_words.size(); ++selector) {
            const auto address = grendel_qsr1_att_config::WORKER_WINDOW.make_address(selector, 0x40);
            EXPECT_EQ(
                noc_att::is_self_address<QSR1>(address, current), QSR1.worker_endpoint_words[selector] == current_word);
        }
    }
}

TEST(QuasarAttAddressQsr1, WorkerMulticastEncodesTheRectangleStart) {
    // Rectangle (2,2)..(9,5): full grid, 32 destinations, start at selector 0.
    constexpr noc_att::NocMulticastAddress mcast = noc_att::make_worker_multicast<QSR1>(2, 2, 9, 5, 0x40, 4);
    static_assert(mcast.rectangle_count == 32);
    static_assert(mcast.start_address == 0x10000000040ull);
    static_assert(mcast.extent_xy == ((4u << 6) | 8u));
    static_assert(mcast.end_address == (0x10000000040ull | (31ull << 24)));
    static_assert(mcast.start_node_xy == QSR1.worker_endpoint_words[0]);
    static_assert(mcast.end_node_xy == QSR1.worker_endpoint_words[31]);
    // Degenerate and out-of-map rectangles resolve to zero destinations.
    static_assert(noc_att::make_worker_multicast<QSR1>(9, 5, 2, 2, 0, 1).rectangle_count == 0);
    static_assert(noc_att::make_worker_multicast<QSR1>(2, 2, 10, 5, 0, 1).rectangle_count == 0);
}

TEST(QuasarAttAddressQsr1, PackedDescriptorRoundTrips) {
    constexpr noc_att::NocAddress descriptor = noc_att::make_multicast_descriptor(2, 2, 9, 5, 0x40);
    constexpr noc_att::NocMulticastAddress mcast = noc_att::resolve_worker_multicast<QSR1>(descriptor, 4);
    static_assert(mcast.rectangle_count == 32);
    static_assert(mcast.start_address == 0x10000000040ull);
}

// ---------------------------------------------------------------------------
// Aether 2x3 bring-up map goldens
// ---------------------------------------------------------------------------

TEST(QuasarAttAddressAether, LocalEncodesThroughTheTranslatingWindow) {
    // The local window has QSR1 slot-14 geometry: base | local at selector 0.
    static_assert(*Address::local(0x40000).encode<AETHER>() == 0x1800040000ull);
    static_assert(noc_att::local_window_base(AETHER) == 0x1800000000ull);
    static_assert(*Address::local(0).encode<AETHER>() == noc_att::local_window_base(AETHER));
}

TEST(QuasarAttAddressAether, WorkerAndDramEncodeThroughTheRemoteWindow) {
    // Worker (1,1) -> selector 1 -> 0x10_0000_0000 | 1<<26 | offset.
    static_assert(*Address::worker(1, 1, 0x1000).encode<AETHER>() == 0x1004001000ull);
    static_assert(*Address::worker(0, 1, 0).encode<AETHER>() == 0x1000000000ull);
    // Logical DRAM bank 1 -> selector 3 (aether_utils configure_aether_dram).
    static_assert(*Address::dram(1, 0x2000).encode<AETHER>() == (0x1000000000ull | (3ull << 26) | 0x2000));
    // The UMD-visible dispatch tile (0,2) -> tile selector 4 (endpoint word 0x80).
    static_assert(*Address::dispatch(0, 2, 0).encode<AETHER>() == (0x1000000000ull | (4ull << 26)));
}

TEST(QuasarAttAddressAether, OutOfMapIdentitiesAreRejected) {
    static_assert(!Address::worker(2, 1, 0).encode<AETHER>().has_value());
    static_assert(!Address::worker(0, 0, 0).encode<AETHER>().has_value());
    static_assert(!Address::dram(2, 0).encode<AETHER>().has_value());
    static_assert(!Address::dispatch(1, 2, 0).encode<AETHER>().has_value());
    static_assert(!Address::dispatch(2, 2, 0).encode<AETHER>().has_value());
}

TEST(QuasarAttAddressAether, SelfDetectionUsesThePatchedEntryZero) {
    // A local-window operand is self by contract; a RAW L1 address matches no
    // window on this map and is never self.
    static_assert(noc_att::is_self_address<AETHER>(0x1800040000ull, noc_att::INVALID_TILE));
    static_assert(!noc_att::is_self_address<AETHER>(0x40000ull, noc_att::INVALID_TILE));
    // This tile's own remote-window address is also self.
    constexpr ResolvedTile current = noc_att::resolve_current(AETHER, 0, 1);
    static_assert(current.valid);
    static_assert(current.window == WindowClass::Worker);
    static_assert(current.selector == 0);
    static_assert(noc_att::is_self_address<AETHER>(0x1000000040ull, current));
    static_assert(!noc_att::is_self_address<AETHER>(0x1004000040ull, current));
}

TEST(QuasarAttAddressAether, ScratchApertureIsUnsupported) {
    static_assert(!Address::loopback_scratch(0x100000).encode<AETHER>().has_value());
    static_assert(!Address::loopback_scratch(0x100040).encode<AETHER>(4).has_value());
    static_assert(!Address::loopback_scratch(0x1FFFFC).encode<AETHER>(4).has_value());
    static_assert(!noc_att::transfer_supported<AETHER>(0x100040, 4));
    static_assert(!noc_att::extract_local_address<AETHER>(0x100040).has_value());
    // The absent scratch role must not make its parked compare a real operand.
    static_assert(!noc_att::is_self_address<AETHER>(UINT64_MAX, noc_att::INVALID_TILE));
    static_assert(!noc_att::transfer_supported<AETHER>(UINT64_MAX, 1));
    static_assert(!noc_att::extract_local_address<AETHER>(UINT64_MAX).has_value());
    // The independent translating local window still supports local operands.
    constexpr auto local = Address::local(0x100040).encode<AETHER>(4);
    static_assert(local.has_value());
    static_assert(noc_att::transfer_supported<AETHER>(*local, 4));
    static_assert(*noc_att::extract_local_address<AETHER>(*local) == 0x100040);
}

TEST(QuasarAttAddressAether, LocalSelectorsFollowTheProgrammedEndpointRows) {
    // Local selectors 1..6 alias remote selectors 0..5 because their table
    // offsets are 0 and 1. Only local selector 0 is patched to self.
    for (std::uint16_t current_word : quasar_aether_2x3_att_config::ATT_FULL_TILE_ENDPOINT_WORDS) {
        const auto current = noc_att::resolve_current(AETHER, current_word & 63, current_word >> 6);
        ASSERT_TRUE(current.valid);
        for (std::uint32_t selector = 0; selector < 8; ++selector) {
            const auto address = quasar_aether_2x3_att_config::LOCAL_WINDOW.make_address(selector, 0x40);
            const bool expected_self = selector == 0 || (selector <= AETHER.full_tile_endpoint_words.size() &&
                                                         AETHER.full_tile_endpoint_words[selector - 1] == current_word);
            EXPECT_EQ(noc_att::is_self_address<AETHER>(address, current), expected_self)
                << "initiator " << current_word << ", local selector " << selector;
            EXPECT_EQ(noc_att::is_self_address<AETHER>(address, noc_att::INVALID_TILE), selector == 0);
        }
    }
}

TEST(QuasarAttAddressAether, MulticastPackingRejectsOverflowBeforeFieldsCanAlias) {
    for (std::uint32_t bad_coordinate : {64u, UINT32_MAX}) {
        const noc_att::NocAddress descriptors[] = {
            noc_att::make_multicast_descriptor(bad_coordinate, 1, 0, 1, 0x40),
            noc_att::make_multicast_descriptor(0, bad_coordinate, 0, 1, 0x40),
            noc_att::make_multicast_descriptor(0, 1, bad_coordinate, 1, 0x40),
            noc_att::make_multicast_descriptor(0, 1, 0, bad_coordinate, 0x40),
        };
        for (auto descriptor : descriptors) {
            EXPECT_EQ(noc_att::resolve_worker_multicast<AETHER>(descriptor, 4).rectangle_count, 0);
            EXPECT_EQ(noc_att::resolve_worker_multicast<QSR1>(descriptor, 4).rectangle_count, 0);
        }
    }
    for (std::uint64_t bad_offset : {std::uint64_t{1} << 36, UINT64_MAX}) {
        const auto descriptor = noc_att::make_multicast_descriptor(0, 1, 0, 1, bad_offset);
        EXPECT_EQ(noc_att::resolve_worker_multicast<AETHER>(descriptor, 4).rectangle_count, 0);
    }
    constexpr auto valid = noc_att::make_multicast_descriptor(0, 1, 0, 1, 0x40);
    static_assert(noc_att::resolve_worker_multicast<AETHER>(valid, 4).rectangle_count == 1);
    static_assert(noc_att::make_multicast_descriptor(63, 63, 63, 63, (1ull << 36) - 1) == (1ull << 60) - 1);
    for (std::uint32_t bit = 60; bit < 64; ++bit) {
        EXPECT_EQ(noc_att::resolve_worker_multicast<AETHER>(valid | (1ull << bit), 4).rectangle_count, 0);
    }
}

TEST(QuasarAttAddressAether, OversizedIdentitiesClampAndReject) {
    // 32-bit identities that do not fit 16 bits clamp to 0xFFFF, which no map
    // resolves - they must not wrap onto a valid selector.
    static_assert(!Address::worker(65536 + 1, 1, 0).encode<AETHER>().has_value());
    static_assert(!Address::worker(65536 + 2, 2, 0).encode<QSR1>().has_value());
    static_assert(!Address::dram(65536, 0).encode<AETHER>().has_value());
    static_assert(!Address::dispatch(65536 + 1, 2, 0).encode<AETHER>().has_value());
}

TEST(QuasarAttAddressAether, PackedDramEndpointsMatchAddressDram) {
    // On this map the NOC_NODE_ID frame is the descriptor frame (no offset),
    // so a host coordinate resolves through the inverse tables unchanged; a
    // DRAM tile coordinate lands on the same remote-window selector
    // Address::dram produces for that bank. Aether DRAM tiles: bank 0
    // -> (0,0), bank 1 -> (1,0). (Under ATT the kernels' DRAM path is typed on
    // every map; this pins the two views of the same tile together.)
    static_assert(AETHER.node_id_offset_x == 0 && AETHER.node_id_offset_y == 0);
    static_assert(noc_att::host_coordinate_is_current(AETHER, 0, 1, 0, 1));
    constexpr ResolvedTile bank0 = noc_att::resolve_host_coordinate(AETHER, 0, 0);
    static_assert(bank0.valid);
    static_assert(bank0.window == WindowClass::FullTile);
    static_assert(
        noc_att::map_window(AETHER, bank0.window).make_address(bank0.selector, 0x2000) ==
        *Address::dram(0, 0x2000).encode<AETHER>());
    constexpr ResolvedTile bank1 = noc_att::resolve_host_coordinate(AETHER, 1, 0);
    static_assert(bank1.valid);
    static_assert(bank1.window == WindowClass::FullTile);
    static_assert(
        noc_att::map_window(AETHER, bank1.window).make_address(bank1.selector, 0x2000) ==
        *Address::dram(1, 0x2000).encode<AETHER>());
}

TEST(QuasarAttAddressAether, WorkerMulticastSpansTheRow) {
    constexpr noc_att::NocMulticastAddress mcast = noc_att::make_worker_multicast<AETHER>(0, 1, 1, 1, 0x80, 4);
    static_assert(mcast.rectangle_count == 2);
    static_assert(mcast.start_address == 0x1000000080ull);
    static_assert(mcast.extent_xy == ((1u << 6) | 2u));
    // Inline-write form: the end tile's operand (selector 1) and both tiles' endpoint words.
    static_assert(mcast.end_address == (0x1000000080ull | (1ull << 26)));
    static_assert(mcast.start_node_xy == 0x40);
    static_assert(mcast.end_node_xy == 0x41);
}

// ---------------------------------------------------------------------------
// Decoding an operand back to the core it reaches, the way the watcher's
// NoC sanitizer and its host report do.
// ---------------------------------------------------------------------------

using noc_att::classify_operand;
using Kind = noc_att::OperandTarget::Kind;

TEST(QuasarAttOperandQsr1, WorkerOperandsClassifyToTheirEndpointRow) {
    // Worker (2,2) is logical (0,0): selector 0 of the worker window, live word 0x104 = (4,4).
    constexpr auto t = classify_operand(QSR1, *Address::worker(2, 2, 0x1234).encode<QSR1>());
    static_assert(t.kind == Kind::Worker);
    static_assert(t.window == WindowClass::Worker);
    static_assert(t.selector == 0);
    static_assert(t.local_address == 0x1234);
    static_assert(t.endpoint_known && t.endpoint_word == 0x104);
    // Worker (9,5) is logical (7,3): selector 31, live word 0x1cb = (11,7).
    constexpr auto corner = classify_operand(QSR1, *Address::worker(9, 5, 0).encode<QSR1>());
    static_assert(corner.kind == Kind::Worker && corner.selector == 31 && corner.endpoint_word == 0x1cb);
    // A worker-window selector past the 32 programmed rows reaches nothing.
    constexpr auto unlisted =
        classify_operand(QSR1, noc_att::map_window(QSR1, WindowClass::Worker).make_address(33, 0));
    static_assert(unlisted.kind == Kind::Invalid);
    static_assert(unlisted.window == WindowClass::Worker && unlisted.selector == 33);
}

TEST(QuasarAttOperandQsr1, DramOperandsClassifyToTheirBank) {
    constexpr auto bank1 = classify_operand(QSR1, *Address::dram(1, 0x4000).encode<QSR1>());
    static_assert(bank1.kind == Kind::Dram);
    static_assert(bank1.window == WindowClass::Dram);
    static_assert(bank1.selector == 1 && bank1.bank == 1);
    static_assert(bank1.local_address == 0x4000);
    static_assert(bank1.endpoint_known && bank1.endpoint_word == 0x24a);
    // Lane B rows (selectors 16..19) are programmed DRAM ingress nodes with no bound bank.
    constexpr auto laneB = classify_operand(QSR1, noc_att::map_window(QSR1, WindowClass::Dram).make_address(16, 0x10));
    static_assert(laneB.kind == Kind::Dram && laneB.bank == noc_att::DRAM_BANK_UNKNOWN);
    static_assert(laneB.endpoint_known && laneB.endpoint_word == 0x247);
    // An unprogrammed DRAM row reaches nothing.
    constexpr auto unprogrammed =
        classify_operand(QSR1, noc_att::map_window(QSR1, WindowClass::Dram).make_address(5, 0x10));
    static_assert(unprogrammed.kind == Kind::Invalid && unprogrammed.window == WindowClass::Dram);
}

TEST(QuasarAttOperandQsr1, SelfAndFullTileOperands) {
    // The local window is the full-tile window at the boot-patched selector 0.
    constexpr auto self = classify_operand(QSR1, *Address::local(0x300000).encode<QSR1>());
    static_assert(self.kind == Kind::Self && self.local_address == 0x300000);
    // The scratch aperture is pass-through: the operand is the absolute L1 address.
    constexpr auto scratch = classify_operand(QSR1, *Address::loopback_scratch(0x150000).encode<QSR1>());
    static_assert(scratch.kind == Kind::Self);
    static_assert(scratch.window == WindowClass::LoopbackScratch && scratch.local_address == 0x150000);
    // A dispatch engine tile reaches its L1 through the full-tile window: (10,6) is selector 56, word 0x20c.
    constexpr auto de = classify_operand(QSR1, *Address::dispatch(10, 6, 0x1000).encode<QSR1>());
    static_assert(de.kind == Kind::FullTile && de.selector == 56 && de.local_address == 0x1000);
    static_assert(de.endpoint_known && de.endpoint_word == 0x20c);
    // Nothing above the top window.
    constexpr auto garbage = classify_operand(QSR1, 0x5555000000000000ull);
    static_assert(garbage.kind == Kind::Invalid && garbage.window == WindowClass::Invalid);
}

TEST(QuasarAttOperandAether, OneSharedWindowIsToldApartBySelector) {
    // Workers: selectors 0 and 1 of the shared remote window.
    constexpr auto w0 = classify_operand(AETHER, *Address::worker(0, 1, 0x80).encode<AETHER>());
    static_assert(w0.kind == Kind::Worker && w0.selector == 0 && w0.endpoint_word == 0x40);
    static_assert(w0.local_address == 0x80);
    constexpr auto w1 = classify_operand(AETHER, *Address::worker(1, 1, 0x80).encode<AETHER>());
    static_assert(w1.kind == Kind::Worker && w1.selector == 1 && w1.endpoint_word == 0x41);
    // DRAM banks: selectors 2 and 3; their tiles live in the full-tile table on this map.
    constexpr auto d0 = classify_operand(AETHER, *Address::dram(0, 0x2000).encode<AETHER>());
    static_assert(d0.kind == Kind::Dram && d0.selector == 2 && d0.bank == 0 && d0.local_address == 0x2000);
    static_assert(d0.endpoint_known && d0.endpoint_word == 0x00);
    constexpr auto d1 = classify_operand(AETHER, *Address::dram(1, 0).encode<AETHER>());
    static_assert(d1.kind == Kind::Dram && d1.bank == 1 && d1.endpoint_known && d1.endpoint_word == 0x01);
    // The dispatch tile (0,2) is full-tile selector 4.
    constexpr auto dispatch = classify_operand(AETHER, *Address::dispatch(0, 2, 0x40).encode<AETHER>());
    static_assert(dispatch.kind == Kind::FullTile && dispatch.selector == 4 && dispatch.endpoint_word == 0x80);
    // Selector 6 is past the six programmed rows.
    constexpr auto unlisted =
        classify_operand(AETHER, noc_att::map_window(AETHER, WindowClass::Worker).make_address(6, 0));
    static_assert(unlisted.kind == Kind::Invalid && unlisted.window == WindowClass::Worker && unlisted.selector == 6);
}

TEST(QuasarAttOperandAether, LocalWindowIsSelfOnlyAtSelectorZero) {
    constexpr auto self = classify_operand(AETHER, *Address::local(0x1000).encode<AETHER>());
    static_assert(self.kind == Kind::Self && self.window == WindowClass::Local && self.local_address == 0x1000);
    // Other selectors of the local window are not an identity kernels can name.
    constexpr auto other = classify_operand(AETHER, noc_att::map_window(AETHER, WindowClass::Local).make_address(1, 0));
    static_assert(other.kind == Kind::Invalid && other.window == WindowClass::Local);
}

TEST(QuasarAttOperand, MulticastDescriptorRoundTrips) {
    constexpr auto rect = noc_att::decode_multicast_descriptor(noc_att::make_multicast_descriptor(0, 1, 1, 1, 0x80));
    static_assert(rect.valid);
    static_assert(rect.start_x == 0 && rect.start_y == 1 && rect.end_x == 1 && rect.end_y == 1);
    static_assert(rect.local_address == 0x80);
    static_assert(!noc_att::decode_multicast_descriptor(noc_att::INVALID_MULTICAST_DESCRIPTOR).valid);
    // The container is not self-identifying: a unicast operand decodes as some rectangle too, which is
    // why the sanitizer is told explicitly whether an operand is a multicast.
    static_assert(noc_att::decode_multicast_descriptor(*Address::worker(2, 2, 0).encode<QSR1>()).valid);
}
}  // namespace
