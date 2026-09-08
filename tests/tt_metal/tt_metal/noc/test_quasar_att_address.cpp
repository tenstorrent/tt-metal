// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only golden tests of the typed ATT resolution layer: exact encoded
// addresses for every address kind on both product maps, self-detection,
// rejection of out-of-map identities, and the worker multicast rectangle.
// No device is opened.

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
    // No logical DRAM or Dispatch binding exists in the checked-in descriptor:
    // the map's tables are empty, so these kinds resolve invalid.
    static_assert(!Address::dram(0, 0).encode<QSR1>().has_value());
    static_assert(!Address::dispatch(0, 0, 0).encode<QSR1>().has_value());
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

TEST(QuasarAttAddressQsr1, WorkerMulticastEncodesTheRectangleStart) {
    // Rectangle (2,2)..(9,5): full grid, 32 destinations, start at selector 0.
    constexpr noc_att::NocMulticastAddress mcast = noc_att::make_worker_multicast<QSR1>(2, 2, 9, 5, 0x40, 4);
    static_assert(mcast.rectangle_count == 32);
    static_assert(mcast.start_address == 0x10000000040ull);
    static_assert(mcast.extent_xy == ((4u << 6) | 8u));
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
    // The UMD-visible dispatch tile (1,2) -> tile selector 4.
    static_assert(*Address::dispatch(1, 2, 0).encode<AETHER>() == (0x1000000000ull | (4ull << 26)));
}

TEST(QuasarAttAddressAether, OutOfMapIdentitiesAreRejected) {
    static_assert(!Address::worker(2, 1, 0).encode<AETHER>().has_value());
    static_assert(!Address::worker(0, 0, 0).encode<AETHER>().has_value());
    static_assert(!Address::dram(2, 0).encode<AETHER>().has_value());
    static_assert(!Address::dispatch(0, 2, 0).encode<AETHER>().has_value());
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

TEST(QuasarAttAddressAether, OversizedIdentitiesClampAndReject) {
    // 32-bit identities that do not fit 16 bits clamp to 0xFFFF, which no map
    // resolves - they must not wrap onto a valid selector.
    static_assert(!Address::worker(65536 + 1, 1, 0).encode<AETHER>().has_value());
    static_assert(!Address::worker(65536 + 2, 2, 0).encode<QSR1>().has_value());
    static_assert(!Address::dram(65536, 0).encode<AETHER>().has_value());
    static_assert(!Address::dispatch(65536 + 1, 2, 0).encode<AETHER>().has_value());
}

TEST(QuasarAttAddressAether, PackedDramEndpointsMatchAddressDram) {
    // The packed bank-table path resolves a DRAM tile coordinate through the
    // inverse endpoint lookup; the result must equal Address::dram for the
    // same bank (regression for the DRAM-bank misrouting found on the first
    // emulator run). Aether DRAM tiles: bank 0 -> (0,0), bank 1 -> (1,0).
    constexpr ResolvedTile bank0 = noc_att::resolve_current(AETHER, 0, 0);
    static_assert(bank0.valid);
    static_assert(bank0.window == WindowClass::FullTile);
    static_assert(
        noc_att::map_window(AETHER, bank0.window).make_address(bank0.selector, 0x2000) ==
        *Address::dram(0, 0x2000).encode<AETHER>());
    constexpr ResolvedTile bank1 = noc_att::resolve_current(AETHER, 1, 0);
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
}

}  // namespace
