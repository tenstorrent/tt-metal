// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-only tests for the tile snapshot and the active-image bound. No
// hardware, no UMD, no Metal.
//
//   ninja -C build x280_runtime_host_tests && build/test/tt_metal/x280/x280_runtime_host_tests

#include "x280/device/x280_tile.hpp"
#include "x280/device/x280_tile_state.hpp"

#include "x280.h"
#include "x280_hw_addrs.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace x280;

// A snapshot of a tile parked in resident idle, as inspect() would build it.
TileState live_idle_tile() {
    TileState state;
    state.id.chip_id = 3;
    state.id.hw_index = 0;
    state.id.noc_x = 8;
    state.id.noc_y = 3;
    state.observation_valid = true;
    state.reset_released = true;
    state.raw_ecc_marker = X280_ECC_PRIME_MARKER_MAGIC;
    state.raw_ecc_extent = kLimEccPrimeBytes;
    state.ecc = LimEccState::Primed;
    state.ecc_primed_bytes = kLimEccPrimeBytes;
    state.lim_backed_bytes = lim_backed_bytes(0);
    for (uint32_t hart_id = 0; hart_id < kHartsPerTile; ++hart_id) {
        auto& hart = state.harts[hart_id];
        hart.hart_id = hart_id;
        hart.observation_valid = true;
        hart.control.run_value = 0x1234;
        hart.control.entry = X280_ACTIVE_FW_LOAD_ADDR;
        hart.state.protocol_version = X280_HART_PROTOCOL_VERSION;
        hart.state.phase = X280_HART_PHASE_IDLE;
        hart.state.heartbeat = X280_HART_IDLE_HEARTBEAT(hart_id);
    }
    state.resident_idle_live = true;
    return state;
}

TileIdentity tile_identity(int chip_id, int hw_index) {
    TileIdentity id;
    id.chip_id = chip_id;
    id.hw_index = hw_index;
    const auto coord = noc0_from_hw_l2cpu(hw_index);
    if (coord.has_value()) {
        id.noc_x = coord->x;
        id.noc_y = coord->y;
    }
    return id;
}

bool mentions(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

}  // namespace

// ---------------------------------------------------------------------------
// Tile identity
// ---------------------------------------------------------------------------

TEST(TileIdentity, EqualValuesCompareEqual) {
    const auto id = tile_identity(2, 1);
    EXPECT_EQ(id, id);
    EXPECT_EQ(id, tile_identity(2, 1));
}

TEST(TileIdentity, ChipIdIsPartOfTheIdentity) { EXPECT_NE(tile_identity(1, 0), tile_identity(2, 0)); }

TEST(TileIdentity, OrdersTilesByChipThenHardwareIndex) {
    std::vector<TileIdentity> actual = {
        tile_identity(2, 0),
        tile_identity(0, 3),
        tile_identity(1, 2),
        tile_identity(0, 1),
    };
    std::sort(actual.begin(), actual.end());

    const std::vector<TileIdentity> expected = {
        tile_identity(0, 1),
        tile_identity(0, 3),
        tile_identity(1, 2),
        tile_identity(2, 0),
    };
    EXPECT_EQ(actual, expected);
}

TEST(TileIdentity, TopologyFieldsParticipateInValueEquality) {
    auto changed_coord = tile_identity(1, 0);
    changed_coord.noc_y++;
    EXPECT_NE(changed_coord, tile_identity(1, 0));

    auto changed_harts = tile_identity(1, 0);
    changed_harts.num_harts++;
    EXPECT_NE(changed_harts, tile_identity(1, 0));
}

// ---------------------------------------------------------------------------
// Active image placement
// ---------------------------------------------------------------------------

constexpr uint64_t kStackBound = X280_ACTIVE_TRAP_DIAG_ADDR - kHartsPerTile * HART_STACK_SIZE;

TEST(ActiveImage, AcceptsAnImageInsideTheReservedRegion) {
    const auto check = validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, 4096, kLimEccPrimeBytes);
    EXPECT_TRUE(check.ok) << check.reason;
    EXPECT_TRUE(check.reason.empty());
}

TEST(ActiveImage, EndsAtTheRecordedPrimedLimBoundary) {
    constexpr uint64_t kPrimedEnd = X280_LIM_BASE + kLimEccPrimeBytes;
    constexpr size_t kMaximumBytes = kPrimedEnd - X280_ACTIVE_FW_LOAD_ADDR;
    EXPECT_TRUE(validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, kMaximumBytes, kLimEccPrimeBytes).ok);
    const auto one_past = validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, kMaximumBytes + 1u, kLimEccPrimeBytes);
    EXPECT_FALSE(one_past.ok);
    EXPECT_TRUE(mentions(one_past.reason, "recorded beside the prime marker")) << one_past.reason;
    EXPECT_LT(kPrimedEnd, kStackBound);
}

// A marker with no extent word is bounded by kLimEccUnrecordedPrimeBytes,
// however far the current prime walks. A 17 680-byte image is refused against
// the unrecorded extent and admitted once a recorded one covers it.
TEST(ActiveImage, UnrecordedExtentBoundsConservatively) {
    constexpr uint64_t kAssumedEnd = X280_LIM_BASE + kLimEccUnrecordedPrimeBytes;
    constexpr size_t kMaximumBytes = kAssumedEnd - X280_ACTIVE_FW_LOAD_ADDR;
    EXPECT_LT(kLimEccUnrecordedPrimeBytes, kLimEccPrimeBytes);
    EXPECT_TRUE(validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, kMaximumBytes, std::nullopt).ok);
    const auto one_past = validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, kMaximumBytes + 1u, std::nullopt);
    EXPECT_FALSE(one_past.ok);
    EXPECT_TRUE(mentions(one_past.reason, "records no extent")) << one_past.reason;

    constexpr size_t kFourTierRouteComputeBytes = 17680;
    EXPECT_FALSE(validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, kFourTierRouteComputeBytes, std::nullopt).ok);
    EXPECT_TRUE(validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, kFourTierRouteComputeBytes, kLimEccPrimeBytes).ok);
}

// A recorded extent that ends at or below the active base admits nothing.
TEST(ActiveImage, ExtentBelowTheActiveBaseAdmitsNothing) {
    EXPECT_FALSE(validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, 4, X280_IDLE_FW_REGION_SIZE).ok);
    EXPECT_FALSE(validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, 4, kLimCacheLineBytes).ok);
}

TEST(ActiveImage, StacksBindWhenTheRecordedExtentReachesThem) {
    constexpr size_t kMaximumBytes = kStackBound - X280_ACTIVE_FW_LOAD_ADDR;
    EXPECT_TRUE(validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, kMaximumBytes, X280_LIM_SIZE).ok);
    const auto one_past = validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, kMaximumBytes + 1u, X280_LIM_SIZE);
    EXPECT_FALSE(one_past.ok);
    EXPECT_TRUE(mentions(one_past.reason, "per-hart stacks")) << one_past.reason;
}

// An image sized so that load_addr + size wraps must not land under the bound.
TEST(ActiveImage, RejectsASizeThatWouldWrap) {
    const auto check = validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, static_cast<size_t>(-1), kLimEccPrimeBytes);
    EXPECT_FALSE(check.ok);
}

TEST(ActiveImage, RejectsAMisplacedImage) {
    const auto below = validate_active_image(X280_ACTIVE_FW_LOAD_ADDR - 0x1000, 4096, kLimEccPrimeBytes);
    EXPECT_FALSE(below.ok);

    // The resident-idle base is a plausible typo and must not pass the active
    // check: loading an active image there overwrites the idle firmware the
    // handoff returns to.
    const auto idle_base = validate_active_image(X280_IDLE_FW_LOAD_ADDR, 4096, kLimEccPrimeBytes);
    EXPECT_FALSE(idle_base.ok);
}

TEST(ActiveImage, RejectsAnEmptyImage) {
    const auto check = validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, 0, kLimEccPrimeBytes);
    EXPECT_FALSE(check.ok);
}

// The two regions are distinct, so an image that fits one may not fit the other.
TEST(ActiveImage, IsABiggerRegionThanResidentIdle) {
    const size_t past_idle = X280_ECC_PRIME_MARKER_OFFSET + 1;
    EXPECT_FALSE(validate_idle_image(X280_IDLE_FW_LOAD_ADDR, past_idle).ok);
    EXPECT_TRUE(validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, past_idle, std::nullopt).ok);
}

// ---------------------------------------------------------------------------
// describe()
// ---------------------------------------------------------------------------

TEST(Describe, LiveIdleTileReportsEveryHart) {
    const auto text = describe(live_idle_tile());
    EXPECT_TRUE(mentions(text, "chip 3 L2CPU 0"));
    EXPECT_TRUE(mentions(text, "reset bit 4"));
    EXPECT_TRUE(mentions(text, "released"));
    EXPECT_TRUE(mentions(text, "live"));
    EXPECT_TRUE(mentions(text, "hart 0")) << text;
    EXPECT_TRUE(mentions(text, "hart 3")) << text;
    EXPECT_TRUE(mentions(text, "version=0x10000")) << text;
    EXPECT_TRUE(mentions(text, "control_run=0x1234")) << text;
    EXPECT_TRUE(mentions(text, "entry=0x8001000")) << text;
}

// An unusable observation invalidates the whole snapshot, so none of the decoded
// fields may be rendered as though they had been read.
TEST(Describe, InvalidObservationPrintsTheReasonAndNothingElse) {
    TileState state;
    state.id.hw_index = 2;
    state.observation_valid = false;
    state.invalid_reason = "L2CPU_RESET read all ones";

    const auto text = describe(state);
    EXPECT_TRUE(mentions(text, "INVALID"));
    EXPECT_TRUE(mentions(text, "L2CPU_RESET read all ones"));
    EXPECT_FALSE(mentions(text, "wayenable"));
    EXPECT_FALSE(mentions(text, "heartbeat"));
}

// With every way committed the marker, mailboxes and trap block are never read.
// Rendering zeroes for them would look like an observation of a silent tile.
TEST(Describe, UnavailableLimSaysTheWordsWereNotRead) {
    TileState state = live_idle_tile();
    state.ecc = LimEccState::LimUnavailable;
    state.raw_wayenable = 0xF;
    state.lim_backed_bytes = lim_backed_bytes(0xF);

    const auto text = describe(state);
    EXPECT_TRUE(mentions(text, "were not read"));
    EXPECT_TRUE(mentions(text, "LIM backed 0 KiB"));
    EXPECT_FALSE(mentions(text, "marker 0x0"));
}

// Mailboxes that match on a tile still in reset are leftovers in SRAM. Saying
// only "not live" invites the reader to distrust a heartbeat that is exactly
// right, so the rendering has to name the cause.
TEST(Describe, StaleMailboxesAreCalledStale) {
    TileState state = live_idle_tile();
    state.reset_released = false;
    state.resident_idle_live = false;

    const auto text = describe(state);
    EXPECT_TRUE(mentions(text, "held"));
    EXPECT_TRUE(mentions(text, "stale LIM contents")) << text;
}

// The extent decides how large an image the tile admits, so the rendering
// says whether it was read from the marker or assumed.
TEST(Describe, PrimedExtentIsNamedAsRecordedOrAssumed) {
    const TileState recorded = live_idle_tile();
    EXPECT_TRUE(mentions(describe(recorded), "128 KiB primed")) << describe(recorded);

    TileState unrecorded = live_idle_tile();
    unrecorded.raw_ecc_extent = 0;
    unrecorded.ecc_primed_bytes.reset();
    EXPECT_TRUE(mentions(describe(unrecorded), "16 KiB assumed")) << describe(unrecorded);
}

TEST(Describe, TrapIsPrintedOnlyWhenSet) {
    TileState clean = live_idle_tile();
    EXPECT_FALSE(mentions(describe(clean), "trap"));

    TileState faulted = live_idle_tile();
    faulted.harts[1].trap.run_value = 9;
    faulted.harts[1].trap.mcause = 0x7;
    faulted.harts[1].trap.mepc = 0x08001234;
    faulted.harts[1].trap.trap_valid = 1;
    const auto text = describe(faulted);
    EXPECT_TRUE(mentions(text, "hart 1"));
    EXPECT_TRUE(mentions(text, "mcause=0x7"));
    EXPECT_TRUE(mentions(text, "mepc=0x8001234"));
}

TEST(Describe, TileStateWithoutHartRecordsIsNotResidentIdle) {
    TileState state;
    state.observation_valid = true;
    state.reset_released = true;
    state.ecc = LimEccState::Primed;
    state.resident_idle_live = false;
    EXPECT_FALSE(state.resident_idle_live);
    EXPECT_TRUE(mentions(describe(state), "hart 0"));
    EXPECT_TRUE(mentions(describe(state), "INVALID"));
}

TEST(Describe, MissingTelemetryIsNotReportedAsZero) {
    TileState state = live_idle_tile();
    EXPECT_TRUE(mentions(describe(state), "not reported"));

    state.pll_mhz = 1000;
    EXPECT_TRUE(mentions(describe(state), "1000 MHz"));
}

TEST(LimRange, RejectsAddressesPastTheEndWithoutUnsignedWrap) {
    const LimRange range{.begin = 0x1000, .end = 0x2000};
    EXPECT_FALSE(range.contains(0x2001, 1));
    EXPECT_FALSE(range.contains(std::numeric_limits<uint64_t>::max(), 1));
}

TEST(LimRange, HandlesEmptyAndInvalidRanges) {
    const LimRange empty{.begin = 0x1000, .end = 0x1000};
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_TRUE(empty.contains(0x1000, 0));
    EXPECT_FALSE(empty.contains(0x1000, 1));

    const LimRange invalid{.begin = 0x2000, .end = 0x1000};
    EXPECT_EQ(invalid.size(), 0u);
    EXPECT_FALSE(invalid.contains(0x2000, 0));
}
