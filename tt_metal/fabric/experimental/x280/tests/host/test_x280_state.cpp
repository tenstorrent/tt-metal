// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-only tests for the X280 component's pure rules. No hardware, no UMD, no
// Metal — these must pass on a machine with no Tenstorrent device.
//
//   ninja -C build x280_runtime_host_tests && build/test/tt_metal/x280/x280_runtime_host_tests

#include "x280/device/x280_state.hpp"
#include "x280/device/x280_deadline.hpp"
#include "x280/device/x280_runtime.hpp"

#include "x280_hart_protocol.h"
#include "x280.h"
#include "x280_hw_addrs.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <vector>

namespace {

using namespace x280;

constexpr uint32_t kAllOnes32 = 0xFFFFFFFFu;
constexpr uint64_t kAllOnes64 = 0xFFFFFFFFFFFFFFFFull;

}  // namespace

// ---------------------------------------------------------------------------
// LIM ECC classification
// ---------------------------------------------------------------------------

// The trap this whole classification exists to avoid: masking WayEnable with
// 0xF before testing it turns an unclocked tile's 0xFFFFFFFF into an
// apparently valid committed 0xF. An all-ones read is an invalid observation,
// never a state.
TEST(LimEcc, AllOnesWayEnableIsInvalidNotCommitted) {
    EXPECT_EQ(classify_lim_ecc(kAllOnes32, X280_ECC_PRIME_MARKER_MAGIC, 0).state, LimEccState::Invalid);
    EXPECT_EQ(classify_lim_ecc(kAllOnes32, 0, 0).state, LimEccState::Invalid);
    EXPECT_NE(classify_lim_ecc(kAllOnes32, 0, 0).state, LimEccState::LimUnavailable);
}

TEST(LimEcc, AllOnesMarkerIsInvalid) { EXPECT_EQ(classify_lim_ecc(0, kAllOnes64, 0).state, LimEccState::Invalid); }

TEST(LimEcc, NoBackedLimDoesNotDependOnMarkerContents) {
    EXPECT_EQ(classify_lim_ecc(0xF, X280_ECC_PRIME_MARKER_MAGIC, 0).state, LimEccState::LimUnavailable);
    EXPECT_EQ(classify_lim_ecc(0xF, kAllOnes64, 0).state, LimEccState::LimUnavailable);
}

TEST(LimEcc, PartialLimUsesTheMarker) {
    EXPECT_EQ(classify_lim_ecc(0x1, X280_ECC_PRIME_MARKER_MAGIC, 0).state, LimEccState::Primed);
    EXPECT_EQ(classify_lim_ecc(0x1, 0, 0).state, LimEccState::Unprimed);
}

TEST(LimEcc, PrimedRequiresMarkerAndBackedLim) {
    EXPECT_EQ(classify_lim_ecc(0, X280_ECC_PRIME_MARKER_MAGIC, 0).state, LimEccState::Primed);
}

TEST(LimEcc, MissingMarkerIsUnprimed) {
    EXPECT_EQ(classify_lim_ecc(0, 0, 0).state, LimEccState::Unprimed);
    EXPECT_EQ(classify_lim_ecc(0, 0x0123456789ABCDEFull, 0).state, LimEccState::Unprimed);
}

TEST(LimEcc, HighBitsOfWayEnableAreIgnored) {
    // The CSR is 4 bits wide inside a 32-bit read.
    EXPECT_EQ(classify_lim_ecc(0xABCD0000u, X280_ECC_PRIME_MARKER_MAGIC, 0).state, LimEccState::Primed);
}

// A marker seeded without an extent word is still a prime, but it attests to
// nothing beyond kLimEccUnrecordedPrimeBytes; the extent stays empty rather
// than reading as the current walk.
TEST(LimEcc, ExtentlessMarkerIsPrimedWithNoRecordedExtent) {
    const auto ecc = classify_lim_ecc(0, X280_ECC_PRIME_MARKER_MAGIC, 0);
    EXPECT_EQ(ecc.state, LimEccState::Primed);
    EXPECT_FALSE(ecc.primed_bytes.has_value());
}

TEST(LimEcc, RecordedExtentIsCarriedWithPrimed) {
    const auto ecc = classify_lim_ecc(0, X280_ECC_PRIME_MARKER_MAGIC, kLimEccPrimeBytes);
    EXPECT_EQ(ecc.state, LimEccState::Primed);
    ASSERT_TRUE(ecc.primed_bytes.has_value());
    EXPECT_EQ(*ecc.primed_bytes, kLimEccPrimeBytes);

    const auto one_way = classify_lim_ecc(0, X280_ECC_PRIME_MARKER_MAGIC, kLimWayBytes);
    EXPECT_EQ(one_way.state, LimEccState::Primed);
    EXPECT_EQ(one_way.primed_bytes, std::optional<uint64_t>{kLimWayBytes});
}

// The walk is whole cache lines within the one pinned way, so an extent that
// is neither cannot have been written by a prime: the record is not intact.
TEST(LimEcc, ImplausibleExtentIsUnprimed) {
    EXPECT_EQ(classify_lim_ecc(0, X280_ECC_PRIME_MARKER_MAGIC, kLimEccPrimeLineSize + 1).state, LimEccState::Unprimed);
    EXPECT_EQ(
        classify_lim_ecc(0, X280_ECC_PRIME_MARKER_MAGIC, kLimWayBytes + kLimEccPrimeLineSize).state,
        LimEccState::Unprimed);
    EXPECT_EQ(classify_lim_ecc(0, X280_ECC_PRIME_MARKER_MAGIC, kAllOnes64).state, LimEccState::Unprimed);
    EXPECT_FALSE(classify_lim_ecc(0, X280_ECC_PRIME_MARKER_MAGIC, kAllOnes64).primed_bytes.has_value());
}

TEST(LimEcc, ExtentDoesNotRescueANonMagicMarker) {
    const auto ecc = classify_lim_ecc(0, 0, kLimEccPrimeBytes);
    EXPECT_EQ(ecc.state, LimEccState::Unprimed);
    EXPECT_FALSE(ecc.primed_bytes.has_value());
}

TEST(LimEcc, BackedCapacityFollowsWayEnable) {
    EXPECT_EQ(lim_backed_bytes(0), 15u * 128u * 1024u);  // 1.875 MiB at reset
    EXPECT_EQ(lim_backed_bytes(1), 14u * 128u * 1024u);
    EXPECT_EQ(lim_backed_bytes(15), 0u);
}

TEST(LimShape, DefaultPreservesTheResetPartition) {
    const auto check = validate_lim_shape(LimShape{}, 0);
    ASSERT_TRUE(check.ok) << check.reason;
    EXPECT_EQ(check.target_wayenable, 0u);
    EXPECT_EQ(check.backed_bytes, 15u * kLimWayBytes);
}

TEST(LimShape, ValidatesHardwareAndRequiredCapacity) {
    constexpr uint64_t required = X280_HART_PROTOCOL_END - X280_LIM_BASE;
    const auto minimum = validate_lim_shape(LimShape{.lim_ways = 10}, required);
    ASSERT_TRUE(minimum.ok) << minimum.reason;
    EXPECT_EQ(minimum.target_wayenable, 5u);

    const auto too_small = validate_lim_shape(LimShape{.lim_ways = 9}, required);
    EXPECT_FALSE(too_small.ok);
    EXPECT_FALSE(too_small.reason.empty());

    const auto too_many = validate_lim_shape(LimShape{.lim_ways = 16}, 0);
    EXPECT_FALSE(too_many.ok);
    EXPECT_FALSE(too_many.reason.empty());
}

TEST(LimShape, IncreaseAppliesOnlyWhileEveryTileIsResetHeld) {
    EXPECT_EQ(classify_lim_shape_transition(0, 5, true), LimShapeTransition::ApplyWhileReset);
    EXPECT_EQ(classify_lim_shape_transition(0, 5, false), LimShapeTransition::ResetRequired);
}

TEST(LimShape, ExpandingLimRequiresReset) {
    EXPECT_EQ(classify_lim_shape_transition(5, 0, true), LimShapeTransition::ResetRequired);
    EXPECT_EQ(classify_lim_shape_transition(5, 5, false), LimShapeTransition::AlreadyConfigured);
    EXPECT_EQ(classify_lim_shape_transition(0xFFFFFFF5u, 5, false), LimShapeTransition::AlreadyConfigured);
}

TEST(LimBlock, ReadRequiresFourByteAlignmentAndAvailableRange) {
    const LimRange available{0x1000, 0x2000};
    EXPECT_TRUE(validate_available_lim_read(available, 0x1000, 4).ok);
    EXPECT_FALSE(validate_available_lim_read(available, 0x1001, 4).ok);
    EXPECT_FALSE(validate_available_lim_read(available, 0x1000, 2).ok);
    EXPECT_FALSE(validate_available_lim_read(available, 0x2000, 4).ok);
}

TEST(LimBlock, WriteRequiresWholeCacheLines) {
    const LimRange available{0x1000, 0x2000};
    EXPECT_TRUE(validate_available_lim_write(available, 0x1000, kLimCacheLineBytes).ok);
    EXPECT_FALSE(validate_available_lim_write(available, 0x1004, kLimCacheLineBytes).ok);
    EXPECT_FALSE(validate_available_lim_write(available, 0x1000, kLimCacheLineBytes - 4).ok);
}

TEST(LimBlock, RejectsEmptyAndOverflowingRanges) {
    const LimRange available{0x1000, 0x2000};
    EXPECT_FALSE(validate_available_lim_read(available, 0x1000, 0).ok);
    EXPECT_FALSE(validate_available_lim_write(available, 0x1000, 0).ok);
    EXPECT_FALSE(validate_available_lim_read(available, UINT64_MAX - 3, 8).ok);
}

// ---------------------------------------------------------------------------
// Argument-block placement
// ---------------------------------------------------------------------------

TEST(ArgumentBlock, NoArgumentsIsAcceptedAndHalfSpecifiedIsNot) {
    const LimRange available{0x1000, 0x2000};
    EXPECT_TRUE(validate_argument_block(available, 0, 0).ok);
    // An address with no extent cannot be bounds-checked, and an extent with no
    // address would be published as "no arguments".
    EXPECT_FALSE(validate_argument_block(available, 0x1000, 0).ok);
    EXPECT_FALSE(validate_argument_block(available, 0, kLimCacheLineBytes).ok);
}

TEST(ArgumentBlock, TakesTheWriteRuleBecauseAWriteIsHowItWasPopulated) {
    const LimRange available{0x1000, 0x2000};
    EXPECT_TRUE(validate_argument_block(available, 0x1000, kLimCacheLineBytes).ok);
    EXPECT_FALSE(validate_argument_block(available, 0x1004, kLimCacheLineBytes).ok);
    EXPECT_FALSE(validate_argument_block(available, 0x1000, kLimCacheLineBytes - 4).ok);
}

TEST(ArgumentBlock, RefusesABlockThatLeavesAvailableLim) {
    const LimRange available{0x1000, 0x2000};
    EXPECT_FALSE(validate_argument_block(available, 0x0FC0, kLimCacheLineBytes).ok);
    EXPECT_FALSE(validate_argument_block(available, 0x2000, kLimCacheLineBytes).ok);
    EXPECT_FALSE(validate_argument_block(available, 0x1FC0, 2 * kLimCacheLineBytes).ok);
    const auto rejected = validate_argument_block(available, 0x2000, kLimCacheLineBytes);
    EXPECT_FALSE(rejected.reason.empty());
}

// ---------------------------------------------------------------------------
// Resident-idle image placement
// ---------------------------------------------------------------------------

TEST(IdleImage, AcceptsAnImageInsideTheReservedRegion) {
    const auto check = validate_idle_image(X280_IDLE_FW_LOAD_ADDR, 512);
    EXPECT_TRUE(check.ok) << check.reason;
    EXPECT_TRUE(check.reason.empty());
}

TEST(IdleImage, AcceptsAnImageEndingExactlyAtTheMarker) {
    // The bound is exclusive: an image ending at the marker address does not
    // touch it.
    const auto check = validate_idle_image(X280_IDLE_FW_LOAD_ADDR, X280_ECC_PRIME_MARKER_OFFSET);
    EXPECT_TRUE(check.ok) << check.reason;
}

TEST(IdleImage, RejectsAnOversizedImage) {
    // One byte past the bound would overwrite the ECC prime marker, which can
    // wedge the tile until an ASIC reset.
    const auto one_past = validate_idle_image(X280_IDLE_FW_LOAD_ADDR, X280_ECC_PRIME_MARKER_OFFSET + 1);
    EXPECT_FALSE(one_past.ok);
    EXPECT_FALSE(one_past.reason.empty());

    // Filling the whole 4 KiB region would also take out the idle stack and the
    // trap-diagnostic block at the top of it.
    const auto whole_region = validate_idle_image(X280_IDLE_FW_LOAD_ADDR, X280_IDLE_FW_REGION_SIZE);
    EXPECT_FALSE(whole_region.ok);
    EXPECT_FALSE(whole_region.reason.empty());

    // A size that makes load_addr + size wrap must not slip under the bound.
    EXPECT_FALSE(validate_idle_image(X280_IDLE_FW_LOAD_ADDR, SIZE_MAX).ok);
}

TEST(IdleImage, RejectsAMisplacedImage) {
    // Active-firmware region: right size, wrong place.
    const auto above = validate_idle_image(X280_ACTIVE_FW_LOAD_ADDR, 512);
    EXPECT_FALSE(above.ok);
    EXPECT_FALSE(above.reason.empty());

    // Below the region base entirely.
    const auto below = validate_idle_image(X280_IDLE_FW_LOAD_ADDR - 0x1000, 512);
    EXPECT_FALSE(below.ok);
    EXPECT_FALSE(below.reason.empty());

    // Inside the region but not at its base: the reset vectors point at the
    // base, so this image would never execute.
    const auto offset = validate_idle_image(X280_IDLE_FW_LOAD_ADDR + 0x100, 512);
    EXPECT_FALSE(offset.ok);
}

TEST(IdleImage, RejectsAnEmptyImage) {
    const auto check = validate_idle_image(X280_IDLE_FW_LOAD_ADDR, 0);
    EXPECT_FALSE(check.ok);
    EXPECT_FALSE(check.reason.empty());
}

// ---------------------------------------------------------------------------
// Tile numbering
// ---------------------------------------------------------------------------

// UMD sorts its L2CPU cores by Y, giving {(8,3),(8,5),(8,7),(8,9)}; the hardware
// numbering is {0:(8,3), 1:(8,9), 2:(8,5), 3:(8,7)}. Conflating them selects the
// wrong tile while looking entirely plausible.
TEST(TileNumbering, MatchesTheDocumentedTable) {
    EXPECT_EQ(hw_l2cpu_from_noc0(8, 3), 0);
    EXPECT_EQ(hw_l2cpu_from_noc0(8, 9), 1);
    EXPECT_EQ(hw_l2cpu_from_noc0(8, 5), 2);
    EXPECT_EQ(hw_l2cpu_from_noc0(8, 7), 3);
}

TEST(TileNumbering, RoundTrips) {
    for (int hw = 0; hw < kNumL2cpuTiles; hw++) {
        const auto coord = noc0_from_hw_l2cpu(hw);
        ASSERT_TRUE(coord.has_value()) << "hw index " << hw;
        EXPECT_EQ(hw_l2cpu_from_noc0(coord->x, coord->y), hw);
    }
}

TEST(TileNumbering, RejectsCoordinatesOutsideTheDocumentedSet) {
    EXPECT_EQ(hw_l2cpu_from_noc0(8, 4), -1);
    EXPECT_EQ(hw_l2cpu_from_noc0(9, 3), -1);
    EXPECT_EQ(hw_l2cpu_from_noc0(0, 0), -1);
    EXPECT_FALSE(noc0_from_hw_l2cpu(-1).has_value());
    EXPECT_FALSE(noc0_from_hw_l2cpu(kNumL2cpuTiles).has_value());
}

TEST(TileNumbering, ResetBitIsFourPlusHardwareIndex) {
    EXPECT_EQ(l2cpu_reset_bit(0), 4);
    EXPECT_EQ(l2cpu_reset_bit(3), 7);
    EXPECT_EQ(l2cpu_reset_bit(4), -1);
    EXPECT_EQ(l2cpu_reset_bit(-1), -1);
}

// ---------------------------------------------------------------------------
// Bounded waiting
// ---------------------------------------------------------------------------

namespace {

// Time only advances when the poll loop sleeps, so the timeout logic is tested
// without spending any.
struct FakeClock {
    using duration = std::chrono::milliseconds;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = std::chrono::time_point<FakeClock, duration>;
    [[maybe_unused]] static constexpr bool is_steady = true;

    static inline duration current{0};

    static time_point now() { return time_point(current); }
    static void reset() { current = duration{0}; }
};

using FakeDeadline = BasicDeadline<FakeClock>;

auto fake_sleep = [](std::chrono::milliseconds d) { FakeClock::current += d; };

}  // namespace

TEST(Poll, ReturnsImmediatelyWhenAlreadySatisfied) {
    FakeClock::reset();
    FakeDeadline deadline{std::chrono::milliseconds(3000)};
    const auto result = poll_until(deadline, std::chrono::milliseconds(5), [] { return true; }, fake_sleep);
    EXPECT_TRUE(result.satisfied);
    EXPECT_EQ(result.polls, 1);
    EXPECT_EQ(FakeClock::current.count(), 0);  // never slept
}

TEST(Poll, SatisfiedPartwayThrough) {
    FakeClock::reset();
    FakeDeadline deadline{std::chrono::milliseconds(3000)};
    int calls = 0;
    const auto result =
        poll_until(deadline, std::chrono::milliseconds(5), [&calls] { return ++calls == 3; }, fake_sleep);
    EXPECT_TRUE(result.satisfied);
    EXPECT_EQ(result.polls, 3);
}

// A slow final iteration must not be reported as a timeout: the predicate is
// evaluated once more after the deadline expires.
TEST(Poll, EvaluatesOnceMoreAfterTheDeadline) {
    FakeClock::reset();
    FakeDeadline deadline{std::chrono::milliseconds(10)};
    std::vector<int64_t> observed_at;
    const auto result = poll_until(
        deadline,
        std::chrono::milliseconds(5),
        [&observed_at] {
            observed_at.push_back(FakeClock::current.count());
            return false;
        },
        fake_sleep);
    EXPECT_FALSE(result.satisfied);
    ASSERT_FALSE(observed_at.empty());
    EXPECT_GE(observed_at.back(), deadline.budget().count()) << "no observation after the deadline expired";
}

// The final predicate call is outside the deadline budget. Its UMD operation
// must have its own timeout if the caller needs a strict wall-clock bound.
TEST(Poll, WallClockBoundIncludesOneFinalPredicateOperation) {
    FakeClock::reset();
    FakeDeadline deadline{std::chrono::milliseconds(10)};
    int calls = 0;
    const auto result = poll_until(
        deadline,
        std::chrono::milliseconds(5),
        [&calls] {
            FakeClock::current += std::chrono::milliseconds(7);
            return ++calls == 2;
        },
        fake_sleep);
    EXPECT_TRUE(result.satisfied);
    EXPECT_EQ(result.polls, 2);
    EXPECT_EQ(FakeClock::current.count(), 17);
}

TEST(Poll, TimesOutAndReportsPollCount) {
    FakeClock::reset();
    FakeDeadline deadline{std::chrono::milliseconds(20)};
    const auto result = poll_until(deadline, std::chrono::milliseconds(5), [] { return false; }, fake_sleep);
    EXPECT_FALSE(result.satisfied);
    EXPECT_GT(result.polls, 1);
    EXPECT_TRUE(deadline.expired());
}

TEST(Poll, RemainingClampsToZeroOnceExpired) {
    FakeClock::reset();
    FakeDeadline deadline{std::chrono::milliseconds(10)};
    EXPECT_EQ(deadline.remaining().count(), 10);
    FakeClock::current += std::chrono::milliseconds(4);
    EXPECT_EQ(deadline.remaining().count(), 6);
    FakeClock::current += std::chrono::milliseconds(20);
    EXPECT_EQ(deadline.remaining().count(), 0);
}

// An interval far coarser than the budget must not stretch the wait: the sleep
// is clamped to the time left, so a 60 s interval on a 10 ms budget costs 10 ms.
TEST(Poll, ACoarseIntervalDoesNotOvershootTheBudget) {
    FakeClock::reset();
    FakeDeadline deadline{std::chrono::milliseconds(10)};
    const auto result = poll_until(deadline, std::chrono::milliseconds(60000), [] { return false; }, fake_sleep);
    EXPECT_FALSE(result.satisfied);
    EXPECT_EQ(FakeClock::current.count(), deadline.budget().count());
    EXPECT_EQ(result.polls, 2) << "one poll before the deadline, one after";
}

TEST(Poll, DeadlineIsNotExpiredBeforeItsBudget) {
    FakeClock::reset();
    FakeDeadline deadline{std::chrono::milliseconds(100)};
    EXPECT_FALSE(deadline.expired());
    FakeClock::current += std::chrono::milliseconds(99);
    EXPECT_FALSE(deadline.expired());
    FakeClock::current += std::chrono::milliseconds(1);
    EXPECT_TRUE(deadline.expired());
}
