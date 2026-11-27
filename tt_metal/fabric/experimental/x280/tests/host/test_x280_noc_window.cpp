// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-only tests for the outbound NOC window encoding.
//
// The interesting ones check the host encoder against noc.h's own bitfield
// union, which is what firmware programs. Getting this wrong produces a window
// that resolves to a plausible wrong address rather than failing.
//
//   ninja -C build x280_runtime_host_tests && build/test/tt_metal/x280/x280_runtime_host_tests

#include "x280/device/x280_noc_window.hpp"
#include "x280/device/x280_tile.hpp"

#include "tests/host/translated_coord_factory.hpp"

#include "noc.h"
#include "rt/x280_noc_map.h"
#include "x280_format.hpp"

#include <gtest/gtest.h>

namespace {

using namespace x280;
using x280::testing::TranslatedCoordFactory;

// The firmware's own encoding, used as the oracle: noc_fill_tlb_2m, the fill
// noc_open_window_2m programs through. It is pure field assignment through the
// union, so it carries whatever layout the compiler picks for noc_tlb_2m_t —
// which is exactly what the host encoder has to reproduce.
NocWindowWords firmware_words(const TranslatedCoord& target, const NocWindow2M& window) {
    noc_tlb_2m_t tlb = {};
    noc_fill_tlb_2m(
        &tlb,
        target.x(),
        target.y(),
        target.x(),
        target.y(),
        window.target_addr,
        window.ordering,
        /*multicast_en=*/0,
        /*num_destinations=*/0u);
    if (window.noc1) {
        tlb.noc_sel = 1;
    }
    NocWindowWords words;
    for (int i = 0; i < 4; i++) {
        words.data[i] = tlb.data[i];
    }
    return words;
}

const TranslatedCoord kTensix = TranslatedCoordFactory::make(1, 11);

NocWindow2M tensix_window() {
    NocWindow2M window;
    window.target_addr = kNoc2mWindowStride;
    window.ordering = NOC_ORDER_STRICT;
    return window;
}

NocWindowWords encode_ok(const TranslatedCoord& target, const NocWindow2M& window) {
    const auto result = encode(target, window);
    EXPECT_TRUE(result.ok()) << describe(result);
    return result.words;
}

NocWindowDecodeResult decode_ok(const NocWindowWords& words) {
    const auto result = decode(words);
    EXPECT_TRUE(result.ok()) << describe(result);
    return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Encoding matches firmware
// ---------------------------------------------------------------------------

TEST(NocWindow, MatchesTheFirmwareUnionForAUnicastWindow) {
    const auto window = tensix_window();
    EXPECT_EQ(encode_ok(kTensix, window), firmware_words(kTensix, window));
}

// Each ordering on either NoC: the host's words equal the ordered fill
// noc_open_window_2m programs, whose words the raw fill pins per ordering below.
TEST(NocWindow, MatchesTheFirmwareUnionForEachOrderingAndNoc) {
    for (noc_ordering_t ordering : {NOC_ORDER_DEFAULT, NOC_ORDER_STRICT, NOC_ORDER_POSTED}) {
        for (bool noc1 : {false, true}) {
            NocWindow2M window = tensix_window();
            window.ordering = ordering;
            window.noc1 = noc1;
            EXPECT_EQ(encode_ok(kTensix, window), firmware_words(kTensix, window))
                << "ordering=" << ordering << " noc1=" << noc1;
        }
    }
}

// Every documented L2CPU and a spread of peer coordinates, since the coordinate
// fields are the ones a wrong shift would corrupt while still looking valid.
TEST(NocWindow, MatchesTheFirmwareUnionAcrossCoordinates) {
    for (uint32_t x : {0u, 1u, 8u, 16u, 63u}) {
        for (uint32_t y : {0u, 3u, 11u, 63u}) {
            const auto target = TranslatedCoordFactory::make(x, y);
            const NocWindow2M window = tensix_window();
            EXPECT_EQ(encode_ok(target, window), firmware_words(target, window)) << "(" << x << "," << y << ")";
        }
    }
}

// The address field is 43 bits of target_addr >> 21, so it straddles words 0
// and 1 — the one place an off-by-one shift is invisible in a small address.
TEST(NocWindow, MatchesTheFirmwareUnionForAHighAddress) {
    NocWindow2M window = tensix_window();
    window.target_addr = 0x4000'3000'0000ull;  // L2CPU DDR front port
    EXPECT_EQ(encode_ok(kTensix, window), firmware_words(kTensix, window));
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST(NocWindow, RejectsAMisalignedTargetInsteadOfTruncatingIt) {
    NocWindow2M window = tensix_window();
    window.target_addr += 0x40;
    const auto result = encode(kTensix, window);
    EXPECT_EQ(result.fault, NocWindowEncodeFault::TargetMisaligned);
    EXPECT_NE(describe(result).find(hex(window.target_addr)), std::string::npos) << describe(result);
}

TEST(NocWindow, AcceptsTheLargestRepresentableTargetBase) {
    NocWindow2M window = tensix_window();
    window.target_addr = kNoc2mMaxTargetBase;
    const auto result = encode(kTensix, window);
    ASSERT_TRUE(result.ok()) << describe(result);
    EXPECT_EQ(decode_ok(result.words).window.target_addr, kNoc2mMaxTargetBase);
}

TEST(NocWindow, RejectsATargetBeyondTheLargestRepresentableBase) {
    NocWindow2M window = tensix_window();
    window.target_addr = kNoc2mMaxTargetBase + 1;
    EXPECT_EQ(encode(kTensix, window).fault, NocWindowEncodeFault::TargetAboveMaxBase);
}

TEST(NocWindow, RejectsCountedWritesInsteadOfEncodingIt) {
    NocWindow2M window = tensix_window();
    window.ordering = static_cast<noc_ordering_t>(3);
    EXPECT_EQ(encode(kTensix, window).fault, NocWindowEncodeFault::OrderingInvalid);
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

TEST(NocWindow, DecodeRoundTripsTheUnicastFields) {
    const auto window = tensix_window();
    const auto back = decode_ok(encode_ok(kTensix, window));
    EXPECT_EQ(back.coord.x, kTensix.x());
    EXPECT_EQ(back.coord.y, kTensix.y());
    EXPECT_EQ(back.window.target_addr, window.target_addr);
    EXPECT_EQ(back.window.ordering, window.ordering);
    EXPECT_EQ(back.window.noc1, window.noc1);
}

TEST(NocWindow, DecodeRejectsDifferentStartAndEndCoordinates) {
    auto words = encode_ok(kTensix, tensix_window());
    words.data[2] ^= 1u << 12;  // x_start
    EXPECT_EQ(decode(words).fault, NocWindowDecodeFault::StartIsNotEnd);
}

TEST(NocWindow, DecodeRejectsMulticastEntries) {
    auto words = encode_ok(kTensix, tensix_window());
    words.data[2] |= 1u << 24;
    EXPECT_EQ(decode(words).fault, NocWindowDecodeFault::Multicast);
}

TEST(NocWindow, DecodeRejectsReservedAddressBitsInsteadOfTruncatingThem) {
    auto words = encode_ok(kTensix, tensix_window());
    words.data[1] |= 1u << 11;
    EXPECT_EQ(decode(words).fault, NocWindowDecodeFault::ReservedAddressBits);
}

// A Counted Writes entry is a real hardware state, but NocWindow2M has no
// value for it; decode refuses rather than report another ordering.
TEST(NocWindow, DecodeRejectsCountedWritesEntries) {
    auto words = encode_ok(kTensix, tensix_window());
    words.data[2] |= 3u << 25;
    EXPECT_EQ(decode(words).fault, NocWindowDecodeFault::CountedWrites);
}

// ---------------------------------------------------------------------------
// Offsets
// ---------------------------------------------------------------------------

TEST(NocWindow, ConfigOffsetIsSixteenBytesPerEntry) {
    EXPECT_EQ(noc_window_config_offset(0), 0u);
    EXPECT_EQ(noc_window_config_offset(1), 0x10u);
    EXPECT_EQ(noc_window_config_offset(X280_TLB_HOST_BASE), X280_TLB_HOST_BASE * 0x10u);
}

TEST(NocWindow, ApertureOffsetCarriesTheLowBitsOfTheAddress) {
    EXPECT_EQ(noc_window_aperture_offset(0, 0x40), 0x40u);
    EXPECT_EQ(noc_window_aperture_offset(2, 0x40), 2 * kNoc2mWindowStride + 0x40);
    // Anything at or above the stride belongs to the next window, not this one.
    EXPECT_EQ(noc_window_aperture_offset(0, kNoc2mWindowStride + 0x40), 0x40u);
}

// ---------------------------------------------------------------------------
// Index reservations
// ---------------------------------------------------------------------------

TEST(NocWindow, AcceptsEveryIndexInTheHostBlock) {
    for (uint32_t i = 0; i < X280_TLB_HOST_COUNT; i++) {
        const auto check = validate_host_window_index(X280_TLB_HOST_BASE + i);
        EXPECT_TRUE(check.ok) << check.reason;
    }
}

// Firmware owns everything below the host block. These are the slots a wrong
// index would land on, and the collision is silent on hardware.
TEST(NocWindow, RejectsIndicesFirmwareOwns) {
    for (uint32_t index : {0u, 1u, 16u, 24u, 34u, 50u, 60u, 100u, uint32_t{X280_TLB_HOST_BASE - 1}}) {
        const auto check = validate_host_window_index(index);
        EXPECT_FALSE(check.ok) << "index " << index;
        EXPECT_NE(check.reason.find("firmware"), std::string::npos) << check.reason;
    }
}

TEST(NocWindow, RejectsIndicesOutsideTheHardwareRange) {
    EXPECT_FALSE(validate_host_window_index(kNoc2mWindowCount).ok);
    EXPECT_FALSE(validate_host_window_index(1000).ok);
}

// The block must fit the hardware, or the last few indices validate and then
// write past the config array.
TEST(NocWindow, HostBlockFitsInsideTheHardwareRange) {
    EXPECT_LE(X280_TLB_HOST_BASE + X280_TLB_HOST_COUNT, kNoc2mWindowCount);
}

// ---------------------------------------------------------------------------
// Unusable readbacks
// ---------------------------------------------------------------------------

// An unclocked tile answers the payload with all ones. Treating that as fields
// would produce a plausible coordinate and address.
TEST(NocWindow, AllOnesReadbackIsUnusable) {
    NocWindowWords all_ones;
    for (auto& word : all_ones.data) {
        word = 0xFFFFFFFFu;
    }
    EXPECT_TRUE(is_unusable(all_ones));
    EXPECT_EQ(decode(all_ones).fault, NocWindowDecodeFault::Unusable);
}

TEST(NocWindow, ReservedWordThreeMayReadAsZeroOnAnUnusableRead) {
    NocWindowWords words;
    words.data[0] = 0xFFFFFFFFu;
    words.data[1] = 0xFFFFFFFFu;
    words.data[2] = 0xFFFFFFFFu;
    words.data[3] = 0;
    EXPECT_TRUE(is_unusable(words));
    EXPECT_EQ(decode(words).fault, NocWindowDecodeFault::Unusable);
}

// Word 0 is the low 32 bits of the shifted address, which is all ones for a real
// target. Rejecting on any single word would refuse a legitimate window.
TEST(NocWindow, AHighAddressAloneIsNotUnusable) {
    NocWindow2M window;
    // Chosen so addr >> 21 is exactly 0xFFFFFFFF, filling word 0 and no more.
    window.target_addr = 0xFFFFFFFFull << 21;
    const auto words = encode_ok(TranslatedCoordFactory::make(8, 3), window);
    ASSERT_EQ(words.data[0], 0xFFFFFFFFu) << "this test is only meaningful if word 0 is all ones";
    EXPECT_FALSE(is_unusable(words));
}

TEST(NocWindow, AZeroedWindowIsUsable) {
    NocWindowWords zeroed;
    EXPECT_FALSE(is_unusable(zeroed)) << "an unprogrammed window reads zero, which is a real observation";
    EXPECT_TRUE(decode(zeroed).ok());
}

// The widened fill: the rectangle, the multicast bit and the destination
// override are distinct fields, and a unicast fill leaves the last two clear.
TEST(NocWindow, FillCarriesTheRectangleUnderMulticast) {
    noc_tlb_2m_t tlb = {};
    noc_fill_tlb_2m(
        &tlb,
        /*x_start=*/2,
        /*y_start=*/3,
        /*x_end=*/9,
        /*y_end=*/11,
        /*addr=*/0,
        NOC_ORDER_POSTED,
        /*multicast_en=*/1,
        /*num_destinations=*/0u);
    EXPECT_EQ(tlb.mcast, 1u);
    EXPECT_EQ(tlb.x_start, 2u);
    EXPECT_EQ(tlb.y_start, 3u);
    EXPECT_EQ(tlb.x_end, 9u);
    EXPECT_EQ(tlb.y_end, 11u);
    EXPECT_EQ(tlb.num_destinations_override, 0u) << "zero is what makes hardware substitute the count";
}

TEST(NocWindow, FillCollapsesToOneTileUnderUnicast) {
    noc_tlb_2m_t tlb = {};
    noc_fill_tlb_2m(
        &tlb,
        /*x_start=*/5,
        /*y_start=*/7,
        /*x_end=*/5,
        /*y_end=*/7,
        /*addr=*/0,
        NOC_ORDER_POSTED,
        /*multicast_en=*/0,
        /*num_destinations=*/0u);
    EXPECT_EQ(tlb.mcast, 0u);
    EXPECT_EQ(tlb.x_start, tlb.x_end);
    EXPECT_EQ(tlb.y_start, tlb.y_end);
}

// The start corner is written as given rather than sorted against the end: the
// direction of data flow is the caller's to choose, and a swap here would aim
// a translated-coordinate broadcast the wrong way.
TEST(NocWindow, FillDoesNotReorderTheCorners) {
    noc_tlb_2m_t tlb = {};
    noc_fill_tlb_2m(
        &tlb,
        /*x_start=*/9,
        /*y_start=*/11,
        /*x_end=*/2,
        /*y_end=*/3,
        /*addr=*/0,
        NOC_ORDER_POSTED,
        /*multicast_en=*/1,
        /*num_destinations=*/0u);
    EXPECT_EQ(tlb.x_start, 9u);
    EXPECT_EQ(tlb.y_start, 11u);
    EXPECT_EQ(tlb.x_end, 2u);
    EXPECT_EQ(tlb.y_end, 3u);
}

TEST(NocWindow, FillCarriesANonZeroDestinationOverride) {
    noc_tlb_2m_t tlb = {};
    noc_fill_tlb_2m(
        &tlb,
        1,
        1,
        4,
        4,
        /*addr=*/0,
        NOC_ORDER_POSTED,
        /*multicast_en=*/1,
        /*num_destinations=*/16u);
    EXPECT_EQ(tlb.num_destinations_override, 16u);
}

// ---------------------------------------------------------------------------
// Ordering encoding and the paired-slot range
// ---------------------------------------------------------------------------

// Each noc_ordering_t value is TLBWindows.md's encoding of its mode in the
// `ordering` field: Default 0, Strict AXI 1, Posted Writes 2.
TEST(NocWindow, EachOrderingFillsItsTlbWindowsEncoding) {
    const struct {
        noc_ordering_t ordering;
        uint32_t mode;
    } cases[] = {{NOC_ORDER_DEFAULT, 0}, {NOC_ORDER_STRICT, 1}, {NOC_ORDER_POSTED, 2}};
    for (const auto& c : cases) {
        noc_tlb_2m_t tlb = {};
        noc_fill_tlb_2m(&tlb, 9, 11, 9, 11, /*addr=*/0, c.ordering, /*multicast_en=*/0, /*num_destinations=*/0u);
        EXPECT_EQ(tlb.ordering, c.mode) << "ordering " << c.ordering;
    }
}

// Counted Writes is outside the enum; the checked entry points refuse it.
TEST(NocWindow, OrderingValidityExcludesCountedWrites) {
    EXPECT_TRUE(noc_ordering_valid(NOC_ORDER_POSTED));
    EXPECT_FALSE(noc_ordering_valid(static_cast<noc_ordering_t>(3)));
}

TEST(NocWindow, PairedSlotPredicateCoversExactlyThePairedRange) {
    EXPECT_FALSE(x280_tlb_paired_slot(X280_TLB_PAIRED_BASE - 1));
    EXPECT_TRUE(x280_tlb_paired_slot(X280_TLB_PAIRED_BASE));
    EXPECT_TRUE(x280_tlb_paired_slot(X280_TLB_PAIRED_BASE + X280_TLB_PAIRED_COUNT - 1));
    EXPECT_FALSE(x280_tlb_paired_slot(X280_TLB_PAIRED_BASE + X280_TLB_PAIRED_COUNT));
}

// A cleared entry is four zero words. static_vc (bit 28 of word 2) is one of
// the fields decode() leaves out, so a check on the decoded fields would pass
// a clear that left it set.
TEST(NocWindow, ClearedMeansEveryWordZeroNotOnlyTheDecodedFields) {
    NocWindowResult window;
    window.ok = true;
    EXPECT_TRUE(window_reads_cleared(window));
    window.words.data[2] = 1u << 28;
    ASSERT_TRUE(decode(window.words).ok());
    EXPECT_EQ(decode(window.words).window.target_addr, 0u);
    EXPECT_FALSE(window_reads_cleared(window));
}

// A read that failed leaves the words zero-initialised; that is no evidence of
// a clear.
TEST(NocWindow, AFailedReadIsNotACleared) {
    NocWindowResult window;
    window.ok = false;
    EXPECT_FALSE(window_reads_cleared(window));
}
