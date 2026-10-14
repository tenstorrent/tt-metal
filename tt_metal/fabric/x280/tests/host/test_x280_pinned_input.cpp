// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rt/x280_pinned_input.h"

#include <gtest/gtest.h>
#include <hostdevcommon/fabric_common.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

// The routing image the pinned input holds is tt-metal's own routing table.
static_assert(X280_PINNED_INPUT_ROUTING_BYTES == sizeof(tt::tt_fabric::routing_l1_info_t));

TEST(X280PinnedInput, FixedRegionConstantsDescribeTheCompleteInputLayout) {
    EXPECT_EQ(X280_PINNED_INPUT_BASE, UINT64_C(0x0A000000));
    EXPECT_EQ(X280_PINNED_INPUT_SPAN, UINT64_C(0x10C0));
    EXPECT_EQ(X280_PINNED_INPUT_END, UINT64_C(0x0A0010C0));
    EXPECT_EQ(X280_PINNED_INPUT_CACHE_LINE_BYTES, 64u);
    EXPECT_EQ(X280_PINNED_INPUT_CACHE_LINES, 67u);
    EXPECT_EQ(X280_PINNED_INPUT_CACHE_LINES * X280_PINNED_INPUT_CACHE_LINE_BYTES, X280_PINNED_INPUT_SPAN);

    EXPECT_EQ(X280_PINNED_INPUT_ROUTING_OFFSET, 0u);
    EXPECT_EQ(X280_PINNED_INPUT_ROUTING_BYTES, 2704u);
    EXPECT_EQ(X280_PINNED_INPUT_HEADERS_OFFSET, UINT64_C(0xAC0));
    EXPECT_EQ(X280_PINNED_INPUT_HEADER_COUNT, 16u);
    EXPECT_EQ(X280_PINNED_INPUT_HEADER_BYTES, 96u);
    EXPECT_EQ(X280_PINNED_INPUT_HEADERS_BYTES, X280_PINNED_INPUT_HEADER_COUNT * X280_PINNED_INPUT_HEADER_BYTES);
    EXPECT_LE(X280_PINNED_INPUT_ROUTING_OFFSET + X280_PINNED_INPUT_ROUTING_BYTES, X280_PINNED_INPUT_HEADERS_OFFSET);
    EXPECT_EQ(X280_PINNED_INPUT_HEADERS_OFFSET + X280_PINNED_INPUT_HEADERS_BYTES, X280_PINNED_INPUT_SPAN);
    EXPECT_EQ(X280_PINNED_INPUT_FIRST_HEADER_BASE, UINT64_C(0x0A000AC0));
    EXPECT_EQ(X280_PINNED_INPUT_FIRST_HEADER_BASE % X280_PINNED_INPUT_CACHE_LINE_BYTES, 0u);
    EXPECT_NE(
        x280_pinned_input_contains(
            &X280_PINNED_INPUT_ROUTER_REGION, X280_PINNED_INPUT_FIRST_HEADER_BASE, X280_PINNED_INPUT_HEADER_BYTES),
        0);
}

TEST(X280PinnedInput, ContainsMatchesEveryLocalHalfOpenBoundarySpan) {
    const uint64_t first_address = X280_PINNED_INPUT_BASE - 1u;
    const uint64_t last_address = X280_PINNED_INPUT_END + 1u;
    const uint64_t largest_local_span = X280_PINNED_INPUT_SPAN + 1u;

    for (uint64_t address = first_address; address <= last_address; ++address) {
        for (uint64_t bytes = 0; bytes <= largest_local_span; ++bytes) {
            const bool expected = address >= X280_PINNED_INPUT_BASE && address <= X280_PINNED_INPUT_END &&
                                  bytes <= X280_PINNED_INPUT_END - address;
            const bool actual = x280_pinned_input_contains(&X280_PINNED_INPUT_ROUTER_REGION, address, bytes) != 0;
            if (actual != expected) {
                FAIL() << "address=" << address << " bytes=" << bytes << " expected=" << expected
                       << " actual=" << actual;
            }
        }
    }
}

TEST(X280PinnedInput, ContainsRejectsRemoteAndOverflowingRanges) {
    struct Range {
        uint64_t address;
        uint64_t bytes;
    };
    constexpr std::array<Range, 10> rejected = {{
        {X280_PINNED_INPUT_BASE - 1u, 0u},
        {X280_PINNED_INPUT_BASE - 1u, 1u},
        {X280_PINNED_INPUT_END + 1u, 0u},
        {X280_PINNED_INPUT_END + 1u, 1u},
        {X280_PINNED_INPUT_BASE, std::numeric_limits<uint64_t>::max()},
        {X280_PINNED_INPUT_BASE, std::numeric_limits<uint64_t>::max() - X280_PINNED_INPUT_BASE},
        {X280_PINNED_INPUT_BASE, std::numeric_limits<uint64_t>::max() - X280_PINNED_INPUT_BASE + 1u},
        {X280_PINNED_INPUT_END, std::numeric_limits<uint64_t>::max()},
        {std::numeric_limits<uint64_t>::max() - 1u, 2u},
        {std::numeric_limits<uint64_t>::max(), 1u},
    }};

    for (const auto& range : rejected) {
        EXPECT_EQ(x280_pinned_input_contains(&X280_PINNED_INPUT_ROUTER_REGION, range.address, range.bytes), 0)
            << "address=" << range.address << " bytes=" << range.bytes;
    }
}

TEST(X280PinnedInput, WritableSamplesRegionValidation) {
    // Valid writable samples descriptor
    x280_pinned_input_region_t valid_region{
        .base = X280_PINNED_INPUT_BASE,
        .bytes = 107520u,  // multiple of 64 <= 131072
        .kind = static_cast<uint32_t>(X280_PINNED_INPUT_WRITABLE_SAMPLES),
        .reserved = 0,
        .reserved_tail = 0,
    };
    EXPECT_NE(x280_pinned_input_validate_region(&valid_region), 0);
    EXPECT_NE(x280_pinned_input_contains(&valid_region, X280_PINNED_INPUT_BASE, 107520u), 0);
    EXPECT_EQ(x280_pinned_input_contains(&valid_region, X280_PINNED_INPUT_BASE, 107521u), 0);

    // Full 128-KiB capacity
    valid_region.bytes = 131072u;
    EXPECT_NE(x280_pinned_input_validate_region(&valid_region), 0);

    // Excess capacity > 131072
    x280_pinned_input_region_t excess_region = valid_region;
    excess_region.bytes = 131072u + 64u;
    EXPECT_EQ(x280_pinned_input_validate_region(&excess_region), 0);

    // Not multiple of 64
    x280_pinned_input_region_t unaligned_region = valid_region;
    unaligned_region.bytes = 1024u + 1u;
    EXPECT_EQ(x280_pinned_input_validate_region(&unaligned_region), 0);

    // Wrong base
    x280_pinned_input_region_t wrong_base = valid_region;
    wrong_base.base = 0x08020000u;
    EXPECT_EQ(x280_pinned_input_validate_region(&wrong_base), 0);

    // Nonzero reserved
    x280_pinned_input_region_t nonzero_reserved = valid_region;
    nonzero_reserved.reserved = 1u;
    EXPECT_EQ(x280_pinned_input_validate_region(&nonzero_reserved), 0);
}

TEST(X280PinnedInput, PreparedPolicyConstantsKeepOneGeneralPurposeWay) {
    EXPECT_EQ(X280_PINNED_INPUT_WAYENABLE, 1u);
    EXPECT_EQ(X280_PINNED_INPUT_GENERAL_WAY, 0u);
    EXPECT_EQ(X280_PINNED_INPUT_RESERVED_WAY, 1u);
    EXPECT_EQ(X280_PINNED_INPUT_ACTIVE_BITS, UINT64_C(0x3));
    EXPECT_EQ(X280_PINNED_INPUT_BASELINE_BITS, UINT64_C(0x3));
    EXPECT_EQ(X280_PINNED_INPUT_PREPARED_BITS, UINT64_C(0x1));
    EXPECT_EQ(X280_PINNED_INPUT_MASTER_COUNT, 38u);
    EXPECT_EQ(X280_PINNED_INPUT_POPULATION_MASTER, 34u);
}

TEST(X280PinnedInput, AdmissionRecordHasTheNativeCheckWireLayout) {
    EXPECT_EQ(X280_PINNED_INPUT_ADMITTED, 0);
    EXPECT_EQ(X280_PINNED_INPUT_WRONG_WAYENABLE, 1);
    EXPECT_EQ(X280_PINNED_INPUT_WRONG_WAYMASK, 2);
    EXPECT_EQ(sizeof(x280_pinned_input_admission_t), 16u);
    EXPECT_EQ(alignof(x280_pinned_input_admission_t), alignof(uint64_t));
    EXPECT_EQ(offsetof(x280_pinned_input_admission_t, code), 0u);
    EXPECT_EQ(offsetof(x280_pinned_input_admission_t, master_id), 4u);
    EXPECT_EQ(offsetof(x280_pinned_input_admission_t, observed), 8u);
}

}  // namespace
