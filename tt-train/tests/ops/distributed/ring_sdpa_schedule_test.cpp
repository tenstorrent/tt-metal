// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Host-side tests for the ring SDPA causal schedule (get_device_execution_info, and
 * get_ring_step_mask with the contiguous and striped step masks).
 *
 * The function is pure, so these run without any device. They pin the schedule
 * both program factories share — in particular the Forward-direction source
 * computation for non-power-of-two ring sizes, where a mod-2^32 wraparound in the
 * source index is not equivalent to mod ring_size.
 */

#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "ops/distributed/ring_attention_sdpa.hpp"

namespace {

using ttml::metal::AttentionMaskType;
using ttml::metal::ops::contiguous_ring_step_masks;
using ttml::metal::ops::get_device_execution_info;
using ttml::metal::ops::get_ring_step_mask;
using ttml::metal::ops::RingStepMasks;
using ttml::ops::distributed::detail::kStripedOnQuery;
using ttml::ops::distributed::detail::kStripedOnShiftedQuery;
using ttml::ttnn_fixed::distributed::RingShiftDirection;

uint32_t source_device(uint32_t device, uint32_t step, uint32_t ring_size, RingShiftDirection direction) {
    return direction == RingShiftDirection::Backward ? (device + step) % ring_size
                                                     : (device + ring_size - (step % ring_size)) % ring_size;
}

struct Expected {
    uint32_t device;
    uint32_t step;
    bool execute;
    AttentionMaskType mask;
};

// Ring size 3, causal: every row except device 1's off-diagonal steps is the same
// in both shift directions — step 0 is always the device's own chunk (causal mask),
// device 0 skips both remote chunks, device 2 attends to both in full.
static constexpr Expected kRing3CausalCommonRows[] = {
    // device 0: only the diagonal executes
    {0, 0, true, AttentionMaskType::Causal},
    {0, 1, false, AttentionMaskType::None},
    {0, 2, false, AttentionMaskType::None},
    // device 1: diagonal; steps 1-2 depend on direction (checked per test)
    {1, 0, true, AttentionMaskType::Causal},
    // device 2: diagonal, then full attention on both earlier chunks
    {2, 0, true, AttentionMaskType::Causal},
    {2, 1, true, AttentionMaskType::None},
    {2, 2, true, AttentionMaskType::None},
};

void expect_ring3_causal_schedule(RingShiftDirection direction, std::initializer_list<Expected> direction_rows) {
    auto check_row = [direction](const Expected& expected) {
        const auto [execute, mask] =
            get_device_execution_info(expected.device, expected.step, 3U, AttentionMaskType::Causal, direction);
        EXPECT_EQ(execute, expected.execute) << "device " << expected.device << " step " << expected.step;
        EXPECT_EQ(mask, expected.mask) << "device " << expected.device << " step " << expected.step;
    };
    for (const auto& expected : kRing3CausalCommonRows) {
        check_row(expected);
    }
    for (const auto& expected : direction_rows) {
        check_row(expected);
    }
}

}  // namespace

TEST(RingSDPAScheduleTest, NonCausalAlwaysExecutesUnmasked) {
    for (auto direction : {RingShiftDirection::Forward, RingShiftDirection::Backward}) {
        for (uint32_t ring_size : {1U, 3U, 8U}) {
            for (uint32_t device = 0; device < ring_size; ++device) {
                for (uint32_t step = 0; step < ring_size; ++step) {
                    const auto [execute, mask] =
                        get_device_execution_info(device, step, ring_size, AttentionMaskType::None, direction);
                    EXPECT_TRUE(execute);
                    EXPECT_EQ(mask, AttentionMaskType::None);
                }
            }
        }
    }
}

// Non-power-of-two ring, Forward direction: the source index must be computed
// mod ring_size, not mod 2^32. src = (device - step) mod 3.
TEST(RingSDPAScheduleTest, ForwardDirectionRingSize3CausalTable) {
    expect_ring3_causal_schedule(
        RingShiftDirection::Forward,
        {
            {1, 1, true, AttentionMaskType::None},   // src 0 < 1: earlier chunk, full attention
            {1, 2, false, AttentionMaskType::None},  // src 2 > 1: later chunk, skip
        });
}

// Backward direction on the same ring size, src = (device + step) mod 3.
TEST(RingSDPAScheduleTest, BackwardDirectionRingSize3CausalTable) {
    expect_ring3_causal_schedule(
        RingShiftDirection::Backward,
        {
            {1, 1, false, AttentionMaskType::None},  // src 2 > 1: later chunk, skip
            {1, 2, true, AttentionMaskType::None},   // src 0 < 1: earlier chunk, full attention
        });
}

// Causal schedule invariants for both directions across power-of-two and
// non-power-of-two ring sizes: over a full ring pass, device d executes exactly
// the d+1 chunks from sources 0..d, applies the causal mask only on its own
// chunk (step 0), and the execute/mask decision agrees with the source index.
TEST(RingSDPAScheduleTest, CausalScheduleInvariants) {
    for (auto direction : {RingShiftDirection::Forward, RingShiftDirection::Backward}) {
        for (uint32_t ring_size : {1U, 2U, 3U, 5U, 7U, 8U}) {
            for (uint32_t device = 0; device < ring_size; ++device) {
                uint32_t executed = 0;
                uint32_t causal_steps = 0;
                for (uint32_t step = 0; step < ring_size; ++step) {
                    const auto [execute, mask] =
                        get_device_execution_info(device, step, ring_size, AttentionMaskType::Causal, direction);
                    const uint32_t src = source_device(device, step, ring_size, direction);
                    EXPECT_EQ(execute, src <= device)
                        << "ring " << ring_size << " device " << device << " step " << step;
                    EXPECT_EQ(mask == AttentionMaskType::Causal, execute && src == device)
                        << "ring " << ring_size << " device " << device << " step " << step;
                    executed += static_cast<uint32_t>(execute);
                    causal_steps += static_cast<uint32_t>(mask == AttentionMaskType::Causal);
                }
                EXPECT_EQ(executed, device + 1) << "ring " << ring_size << " device " << device;
                EXPECT_EQ(causal_steps, 1U) << "ring " << ring_size << " device " << device;
            }
        }
    }
}

// Striped ring of 4, Backward direction (src = (d + step) mod 4): every device runs a causal
// block at every step under exactly one of the two striped tables, the shifted-query one
// exactly when src > d.
TEST(RingSDPAScheduleTest, StripedRing4BackwardTable) {
    struct Row {
        uint32_t device;
        uint32_t step;
        bool shifted;
    };
    static constexpr Row kRows[] = {
        {0, 0, false},
        {0, 1, true},
        {0, 2, true},
        {0, 3, true},  // device 0: src 1, 2, 3 > 0
        {1, 0, false},
        {1, 1, true},
        {1, 2, true},
        {1, 3, false},  // src 2, 3 > 1; src 0 < 1
        {2, 0, false},
        {2, 1, true},
        {2, 2, false},
        {2, 3, false},  // src 3 > 2; src 0, 1 < 2
        {3, 0, false},
        {3, 1, false},
        {3, 2, false},
        {3, 3, false},  // device 3: every src <= 3
    };
    for (const auto& row : kRows) {
        const auto on_query =
            get_ring_step_mask(row.device, row.step, 4U, RingShiftDirection::Backward, kStripedOnQuery);
        const auto on_shifted =
            get_ring_step_mask(row.device, row.step, 4U, RingShiftDirection::Backward, kStripedOnShiftedQuery);
        EXPECT_NE(on_query.has_value(), on_shifted.has_value()) << "device " << row.device << " step " << row.step;
        EXPECT_EQ(on_shifted.has_value(), row.shifted) << "device " << row.device << " step " << row.step;
        EXPECT_EQ(on_query.value_or(on_shifted.value_or(AttentionMaskType::None)), AttentionMaskType::Causal)
            << "device " << row.device << " step " << row.step;
    }
}

// Token-level check of the whole striped ring attention: global token p lives on device p mod N
// as local row p div N. At each (device d, step) the scheduled block is a causal SDPA call over
// local rows, on the query itself (row l sees keys m <= l) or on the query shifted up one row
// (kernel row r is query r + 1 and sees keys m <= r; the last kernel row is junk and discarded).
// Every causally visible (query, key) pair of the global sequence must be computed exactly once
// and no masked pair ever, for both ring directions and ring sizes that are and are not powers
// of two.
TEST(RingSDPAScheduleTest, StripedCoversCausalTriangleExactlyOnce) {
    for (auto direction : {RingShiftDirection::Forward, RingShiftDirection::Backward}) {
        for (uint32_t ring_size : {1U, 2U, 3U, 4U, 5U, 8U}) {
            for (uint32_t local_len : {1U, 2U, 5U}) {
                const uint32_t seq_len = ring_size * local_len;
                std::vector<int> count(static_cast<size_t>(seq_len) * seq_len, 0);
                for (uint32_t device = 0; device < ring_size; ++device) {
                    for (uint32_t step = 0; step < ring_size; ++step) {
                        const auto on_query = get_ring_step_mask(device, step, ring_size, direction, kStripedOnQuery);
                        const auto on_shifted =
                            get_ring_step_mask(device, step, ring_size, direction, kStripedOnShiftedQuery);
                        ASSERT_NE(on_query.has_value(), on_shifted.has_value());
                        ASSERT_EQ(
                            on_query.value_or(on_shifted.value_or(AttentionMaskType::None)), AttentionMaskType::Causal);
                        const bool shifted = on_shifted.has_value();
                        const uint32_t src = source_device(device, step, ring_size, direction);
                        for (uint32_t row = 0; row < local_len; ++row) {
                            // Real local query row this kernel row computes, if any.
                            const uint32_t query_row = shifted ? row + 1U : row;
                            if (query_row >= local_len) {
                                continue;  // junk row of the shifted query
                            }
                            for (uint32_t key_row = 0; key_row <= row; ++key_row) {
                                const uint32_t q = query_row * ring_size + device;
                                const uint32_t k = key_row * ring_size + src;
                                ++count[static_cast<size_t>(q) * seq_len + k];
                            }
                        }
                    }
                }
                for (uint32_t q = 0; q < seq_len; ++q) {
                    for (uint32_t k = 0; k < seq_len; ++k) {
                        EXPECT_EQ(count[static_cast<size_t>(q) * seq_len + k], k <= q ? 1 : 0)
                            << "ring " << ring_size << " local " << local_len << " direction "
                            << static_cast<int>(direction) << " query " << q << " key " << k;
                    }
                }
            }
        }
    }
}

// The program hash must tell a skipped step from an unmasked one.
TEST(RingSDPAScheduleTest, StepMasksHashDistinguishesSkipFromNone) {
    const RingStepMasks skip_after{.src_before = AttentionMaskType::None, .src_self = AttentionMaskType::None};
    const RingStepMasks none_after{
        .src_before = AttentionMaskType::None,
        .src_self = AttentionMaskType::None,
        .src_after = AttentionMaskType::None};
    EXPECT_NE(skip_after.to_hash(), none_after.to_hash());
    EXPECT_NE(kStripedOnQuery.to_hash(), kStripedOnShiftedQuery.to_hash());
    EXPECT_NE(kStripedOnQuery.to_hash(), contiguous_ring_step_masks(AttentionMaskType::Causal).to_hash());
}
