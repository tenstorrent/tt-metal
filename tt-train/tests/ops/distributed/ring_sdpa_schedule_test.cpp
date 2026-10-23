// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Host-side tests for the ring SDPA causal schedule (get_device_execution_info).
 *
 * The function is pure, so these run without any device. They pin the schedule
 * both program factories share — in particular the Forward-direction source
 * computation for non-power-of-two ring sizes, where a mod-2^32 wraparound in the
 * source index is not equivalent to mod ring_size.
 */

#include <gtest/gtest.h>

#include <initializer_list>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace {

using ttml::metal::AttentionMaskType;
using ttml::metal::ops::get_device_execution_info;
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
