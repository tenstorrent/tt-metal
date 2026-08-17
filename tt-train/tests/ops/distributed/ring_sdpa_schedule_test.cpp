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

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace {

using ttml::metal::AttentionMaskType;
using ttml::metal::ops::get_device_execution_info;
using ttml::ttnn_fixed::distributed::RingShiftDirection;

uint32_t source_device(uint32_t device, uint32_t step, uint32_t ring_size, RingShiftDirection direction) {
    return direction == RingShiftDirection::Backward ? (device + step) % ring_size
                                                     : (device + ring_size - (step % ring_size)) % ring_size;
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
// mod ring_size, not mod 2^32. Full expected table for ring size 3,
// src = (device - step) mod 3.
TEST(RingSDPAScheduleTest, ForwardDirectionRingSize3CausalTable) {
    struct Expected {
        uint32_t device;
        uint32_t step;
        bool execute;
        AttentionMaskType mask;
    };
    const Expected table[] = {
        // device 0: sources 0, 2, 1 -> diagonal, skip, skip
        {0, 0, true, AttentionMaskType::Causal},
        {0, 1, false, AttentionMaskType::None},  // src 2 > 0: later chunk, skip
        {0, 2, false, AttentionMaskType::None},  // src 1 > 0: later chunk, skip
        // device 1: sources 1, 0, 2 -> diagonal, full, skip
        {1, 0, true, AttentionMaskType::Causal},
        {1, 1, true, AttentionMaskType::None},  // src 0 < 1: earlier chunk, full attention
        {1, 2, false, AttentionMaskType::None},
        // device 2: sources 2, 1, 0 -> diagonal, full, full
        {2, 0, true, AttentionMaskType::Causal},
        {2, 1, true, AttentionMaskType::None},
        {2, 2, true, AttentionMaskType::None},
    };
    for (const auto& expected : table) {
        const auto [execute, mask] = get_device_execution_info(
            expected.device, expected.step, 3U, AttentionMaskType::Causal, RingShiftDirection::Forward);
        EXPECT_EQ(execute, expected.execute) << "device " << expected.device << " step " << expected.step;
        EXPECT_EQ(mask, expected.mask) << "device " << expected.device << " step " << expected.step;
    }
}

// Backward direction on the same ring size, src = (device + step) mod 3.
TEST(RingSDPAScheduleTest, BackwardDirectionRingSize3CausalTable) {
    struct Expected {
        uint32_t device;
        uint32_t step;
        bool execute;
        AttentionMaskType mask;
    };
    const Expected table[] = {
        {0, 0, true, AttentionMaskType::Causal},
        {0, 1, false, AttentionMaskType::None},  // src 1 > 0
        {0, 2, false, AttentionMaskType::None},  // src 2 > 0
        {1, 0, true, AttentionMaskType::Causal},
        {1, 1, false, AttentionMaskType::None},  // src 2 > 1
        {1, 2, true, AttentionMaskType::None},   // src 0 < 1
        {2, 0, true, AttentionMaskType::Causal},
        {2, 1, true, AttentionMaskType::None},  // src 0 < 2
        {2, 2, true, AttentionMaskType::None},  // src 1 < 2
    };
    for (const auto& expected : table) {
        const auto [execute, mask] = get_device_execution_info(
            expected.device, expected.step, 3U, AttentionMaskType::Causal, RingShiftDirection::Backward);
        EXPECT_EQ(execute, expected.execute) << "device " << expected.device << " step " << expected.step;
        EXPECT_EQ(mask, expected.mask) << "device " << expected.device << " step " << expected.step;
    }
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
