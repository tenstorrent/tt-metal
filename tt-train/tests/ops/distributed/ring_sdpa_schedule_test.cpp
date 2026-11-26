// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Host-side tests for the ring SDPA schedule (get_device_execution_info, get_ring_step_work) under None and ZigZag.
 *
 * The functions are pure, so these run without any device.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace {

using ttml::metal::AttentionMaskType;
using ttml::metal::ops::get_device_execution_info;
using ttml::metal::ops::get_ring_chunk_count;
using ttml::metal::ops::get_ring_step_work;
using ttml::metal::ops::RingLoadBalancing;
using ttml::metal::ops::RingStepWork;
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

// None mode gives the same work as the contiguous causal schedule, always on chunk 0.
TEST(RingSDPAScheduleTest, LoadBalancingNoneMatchesExecutionInfo) {
    for (auto direction : {RingShiftDirection::Forward, RingShiftDirection::Backward}) {
        for (auto mask_type : {AttentionMaskType::None, AttentionMaskType::Causal}) {
            for (uint32_t ring_size : {1U, 2U, 3U, 8U}) {
                for (uint32_t device = 0; device < ring_size; ++device) {
                    for (uint32_t step = 0; step < ring_size; ++step) {
                        const auto [execute, mask] =
                            get_device_execution_info(device, step, ring_size, mask_type, direction);
                        const RingStepWork expected{.execute = execute, .mask_type = mask, .chunk = 0U};
                        EXPECT_EQ(
                            get_ring_step_work(device, step, ring_size, mask_type, direction, RingLoadBalancing::None),
                            expected)
                            << "ring " << ring_size << " device " << device << " step " << step;
                    }
                }
            }
        }
    }
}

// ZigZag on a ring of 4, Backward: skip at step 0, then the expected chunk per device and step.
TEST(RingSDPAScheduleTest, ZigZagRing4BackwardTable) {
    struct Row {
        uint32_t device;
        uint32_t step;
        bool execute;
        uint32_t chunk;
    };
    static constexpr Row kRows[] = {
        {0, 0, false, 0},
        {0, 1, true, 1},
        {0, 2, true, 1},
        {0, 3, true, 1},
        {1, 0, false, 0},
        {1, 1, true, 1},
        {1, 2, true, 1},
        {1, 3, true, 0},
        {2, 0, false, 0},
        {2, 1, true, 1},
        {2, 2, true, 0},
        {2, 3, true, 0},
        {3, 0, false, 0},
        {3, 1, true, 0},
        {3, 2, true, 0},
        {3, 3, true, 0},
    };
    for (const auto& row : kRows) {
        const auto work = get_ring_step_work(
            row.device,
            row.step,
            4U,
            AttentionMaskType::Causal,
            RingShiftDirection::Backward,
            RingLoadBalancing::ZigZag);
        EXPECT_EQ(work.execute, row.execute) << "device " << row.device << " step " << row.step;
        EXPECT_EQ(work.mask_type, AttentionMaskType::None) << "device " << row.device << " step " << row.step;
        if (row.execute) {
            EXPECT_EQ(work.chunk, row.chunk) << "device " << row.device << " step " << row.step;
        }
    }
}

// Step 0 + cross block + ZigZag chunk cover every causal block once, with equal work per device and step.
TEST(RingSDPAScheduleTest, ZigZagCoversCausalTriangleExactlyOnceAndBalances) {
    for (auto direction : {RingShiftDirection::Forward, RingShiftDirection::Backward}) {
        for (uint32_t ring_size : {1U, 2U, 3U, 4U, 5U, 8U}) {
            const uint32_t num_chunks = 2U * ring_size;
            std::vector<std::vector<int>> full_count(num_chunks, std::vector<int>(num_chunks, 0));
            std::vector<std::vector<int>> diag_count(num_chunks, std::vector<int>(num_chunks, 0));
            auto first_chunk = [](uint32_t device) { return device; };
            auto second_chunk = [num_chunks](uint32_t device) { return num_chunks - 1U - device; };

            for (uint32_t step = 0; step < ring_size; ++step) {
                for (uint32_t device = 0; device < ring_size; ++device) {
                    const uint32_t src = source_device(device, step, ring_size, direction);
                    double work_units = 0.0;
                    if (step == 0) {
                        ASSERT_EQ(src, device);
                        ++diag_count[first_chunk(device)][first_chunk(device)];
                        ++diag_count[second_chunk(device)][second_chunk(device)];
                        ++full_count[second_chunk(device)][first_chunk(device)];
                        work_units += 2.0;
                    } else {
                        ++full_count[second_chunk(device)][first_chunk(src)];
                        work_units += 1.0;
                    }
                    const auto work = get_ring_step_work(
                        device, step, ring_size, AttentionMaskType::Causal, direction, RingLoadBalancing::ZigZag);
                    EXPECT_EQ(work.execute, step != 0)
                        << "ring " << ring_size << " device " << device << " step " << step;
                    if (work.execute) {
                        EXPECT_EQ(work.mask_type, AttentionMaskType::None);
                        if (work.chunk == 0U) {
                            ++full_count[first_chunk(device)][first_chunk(src)];
                        } else {
                            ASSERT_EQ(work.chunk, 1U);
                            ++full_count[second_chunk(device)][second_chunk(src)];
                        }
                        work_units += 1.0;
                    }
                    EXPECT_DOUBLE_EQ(work_units, 2.0)
                        << "ring " << ring_size << " device " << device << " step " << step;
                }
            }

            for (uint32_t q_chunk = 0; q_chunk < num_chunks; ++q_chunk) {
                for (uint32_t k_chunk = 0; k_chunk < num_chunks; ++k_chunk) {
                    const int expected_full = k_chunk < q_chunk ? 1 : 0;
                    const int expected_diag = k_chunk == q_chunk ? 1 : 0;
                    EXPECT_EQ(full_count[q_chunk][k_chunk], expected_full)
                        << "ring " << ring_size << " q chunk " << q_chunk << " k chunk " << k_chunk;
                    EXPECT_EQ(diag_count[q_chunk][k_chunk], expected_diag)
                        << "ring " << ring_size << " q chunk " << q_chunk << " k chunk " << k_chunk;
                }
            }
        }
    }
}

// ZigZag with a non-causal mask throws.
TEST(RingSDPAScheduleTest, ZigZagRequiresCausalMask) {
    EXPECT_THAT(
        [] {
            get_ring_step_work(
                1U, 1U, 4U, AttentionMaskType::None, RingShiftDirection::Backward, RingLoadBalancing::ZigZag);
        },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("only defined for causal attention")));
}

// Every mode only assigns chunk indices inside its chunk list.
TEST(RingSDPAScheduleTest, AssignedChunkIndexIsBelowChunkCountOfLoadBalancing) {
    struct Mode {
        RingLoadBalancing load_balancing;
        AttentionMaskType mask_type;
        uint32_t first_step;
    };
    static constexpr Mode kModes[] = {
        {RingLoadBalancing::None, AttentionMaskType::None, 0U},
        {RingLoadBalancing::None, AttentionMaskType::Causal, 0U},
        {RingLoadBalancing::ZigZag, AttentionMaskType::Causal, 1U},
    };
    for (const auto& mode : kModes) {
        for (auto direction : {RingShiftDirection::Forward, RingShiftDirection::Backward}) {
            for (uint32_t ring_size : {2U, 3U, 4U, 8U}) {
                for (uint32_t device = 0; device < ring_size; ++device) {
                    for (uint32_t step = mode.first_step; step < ring_size; ++step) {
                        const auto work =
                            get_ring_step_work(device, step, ring_size, mode.mask_type, direction, mode.load_balancing);
                        EXPECT_LT(work.chunk, get_ring_chunk_count(mode.load_balancing))
                            << "load balancing " << static_cast<int>(mode.load_balancing) << " ring " << ring_size
                            << " device " << device << " step " << step;
                    }
                }
            }
        }
    }
}
