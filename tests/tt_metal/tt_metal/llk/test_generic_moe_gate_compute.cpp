// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>

#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {
namespace {

float selection_score(std::uint32_t expert) {
    // Spread the top sixteen keys so uniform normalized weights cannot pass.
    return expert < 240 ? static_cast<float>(static_cast<int>(expert) - 240) / 64
                        : 1.0f + static_cast<float>(expert - 240) / 4;
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, GenericMoeGateBiasedScorePayload) {
    constexpr std::uint32_t experts = 256;
    constexpr float raw_score = 1.0f / 16;
    std::vector<bfloat16> scores(tt::constants::TILE_HW, bfloat16(0.0f));
    std::vector<bfloat16> biases(tt::constants::TILE_HW, bfloat16(0.0f));
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        // The gate reads the first face. All raw scores are equal; the bias
        // produces distinct, exactly representable BF16 keys in expert order.
        scores[expert] = bfloat16(raw_score);
        biases[expert] = bfloat16(selection_score(expert) - raw_score);
    }
    for (const std::uint32_t selected : {4u, 12u}) {
        for (const bool normalize : {false, true}) {
            for (const bool scores_include_bias : {false, true}) {
                SCOPED_TRACE(
                    ::testing::Message() << "selected=" << selected << ", normalize=" << normalize
                                         << ", scores_include_bias=" << scores_include_bias);
                const auto packed = unit_tests::llk::single_core::run_binary(
                    *devices_.at(0),
                    pack_bfloat16_vec_into_uint32_vec(scores),
                    pack_bfloat16_vec_into_uint32_vec(biases),
                    1,
                    "tests/tt_metal/tt_metal/test_kernels/compute/generic_moe_gate_compute.cpp",
                    {},
                    1,
                    1,
                    {selected, static_cast<std::uint32_t>(normalize), static_cast<std::uint32_t>(scores_include_bias)});
                ASSERT_EQ(packed.size(), tt::constants::TILE_HW / 2);
                float denominator = 0.0f;
                for (std::uint32_t rank = 0; rank < selected; ++rank) {
                    denominator += scores_include_bias ? selection_score(experts - 1 - rank) : raw_score;
                }
                const std::uint32_t emitted_rows = selected <= 8 ? 8 : 16;
                for (std::uint32_t rank = 0; rank < emitted_rows; ++rank) {
                    // Winners occupy column zero in descending key order.
                    const auto offset = rank * tt::constants::FACE_WIDTH;
                    const auto bits = ((packed[offset / 2] >> (16 * (offset % 2))) & 0xffff) << 16;
                    const float actual = std::bit_cast<float>(bits);
                    if (rank >= selected) {
                        EXPECT_EQ(actual, 0.0f) << "rank=" << rank;
                    } else {
                        const float payload = scores_include_bias ? selection_score(experts - 1 - rank) : raw_score;
                        const float expected = normalize ? payload / denominator : payload;
                        if (normalize) {
                            // SFPU reciprocal approximation and BF16 truncation.
                            EXPECT_NEAR(actual, expected, expected / 64) << "rank=" << rank;
                        } else {
                            EXPECT_EQ(actual, expected) << "rank=" << rank;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
