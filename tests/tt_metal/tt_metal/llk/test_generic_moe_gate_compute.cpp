// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>

#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t experts = tt::constants::FACE_HW;
constexpr std::uint32_t top_score_count = 16;
constexpr std::uint32_t first_top_expert = experts - top_score_count;

float selection_score(std::uint32_t expert) {
    constexpr float lower_score_step = 1.0f / 64;
    constexpr float top_score_step = 1.0f / 4;
    constexpr float first_top_score = 1.0f;
    // Keep lower-ranked keys negative and spread the top sixteen from 1 to 4.75,
    // so uniform normalized weights cannot pass. Every key is exactly BF16-representable.
    return expert < first_top_expert
               ? static_cast<float>(static_cast<int>(expert) - static_cast<int>(first_top_expert)) * lower_score_step
               : first_top_score + static_cast<float>(expert - first_top_expert) * top_score_step;
}

// Compile-only program for the gate kernel; never launched. Mirrors the layout run_binary builds
// (c_0 scores, c_1 bias, c_16 out) so only the Dest accumulation mode differs between cases.
Program make_generic_moe_gate_compile_program(bool fp32_dest_acc_en, bool explicit_fp32_false = false) {
    Program program = CreateProgram();
    const CoreCoord core{0, 0};
    const std::uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    for (const auto cb : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16}) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tile_bytes, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, tile_bytes));
    }
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/generic_moe_gate_compute.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            // num_tiles, selected, normalize, scores_include_bias
            .compile_args = {1, 8, 0, 0},
            .defines = explicit_fp32_false
                           ? std::map<std::string, std::string>{{"GENERIC_MOE_GATE_EXPLICIT_FP32_FALSE", "1"}}
                           : std::map<std::string, std::string>{}});
    return program;
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, GenericMoeGateRejectsFp32DestAcc) {
    // Compile-time contract test; these programs are never launched. The SFPU top-k pairs the
    // expert id and score in one LReg lane and moves them with LO16_ONLY/HI16_ONLY loads and
    // stores, which address Dest as 16-bit rows, so the gate only works with a 16-bit Dest; an
    // fp32 Dest build must not compile, whether is_fp32_dest_acc_en follows DST_ACCUM_MODE
    // (the default) or is passed as an explicit false.
    struct Case {
        bool fp32_dest_acc_en;
        bool explicit_fp32_false;
        bool must_compile;
    };
    for (const Case c : {Case{false, false, true}, Case{true, false, false}, Case{true, true, false}}) {
        SCOPED_TRACE(
            ::testing::Message() << "fp32_dest_acc_en=" << c.fp32_dest_acc_en
                                 << " explicit_fp32_false=" << c.explicit_fp32_false);
        auto program = make_generic_moe_gate_compile_program(c.fp32_dest_acc_en, c.explicit_fp32_false);
        if (c.must_compile) {
            EXPECT_NO_THROW(program.impl().compile(&this->device()));
        } else {
            EXPECT_THROW(program.impl().compile(&this->device()), std::runtime_error);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, GenericMoeGateBiasedScorePayload) {
    constexpr float raw_score = 1.0f / 16;
    // Allow for SFPU reciprocal approximation and BF16 truncation when normalizing.
    constexpr float normalized_relative_tolerance = 1.0f / 64;
    std::vector<bfloat16> scores(tt::constants::TILE_HW, bfloat16(0.0f));
    std::vector<bfloat16> biases(tt::constants::TILE_HW, bfloat16(0.0f));
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        // The gate reads the first face. All raw scores are equal; the bias
        // produces distinct, exactly representable BF16 keys in expert order.
        scores[expert] = bfloat16(raw_score);
        biases[expert] = bfloat16(selection_score(expert) - raw_score);
    }
    // Exercise both top-8 and top-16 paths, with unused output rows to check zero_tail.
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
                    /*num_tiles=*/1,
                    "tests/tt_metal/tt_metal/test_kernels/compute/generic_moe_gate_compute.cpp",
                    /*compute_defines=*/{},
                    /*cb_depth_tiles=*/1,
                    /*out_tiles=*/1,
                    {selected, static_cast<std::uint32_t>(normalize), static_cast<std::uint32_t>(scores_include_bias)});
                const auto result = unpack_uint32_vec_into_bfloat16_vec(packed);
                ASSERT_EQ(result.size(), tt::constants::TILE_HW);
                float denominator = 0.0f;
                for (std::uint32_t rank = 0; rank < selected; ++rank) {
                    denominator += scores_include_bias ? selection_score(experts - 1 - rank) : raw_score;
                }
                // The gate emits eight or sixteen rows, including the zeroed tail.
                const std::uint32_t emitted_rows = selected <= 8 ? 8 : 16;
                for (std::uint32_t rank = 0; rank < emitted_rows; ++rank) {
                    // Winners occupy column zero in descending key order.
                    const auto offset = rank * tt::constants::FACE_WIDTH;
                    const float actual = static_cast<float>(result[offset]);
                    if (rank >= selected) {
                        EXPECT_EQ(actual, 0.0f) << "rank=" << rank;
                    } else {
                        const float payload = scores_include_bias ? selection_score(experts - 1 - rank) : raw_score;
                        const float expected = normalize ? payload / denominator : payload;
                        if (normalize) {
                            EXPECT_NEAR(actual, expected, expected * normalized_relative_tolerance) << "rank=" << rank;
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
