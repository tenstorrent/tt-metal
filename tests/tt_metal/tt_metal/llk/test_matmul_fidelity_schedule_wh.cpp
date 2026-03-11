// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_llk_wormhole_b0/llk_lib/llk_math_matmul_fidelity.h"

namespace tt::tt_metal::unit_tests::compute::matmul {

TEST(WormholeMatmulFidelitySchedule, SkipsHiFi2ForBfloat16TimesBfp4AtHiFi3) {
    constexpr std::uint32_t format_extra_bit = 1u << DATA_FORMAT_BIT_COUNT;

    EXPECT_TRUE(ckernel::should_skip_hifi2_for_bf16_bfp4_matmul(
        ckernel::MathFidelity::HiFi3, ::DataFormat::Float16_b, ::DataFormat::Bfp4_b));
    EXPECT_EQ(
        ckernel::get_matmul_fidelity_phase_count(
            ckernel::MathFidelity::HiFi3,
            static_cast<std::uint32_t>(::DataFormat::Float16_b),
            static_cast<std::uint32_t>(::DataFormat::Bfp4_b)),
        2u);
    EXPECT_EQ(
        ckernel::get_matmul_fidelity_increment(
            ckernel::MathFidelity::HiFi3,
            static_cast<std::uint32_t>(::DataFormat::Float16_b),
            static_cast<std::uint32_t>(::DataFormat::Bfp4_b)),
        2u);
    EXPECT_TRUE(ckernel::should_skip_hifi2_for_bf16_bfp4_matmul(
        ckernel::MathFidelity::HiFi3,
        static_cast<std::uint32_t>(::DataFormat::Float16_b) | format_extra_bit,
        static_cast<std::uint32_t>(::DataFormat::Bfp4_b) | format_extra_bit));
}

TEST(WormholeMatmulFidelitySchedule, KeepsStandardHiFi3ScheduleForNonBfp4Inputs) {
    EXPECT_FALSE(ckernel::should_skip_hifi2_for_bf16_bfp4_matmul(
        ckernel::MathFidelity::HiFi3, ::DataFormat::Float16_b, ::DataFormat::Float16_b));
    EXPECT_EQ(
        ckernel::get_matmul_fidelity_phase_count(
            ckernel::MathFidelity::HiFi3,
            static_cast<std::uint32_t>(::DataFormat::Float16_b),
            static_cast<std::uint32_t>(::DataFormat::Float16_b)),
        3u);
    EXPECT_EQ(
        ckernel::get_matmul_fidelity_increment(
            ckernel::MathFidelity::HiFi3,
            static_cast<std::uint32_t>(::DataFormat::Float16_b),
            static_cast<std::uint32_t>(::DataFormat::Float16_b)),
        1u);
}

TEST(WormholeMatmulFidelitySchedule, KeepsStandardHiFi2ScheduleForBfloat16TimesBfp4) {
    EXPECT_FALSE(ckernel::should_skip_hifi2_for_bf16_bfp4_matmul(
        ckernel::MathFidelity::HiFi2, ::DataFormat::Float16_b, ::DataFormat::Bfp4_b));
    EXPECT_EQ(
        ckernel::get_matmul_fidelity_phase_count(
            ckernel::MathFidelity::HiFi2,
            static_cast<std::uint32_t>(::DataFormat::Float16_b),
            static_cast<std::uint32_t>(::DataFormat::Bfp4_b)),
        2u);
    EXPECT_EQ(
        ckernel::get_matmul_fidelity_increment(
            ckernel::MathFidelity::HiFi2,
            static_cast<std::uint32_t>(::DataFormat::Float16_b),
            static_cast<std::uint32_t>(::DataFormat::Bfp4_b)),
        1u);
}

}  // namespace tt::tt_metal::unit_tests::compute::matmul
