// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>

#include "ttnn/operations/wavelet/common/signal_extension.hpp"
#include "ttnn/operations/wavelet/planner/inverse_plan_2d.hpp"
#include "ttnn/operations/wavelet/planner/plan_2d.hpp"
#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace {

namespace wavelet = ttnn::operations::wavelet;

struct PlannerTestScheme {
    static constexpr uint32_t tap_size = 2;
    static constexpr int32_t delay_even = 0;
    static constexpr int32_t delay_odd = 1;
    static constexpr uint32_t num_steps = 5;

    template <std::size_t I>
    struct step;
};

template <wavelet::BoundaryMode Mode>
float extended_value(const int32_t index) {
    constexpr std::array<float, 4> source = {1.0F, 2.0F, 3.0F, 4.0F};
    const auto extended = wavelet::make_extended_index_i32<Mode>(index, source.size());
    return wavelet::evaluate_extended_index_i32<Mode>(
        extended, source.size(), [&](const uint32_t source_index) { return source[source_index]; });
}

template <>
struct PlannerTestScheme::step<0> {
    using type = wavelet::StaticStep<wavelet::StepType::kPredict, 0, 0x3f800000U>;
};

template <>
struct PlannerTestScheme::step<1> {
    using type = wavelet::StaticStep<wavelet::StepType::kSwap, 0>;
};

template <>
struct PlannerTestScheme::step<2> {
    using type = wavelet::StaticStep<wavelet::StepType::kPredict, 0, 0xbf000000U>;
};

template <>
struct PlannerTestScheme::step<3> {
    using type = wavelet::StaticStep<wavelet::StepType::kScaleEven, 0, 0x3f3504f3U>;
};

template <>
struct PlannerTestScheme::step<4> {
    using type = wavelet::StaticStep<wavelet::StepType::kScaleOdd, 0, 0xbfb504f3U>;
};

TEST(WaveletPlanner, CeilDivHandlesMaximumSizeWithoutOverflow) {
    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    EXPECT_EQ(wavelet::ceil_div(maximum, size_t{1}), maximum);
    EXPECT_EQ(wavelet::ceil_div(maximum, size_t{2}), maximum / 2 + 1);
    EXPECT_EQ(wavelet::ceil_div(maximum - 1, size_t{2}), maximum / 2);
    EXPECT_EQ(wavelet::ceil_div(size_t{17}, size_t{0}), 0);
}

TEST(WaveletPlanner, BoundaryExtensionCoversAllSupportedModes) {
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kZero>(-1), 0.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kZero>(4), 0.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kConstant>(-1), 1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kConstant>(4), 4.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSymmetric>(-1), 1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSymmetric>(4), 4.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kReflect>(-1), 2.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kReflect>(4), 3.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kPeriodic>(-1), 4.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kPeriodic>(4), 1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSmooth>(-1), 0.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSmooth>(4), 5.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntisymmetric>(-1), -1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntisymmetric>(4), -4.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntireflect>(-1), 0.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntireflect>(4), 5.0F);
}

TEST(WaveletPlanner, ForwardPlanEnforcesStickAndSignedIndexContracts) {
    EXPECT_ANY_THROW({
        [[maybe_unused]] const auto plan = wavelet::make_forward_lifting_plan<PlannerTestScheme>(wavelet::SignalBuffer{
            .length = 64,
            .stick_width = 16,
            .element_size_bytes = sizeof(float),
        });
    });

    constexpr size_t maximum_length = static_cast<size_t>(wavelet::kMaxSignedDeviceIndex) - 2;
    EXPECT_NO_THROW({
        [[maybe_unused]] const auto plan = wavelet::make_forward_lifting_plan<PlannerTestScheme>(wavelet::SignalBuffer{
            .length = maximum_length,
            .stick_width = wavelet::kStickWidth,
            .element_size_bytes = sizeof(float),
        });
    });
    EXPECT_ANY_THROW({
        [[maybe_unused]] const auto plan = wavelet::make_forward_lifting_plan<PlannerTestScheme>(wavelet::SignalBuffer{
            .length = maximum_length + 1,
            .stick_width = wavelet::kStickWidth,
            .element_size_bytes = sizeof(float),
        });
    });
}

TEST(WaveletPlanner, TwoDimensionalPlansRejectUnsupportedSignedGeometryEarly) {
    constexpr size_t invalid_extent = static_cast<size_t>(wavelet::kMax2DLogicalExtent) + 1;
    EXPECT_ANY_THROW({
        [[maybe_unused]] const auto plan = wavelet::make_lwt_2d_execution_plan<PlannerTestScheme>(
            invalid_extent, 1, 1, 768 * 1024, wavelet::BoundaryMode::kZero);
    });
    EXPECT_ANY_THROW({
        [[maybe_unused]] const auto plan = wavelet::make_ilwt_2d_execution_plan<PlannerTestScheme>(
            1, invalid_extent, 1, 768 * 1024, wavelet::BoundaryMode::kZero);
    });
}

TEST(WaveletPlanner, OddRectangularPlansProduceBoundedSerializableChunks) {
    constexpr size_t height = 33;
    constexpr size_t width = 35;
    constexpr uint64_t l1_budget_bytes = 768 * 1024;
    const auto forward = wavelet::make_lwt_2d_execution_plan<PlannerTestScheme>(
        height,
        width,
        8,
        l1_budget_bytes,
        wavelet::BoundaryMode::kSymmetric,
        true,
        true,
        wavelet::Lwt2DRouteDomainPolicy::kExact);
    ASSERT_FALSE(forward.chunks.empty());
    for (const auto& chunk : forward.chunks) {
        EXPECT_LE(chunk.final_band_rect.y.end, forward.tiling.band.logical.height);
        EXPECT_LE(chunk.final_band_rect.x.end, forward.tiling.band.logical.width);
        EXPECT_LE(chunk.resources.total_l1_bytes, l1_budget_bytes);
    }
    EXPECT_EQ(
        wavelet::build_lwt_2d_chunk_config_words(forward).size(),
        forward.chunks.size() * wavelet::device_protocol::kLwt2DChunkConfigWordCount);
    EXPECT_FALSE(wavelet::build_lwt_2d_route_config_words(forward).empty());

    const auto inverse = wavelet::make_ilwt_2d_execution_plan<PlannerTestScheme>(
        height, width, 8, l1_budget_bytes, wavelet::BoundaryMode::kSymmetric);
    ASSERT_FALSE(inverse.chunks.empty());
    for (const auto& chunk : inverse.chunks) {
        EXPECT_LE(chunk.final_band_rect.y.end, height);
        EXPECT_LE(chunk.final_band_rect.x.end, width);
        EXPECT_LE(chunk.resources.total_l1_bytes, l1_budget_bytes);
    }
    EXPECT_EQ(
        wavelet::build_ilwt_2d_chunk_config_words(inverse).size(),
        inverse.chunks.size() * wavelet::device_protocol::kLwt2DChunkConfigWordCount);
    EXPECT_FALSE(wavelet::build_ilwt_2d_route_config_words(inverse).empty());
}

}  // namespace
