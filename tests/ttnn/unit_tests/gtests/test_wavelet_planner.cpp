// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <vector>

#include "ttnn/operations/wavelet/common/signal_extension.hpp"
#include "ttnn/operations/wavelet/planner/inverse_plan.hpp"
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

using CpuStreams = std::array<std::vector<float>, 3>;

struct CpuCoefficients {
    std::vector<float> approximation;
    std::vector<float> detail;
};

[[nodiscard]] size_t stream_index(const wavelet::StorageSlot slot) { return static_cast<size_t>(slot); }

template <typename Scheme, size_t Index = 0>
void execute_forward_routes(
    const wavelet::LiftingForwardPlan& plan, CpuStreams& streams, CpuCoefficients& coefficients) {
    if constexpr (Index < Scheme::num_steps) {
        using Step = wavelet::SchemeStep<Scheme, Index>;
        const auto& route = plan.routes[Index];
        if constexpr (Step::type != wavelet::StepType::kSwap) {
            const auto& source = streams[stream_index(route.source.slot)];
            const auto& base = streams[stream_index(route.base.slot)];
            std::vector<float> output(route.output_length);
            for (size_t index = 0; index < route.output_length; ++index) {
                if constexpr (wavelet::is_predict_update_step(Step::type)) {
                    float value = base[route.base_offset + index];
                    for (size_t coefficient = 0; coefficient < Step::k; ++coefficient) {
                        value += std::bit_cast<float>(Step::coeff_bits[coefficient]) *
                                 source[route.source_offset + index + coefficient];
                    }
                    output[index] = value;
                } else {
                    output[index] = source[route.source_offset + index] * std::bit_cast<float>(Step::coeff_bits[0]);
                }
            }
            streams[stream_index(route.output.slot)] = std::move(output);
            if (route.output.storage == wavelet::RouteOutputStorage::kFinalEvenDram) {
                coefficients.approximation = streams[stream_index(route.output.slot)];
            } else if (route.output.storage == wavelet::RouteOutputStorage::kFinalOddDram) {
                coefficients.detail = streams[stream_index(route.output.slot)];
            }
        }
        execute_forward_routes<Scheme, Index + 1>(plan, streams, coefficients);
    }
}

template <typename Scheme, size_t Index = 0>
void append_predict_update_coefficients(std::vector<std::vector<float>>& coefficients) {
    if constexpr (Index < Scheme::num_steps) {
        using Step = wavelet::SchemeStep<Scheme, Index>;
        if constexpr (wavelet::is_predict_update_step(Step::type)) {
            std::vector<float> step_coefficients;
            step_coefficients.reserve(Step::k);
            for (const uint32_t bits : Step::coeff_bits) {
                step_coefficients.push_back(std::bit_cast<float>(bits));
            }
            coefficients.push_back(std::move(step_coefficients));
        }
        append_predict_update_coefficients<Scheme, Index + 1>(coefficients);
    }
}

template <typename Scheme, size_t Index = 0>
[[nodiscard]] float scale_coefficient(const wavelet::StepType scale_type) {
    if constexpr (Index < Scheme::num_steps) {
        using Step = wavelet::SchemeStep<Scheme, Index>;
        if constexpr (wavelet::is_scale_step(Step::type)) {
            if (Step::type == scale_type) {
                return std::bit_cast<float>(Step::coeff_bits[0]);
            }
        }
        return scale_coefficient<Scheme, Index + 1>(scale_type);
    }
    return 1.0F;
}

template <typename Scheme>
[[nodiscard]] CpuCoefficients cpu_forward(const std::vector<float>& input) {
    const auto plan = wavelet::make_forward_lifting_plan<Scheme>(
        wavelet::SignalBuffer{.length = input.size()}, wavelet::BoundaryMode::kSymmetric);
    CpuStreams streams;
    const size_t padded_length = plan.preprocess_layout.padded_length();
    streams[stream_index(wavelet::StorageSlot::kA)].reserve(plan.preprocess_layout.output.even.length);
    streams[stream_index(wavelet::StorageSlot::kB)].reserve(plan.preprocess_layout.output.odd.length);
    for (size_t index = 0; index < padded_length; ++index) {
        const int32_t source_index =
            static_cast<int32_t>(index) - static_cast<int32_t>(plan.preprocess_layout.pad_config.left);
        const auto extended =
            wavelet::make_extended_index_i32<wavelet::BoundaryMode::kSymmetric>(source_index, input.size());
        const float value = wavelet::evaluate_extended_index_i32<wavelet::BoundaryMode::kSymmetric>(
            extended, input.size(), [&](const uint32_t source) { return input[source]; });
        streams[stream_index(index % 2 == 0 ? wavelet::StorageSlot::kA : wavelet::StorageSlot::kB)].push_back(value);
    }
    CpuCoefficients coefficients;
    execute_forward_routes<Scheme>(plan, streams, coefficients);
    return coefficients;
}

template <typename Scheme>
[[nodiscard]] std::vector<float> cpu_inverse(const CpuCoefficients& coefficients, const size_t output_length) {
    auto inverse = wavelet::make_inverse_lifting_plan<Scheme>(
        output_length, coefficients.approximation.size(), wavelet::BoundaryMode::kSymmetric);
    auto plan = wavelet::make_ilwt_execution_plan(
        std::move(inverse), 1, 1024 * 1024, wavelet::WorkspaceLayout::kRowMajor, false);
    EXPECT_EQ(plan.chunks.size(), 1U);
    const auto& chunk = plan.chunks.front();
    CpuStreams streams;
    streams[stream_index(wavelet::StorageSlot::kA)] = std::vector<float>(
        coefficients.approximation.begin() + chunk.canonical_approximation.begin,
        coefficients.approximation.begin() + chunk.canonical_approximation.end);
    streams[stream_index(wavelet::StorageSlot::kB)] = std::vector<float>(
        coefficients.detail.begin() + chunk.canonical_detail.begin,
        coefficients.detail.begin() + chunk.canonical_detail.end);

    const float even_scale = scale_coefficient<Scheme>(wavelet::StepType::kScaleEven);
    const float odd_scale = scale_coefficient<Scheme>(wavelet::StepType::kScaleOdd);
    for (float& value : streams[stream_index(wavelet::StorageSlot::kA)]) {
        value /= even_scale;
    }
    for (float& value : streams[stream_index(wavelet::StorageSlot::kB)]) {
        value /= odd_scale;
    }

    std::vector<std::vector<float>> step_coefficients;
    append_predict_update_coefficients<Scheme>(step_coefficients);
    size_t coefficient_index = step_coefficients.size();
    for (const auto& route : chunk.routes) {
        const auto& source = streams[stream_index(route.source.slot)];
        const auto& base = streams[stream_index(route.base.slot)];
        const auto& route_coefficients = step_coefficients[--coefficient_index];
        std::vector<float> output(route.output_length);
        for (size_t index = 0; index < route.output_length; ++index) {
            float value = base[route.base_offset_elements + index];
            for (size_t coefficient = 0; coefficient < route_coefficients.size(); ++coefficient) {
                value -= route_coefficients[coefficient] * source[route.source_offset_elements + index + coefficient];
            }
            output[index] = value;
        }
        streams[stream_index(route.output.slot)] = std::move(output);
    }

    std::vector<float> output(output_length);
    const size_t left_pad = plan.full_plan.forward_trace.preprocess_layout.pad_config.left;
    for (size_t index = 0; index < output.size(); ++index) {
        const size_t padded_index = index + left_pad;
        const bool even = padded_index % 2 == 0;
        const auto interval = even ? chunk.reconstructed_even : chunk.reconstructed_odd;
        const size_t offset = even ? chunk.final_even_offset_elements : chunk.final_odd_offset_elements;
        const auto stream = even ? chunk.final_even : chunk.final_odd;
        output[index] = streams[stream_index(stream.slot)][offset + padded_index / 2 - interval.begin];
    }
    return output;
}

void expect_values_near(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1.0e-5F);
    }
}

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

TEST(WaveletPlanner, HaarSymmetricMatchesPyWaveletsAndRoundTripsEvenAndOddLengths) {
    const std::array inputs = {
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F},
    };
    const std::array expected_approximation = {
        std::vector<float>{2.1213202477F, 4.9497470856F, 7.7781744003F},
        std::vector<float>{2.1213202477F, 4.9497470856F, 7.0710678101F},
    };
    const std::array expected_detail = {
        std::vector<float>{-0.7071067691F, -0.7071068287F, -0.7071065903F},
        std::vector<float>{-0.7071067691F, -0.7071068287F, 0.0F},
    };

    for (size_t index = 0; index < inputs.size(); ++index) {
        const CpuCoefficients coefficients = cpu_forward<PlannerTestScheme>(inputs[index]);
        expect_values_near(coefficients.approximation, expected_approximation[index]);
        expect_values_near(coefficients.detail, expected_detail[index]);
        expect_values_near(cpu_inverse<PlannerTestScheme>(coefficients, inputs[index].size()), inputs[index]);
    }
}

TEST(WaveletPlanner, HaarSymmetricTwoDimensionalApproximationMatchesPyWavelets) {
    constexpr size_t side = 4;
    const std::array<float, side * side> input = {
        1.0F,
        2.0F,
        3.0F,
        4.0F,
        5.0F,
        6.0F,
        7.0F,
        8.0F,
        9.0F,
        10.0F,
        11.0F,
        12.0F,
        13.0F,
        14.0F,
        15.0F,
        16.0F,
    };
    std::array<std::vector<float>, side> low_rows;
    for (size_t row = 0; row < side; ++row) {
        low_rows[row] = cpu_forward<PlannerTestScheme>(
                            std::vector<float>(input.begin() + row * side, input.begin() + (row + 1) * side))
                            .approximation;
    }

    std::array<std::vector<float>, 2> low_columns;
    for (size_t column = 0; column < low_rows.front().size(); ++column) {
        std::vector<float> values(side);
        for (size_t row = 0; row < side; ++row) {
            values[row] = low_rows[row][column];
        }
        low_columns[column] = cpu_forward<PlannerTestScheme>(values).approximation;
    }
    std::vector<float> approximation;
    for (size_t row = 0; row < low_columns.front().size(); ++row) {
        for (const auto& column : low_columns) {
            approximation.push_back(column[row]);
        }
    }
    expect_values_near(approximation, {6.9999995232F, 11.0F, 23.0F, 27.0F});
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
