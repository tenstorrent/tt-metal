// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
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

template <wavelet::BoundaryMode Mode, size_t Size>
float extended_value(const int32_t index, const std::array<float, Size>& source) {
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

struct MultiTapTestScheme {
    static constexpr uint32_t tap_size = 9;
    static constexpr int32_t delay_even = 3;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 4;

    template <std::size_t I>
    struct step;
};

template <>
struct MultiTapTestScheme::step<0> {
    using type = wavelet::StaticStep<wavelet::StepType::kPredict, -1, 0x3e800000U, 0x3f000000U, 0x3e800000U>;
};

template <>
struct MultiTapTestScheme::step<1> {
    using type = wavelet::StaticStep<wavelet::StepType::kUpdate, -2, 0xbe000000U, 0x3e800000U, 0xbe000000U>;
};

template <>
struct MultiTapTestScheme::step<2> {
    using type = wavelet::StaticStep<wavelet::StepType::kScaleEven, 0, 0x3f800000U>;
};

template <>
struct MultiTapTestScheme::step<3> {
    using type = wavelet::StaticStep<wavelet::StepType::kScaleOdd, 0, 0x3f800000U>;
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

void expect_interval_eq(const wavelet::IndexInterval actual, const wavelet::IndexInterval expected) {
    EXPECT_EQ(actual.begin, expected.begin);
    EXPECT_EQ(actual.end, expected.end);
}

void expect_rectangle_eq(const wavelet::IndexRectangle& actual, const wavelet::IndexRectangle& expected) {
    expect_interval_eq(actual.y, expected.y);
    expect_interval_eq(actual.x, expected.x);
}

void expect_route_eq(const wavelet::Lwt2DRoutePlan& actual, const wavelet::Lwt2DRoutePlan& expected) {
    EXPECT_EQ(actual.axis, expected.axis);
    EXPECT_EQ(actual.axis_route_index, expected.axis_route_index);
    EXPECT_EQ(actual.type, expected.type);
    EXPECT_EQ(actual.source_slot, expected.source_slot);
    EXPECT_EQ(actual.base_slot, expected.base_slot);
    EXPECT_EQ(actual.output_slot, expected.output_slot);
    expect_rectangle_eq(actual.source, expected.source);
    expect_rectangle_eq(actual.base, expected.base);
    expect_rectangle_eq(actual.output, expected.output);
    EXPECT_EQ(actual.inline_terminal_scale, expected.inline_terminal_scale);
}

void expect_chunk_eq(const wavelet::Lwt2DChunkPlan& actual, const wavelet::Lwt2DChunkPlan& expected) {
    expect_rectangle_eq(actual.final_band_rect, expected.final_band_rect);
    expect_rectangle_eq(actual.execution_band_rect, expected.execution_band_rect);
    expect_rectangle_eq(actual.initial.ee, expected.initial.ee);
    expect_rectangle_eq(actual.initial.eo, expected.initial.eo);
    expect_rectangle_eq(actual.initial.oe, expected.initial.oe);
    expect_rectangle_eq(actual.initial.oo, expected.initial.oo);
    ASSERT_EQ(actual.routes.size(), expected.routes.size());
    for (size_t index = 0; index < expected.routes.size(); ++index) {
        SCOPED_TRACE("route " + std::to_string(index));
        expect_route_eq(actual.routes[index], expected.routes[index]);
    }
    EXPECT_EQ(actual.final_bands.ll, expected.final_bands.ll);
    EXPECT_EQ(actual.final_bands.lh, expected.final_bands.lh);
    EXPECT_EQ(actual.final_bands.hl, expected.final_bands.hl);
    EXPECT_EQ(actual.final_bands.hh, expected.final_bands.hh);
    expect_rectangle_eq(actual.final_band_sources.ll, expected.final_band_sources.ll);
    expect_rectangle_eq(actual.final_band_sources.lh, expected.final_band_sources.lh);
    expect_rectangle_eq(actual.final_band_sources.hl, expected.final_band_sources.hl);
    expect_rectangle_eq(actual.final_band_sources.hh, expected.final_band_sources.hh);
    EXPECT_EQ(actual.resources.plane_heights_elements, expected.resources.plane_heights_elements);
    EXPECT_EQ(actual.resources.plane_widths_elements, expected.resources.plane_widths_elements);
    EXPECT_EQ(actual.resources.total_l1_bytes, expected.resources.total_l1_bytes);
    EXPECT_DOUBLE_EQ(actual.dependency_overhead, expected.dependency_overhead);
}

void expect_chunks_eq(
    const std::vector<wavelet::Lwt2DChunkPlan>& actual, const std::vector<wavelet::Lwt2DChunkPlan>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        SCOPED_TRACE("chunk " + std::to_string(index));
        expect_chunk_eq(actual[index], expected[index]);
    }
}

[[nodiscard]] uint64_t exhaustive_schedule_cost(
    const std::vector<wavelet::Lwt2DChunkPlan>& chunks,
    const uint32_t active_core_count,
    const wavelet::LiftingForwardPlan& y_plan,
    const wavelet::LiftingForwardPlan& x_plan,
    const bool inverse,
    const uint64_t penalty_per_core) {
    std::vector<uint64_t> chunk_costs;
    chunk_costs.reserve(chunks.size());
    for (const auto& chunk : chunks) {
        chunk_costs.push_back(wavelet::plan_2d_detail::estimate_chunk_cost(chunk, y_plan, x_plan, inverse));
    }
    const size_t base = chunks.size() / active_core_count;
    const size_t extra = chunks.size() % active_core_count;
    size_t begin = 0;
    uint64_t maximum = 0;
    for (uint32_t core = 0; core < active_core_count; ++core) {
        const size_t count = base + (core < extra ? 1U : 0U);
        uint64_t core_cost = wavelet::planner_cost_model::kCoreStartup;
        for (size_t index = 0; index < count; ++index) {
            core_cost += chunk_costs[begin + index];
        }
        maximum = std::max(maximum, core_cost);
        begin += count;
    }
    if (penalty_per_core > 0 && active_core_count > 64) {
        maximum += static_cast<uint64_t>(active_core_count - 64) * penalty_per_core;
    }
    return maximum;
}

[[nodiscard]] wavelet::plan_2d_detail::Candidate exhaustive_forward_candidate(
    const wavelet::LiftingForwardPlan& y_plan,
    const wavelet::LiftingForwardPlan& x_plan,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    const bool fuse_terminal_scale,
    const bool latency_oriented,
    const wavelet::Lwt2DRouteDomainPolicy route_domain) {
    const uint32_t band_tiles_y = static_cast<uint32_t>(
        tt::div_up(y_plan.output_length, static_cast<size_t>(wavelet::plan_2d_detail::kTileHeight)));
    const uint32_t band_tiles_x = static_cast<uint32_t>(
        tt::div_up(x_plan.output_length, static_cast<size_t>(wavelet::plan_2d_detail::kTileWidth)));
    const auto y_classes = wavelet::plan_2d_detail::make_axis_candidate_classes(
        y_plan.output_length,
        band_tiles_y,
        wavelet::plan_2d_detail::kTileHeight,
        [&](const wavelet::IndexInterval output) {
            return wavelet::plan_2d_detail::make_forward_axis_signature(
                y_plan,
                output,
                wavelet::plan_2d_detail::kTileHeight,
                route_domain,
                "oracle vertical even",
                "oracle vertical odd");
        });
    const auto x_classes = wavelet::plan_2d_detail::make_axis_candidate_classes(
        x_plan.output_length,
        band_tiles_x,
        wavelet::plan_2d_detail::kTileWidth,
        [&](const wavelet::IndexInterval output) {
            return wavelet::plan_2d_detail::make_forward_axis_signature(
                x_plan,
                output,
                wavelet::plan_2d_detail::kTileWidth,
                route_domain,
                "oracle horizontal even",
                "oracle horizontal odd");
        });
    wavelet::plan_2d_detail::Candidate best{};
    bool found = false;
    for (uint32_t tiles_y = 1; tiles_y <= band_tiles_y; ++tiles_y) {
        for (uint32_t tiles_x = 1; tiles_x <= band_tiles_x; ++tiles_x) {
            auto chunks = wavelet::plan_2d_detail::build_chunks(
                y_plan, x_plan, tiles_y, tiles_x, fuse_terminal_scale, route_domain);
            double max_dependency_overhead = 0.0;
            bool fits = true;
            for (const auto& chunk : chunks) {
                max_dependency_overhead = std::max(max_dependency_overhead, chunk.dependency_overhead);
                fits = fits && chunk.resources.total_l1_bytes <= l1_budget_bytes;
            }
            const auto representative = wavelet::plan_2d_detail::evaluate_candidate(
                y_classes[tiles_y],
                x_classes[tiles_x],
                tiles_y,
                tiles_x,
                core_limit,
                l1_budget_bytes,
                [&](const wavelet::IndexRectangle output) {
                    return wavelet::plan_2d_detail::build_chunk(
                        y_plan, x_plan, output, fuse_terminal_scale, route_domain);
                },
                [&](const wavelet::Lwt2DChunkPlan& chunk) {
                    return wavelet::plan_2d_detail::estimate_chunk_cost(chunk, y_plan, x_plan);
                },
                0);
            if (!fits) {
                EXPECT_FALSE(representative.has_value());
                continue;
            }
            const uint32_t active_core_count =
                static_cast<uint32_t>(std::min(chunks.size(), static_cast<size_t>(core_limit)));
            wavelet::plan_2d_detail::Candidate candidate{
                .chunk_tiles_y = tiles_y,
                .chunk_tiles_x = tiles_x,
                .active_core_count = active_core_count,
                .max_dependency_overhead = max_dependency_overhead,
                .estimated_cost = exhaustive_schedule_cost(chunks, active_core_count, y_plan, x_plan, false, 0),
                .chunks = std::move(chunks),
            };
            EXPECT_TRUE(representative.has_value());
            if (representative.has_value()) {
                EXPECT_EQ(representative->active_core_count, candidate.active_core_count);
                EXPECT_EQ(representative->max_dependency_overhead, candidate.max_dependency_overhead);
                EXPECT_EQ(representative->estimated_cost, candidate.estimated_cost);
            }
            if (!found || wavelet::plan_2d_detail::is_better_candidate(candidate, best, latency_oriented)) {
                best = std::move(candidate);
                found = true;
            }
        }
    }
    EXPECT_TRUE(found);
    return best;
}

[[nodiscard]] wavelet::plan_2d_detail::Candidate exhaustive_inverse_candidate(
    const wavelet::LiftingInversePlan& y_plan,
    const wavelet::LiftingInversePlan& x_plan,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    const uint64_t penalty_per_core) {
    const uint32_t output_tiles_y =
        static_cast<uint32_t>(tt::div_up(y_plan.original_length, static_cast<size_t>(wavelet::kTileHeight2D)));
    const uint32_t output_tiles_x =
        static_cast<uint32_t>(tt::div_up(x_plan.original_length, static_cast<size_t>(wavelet::kTileWidth2D)));
    const auto y_classes = wavelet::plan_2d_detail::make_axis_candidate_classes(
        y_plan.original_length, output_tiles_y, wavelet::kTileHeight2D, [&](const wavelet::IndexInterval output) {
            return wavelet::inverse_2d_detail::make_inverse_axis_signature(y_plan, output, wavelet::kTileHeight2D);
        });
    const auto x_classes = wavelet::plan_2d_detail::make_axis_candidate_classes(
        x_plan.original_length, output_tiles_x, wavelet::kTileWidth2D, [&](const wavelet::IndexInterval output) {
            return wavelet::inverse_2d_detail::make_inverse_axis_signature(x_plan, output, wavelet::kTileWidth2D);
        });
    wavelet::plan_2d_detail::Candidate best{};
    bool found = false;
    for (uint32_t tiles_y = 1; tiles_y <= output_tiles_y; ++tiles_y) {
        for (uint32_t tiles_x = 1; tiles_x <= output_tiles_x; ++tiles_x) {
            auto chunks = wavelet::inverse_2d_detail::build_chunks(y_plan, x_plan, tiles_y, tiles_x);
            double max_dependency_overhead = 0.0;
            bool fits = true;
            for (const auto& chunk : chunks) {
                max_dependency_overhead = std::max(max_dependency_overhead, chunk.dependency_overhead);
                fits = fits && chunk.resources.total_l1_bytes <= l1_budget_bytes;
            }
            const auto representative = wavelet::plan_2d_detail::evaluate_candidate(
                y_classes[tiles_y],
                x_classes[tiles_x],
                tiles_y,
                tiles_x,
                core_limit,
                l1_budget_bytes,
                [&](const wavelet::IndexRectangle output) {
                    return wavelet::inverse_2d_detail::build_chunk(y_plan, x_plan, output);
                },
                [&](const wavelet::Lwt2DChunkPlan& chunk) {
                    return wavelet::plan_2d_detail::estimate_chunk_cost(
                        chunk, y_plan.forward_trace, x_plan.forward_trace, true);
                },
                penalty_per_core);
            if (!fits) {
                EXPECT_FALSE(representative.has_value());
                continue;
            }
            const uint32_t active_core_count =
                static_cast<uint32_t>(std::min(chunks.size(), static_cast<size_t>(core_limit)));
            wavelet::plan_2d_detail::Candidate candidate{
                .chunk_tiles_y = tiles_y,
                .chunk_tiles_x = tiles_x,
                .active_core_count = active_core_count,
                .max_dependency_overhead = max_dependency_overhead,
                .estimated_cost = exhaustive_schedule_cost(
                    chunks, active_core_count, y_plan.forward_trace, x_plan.forward_trace, true, penalty_per_core),
                .chunks = std::move(chunks),
            };
            EXPECT_TRUE(representative.has_value());
            if (representative.has_value()) {
                EXPECT_EQ(representative->active_core_count, candidate.active_core_count);
                EXPECT_EQ(representative->max_dependency_overhead, candidate.max_dependency_overhead);
                EXPECT_EQ(representative->estimated_cost, candidate.estimated_cost);
            }
            if (!found || wavelet::plan_2d_detail::is_better_candidate(candidate, best, true)) {
                best = std::move(candidate);
                found = true;
            }
        }
    }
    EXPECT_TRUE(found);
    return best;
}

template <typename Scheme>
void expect_forward_matches_exhaustive(
    const size_t height,
    const size_t width,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    const bool latency_oriented,
    const wavelet::BoundaryMode boundary_mode,
    const wavelet::Lwt2DRouteDomainPolicy route_domain) {
    const auto y_plan =
        wavelet::make_forward_lifting_plan<Scheme>(wavelet::SignalBuffer{.length = height}, boundary_mode);
    const auto x_plan =
        wavelet::make_forward_lifting_plan<Scheme>(wavelet::SignalBuffer{.length = width}, boundary_mode);
    const auto expected =
        exhaustive_forward_candidate(y_plan, x_plan, core_limit, l1_budget_bytes, true, latency_oriented, route_domain);
    const auto actual = wavelet::make_lwt_2d_execution_plan(
        y_plan, x_plan, core_limit, l1_budget_bytes, true, latency_oriented, route_domain);

    ASSERT_FALSE(actual.chunks.empty());
    EXPECT_EQ(
        expected.chunk_tiles_y,
        tt::div_up(actual.chunks.front().final_band_rect.height(), static_cast<size_t>(wavelet::kTileHeight2D)));
    EXPECT_EQ(
        expected.chunk_tiles_x,
        tt::div_up(actual.chunks.front().final_band_rect.width(), static_cast<size_t>(wavelet::kTileWidth2D)));
    EXPECT_EQ(
        expected.active_core_count,
        static_cast<uint32_t>(std::min(actual.chunks.size(), static_cast<size_t>(core_limit))));
    EXPECT_EQ(
        expected.estimated_cost,
        exhaustive_schedule_cost(actual.chunks, expected.active_core_count, y_plan, x_plan, false, 0));
    expect_chunks_eq(actual.chunks, expected.chunks);
}

template <typename Scheme>
void expect_inverse_matches_exhaustive(
    const size_t height,
    const size_t width,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    const wavelet::BoundaryMode boundary_mode,
    const uint64_t penalty_per_core) {
    const auto y_forward =
        wavelet::make_forward_lifting_plan<Scheme>(wavelet::SignalBuffer{.length = height}, boundary_mode);
    const auto x_forward =
        wavelet::make_forward_lifting_plan<Scheme>(wavelet::SignalBuffer{.length = width}, boundary_mode);
    const wavelet::LiftingInversePlan y_plan{
        .forward_trace = y_forward,
        .original_length = height,
        .coefficient_length = y_forward.output_length,
    };
    const wavelet::LiftingInversePlan x_plan{
        .forward_trace = x_forward,
        .original_length = width,
        .coefficient_length = x_forward.output_length,
    };
    const auto expected = exhaustive_inverse_candidate(y_plan, x_plan, core_limit, l1_budget_bytes, penalty_per_core);
    const auto actual =
        wavelet::make_ilwt_2d_execution_plan(y_plan, x_plan, core_limit, l1_budget_bytes, penalty_per_core);

    ASSERT_FALSE(actual.chunks.empty());
    EXPECT_EQ(
        expected.chunk_tiles_y,
        tt::div_up(actual.chunks.front().final_band_rect.height(), static_cast<size_t>(wavelet::kTileHeight2D)));
    EXPECT_EQ(
        expected.chunk_tiles_x,
        tt::div_up(actual.chunks.front().final_band_rect.width(), static_cast<size_t>(wavelet::kTileWidth2D)));
    EXPECT_EQ(
        expected.active_core_count,
        static_cast<uint32_t>(std::min(actual.chunks.size(), static_cast<size_t>(core_limit))));
    EXPECT_EQ(
        expected.estimated_cost,
        exhaustive_schedule_cost(
            actual.chunks, expected.active_core_count, y_forward, x_forward, true, penalty_per_core));
    expect_chunks_eq(actual.chunks, expected.chunks);
}

TEST(WaveletPlanner, BoundaryExtensionCoversAllSupportedModes) {
    constexpr std::array<float, 4> source = {1.0F, 2.0F, 3.0F, 4.0F};
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kZero>(-1, source), 0.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kZero>(4, source), 0.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kConstant>(-9, source), 1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kConstant>(12, source), 4.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSymmetric>(-9, source), 1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSymmetric>(12, source), 4.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kReflect>(-7, source), 2.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kReflect>(10, source), 3.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kPeriodic>(-9, source), 4.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kPeriodic>(12, source), 1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSmooth>(-3, source), -2.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSmooth>(6, source), 7.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntisymmetric>(-9, source), -1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntisymmetric>(12, source), -4.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntireflect>(-7, source), -6.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntireflect>(10, source), 11.0F);

    constexpr std::array<float, 1> singleton = {7.0F};
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kReflect>(-17, singleton), 7.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kSmooth>(17, singleton), 7.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntireflect>(-17, singleton), 7.0F);

    constexpr std::array<float, 2> pair = {1.0F, 2.0F};
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kReflect>(-3, pair), 2.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kReflect>(4, pair), 1.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntireflect>(-3, pair), -2.0F);
    EXPECT_EQ(extended_value<wavelet::BoundaryMode::kAntireflect>(4, pair), 5.0F);

    const auto empty = wavelet::make_extended_index_i32<wavelet::BoundaryMode::kAntireflect>(5, 0);
    EXPECT_EQ(empty.source_index, 0U);
    EXPECT_EQ(empty.operation, wavelet::ExtensionOperation::kZero);

    constexpr int32_t minimum = std::numeric_limits<int32_t>::min();
    EXPECT_EQ(wavelet::extension_positive_mod_i32(minimum, 6), 4U);
    const auto minimum_antireflect = wavelet::make_antireflect_index_i32(minimum, source.size());
    EXPECT_EQ(minimum_antireflect.source_index, 2U);
    EXPECT_EQ(minimum_antireflect.period_quotient, -357913942);
    EXPECT_TRUE(minimum_antireflect.reflected);

    const auto wrapped = wavelet::make_extended_index_i32<wavelet::BoundaryMode::kAntireflect>(10, source.size());
    std::vector<uint32_t> visited;
    wavelet::visit_extended_source_indices_i32<wavelet::BoundaryMode::kAntireflect>(
        wrapped, source.size(), [&](const uint32_t index) { visited.push_back(index); });
    EXPECT_EQ(visited, (std::vector<uint32_t>{2, 0, 3}));
}

TEST(WaveletPlanner, HaarSymmetricMatchesPyWaveletsAndRoundTripsEvenAndOddLengths) {
    constexpr float sqrt2 = std::numbers::sqrt2_v<float>;
    const std::array inputs = {
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F},
    };
    const std::array expected_approximation = {
        std::vector<float>{3.0F / sqrt2, 7.0F / sqrt2, 11.0F / sqrt2},
        std::vector<float>{3.0F / sqrt2, 7.0F / sqrt2, 10.0F / sqrt2},
    };
    const std::array expected_detail = {
        std::vector<float>{-1.0F / sqrt2, -1.0F / sqrt2, -1.0F / sqrt2},
        std::vector<float>{-1.0F / sqrt2, -1.0F / sqrt2, 0.0F},
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
    expect_values_near(approximation, {7.0F, 11.0F, 23.0F, 27.0F});
}

TEST(WaveletPlanner, MultiTapPredictAndUpdateWidenDependencyCones) {
    const auto forward = wavelet::make_forward_lifting_plan<MultiTapTestScheme>(
        wavelet::SignalBuffer{.length = 128}, wavelet::BoundaryMode::kSymmetric);
    const auto cone = wavelet::build_axis_cone(
        forward, wavelet::IndexInterval{.begin = 10, .end = 14}, wavelet::IndexInterval{.begin = 10, .end = 14});
    expect_interval_eq(cone.initial_even, {.begin = 10, .end = 18});
    expect_interval_eq(cone.initial_odd, {.begin = 11, .end = 17});
    ASSERT_EQ(cone.routes.size(), 4U);

    const auto& predict = cone.routes[0];
    EXPECT_EQ(predict.type, wavelet::StepType::kPredict);
    expect_interval_eq(predict.source, {.begin = 10, .end = 18});
    expect_interval_eq(predict.base, {.begin = 11, .end = 17});
    expect_interval_eq(predict.output, {.begin = 10, .end = 16});

    const auto& update = cone.routes[1];
    EXPECT_EQ(update.type, wavelet::StepType::kUpdate);
    expect_interval_eq(update.source, {.begin = 10, .end = 16});
    expect_interval_eq(update.base, {.begin = 11, .end = 15});
    expect_interval_eq(update.output, {.begin = 10, .end = 14});

    const wavelet::LiftingInversePlan inverse{
        .forward_trace = forward,
        .original_length = 128,
        .coefficient_length = forward.output_length,
    };
    const auto inverse_cone = wavelet::inverse_2d_detail::build_axis_cone(
        inverse, wavelet::IndexInterval{.begin = 10, .end = 18}, wavelet::IndexInterval{.begin = 11, .end = 17});
    expect_interval_eq(inverse_cone.initial_even, {.begin = 9, .end = 17});
    expect_interval_eq(inverse_cone.initial_odd, {.begin = 9, .end = 19});
    ASSERT_EQ(inverse_cone.routes.size(), 4U);
    const auto& inverse_update = inverse_cone.routes[2];
    EXPECT_EQ(inverse_update.type, wavelet::StepType::kUpdate);
    expect_interval_eq(inverse_update.source, {.begin = 9, .end = 19});
    expect_interval_eq(inverse_update.base, {.begin = 9, .end = 17});
    expect_interval_eq(inverse_update.output, {.begin = 10, .end = 18});
    const auto& inverse_predict = inverse_cone.routes[3];
    EXPECT_EQ(inverse_predict.type, wavelet::StepType::kPredict);
    expect_interval_eq(inverse_predict.source, {.begin = 10, .end = 18});
    expect_interval_eq(inverse_predict.base, {.begin = 10, .end = 16});
    expect_interval_eq(inverse_predict.output, {.begin = 11, .end = 17});
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

TEST(WaveletPlanner, OptimizedTwoDimensionalPlannerMatchesExhaustiveSearch) {
    constexpr std::array shapes = {
        std::pair<size_t, size_t>{5, 7},
        std::pair<size_t, size_t>{33, 35},
        std::pair<size_t, size_t>{64, 96},
        std::pair<size_t, size_t>{97, 65},
    };
    constexpr std::array<uint32_t, 2> core_limits = {1, 8};
    constexpr std::array<uint64_t, 2> l1_budgets = {512 * 1024, 768 * 1024};
    constexpr std::array route_domains = {
        wavelet::Lwt2DRouteDomainPolicy::kExact,
        wavelet::Lwt2DRouteDomainPolicy::kTileClosed,
    };
    constexpr std::array boundary_modes = {
        wavelet::BoundaryMode::kZero,
        wavelet::BoundaryMode::kConstant,
        wavelet::BoundaryMode::kSymmetric,
        wavelet::BoundaryMode::kReflect,
        wavelet::BoundaryMode::kPeriodic,
        wavelet::BoundaryMode::kSmooth,
        wavelet::BoundaryMode::kAntisymmetric,
        wavelet::BoundaryMode::kAntireflect,
    };
    for (const auto [height, width] : shapes) {
        for (const uint32_t core_limit : core_limits) {
            for (const uint64_t l1_budget : l1_budgets) {
                for (const auto boundary_mode : boundary_modes) {
                    for (const auto route_domain : route_domains) {
                        for (const bool latency_oriented : {false, true}) {
                            SCOPED_TRACE(
                                "forward " + std::to_string(height) + "x" + std::to_string(width) +
                                " cores=" + std::to_string(core_limit) + " budget=" + std::to_string(l1_budget));
                            expect_forward_matches_exhaustive<PlannerTestScheme>(
                                height, width, core_limit, l1_budget, latency_oriented, boundary_mode, route_domain);
                            expect_forward_matches_exhaustive<MultiTapTestScheme>(
                                height, width, core_limit, l1_budget, latency_oriented, boundary_mode, route_domain);
                        }
                    }
                    SCOPED_TRACE(
                        "inverse " + std::to_string(height) + "x" + std::to_string(width) +
                        " cores=" + std::to_string(core_limit) + " budget=" + std::to_string(l1_budget));
                    expect_inverse_matches_exhaustive<PlannerTestScheme>(
                        height, width, core_limit, l1_budget, boundary_mode, 0);
                    expect_inverse_matches_exhaustive<MultiTapTestScheme>(
                        height, width, core_limit, l1_budget, boundary_mode, 2'000);
                }
            }
        }
    }
}

TEST(WaveletPlanner, TwoDimensionalProtocolSerializationMatchesGoldenWords) {
    const wavelet::Lwt2DChunkPlan chunk{
        .final_band_rect = {.y = {.begin = 3, .end = 8}, .x = {.begin = 5, .end = 12}},
        .execution_band_rect = {.y = {.begin = 32, .end = 96}, .x = {.begin = 64, .end = 96}},
        .initial =
            {
                .ee = {.y = {.begin = 1, .end = 3}, .x = {.begin = 2, .end = 6}},
                .eo = {.y = {.begin = 4, .end = 7}, .x = {.begin = 8, .end = 13}},
                .oe = {.y = {.begin = 14, .end = 18}, .x = {.begin = 19, .end = 25}},
                .oo = {.y = {.begin = 26, .end = 31}, .x = {.begin = 32, .end = 39}},
            },
        .routes =
            {
                {
                    .axis = wavelet::Lwt2DAxis::kVertical,
                    .axis_route_index = 3,
                    .type = wavelet::StepType::kPredict,
                    .source_slot = wavelet::Lwt2DPlaneSlot::kP1,
                    .base_slot = wavelet::Lwt2DPlaneSlot::kP2,
                    .output_slot = wavelet::Lwt2DPlaneSlot::kScratch,
                    .source = {.y = {.begin = 2, .end = 7}, .x = {.begin = 11, .end = 17}},
                    .base = {.y = {.begin = 3, .end = 9}, .x = {.begin = 12, .end = 19}},
                    .output = {.y = {.begin = 4, .end = 11}, .x = {.begin = 13, .end = 21}},
                    .inline_terminal_scale = true,
                },
                {
                    .axis = wavelet::Lwt2DAxis::kHorizontal,
                    .axis_route_index = 4,
                    .type = wavelet::StepType::kScaleEven,
                    .source_slot = wavelet::Lwt2DPlaneSlot::kP3,
                    .base_slot = wavelet::Lwt2DPlaneSlot::kP3,
                    .output_slot = wavelet::Lwt2DPlaneSlot::kP0,
                    .source = {.y = {.begin = 5, .end = 13}, .x = {.begin = 14, .end = 23}},
                    .base = {.y = {.begin = 6, .end = 15}, .x = {.begin = 16, .end = 26}},
                    .output = {.y = {.begin = 7, .end = 17}, .x = {.begin = 18, .end = 29}},
                },
            },
        .final_bands =
            {
                .ll = wavelet::Lwt2DPlaneSlot::kP0,
                .lh = wavelet::Lwt2DPlaneSlot::kP1,
                .hl = wavelet::Lwt2DPlaneSlot::kP2,
                .hh = wavelet::Lwt2DPlaneSlot::kP3,
            },
        .final_band_sources =
            {
                .ll = {.y = {.begin = 1, .end = 3}, .x = {.begin = 4, .end = 7}},
                .lh = {.y = {.begin = 8, .end = 11}, .x = {.begin = 12, .end = 16}},
                .hl = {.y = {.begin = 17, .end = 21}, .x = {.begin = 22, .end = 27}},
                .hh = {.y = {.begin = 28, .end = 33}, .x = {.begin = 34, .end = 40}},
            },
    };
    wavelet::Lwt2DExecutionPlan forward;
    forward.y_plan.routes.resize(1);
    forward.chunks = {chunk};
    wavelet::Ilwt2DExecutionPlan inverse;
    inverse.y_plan.forward_trace.routes.resize(1);
    inverse.chunks = {chunk};

    const auto write_rectangle =
        [](std::vector<uint32_t>& words, const size_t offset, const wavelet::IndexRectangle& r) {
            words[offset] = static_cast<uint32_t>(r.y.begin);
            words[offset + 1] = static_cast<uint32_t>(r.height());
            words[offset + 2] = static_cast<uint32_t>(r.x.begin);
            words[offset + 3] = static_cast<uint32_t>(r.width());
        };
    std::vector<uint32_t> expected_chunk(32, 0);
    expected_chunk[0] = 3;
    expected_chunk[1] = 5;
    expected_chunk[2] = 5;
    expected_chunk[3] = 7;
    expected_chunk[4] = 1;
    expected_chunk[5] = 2;
    expected_chunk[6] = 2;
    expected_chunk[7] = 1;
    write_rectangle(expected_chunk, 8, chunk.initial.ee);
    write_rectangle(expected_chunk, 12, chunk.initial.eo);
    write_rectangle(expected_chunk, 16, chunk.initial.oe);
    write_rectangle(expected_chunk, 20, chunk.initial.oo);
    EXPECT_EQ(wavelet::build_lwt_2d_chunk_config_words(forward), expected_chunk);
    EXPECT_EQ(wavelet::build_ilwt_2d_chunk_config_words(inverse), expected_chunk);

    std::vector<uint32_t> expected_forward_routes(64, 0);
    expected_forward_routes[0] = 0;
    expected_forward_routes[1] = 0;
    expected_forward_routes[2] = 1;
    expected_forward_routes[3] = 2;
    expected_forward_routes[4] = 4;
    write_rectangle(expected_forward_routes, 5, chunk.routes[0].source);
    write_rectangle(expected_forward_routes, 9, chunk.routes[0].base);
    write_rectangle(expected_forward_routes, 13, chunk.routes[0].output);
    expected_forward_routes[17] = 4;
    expected_forward_routes[18] = 3;
    constexpr size_t second_route = 32;
    expected_forward_routes[second_route] = 1;
    expected_forward_routes[second_route + 1] = 2;
    expected_forward_routes[second_route + 2] = 3;
    expected_forward_routes[second_route + 3] = 3;
    expected_forward_routes[second_route + 4] = 0;
    write_rectangle(expected_forward_routes, second_route + 5, chunk.routes[1].source);
    write_rectangle(expected_forward_routes, second_route + 9, chunk.routes[1].base);
    write_rectangle(expected_forward_routes, second_route + 13, chunk.routes[1].output);
    expected_forward_routes[second_route + 17] = 2;
    expected_forward_routes[second_route + 18] = 4;
    EXPECT_EQ(wavelet::build_lwt_2d_route_config_words(forward), expected_forward_routes);
    auto expected_inverse_routes = expected_forward_routes;
    expected_inverse_routes[17] = 0;
    EXPECT_EQ(wavelet::build_ilwt_2d_route_config_words(inverse), expected_inverse_routes);

    std::vector<uint32_t> expected_bands(32, 0);
    expected_bands[0] = 3;
    expected_bands[1] = 5;
    expected_bands[2] = 5;
    expected_bands[3] = 7;
    const auto write_band = [&](const size_t offset, const uint32_t slot, const wavelet::IndexRectangle& source) {
        expected_bands[offset] = slot;
        write_rectangle(expected_bands, offset + 1, source);
    };
    write_band(4, 0, chunk.final_band_sources.ll);
    write_band(9, 1, chunk.final_band_sources.lh);
    write_band(14, 2, chunk.final_band_sources.hl);
    write_band(19, 3, chunk.final_band_sources.hh);
    EXPECT_EQ(wavelet::build_lwt_2d_band_config_words(forward), expected_bands);
    EXPECT_EQ(wavelet::build_ilwt_2d_band_config_words(inverse), expected_bands);
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
