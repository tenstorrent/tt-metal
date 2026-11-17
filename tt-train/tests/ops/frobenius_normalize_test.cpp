// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/frobenius_normalize/device/frobenius_normalize_device_operation.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

namespace {

struct FrobeniusCase {
    std::array<uint32_t, 4> shape;
    std::string name;
};

xt::xarray<float> frobenius_normalize_ref(const xt::xarray<float>& X, float eps) {
    const auto squares = X * X;
    const float sum_sq = xt::sum(squares)();
    const float norm = std::sqrt(sum_sq) + eps;
    return X / norm;
}

}  // namespace

class FrobeniusNormalizeTest : public ::testing::TestWithParam<FrobeniusCase> {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(FrobeniusNormalizeTest, MatchesCpuReference) {
    using namespace ttml;

    const auto& c = GetParam();
    constexpr float kEps = 1e-7f;

    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(c.name));
    const auto data = ttml::test_utils::make_uniform_xarray<float>(c.shape, -1.0f, 1.0f, seed);
    const auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());

    const auto bf16_data = core::to_xtensor(input_tensor);
    const auto expected = frobenius_normalize_ref(bf16_data, kEps);

    const auto result_tensor = metal::frobenius_normalize(input_tensor, kEps);
    const auto result = core::to_xtensor(result_tensor);

    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f));
}

static std::string CaseName(const ::testing::TestParamInfo<FrobeniusCase>& info) {
    return info.param.name;
}

static const FrobeniusCase kCases[] = {
    {{1, 1, 32, 32}, "SingleTile"},
    {{1, 1, 32, 64}, "TwoTiles"},
    {{1, 1, 32, 96}, "ThreeTiles"},
    {{1, 1, 64, 64}, "SmallMatrix"},
    {{1, 1, 256, 320}, "MediumMatrix"},
    {{1, 1, 2048, 5632}, "ProductionSize"},
};

INSTANTIATE_TEST_SUITE_P(All, FrobeniusNormalizeTest, ::testing::ValuesIn(kCases), CaseName);

// ============================================================================
// Program-cache hits
// ============================================================================
// A launch on new tensors with cached specs reuses the program, so the buffer addresses are set again on every hit.
// Epsilon reaches the kernels as a runtime arg; whether or not the program-cache key includes it, every launch must
// use its own epsilon. Each test starts from an empty program cache so that its first launch builds the program, and
// keeps every tensor of the earlier launches alive so that a later launch cannot land on a freed address: a hit must
// read and write the new buffers and leave the earlier tensors as they were.
// ============================================================================

class FrobeniusNormalizeCacheHitTest : public ttml::test_utils::ProgramCacheSuiteTest {};

namespace {

using ttml::test_utils::expect_unchanged;

using Shape = std::array<uint32_t, 4>;

// MatchesCpuReference's epsilon and tolerances. The tolerances apply in units of the largest expected entry (see
// expect_close).
constexpr float kCacheTestEps = 1e-7F;
constexpr float kCacheTestRtol = 1e-2F;
constexpr float kCacheTestAtol = 1e-2F;

// The factory splits the input's tiles over the cores. 2 * cores + 3 tiles put three tiles on three cores and two on
// every other core, so split_work_to_cores hands out two core groups (two non-origin compute kernels) on any grid.
Shape two_core_group_shape() {
    return {1U, 1U, 32U, ttml::test_utils::two_core_group_units(3U) * 32U};
}

// One tile runs on the origin core alone: no non-origin reader or compute core.
constexpr Shape kSingleTileShape = {1U, 1U, 32U, 32U};

// A device input and the values the kernels read from it: the host values rounded to bf16.
struct Input {
    ttnn::Tensor tensor;
    xt::xarray<float> values;
};

Input make_input(const Shape& shape, uint32_t seed) {
    const auto data = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed);
    auto tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &ttml::autograd::ctx().get_device());
    auto values = ttml::core::to_xtensor(tensor);
    return {std::move(tensor), std::move(values)};
}

float frobenius_norm(const xt::xarray<float>& x) {
    return std::sqrt(xt::sum(x * x)());
}

// The output has unit Frobenius norm, so its entries shrink as 1 / sqrt(numel). They are compared in units of the
// largest expected entry, or an all-zero output would pass the absolute tolerance on any large input.
void expect_close(const xt::xarray<float>& actual, const xt::xarray<float>& expected, const std::string& label) {
    ASSERT_EQ(actual.shape(), expected.shape()) << label;
    const float scale = xt::amax(xt::abs(expected))();
    EXPECT_TRUE(xt::allclose(actual / scale, expected / scale, kCacheTestRtol, kCacheTestAtol))
        << label << ": max abs diff " << xt::amax(xt::abs(actual - expected))() << ", largest expected entry " << scale;
}

// Launch A builds the program. Launch B, on a new input of the same shape while A's input and output stay alive, must
// reuse it, write its own output and leave A's tensors as they were.
void expect_cache_hit_with_fresh_tensors(const Shape& shape) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();

    const auto a = make_input(shape, 101U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_out = metal::frobenius_normalize(a.tensor, kCacheTestEps);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "launch A must build a program";
    const auto a_out_values = core::to_xtensor(a_out);
    expect_close(a_out_values, frobenius_normalize_ref(a.values, kCacheTestEps), "launch A");

    const auto b = make_input(shape, 102U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto b_out = metal::frobenius_normalize(b.tensor, kCacheTestEps);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    test_utils::expect_new_buffers({a.tensor, a_out}, {b.tensor, b_out}, "launch B");
    expect_close(core::to_xtensor(b_out), frobenius_normalize_ref(b.values, kCacheTestEps), "launch B");

    expect_unchanged(a.tensor, a.values, "launch A's input");
    expect_unchanged(a_out, a_out_values, "launch A's output");
}

// Every launch must use its own epsilon. Launch B's epsilon is the norm of its input, which halves the output, so a
// launch that kept launch A's epsilon would be off by a factor of two. Launch B hits launch A's program or builds its
// own, depending on whether the key includes epsilon. Launch C goes back to launch A's epsilon, so it repeats
// launch A's key under either key and must hit, after a launch that set another epsilon.
void expect_each_launch_uses_its_epsilon(const Shape& shape) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();

    const auto a = make_input(shape, 201U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_out = metal::frobenius_normalize(a.tensor, kCacheTestEps);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "launch A must build a program";
    const auto a_out_values = core::to_xtensor(a_out);
    expect_close(a_out_values, frobenius_normalize_ref(a.values, kCacheTestEps), "launch A");

    const auto b = make_input(shape, 202U);
    const float b_eps = frobenius_norm(b.values);
    const auto entries_before_b = device.num_program_cache_entries();
    const auto b_out = metal::frobenius_normalize(b.tensor, b_eps);
    EXPECT_LE(device.num_program_cache_entries(), entries_before_b + 1U) << "launch B builds at most one program";
    const auto b_out_values = core::to_xtensor(b_out);
    expect_close(b_out_values, frobenius_normalize_ref(b.values, b_eps), "launch B, epsilon = norm of B");

    const auto c = make_input(shape, 203U);
    const auto entries_before_c = device.num_program_cache_entries();
    const auto c_out = metal::frobenius_normalize(c.tensor, kCacheTestEps);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_c) << "launch C repeats launch A's key";
    expect_close(core::to_xtensor(c_out), frobenius_normalize_ref(c.values, kCacheTestEps), "launch C");

    expect_unchanged(a_out, a_out_values, "launch A's output");
    expect_unchanged(b_out, b_out_values, "launch B's output");
}

}  // namespace

TEST_F(FrobeniusNormalizeCacheHitTest, FreshTensorsTwoCoreGroups) {
    expect_cache_hit_with_fresh_tensors(two_core_group_shape());
}

TEST_F(FrobeniusNormalizeCacheHitTest, FreshTensorsSingleTile) {
    expect_cache_hit_with_fresh_tensors(kSingleTileShape);
}

TEST_F(FrobeniusNormalizeCacheHitTest, EpsilonChangeTwoCoreGroups) {
    expect_each_launch_uses_its_epsilon(two_core_group_shape());
}

TEST_F(FrobeniusNormalizeCacheHitTest, EpsilonChangeSingleTile) {
    expect_each_launch_uses_its_epsilon(kSingleTileShape);
}

// Launches alternate between an allocated and a preallocated output, in both orders, with every tensor kept alive.
// Whether or not the key tells the two apart, each launch must write the output it was given (or allocated) and
// nothing else, and launches 2 and 3 repeat launches 0 and 1, so they must hit.
TEST_F(FrobeniusNormalizeCacheHitTest, PreallocatedOutput) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();
    const auto shape = two_core_group_shape();

    struct Launch {
        Input input;
        ttnn::Tensor output;
        xt::xarray<float> output_values;
    };
    uint32_t seed = 300U;
    for (const bool first_preallocated : {false, true}) {
        SCOPED_TRACE(first_preallocated ? "preallocated output first" : "allocated output first");
        device.clear_program_cache();
        std::vector<Launch> launches;
        std::size_t entries_after_second = 0U;
        for (uint32_t i = 0; i < 4U; ++i) {
            const bool preallocated = (i % 2U == 0U) == first_preallocated;
            auto input = make_input(shape, ++seed);
            std::optional<ttnn::Tensor> preallocated_output;
            if (preallocated) {
                // A unit-norm output has no entry above 1, so a preallocated output the kernels skipped keeps the 7.
                preallocated_output = test_utils::filled_on_device({shape.begin(), shape.end()}, 7.0F);
            }
            const auto entries_before = device.num_program_cache_entries();
            auto output = ttnn::prim::ttml_frobenius_normalize(input.tensor, kCacheTestEps, preallocated_output)[0];
            const auto entries_after = device.num_program_cache_entries();
            if (i == 0U) {
                ASSERT_GT(entries_after, entries_before) << "launch 0 must build a program";
            } else if (i == 1U) {
                entries_after_second = entries_after;  // a hit or a second program, depending on the key
            } else {
                EXPECT_EQ(entries_after, entries_after_second) << "launch " << i << " repeats launch " << i - 2U;
            }
            if (preallocated) {
                EXPECT_EQ(output.buffer()->address(), preallocated_output->buffer()->address())
                    << "launch " << i << " did not return its preallocated output";
            }
            // Read the caller's tensor itself, not only the returned handle.
            auto output_values = core::to_xtensor(preallocated_output.value_or(output));
            expect_close(
                output_values, frobenius_normalize_ref(input.values, kCacheTestEps), "launch " + std::to_string(i));
            launches.push_back({std::move(input), std::move(output), std::move(output_values)});
        }
        for (std::size_t i = 0; i < launches.size(); ++i) {
            expect_unchanged(
                launches[i].input.tensor, launches[i].input.values, "input of launch " + std::to_string(i));
            expect_unchanged(launches[i].output, launches[i].output_values, "output of launch " + std::to_string(i));
        }
    }
}

// The public wrapper never normalizes in place, but the prim accepts the input as its preallocated output, and it is
// safe: each core re-reads only its own tiles, after the norm is known. The input then appears twice among the op's
// tensors, which the cache-hit path must handle as well. In place on A, then in place on B while A stays alive.
TEST_F(FrobeniusNormalizeCacheHitTest, InPlace) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();
    const auto shape = two_core_group_shape();

    const auto a = make_input(shape, 401U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_out = ttnn::prim::ttml_frobenius_normalize(a.tensor, kCacheTestEps, a.tensor)[0];
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "launch A must build a program";
    EXPECT_EQ(a_out.buffer()->address(), a.tensor.buffer()->address());
    const auto a_values = core::to_xtensor(a.tensor);
    expect_close(a_values, frobenius_normalize_ref(a.values, kCacheTestEps), "in place on A");

    const auto b = make_input(shape, 402U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto b_out = ttnn::prim::ttml_frobenius_normalize(b.tensor, kCacheTestEps, b.tensor)[0];
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    EXPECT_EQ(b_out.buffer()->address(), b.tensor.buffer()->address());
    EXPECT_NE(b.tensor.buffer()->address(), a.tensor.buffer()->address());
    expect_close(core::to_xtensor(b.tensor), frobenius_normalize_ref(b.values, kCacheTestEps), "in place on B");

    expect_unchanged(a.tensor, a_values, "launch A's tensor");
}

// A preallocated output that is a distinct tensor and one that is the input itself have the same spec, so launches
// that alternate between the two share one program whether or not the key includes that spec. In either order every
// hit must take its addresses from its own call: an in-place hit on a program built for a distinct output must write
// its input, and a distinct-output hit on a program built in place must write the preallocated output and leave its
// input as it was. Launch 2 repeats launch 0's form.
TEST_F(FrobeniusNormalizeCacheHitTest, InPlaceAndPreallocatedOutputAlternate) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();
    const auto shape = two_core_group_shape();

    struct Launch {
        Input input;
        bool in_place;
        ttnn::Tensor output;  // the input itself when in place
        xt::xarray<float> output_values;
    };
    uint32_t seed = 500U;
    for (const bool in_place_first : {false, true}) {
        SCOPED_TRACE(in_place_first ? "in place first" : "distinct output first");
        device.clear_program_cache();
        std::vector<Launch> launches;
        for (uint32_t i = 0; i < 3U; ++i) {
            const bool in_place = (i % 2U == 0U) == in_place_first;
            const std::string label = "launch " + std::to_string(i) + (in_place ? " (in place)" : " (distinct output)");
            auto input = make_input(shape, ++seed);
            // A unit-norm output has no entry above 1, so a preallocated output the kernels skipped keeps the 7.
            auto output = in_place ? input.tensor : test_utils::filled_on_device({shape.begin(), shape.end()}, 7.0F);
            const auto entries_before = device.num_program_cache_entries();
            const auto result = ttnn::prim::ttml_frobenius_normalize(input.tensor, kCacheTestEps, output)[0];
            const auto entries_after = device.num_program_cache_entries();
            if (i == 0U) {
                ASSERT_GT(entries_after, entries_before) << label << " must build a program";
            } else {
                EXPECT_EQ(entries_after, entries_before) << label << " must reuse launch 0's program";
            }
            EXPECT_EQ(result.buffer()->address(), output.buffer()->address()) << label << " did not return its output";
            auto output_values = core::to_xtensor(output);
            expect_close(output_values, frobenius_normalize_ref(input.values, kCacheTestEps), label);
            if (!in_place) {
                EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor(input.tensor), input.values)))
                    << label << " wrote into its input";
            }
            launches.push_back({std::move(input), in_place, std::move(output), std::move(output_values)});
        }
        for (std::size_t i = 0; i < launches.size(); ++i) {
            const auto& launch = launches[i];
            expect_unchanged(launch.output, launch.output_values, "output of launch " + std::to_string(i));
            if (!launch.in_place) {
                expect_unchanged(launch.input.tensor, launch.input.values, "input of launch " + std::to_string(i));
            }
        }
    }
}
