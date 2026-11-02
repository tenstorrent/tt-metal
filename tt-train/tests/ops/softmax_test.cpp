// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/softmax/device/softmax_device_operation.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

using shape_type = std::array<std::size_t, 4>;

class SoftmaxTest : public ttml::test_utils::DeviceSuiteTest {
protected:
    // Reseed per test so each test's data does not depend on which tests ran before it.
    void SetUp() override {
        ttml::autograd::ctx().set_seed(42);
    }
};

xt::xarray<float> xt_softmax(const xt::xarray<float>& input, uint32_t dim = 3U) {
    xt::xarray<float> max_value = xt::amax(input, dim, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_value;  // for numerical stability
    xt::xarray<float> exp_shifted_input = xt::exp(shifted_input);
    xt::xarray<float> exp_sum = xt::sum(exp_shifted_input, dim, xt::keep_dims);
    xt::xarray<float> result = exp_shifted_input / exp_sum;
    return result;
}

namespace {

constexpr int32_t kLastDim = 3;
constexpr float kRtol = 3e-2F;
constexpr float kAtol = 1e-2F;

// A fresh uniform [-10, 10] input. `values` is what landed on device, so references computed from it are not charged
// for the bf16 rounding of the input.
struct SoftmaxInput {
    ttnn::Tensor tensor;
    xt::xarray<float> values;
};

SoftmaxInput make_softmax_input(const shape_type& shape) {
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    const xt::xarray<float> data = ttml::test_utils::make_uniform_xarray<float>(shape, -10.0F, 10.0F, seed);
    auto tensor = ttml::core::from_xtensor(data, &ttml::autograd::ctx().get_device());
    auto values = ttml::core::to_xtensor(tensor);
    return {std::move(tensor), std::move(values)};
}

// Launches softmax on `shape` twice from an empty program cache. The first launch builds the program. The second runs
// on fresh tensors while every tensor of the first stays alive, so all of its buffers are new: it must reuse the
// program, compute its own result and leave the first result untouched.
void expect_cache_hit_with_fresh_tensors(const shape_type& shape) {
    using namespace ttml;
    auto& device = test_utils::device_with_empty_program_cache();

    const auto first = make_softmax_input(shape);
    const auto entries_before = device.num_program_cache_entries();
    const auto first_result = metal::softmax(first.tensor, kLastDim);
    const auto entries_after_miss = device.num_program_cache_entries();
    ASSERT_GT(entries_after_miss, entries_before) << "the first launch did not build a program";
    const auto first_expected = xt_softmax(first.values);
    const auto first_result_values = core::to_xtensor(first_result);
    EXPECT_TRUE(xt::allclose(first_result_values, first_expected, kRtol, kAtol));

    const auto second = make_softmax_input(shape);
    const auto second_result = metal::softmax(second.tensor, kLastDim);
    EXPECT_EQ(device.num_program_cache_entries(), entries_after_miss) << "the second launch did not hit";
    test_utils::expect_new_buffers({first.tensor, first_result}, {second.tensor, second_result}, "the second launch");
    const auto second_expected = xt_softmax(second.values);
    ASSERT_FALSE(xt::allclose(second_expected, first_expected, kRtol, kAtol)) << "the launches must be told apart";
    EXPECT_TRUE(xt::allclose(core::to_xtensor(second_result), second_expected, kRtol, kAtol));
    EXPECT_TRUE(core::to_xtensor(first_result) == first_result_values) << "the hit wrote into the first result";
}

}  // namespace

// Disabled: flaky — https://github.com/tenstorrent/tt-metal/issues/46422
TEST_F(SoftmaxTest, DISABLED_SoftmaxTest_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 59U, W = 197U;
    const auto shape = shape_type{N, C, H, W};
    int32_t dim = 3U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float, shape_type>(shape, -10.0F, 10.0F, seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    ttnn::Tensor ttml_softmax = ttml::metal::softmax(input, dim);
    auto ttml_softmax_xtensor = core::to_xtensor(ttml_softmax);

    ttnn::Tensor ttnn_softmax = ttnn_fixed::softmax(input, dim);
    auto ttnn_softmax_xtensor = core::to_xtensor(ttnn_softmax);

    // Host side reference using FP32 and xtensor
    auto expected_result = xt_softmax(input_tensor, dim);

    ASSERT_EQ(ttml_softmax_xtensor.shape(), expected_result.shape());

    // ttml vs host
    EXPECT_TRUE(xt::allclose(ttml_softmax_xtensor, expected_result, 3e-2F, 1e-2F));

    // ttnn vs host
    EXPECT_TRUE(xt::allclose(ttnn_softmax_xtensor, expected_result, 3e-2F, 1e-2F));

    // ttml vs ttnn
    EXPECT_TRUE(xt::allclose(ttml_softmax_xtensor, ttnn_softmax_xtensor, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, SoftmaxTest_Big_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 32U, W = 128007U;
    const auto shape = shape_type{N, C, H, W};
    int32_t dim = 3U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor = ttml::test_utils::make_uniform_xarray<float>(shape, -10.0F, 10.0F, seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto result = ttml::metal::softmax(input, dim);

    auto expected_result = xt_softmax(input_tensor, dim);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    ASSERT_EQ(result_xtensor.shape(), expected_result.shape());
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, NIGHTLY_SoftmaxTest_Huge_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    const auto shape = shape_type{N, C, H, W};
    int32_t dim = 3U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float, shape_type>(shape, -10.0F, 10.0F, seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto result = ttml::metal::softmax(input, dim);

    auto expected_result = xt_softmax(input_tensor, dim);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    ASSERT_EQ(result_xtensor.shape(), expected_result.shape());
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, SoftmaxTest_Large_Values) {
    using namespace ttml;

    int32_t dim = 3U;

    xt::xarray<float> input_tensor = {
        {{{5.36871e+08,  -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08}}}};

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    ttnn::Tensor ttml_softmax = ttml::metal::softmax(input, dim);
    auto ttml_softmax_xtensor = core::to_xtensor(ttml_softmax);

    ttnn::Tensor ttnn_softmax = ttnn_fixed::softmax(input, dim);
    auto ttnn_softmax_xtensor = core::to_xtensor(ttnn_softmax);

    // Host side reference using FP32 and xtensor
    auto expected_result = xt_softmax(input_tensor, dim);

    ASSERT_EQ(ttml_softmax_xtensor.shape(), expected_result.shape());

    // ttml vs host
    EXPECT_TRUE(xt::allclose(ttml_softmax_xtensor, expected_result, 3e-2F, 1e-2F));

    // ttnn vs host
    EXPECT_TRUE(xt::allclose(ttnn_softmax_xtensor, expected_result, 3e-2F, 1e-2F));

    // ttml vs ttnn
    EXPECT_TRUE(xt::allclose(ttml_softmax_xtensor, ttnn_softmax_xtensor, 3e-2F, 1e-2F));
}

// One tile per row, so the kernel's row max and exp sum never combine two tiles: many multi-tile rows over many cores
// is the shape DISABLED_SoftmaxTest_Batch flaked on (#46422, analysis in #53208).
TEST_F(SoftmaxTest, SoftmaxTest_CacheHit_FreshTensors_TwoCoreGroups) {
    expect_cache_hit_with_fresh_tensors(shape_type{1U, 1U, ttml::test_utils::two_core_group_units(3U) * 32U, 32U});
}

TEST_F(SoftmaxTest, SoftmaxTest_CacheHit_FreshTensors_SingleCore) {
    // One tile-row runs on a single core and leaves the second core group empty. W = 50 is not tile-aligned, so the
    // masked path runs too.
    expect_cache_hit_with_fresh_tensors(shape_type{1U, 1U, 32U, 50U});
}

// Whether or not the program-cache key records a preallocated output, each launch must write the output it was given
// (or allocated) and nothing else. Both orders run with every tensor kept alive; the third and fourth launches repeat
// the first two, so they must hit. One tile-row is enough: which output a launch writes does not depend on the core
// count, and TwoCoreGroups covers every core.
TEST_F(SoftmaxTest, SoftmaxTest_CacheHit_PreallocatedOutput) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();
    const auto shape = shape_type{1U, 1U, 32U, 64U};

    struct Launch {
        SoftmaxInput input;
        ttnn::Tensor result;
        xt::xarray<float> result_values;
    };
    for (const bool first_preallocated : {false, true}) {
        SCOPED_TRACE(first_preallocated ? "preallocated output first" : "allocated output first");
        device.clear_program_cache();
        std::vector<Launch> launches;
        std::size_t entries_after_second = 0U;
        for (uint32_t i = 0; i < 4U; ++i) {
            const bool preallocated = (i % 2U == 0U) == first_preallocated;
            auto input = make_softmax_input(shape);
            std::optional<ttnn::Tensor> output;
            if (preallocated) {
                // softmax never yields 7, so a preallocated output the kernels skipped keeps it.
                output = test_utils::filled_on_device({shape.begin(), shape.end()}, 7.0F);
            }
            const auto entries_before = device.num_program_cache_entries();
            auto result = ttnn::prim::ttml_softmax(input.tensor, kLastDim, output);
            const auto entries_after = device.num_program_cache_entries();
            if (i == 0U) {
                ASSERT_GT(entries_after, entries_before) << "the first launch did not build a program";
            } else if (i == 1U) {
                entries_after_second = entries_after;  // a hit or a second program, depending on the key
            } else {
                EXPECT_EQ(entries_after, entries_after_second) << "launch " << i << " repeats launch " << i - 2U;
            }
            if (preallocated) {
                EXPECT_EQ(result.buffer()->address(), output->buffer()->address())
                    << "launch " << i << " did not write into its preallocated output";
            }
            // Read the caller's tensor itself, not only the returned handle.
            auto result_values = core::to_xtensor(output.value_or(result));
            EXPECT_TRUE(xt::allclose(result_values, xt_softmax(input.values), kRtol, kAtol)) << "launch " << i;
            launches.push_back({std::move(input), std::move(result), std::move(result_values)});
        }
        for (std::size_t i = 0; i < launches.size(); ++i) {
            EXPECT_TRUE(core::to_xtensor(launches[i].result) == launches[i].result_values)
                << "a later launch wrote into the result of launch " << i;
        }
    }
}
