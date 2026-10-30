// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/layernorm_fw/device/layernorm_fw_device_operation.hpp"
#include "metal/ops/layernorm_fw/layernorm_fw.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

// Reference implementation using xtensor
std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_forward_reference_(
    const xt::xarray<float>& x,
    const xt::xarray<float>& gamma,
    const xt::xarray<float>& beta,
    uint32_t batch_size,
    uint32_t features,
    float eps = 1e-6f) {
    // Reshape input to (batch_size, features) for easier manipulation
    auto x_reshaped = xt::reshape_view(x, {batch_size, features});

    // Compute mean along features axis
    xt::xarray<float> mu = xt::mean(x_reshaped, {1});

    // Compute variance along features axis
    xt::xarray<float> x_centered = x_reshaped - xt::view(mu, xt::all(), xt::newaxis());
    xt::xarray<float> var = xt::mean(xt::square(x_centered), {1});

    // Compute reciprocal standard deviation (rstd)
    xt::xarray<float> rstd = 1.0f / xt::sqrt(var + eps);

    // Normalize
    xt::xarray<float> x_hat = x_centered * xt::view(rstd, xt::all(), xt::newaxis());

    // Scale and shift
    xt::xarray<float> y = x_hat * xt::view(gamma, xt::newaxis(), xt::all()) + xt::view(beta, xt::newaxis(), xt::all());

    // Flatten outputs back to 1D
    y = xt::flatten(y);

    return std::make_tuple(y, mu, rstd);
}

class LayerNormForwardOpTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// Helper function to compare metal kernel results against xtensor reference
static void CompareKernelVsXArray(
    uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 3) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Generate test data using xtensor
        uint32_t total_elements = batch_size * seq_len * heads * features;
        uint32_t combined_batch = batch_size * seq_len * heads;

        auto& rng = autograd::ctx().get_generator();
        uint32_t seed1 = rng();
        xt::xarray<float> x_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed1);

        uint32_t seed2 = rng();
        xt::xarray<float> gamma_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, 0.5F, 1.5F, seed2);

        uint32_t seed3 = rng();
        xt::xarray<float> beta_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, -0.1F, 0.1F, seed3);

        // Compute reference results
        auto [y_ref, mu_ref, rstd_ref] =
            layernorm_forward_reference_(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);

        // Copy and reshape data to 4D for device tensors (copy to avoid corrupting reference data)
        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> beta_4d = beta_data;
        beta_4d.reshape({1, 1, 1, features});

        // Create tensors on device using from_xtensor
        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto beta_tensor = core::from_xtensor(beta_4d, &autograd::ctx().get_device());

        // Run metal kernel
        auto output_tensors =
            metal::layernorm_fw(input_tensor, gamma_tensor, beta_tensor, 1e-6f, /* return_mean_rstd */ true);

        auto metal_y_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_mu_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_rstd_xtensor = core::to_xtensor(output_tensors[2].value());

        // Flatten metal results for comparison
        xt::xarray<float> metal_y_flat = xt::flatten(metal_y_xtensor);
        xt::xarray<float> metal_mu_flat = xt::flatten(metal_mu_xtensor);
        xt::xarray<float> metal_rstd_flat = xt::flatten(metal_rstd_xtensor);

        // Compare shapes
        ASSERT_EQ(y_ref.shape(), metal_y_flat.shape());
        ASSERT_EQ(mu_ref.shape(), metal_mu_flat.shape());
        ASSERT_EQ(rstd_ref.shape(), metal_rstd_flat.shape());

        // Compare values
        EXPECT_TRUE(xt::allclose(metal_y_flat, y_ref, 1.0e-3F, 5e-2F));
        EXPECT_TRUE(xt::allclose(metal_mu_flat, mu_ref, 1.0e-3F, 5e-2F));
        EXPECT_TRUE(xt::allclose(metal_rstd_flat, rstd_ref, 1.0e-3F, 5e-2F));
    }
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_OneTile) {
    CompareKernelVsXArray(1, 32, 1, 32);
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_OneIncompleteTile) {
    CompareKernelVsXArray(1, 12, 1, 19);
}

TEST_F(LayerNormForwardOpTest, NIGHTLY_MetalLayerNormFw_MediumTensorFitsInL1) {
    CompareKernelVsXArray(2, 182, 1, 2083);
}

TEST_F(LayerNormForwardOpTest, NIGHTLY_MetalLayerNormFw_LargeTensor_DoesNotFitInL1) {
    CompareKernelVsXArray(4, 324, 1, 9132);
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_HeadsDimNot1) {
    CompareKernelVsXArray(2, 8, 4, 512);
}

// ============================================================================
// Program cache hits: launches on fresh tensors of already cached specs, with every earlier launch alive
// ============================================================================

namespace {

constexpr float kEps = 1e-6F;
constexpr float kRtol = 1.0e-3F;
constexpr float kAtol = 5e-2F;
// Far outside any layernorm output, so a preallocated tensor the op did not write stands out.
constexpr float kPreallocatedSentinel = 1000.0F;

// Bit mask of the optional preallocated tensors a launch passes.
constexpr uint32_t kPreallocNone = 0U;
constexpr uint32_t kPreallocOutput = 1U;
constexpr uint32_t kPreallocMean = 2U;
constexpr uint32_t kPreallocRstd = 4U;
constexpr uint32_t kPreallocAll = kPreallocOutput | kPreallocMean | kPreallocRstd;

struct LayerNormFwShape {
    uint32_t batch = 1U;
    uint32_t heads = 1U;
    uint32_t seq_len = 1U;
    uint32_t features = 1U;
};

// 6 tile-rows on 6 cores (one work group); 72 features are three tiles with a masked tail.
constexpr LayerNormFwShape kSixTileRows{.batch = 3U, .heads = 1U, .seq_len = 40U, .features = 72U};

// The factory splits tile-rows over the grid. 2 * num_cores + 3 of them gives three cores 3 rows (core_group_1)
// and every other core 2 rows (core_group_2) on any grid, so both compute kernels run. One 8-row sequence per
// tile-row keeps the host work small; 72 features are three tiles with a masked tail.
LayerNormFwShape two_core_group_shape() {
    return {.batch = ttml::test_utils::two_core_group_units(3U), .heads = 1U, .seq_len = 8U, .features = 72U};
}

// One launch: its tensors (alive as long as the struct), what the op returned and the host reference.
struct LayerNormFwLaunch {
    float epsilon = kEps;
    bool return_mean_rstd = true;
    ttnn::Tensor input;
    ttnn::Tensor gamma;
    ttnn::Tensor beta;
    std::optional<ttnn::Tensor> preallocated_output;
    std::optional<ttnn::Tensor> preallocated_mean;
    std::optional<ttnn::Tensor> preallocated_rstd;
    std::vector<std::optional<ttnn::Tensor>> outputs;  // [output, mean, rstd]
    xt::xarray<float> output_ref;
    xt::xarray<float> mean_ref;
    xt::xarray<float> rstd_ref;
};

LayerNormFwLaunch make_layernorm_fw_launch(
    const LayerNormFwShape& shape,
    const uint32_t seed,
    const float epsilon,
    const bool return_mean_rstd,
    const uint32_t preallocated = kPreallocNone) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const uint32_t rows = shape.batch * shape.heads * shape.seq_len;

    xt::xarray<float> x = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 1>{static_cast<std::size_t>(rows) * shape.features}, -1.0F, 1.0F, seed);
    xt::xarray<float> gamma =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{shape.features}, 0.5F, 1.5F, seed + 1U);
    xt::xarray<float> beta =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{shape.features}, -0.1F, 0.1F, seed + 2U);

    LayerNormFwLaunch launch;
    launch.epsilon = epsilon;
    launch.return_mean_rstd = return_mean_rstd;
    std::tie(launch.output_ref, launch.mean_ref, launch.rstd_ref) =
        layernorm_forward_reference_(x, gamma, beta, rows, shape.features, epsilon);

    x.reshape({shape.batch, shape.heads, shape.seq_len, shape.features});
    gamma.reshape({1U, 1U, 1U, shape.features});
    beta.reshape({1U, 1U, 1U, shape.features});
    launch.input = core::from_xtensor(x, device);
    launch.gamma = core::from_xtensor(gamma, device);
    launch.beta = core::from_xtensor(beta, device);

    const ttnn::Shape stat_shape({shape.batch, shape.heads, shape.seq_len, 1U});
    if ((preallocated & kPreallocOutput) != 0U) {
        launch.preallocated_output = core::full(launch.input.logical_shape(), kPreallocatedSentinel, device);
    }
    if ((preallocated & kPreallocMean) != 0U) {
        launch.preallocated_mean = core::full(stat_shape, kPreallocatedSentinel, device);
    }
    if ((preallocated & kPreallocRstd) != 0U) {
        launch.preallocated_rstd = core::full(stat_shape, kPreallocatedSentinel, device);
    }
    return launch;
}

void run_layernorm_fw(LayerNormFwLaunch& launch) {
    launch.outputs = ttnn::prim::ttml_layernorm_fw(
        launch.input,
        launch.gamma,
        launch.beta,
        launch.epsilon,
        launch.return_mean_rstd,
        launch.preallocated_output,
        launch.preallocated_mean,
        launch.preallocated_rstd);
}

void expect_close(const ttnn::Tensor& tensor, const xt::xarray<float>& expected, const std::string& name) {
    const xt::xarray<float> actual_nd = ttml::core::to_xtensor(tensor);
    const xt::xarray<float> actual = xt::flatten(actual_nd);
    ASSERT_EQ(actual.shape(), expected.shape()) << name;
    EXPECT_TRUE(xt::allclose(actual, expected, kRtol, kAtol))
        << name << " max_abs_diff=" << xt::amax(xt::abs(actual - expected))();
}

void expect_untouched(const ttnn::Tensor& tensor, const std::string& name) {
    const xt::xarray<float> values = ttml::core::to_xtensor(tensor);
    EXPECT_TRUE(xt::all(xt::equal(values, kPreallocatedSentinel))) << name << " was written";
}

void expect_layernorm_fw_launch_matches(const LayerNormFwLaunch& launch) {
    ASSERT_EQ(launch.outputs.size(), 3U);
    ASSERT_TRUE(launch.outputs[0].has_value());
    expect_close(*launch.outputs[0], launch.output_ref, "output");
    if (launch.preallocated_output.has_value()) {
        EXPECT_EQ(launch.outputs[0]->buffer(), launch.preallocated_output->buffer())
            << "output is not the preallocated tensor";
    }
    if (!launch.return_mean_rstd) {
        EXPECT_FALSE(launch.outputs[1].has_value());
        EXPECT_FALSE(launch.outputs[2].has_value());
        // Without return_mean_rstd the op ignores preallocated mean/rstd and must leave them alone.
        if (launch.preallocated_mean.has_value()) {
            expect_untouched(*launch.preallocated_mean, "ignored preallocated mean");
        }
        if (launch.preallocated_rstd.has_value()) {
            expect_untouched(*launch.preallocated_rstd, "ignored preallocated rstd");
        }
        return;
    }
    ASSERT_TRUE(launch.outputs[1].has_value());
    ASSERT_TRUE(launch.outputs[2].has_value());
    expect_close(*launch.outputs[1], launch.mean_ref, "mean");
    expect_close(*launch.outputs[2], launch.rstd_ref, "rstd");
    if (launch.preallocated_mean.has_value()) {
        EXPECT_EQ(launch.outputs[1]->buffer(), launch.preallocated_mean->buffer())
            << "mean is not the preallocated tensor";
    }
    if (launch.preallocated_rstd.has_value()) {
        EXPECT_EQ(launch.outputs[2]->buffer(), launch.preallocated_rstd->buffer())
            << "rstd is not the preallocated tensor";
    }
}

// Proves a hit on `b` ran on new addresses: `a` is alive, so none of b's buffers can reuse a's.
void expect_distinct_buffers(const LayerNormFwLaunch& a, const LayerNormFwLaunch& b) {
    EXPECT_NE(a.input.buffer()->address(), b.input.buffer()->address()) << "input";
    EXPECT_NE(a.gamma.buffer()->address(), b.gamma.buffer()->address()) << "gamma";
    EXPECT_NE(a.beta.buffer()->address(), b.beta.buffer()->address()) << "beta";
    for (size_t i = 0; i < std::min(a.outputs.size(), b.outputs.size()); ++i) {
        if (a.outputs[i].has_value() && b.outputs[i].has_value()) {
            EXPECT_NE(a.outputs[i]->buffer()->address(), b.outputs[i]->buffer()->address()) << "output " << i;
        }
    }
}

std::string describe_preallocated(const uint32_t preallocated) {
    if (preallocated == kPreallocNone) {
        return "nothing preallocated";
    }
    std::string description = "preallocated";
    description += (preallocated & kPreallocOutput) != 0U ? " output" : "";
    description += (preallocated & kPreallocMean) != 0U ? " mean" : "";
    description += (preallocated & kPreallocRstd) != 0U ? " rstd" : "";
    return description;
}

// Launches once per entry of `preallocation_patterns` and then once more per entry, on fresh tensors of one shape
// with new values each time, from an empty cache and with every launch kept alive. If the hash ignores which
// preallocated tensors are passed, every launch after the first hits the first launch's program; if it keys them, the
// second pass hits the program each pattern built in the first. Either way the second pass adds no cache entry, and
// every launch must return its own results and leave the other launches' tensors alone. A hit that keeps an earlier
// launch's address, or patches addresses by position in a tensor list whose layout changed, fails that.
void run_layernorm_fw_preallocation_sequence(
    const LayerNormFwShape& shape,
    const bool return_mean_rstd,
    const std::vector<uint32_t>& preallocation_patterns,
    const uint32_t seed) {
    auto& device = ttml::test_utils::device_with_empty_program_cache();
    const size_t num_patterns = preallocation_patterns.size();
    std::vector<LayerNormFwLaunch> launches;
    launches.reserve(2U * num_patterns);
    for (size_t i = 0; i < 2U * num_patterns; ++i) {
        const uint32_t preallocated = preallocation_patterns[i % num_patterns];
        SCOPED_TRACE("launch " + std::to_string(i) + ", " + describe_preallocated(preallocated));
        launches.push_back(make_layernorm_fw_launch(
            shape, seed + 10U * static_cast<uint32_t>(i), kEps, return_mean_rstd, preallocated));
        const auto entries_before = device.num_program_cache_entries();
        run_layernorm_fw(launches[i]);
        if (i == 0U) {
            ASSERT_GT(device.num_program_cache_entries(), entries_before)
                << "the first launch did not populate the cache";
        } else if (i >= num_patterns) {
            EXPECT_EQ(device.num_program_cache_entries(), entries_before) << "a repeated pattern was not a cache hit";
            expect_distinct_buffers(launches[i - num_patterns], launches[i]);
        }
        expect_layernorm_fw_launch_matches(launches[i]);
    }
    for (size_t i = 0; i < launches.size(); ++i) {
        SCOPED_TRACE("launch " + std::to_string(i) + " re-checked after the last launch");
        expect_layernorm_fw_launch_matches(launches[i]);
    }
}

struct LayerNormFwAttributes {
    float epsilon = kEps;
    bool return_mean_rstd = true;
};

// Launches with `cached`, then with `changed`, then with `cached` again, on fresh tensors from an empty cache and with
// every launch kept alive. Whether `changed` makes a new cache entry or is re-applied on a hit, each launch must run
// with its own attributes; the last launch must reuse the first one's program.
void run_layernorm_fw_attribute_change(
    const LayerNormFwAttributes& cached, const LayerNormFwAttributes& changed, const uint32_t seed) {
    auto& device = ttml::test_utils::device_with_empty_program_cache();
    const std::array<LayerNormFwAttributes, 3> attributes{cached, changed, cached};
    std::vector<LayerNormFwLaunch> launches;
    launches.reserve(attributes.size());
    for (size_t i = 0; i < attributes.size(); ++i) {
        SCOPED_TRACE(
            "launch " + std::to_string(i) + ", epsilon " + std::to_string(attributes[i].epsilon) +
            ", return_mean_rstd " + std::to_string(attributes[i].return_mean_rstd));
        launches.push_back(make_layernorm_fw_launch(
            kSixTileRows,
            seed + 10U * static_cast<uint32_t>(i),
            attributes[i].epsilon,
            attributes[i].return_mean_rstd));
        const auto entries_before = device.num_program_cache_entries();
        run_layernorm_fw(launches[i]);
        if (i + 1U == attributes.size()) {
            EXPECT_EQ(device.num_program_cache_entries(), entries_before)
                << "the first launch's attributes did not hit its cached program";
        }
        expect_layernorm_fw_launch_matches(launches[i]);
    }
    for (size_t i = 0; i < launches.size(); ++i) {
        SCOPED_TRACE("launch " + std::to_string(i) + " re-checked after the last launch");
        expect_layernorm_fw_launch_matches(launches[i]);
    }
}

}  // namespace

// On cores of both work groups, without and with all three outputs preallocated: a hit that keeps any reader or
// writer address of an earlier launch returns that launch's results or writes into its outputs.
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_CacheHit_FreshLiveTensors_TwoCoreGroups) {
    run_layernorm_fw_preallocation_sequence(
        two_core_group_shape(), /*return_mean_rstd=*/true, {kPreallocNone, kPreallocAll}, /*seed=*/11U);
}

// Without mean/rstd the writer gets no mean/rstd address and the program is built without RETURN_MEAN_RSTD;
// preallocated mean/rstd are then ignored but still passed.
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_CacheHit_FreshLiveTensors_TwoCoreGroups_NoMeanRstd) {
    run_layernorm_fw_preallocation_sequence(
        two_core_group_shape(), /*return_mean_rstd=*/false, {kPreallocNone, kPreallocAll}, /*seed=*/11U);
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_CacheHit_PreallocatedOutputsAddedAndRemoved) {
    // Cached without preallocated tensors, then launches that add some or all of them and drop them again.
    run_layernorm_fw_preallocation_sequence(
        kSixTileRows,
        /*return_mean_rstd=*/true,
        {kPreallocNone, kPreallocOutput, kPreallocMean | kPreallocRstd, kPreallocRstd, kPreallocAll},
        /*seed=*/101U);
    // Cached with a preallocated output, then launches without it and with other subsets.
    run_layernorm_fw_preallocation_sequence(
        kSixTileRows,
        /*return_mean_rstd=*/true,
        {kPreallocOutput, kPreallocNone, kPreallocMean, kPreallocAll},
        /*seed=*/201U);
}

// With return_mean_rstd=false the op ignores preallocated mean/rstd and must not write them, but they are still in its
// tensor list: dropping or adding them moves the output's position in that list.
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_CacheHit_NoMeanRstd_IgnoredPreallocatedMeanRstd) {
    run_layernorm_fw_preallocation_sequence(
        kSixTileRows,
        /*return_mean_rstd=*/false,
        {kPreallocMean | kPreallocRstd, kPreallocNone, kPreallocOutput | kPreallocMean, kPreallocOutput},
        /*seed=*/301U);
}

// epsilon reaches the reader as a compile-time arg.
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_EpsilonChangeOnCachedShape) {
    // 0.125 is exact in bf16 and moves rstd by ~15% for inputs in [-1, 1].
    run_layernorm_fw_attribute_change({.epsilon = kEps}, {.epsilon = 0.125F}, /*seed=*/401U);
}

// return_mean_rstd decides whether the kernels are built with RETURN_MEAN_RSTD and whether mean/rstd are returned.
TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_ReturnMeanRstdChangeOnCachedShape) {
    run_layernorm_fw_attribute_change({.return_mean_rstd = true}, {.return_mean_rstd = false}, /*seed=*/451U);
}
