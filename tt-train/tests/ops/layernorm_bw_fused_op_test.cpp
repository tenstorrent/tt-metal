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
#include "metal/ops/layernorm_bw/device/layernorm_bw_device_operation.hpp"
#include "metal/ops/layernorm_bw/layernorm_bw.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

// Reference implementation using xtensor
struct LayerNormCache {
    xt::xarray<float> x;
    xt::xarray<float> x_hat;
    xt::xarray<float> mu;
    xt::xarray<float> s;
    xt::xarray<float> gamma;
    xt::xarray<float> beta;
    uint32_t batch_size;
    uint32_t features;
    float eps;
};

std::tuple<xt::xarray<float>, LayerNormCache> layernorm_forward_reference(
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

    // Compute standard deviation with epsilon
    xt::xarray<float> s = xt::sqrt(var + eps);

    // Normalize
    xt::xarray<float> x_hat = x_centered / xt::view(s, xt::all(), xt::newaxis());

    // Scale and shift
    xt::xarray<float> y = x_hat * xt::view(gamma, xt::newaxis(), xt::all()) + xt::view(beta, xt::newaxis(), xt::all());

    // Flatten outputs back to 1D
    y = xt::flatten(y);
    x_hat = xt::flatten(x_hat);

    LayerNormCache cache{x, x_hat, mu, s, gamma, beta, batch_size, features, eps};
    return std::make_tuple(y, cache);
}

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_backward_reference(
    const xt::xarray<float>& dy, const LayerNormCache& cache) {
    // Reshape to (batch_size, features)
    auto dy_reshaped = xt::reshape_view(dy, {cache.batch_size, cache.features});
    auto x_hat_reshaped = xt::reshape_view(cache.x_hat, {cache.batch_size, cache.features});

    // Compute dgamma and dbeta - sum over batch dimension
    xt::xarray<float> dgamma = xt::sum(dy_reshaped * x_hat_reshaped, {0});
    xt::xarray<float> dbeta = xt::sum(dy_reshaped, {0});

    // Compute dxhat = dy * gamma
    xt::xarray<float> dxhat = dy_reshaped * xt::view(cache.gamma, xt::newaxis(), xt::all());

    // Compute mean_dxhat - mean along features axis
    xt::xarray<float> mean_dxhat = xt::mean(dxhat, {1});

    // Compute mean_dxhat_xhat - mean along features axis
    xt::xarray<float> mean_dxhat_xhat = xt::mean(dxhat * x_hat_reshaped, {1});

    // Compute final dx
    xt::xarray<float> dx = (dxhat - xt::view(mean_dxhat, xt::all(), xt::newaxis()) -
                            x_hat_reshaped * xt::view(mean_dxhat_xhat, xt::all(), xt::newaxis())) /
                           xt::view(cache.s, xt::all(), xt::newaxis());

    // Flatten dx back to 1D
    dx = xt::flatten(dx);

    return std::make_tuple(dx, dgamma, dbeta);
}

class LayerNormBackwardOpTest : public ::testing::Test {
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
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 3,
    // dx comes straight from the kernel (<=0.014 max abs error on device), while
    // dgamma/dbeta are host-side sums of bf16 per-row components whose noise grows
    // with rows*features (~0.29 at features~8k on device) - graded separately.
    const float dx_atol = 2e-2F,
    const float dgb_atol = 5e-2F) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Generate test data using flattened 1D arrays
        uint32_t total_elements = batch_size * seq_len * heads * features;
        uint32_t combined_batch = batch_size * seq_len * heads;

        auto& rng = autograd::ctx().get_generator();
        uint32_t seed1 = rng();
        xt::xarray<float> x_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed1);

        uint32_t seed2 = rng();
        xt::xarray<float> gamma_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, 0.0F, 1.0F, seed2);

        uint32_t seed3 = rng();
        xt::xarray<float> beta_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, 0.0F, 1.0F, seed3);

        uint32_t seed4 = rng();
        xt::xarray<float> dy_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed4);

        // Compute reference results
        auto [y_ref, cache] =
            layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        // Copy and reshape data to 4D for device tensors (copy to avoid corrupting reference data)
        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> dy_4d = dy_data;
        dy_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> mu_4d = cache.mu;
        mu_4d.reshape({batch_size, heads, seq_len, 1});

        // Compute rstd from s and reshape
        xt::xarray<float> rstd_data = 1.0f / cache.s;
        rstd_data.reshape({batch_size, heads, seq_len, 1});

        // Create tensors on device using from_xtensor
        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto mean_tensor = core::from_xtensor(mu_4d, &autograd::ctx().get_device());
        auto rstd_tensor = core::from_xtensor(rstd_data, &autograd::ctx().get_device());
        auto dy_tensor = core::from_xtensor(dy_4d, &autograd::ctx().get_device());

        auto output_tensors = metal::layernorm_bw(input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_dgamma_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_dbeta_xtensor = core::to_xtensor(output_tensors[2].value());

        // Flatten metal results for comparison
        xt::xarray<float> metal_dx_flat = xt::flatten(metal_dx_xtensor);
        xt::xarray<float> metal_dgamma_flat = xt::flatten(metal_dgamma_xtensor);
        xt::xarray<float> metal_dbeta_flat = xt::flatten(metal_dbeta_xtensor);

        // Compare shapes
        ASSERT_EQ(dx_ref.shape(), metal_dx_flat.shape());
        ASSERT_EQ(dgamma_ref.shape(), metal_dgamma_flat.shape());
        ASSERT_EQ(dbeta_ref.shape(), metal_dbeta_flat.shape());

        // Compare values
        EXPECT_TRUE(xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, dx_atol))
            << "dx max_abs_diff=" << xt::amax(xt::abs(metal_dx_flat - dx_ref))();
        EXPECT_TRUE(xt::allclose(metal_dgamma_flat, dgamma_ref, 1.0e-3F, dgb_atol))
            << "dgamma max_abs_diff=" << xt::amax(xt::abs(metal_dgamma_flat - dgamma_ref))();
        EXPECT_TRUE(xt::allclose(metal_dbeta_flat, dbeta_ref, 1.0e-3F, dgb_atol))
            << "dbeta max_abs_diff=" << xt::amax(xt::abs(metal_dbeta_flat - dbeta_ref))();
    }
}

// ============================================================================
// Test Cases - LayerNorm Backward Metal Kernel vs XArray Reference
// ============================================================================

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_OneTile) {
    CompareKernelVsXArray(1, 13, 1, 20);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_TwoIncompleteTiles) {
    CompareKernelVsXArray(1, 32, 1, 33);
}

TEST_F(LayerNormBackwardOpTest, NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit) {
    CompareKernelVsXArray(3, 273, 1, 8462, 3, /*dx_atol=*/2e-2F, /*dgb_atol=*/4e-1F);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4) {
    CompareKernelVsXArray(3, 100, 1, 8191, 10, /*dx_atol=*/2e-2F, /*dgb_atol=*/4e-1F);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_OneTilePerRow) {
    CompareKernelVsXArray(1, 19, 1, 213, 10);
}

// ============================================================================
// Program cache hits: launches on fresh tensors of already cached specs, with every earlier launch alive.
// These call the device op directly and check the per-element dgamma/dbeta components it writes, which the
// public wrapper only returns summed over rows.
// ============================================================================

namespace {

constexpr float kEps = 1e-6F;
constexpr float kRtol = 1.0e-3F;
constexpr float kDxAtol = 2e-2F;
// dgamma_components = dy * x_hat and dbeta_components = dy, element by element, held to the dgamma/dbeta tolerance.
constexpr float kComponentsAtol = 5e-2F;
// Far outside any gradient here, so a preallocated tensor the op did not write stands out.
constexpr float kPreallocatedSentinel = 1000.0F;

// Bit mask of the optional preallocated outputs a launch passes.
constexpr uint32_t kPreallocNone = 0U;
constexpr uint32_t kPreallocDx = 1U;
constexpr uint32_t kPreallocDgammaComponents = 2U;
constexpr uint32_t kPreallocDbetaComponents = 4U;
constexpr uint32_t kPreallocAll = kPreallocDx | kPreallocDgammaComponents | kPreallocDbetaComponents;

struct LayerNormBwShape {
    uint32_t batch = 1U;
    uint32_t heads = 1U;
    uint32_t seq_len = 1U;
    uint32_t features = 1U;
};

// 6 tile-rows on 6 cores (one work group); 72 features are three tiles with a masked tail.
constexpr LayerNormBwShape kSixTileRows{.batch = 3U, .heads = 1U, .seq_len = 40U, .features = 72U};

// The factory splits tile-rows over the grid. 2 * num_cores + 3 of them gives three cores 3 rows (core_group_1)
// and every other core 2 rows (core_group_2) on any grid, so both compute kernels run. One 8-row sequence per
// tile-row keeps the host work small; 72 features are three tiles with a masked tail.
LayerNormBwShape two_core_group_shape() {
    return {.batch = ttml::test_utils::two_core_group_units(3U), .heads = 1U, .seq_len = 8U, .features = 72U};
}

// One launch: its tensors (alive as long as the struct), what the op returned and the host reference.
struct LayerNormBwLaunch {
    ttnn::Tensor input;
    ttnn::Tensor gamma;
    ttnn::Tensor mean;
    ttnn::Tensor rstd;
    ttnn::Tensor dL_dout;
    std::optional<ttnn::Tensor> preallocated_dx;
    std::optional<ttnn::Tensor> preallocated_dgamma_components;
    std::optional<ttnn::Tensor> preallocated_dbeta_components;
    std::vector<ttnn::Tensor> outputs;  // [dx, dgamma_components, dbeta_components]
    xt::xarray<float> dx_ref;
    xt::xarray<float> dgamma_components_ref;
    xt::xarray<float> dbeta_components_ref;
};

LayerNormBwLaunch make_layernorm_bw_launch(
    const LayerNormBwShape& shape, const uint32_t seed, const uint32_t preallocated = kPreallocNone) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const uint32_t rows = shape.batch * shape.heads * shape.seq_len;
    const std::size_t total_elements = static_cast<std::size_t>(rows) * shape.features;

    const xt::xarray<float> x =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed);
    const xt::xarray<float> gamma =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{shape.features}, 0.0F, 1.0F, seed + 1U);
    const xt::xarray<float> beta =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{shape.features}, 0.0F, 1.0F, seed + 2U);
    const xt::xarray<float> dy =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed + 3U);

    [[maybe_unused]] const auto [y_ref, cache] =
        layernorm_forward_reference(x, gamma, beta, rows, shape.features, kEps);
    LayerNormBwLaunch launch;
    std::tie(launch.dx_ref, std::ignore, std::ignore) = layernorm_backward_reference(dy, cache);
    launch.dgamma_components_ref = dy * cache.x_hat;
    launch.dbeta_components_ref = dy;

    xt::xarray<float> x_4d = x;
    x_4d.reshape({shape.batch, shape.heads, shape.seq_len, shape.features});
    xt::xarray<float> gamma_4d = gamma;
    gamma_4d.reshape({1U, 1U, 1U, shape.features});
    xt::xarray<float> mean_4d = cache.mu;
    mean_4d.reshape({shape.batch, shape.heads, shape.seq_len, 1U});
    xt::xarray<float> rstd_4d = 1.0F / cache.s;
    rstd_4d.reshape({shape.batch, shape.heads, shape.seq_len, 1U});
    xt::xarray<float> dy_4d = dy;
    dy_4d.reshape({shape.batch, shape.heads, shape.seq_len, shape.features});

    launch.input = core::from_xtensor(x_4d, device);
    launch.gamma = core::from_xtensor(gamma_4d, device);
    launch.mean = core::from_xtensor(mean_4d, device);
    launch.rstd = core::from_xtensor(rstd_4d, device);
    launch.dL_dout = core::from_xtensor(dy_4d, device);

    const ttnn::Shape& output_shape = launch.input.logical_shape();
    if ((preallocated & kPreallocDx) != 0U) {
        launch.preallocated_dx = core::full(output_shape, kPreallocatedSentinel, device);
    }
    if ((preallocated & kPreallocDgammaComponents) != 0U) {
        launch.preallocated_dgamma_components = core::full(output_shape, kPreallocatedSentinel, device);
    }
    if ((preallocated & kPreallocDbetaComponents) != 0U) {
        launch.preallocated_dbeta_components = core::full(output_shape, kPreallocatedSentinel, device);
    }
    return launch;
}

void run_layernorm_bw(LayerNormBwLaunch& launch) {
    launch.outputs = ttnn::prim::ttml_layernorm_bw(
        launch.input,
        launch.gamma,
        launch.mean,
        launch.rstd,
        launch.dL_dout,
        launch.preallocated_dx,
        launch.preallocated_dgamma_components,
        launch.preallocated_dbeta_components);
}

void expect_close(
    const ttnn::Tensor& tensor, const xt::xarray<float>& expected, const float atol, const std::string& name) {
    const xt::xarray<float> actual_nd = ttml::core::to_xtensor(tensor);
    const xt::xarray<float> actual = xt::flatten(actual_nd);
    ASSERT_EQ(actual.shape(), expected.shape()) << name;
    EXPECT_TRUE(xt::allclose(actual, expected, kRtol, atol))
        << name << " max_abs_diff=" << xt::amax(xt::abs(actual - expected))();
}

void expect_layernorm_bw_launch_matches(const LayerNormBwLaunch& launch) {
    ASSERT_EQ(launch.outputs.size(), 3U);
    expect_close(launch.outputs[0], launch.dx_ref, kDxAtol, "dx");
    expect_close(launch.outputs[1], launch.dgamma_components_ref, kComponentsAtol, "dgamma_components");
    expect_close(launch.outputs[2], launch.dbeta_components_ref, kComponentsAtol, "dbeta_components");
    const std::array<std::optional<ttnn::Tensor>, 3> preallocated{
        launch.preallocated_dx, launch.preallocated_dgamma_components, launch.preallocated_dbeta_components};
    for (size_t i = 0; i < preallocated.size(); ++i) {
        if (preallocated[i].has_value()) {
            EXPECT_EQ(launch.outputs[i].buffer(), preallocated[i]->buffer())
                << "output " << i << " is not the preallocated tensor";
        }
    }
}

// Proves a hit on `b` ran on new addresses: `a` is alive, so none of b's buffers can reuse a's.
void expect_distinct_buffers(const LayerNormBwLaunch& a, const LayerNormBwLaunch& b) {
    EXPECT_NE(a.input.buffer()->address(), b.input.buffer()->address()) << "input";
    EXPECT_NE(a.gamma.buffer()->address(), b.gamma.buffer()->address()) << "gamma";
    EXPECT_NE(a.mean.buffer()->address(), b.mean.buffer()->address()) << "mean";
    EXPECT_NE(a.rstd.buffer()->address(), b.rstd.buffer()->address()) << "rstd";
    EXPECT_NE(a.dL_dout.buffer()->address(), b.dL_dout.buffer()->address()) << "dL_dout";
    for (size_t i = 0; i < std::min(a.outputs.size(), b.outputs.size()); ++i) {
        EXPECT_NE(a.outputs[i].buffer()->address(), b.outputs[i].buffer()->address()) << "output " << i;
    }
}

std::string describe_preallocated(const uint32_t preallocated) {
    if (preallocated == kPreallocNone) {
        return "nothing preallocated";
    }
    std::string description = "preallocated";
    description += (preallocated & kPreallocDx) != 0U ? " dx" : "";
    description += (preallocated & kPreallocDgammaComponents) != 0U ? " dgamma_components" : "";
    description += (preallocated & kPreallocDbetaComponents) != 0U ? " dbeta_components" : "";
    return description;
}

// Launches once per entry of `preallocation_patterns` and then once more per entry, on fresh tensors of one shape
// with new values each time, from an empty cache and with every launch kept alive. If the hash ignores which
// preallocated outputs are passed, every launch after the first hits the first launch's program; if it keys them, the
// second pass hits the program each pattern built in the first. Either way the second pass adds no cache entry, and
// every launch must return its own results and leave the other launches' tensors alone. A hit that keeps an earlier
// launch's address, or patches addresses by position in a tensor list whose layout changed, fails that.
void run_layernorm_bw_preallocation_sequence(
    const LayerNormBwShape& shape, const std::vector<uint32_t>& preallocation_patterns, const uint32_t seed) {
    auto& device = ttml::test_utils::device_with_empty_program_cache();
    const size_t num_patterns = preallocation_patterns.size();
    std::vector<LayerNormBwLaunch> launches;
    launches.reserve(2U * num_patterns);
    for (size_t i = 0; i < 2U * num_patterns; ++i) {
        const uint32_t preallocated = preallocation_patterns[i % num_patterns];
        SCOPED_TRACE("launch " + std::to_string(i) + ", " + describe_preallocated(preallocated));
        launches.push_back(make_layernorm_bw_launch(shape, seed + 10U * static_cast<uint32_t>(i), preallocated));
        const auto entries_before = device.num_program_cache_entries();
        run_layernorm_bw(launches[i]);
        if (i == 0U) {
            ASSERT_GT(device.num_program_cache_entries(), entries_before)
                << "the first launch did not populate the cache";
        } else if (i >= num_patterns) {
            EXPECT_EQ(device.num_program_cache_entries(), entries_before) << "a repeated pattern was not a cache hit";
            expect_distinct_buffers(launches[i - num_patterns], launches[i]);
        }
        expect_layernorm_bw_launch_matches(launches[i]);
    }
    for (size_t i = 0; i < launches.size(); ++i) {
        SCOPED_TRACE("launch " + std::to_string(i) + " re-checked after the last launch");
        expect_layernorm_bw_launch_matches(launches[i]);
    }
}

}  // namespace

// On cores of both work groups, without and with every output preallocated: a hit that keeps any of the five input or
// three output addresses of an earlier launch computes from that launch's tensors or writes into its outputs.
TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_CacheHit_FreshLiveTensors_TwoCoreGroups) {
    run_layernorm_bw_preallocation_sequence(two_core_group_shape(), {kPreallocNone, kPreallocAll}, /*seed=*/11U);
}

// 768 features are 24 whole tiles, as model widths usually are, so the kernels are built without DO_MASK_W.
// One tile-row: a single core.
TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_CacheHit_FreshLiveTensors_OneCore_WholeTiles) {
    run_layernorm_bw_preallocation_sequence(
        {.batch = 1U, .heads = 1U, .seq_len = 32U, .features = 768U}, {kPreallocNone, kPreallocAll}, /*seed=*/31U);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_CacheHit_PreallocatedOutputsAddedAndRemoved) {
    // Cached without preallocated outputs, then launches that add some or all of them and drop them again.
    run_layernorm_bw_preallocation_sequence(
        kSixTileRows,
        {kPreallocNone,
         kPreallocDbetaComponents,
         kPreallocDx,
         kPreallocDgammaComponents | kPreallocDbetaComponents,
         kPreallocAll},
        /*seed=*/101U);
    // Cached with a preallocated dx, then launches without it and with other subsets.
    run_layernorm_bw_preallocation_sequence(
        kSixTileRows, {kPreallocDx, kPreallocNone, kPreallocDgammaComponents, kPreallocAll}, /*seed=*/201U);
}
