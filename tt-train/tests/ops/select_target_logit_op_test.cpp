// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/select_target_logit/device/select_target_logit_device_operation.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

class SelectTargetLogitTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// CPU reference: logit holds the LOCAL shard [first_v, last_v) of the vocabulary.
// For each (n, s): output = logit[n, 0, s, c - first_v] if c in [first_v, last_v), else 0.
static xt::xarray<float> select_target_logit_reference(
    const xt::xarray<float>& logit,      // [N, 1, S, local_V]  where local_V = last_v - first_v
    const xt::xarray<uint32_t>& target,  // [N, S]
    uint32_t first_v,
    uint32_t last_v) {
    const uint32_t N = static_cast<uint32_t>(target.shape(0));
    const uint32_t S = static_cast<uint32_t>(target.shape(1));
    xt::xarray<float> result = xt::zeros<float>({N, 1U, S, 1U});
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t c = target(n, s);
            if (c >= first_v && c < last_v) {
                result(n, 0U, s, 0U) = logit(n, 0U, s, c - first_v);
            }
        }
    }
    return result;
}

TEST_F(SelectTargetLogitTest, SmallFullVocab) {
    using namespace ttml;

    // logit [1, 1, 1, 8], target [1, 1], full vocab (S=1)
    xt::xarray<float> logit_t = {{{{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F}}}};
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({1U, 1U});
    target_t(0, 0) = 3U;

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::select_target_logit(logit_dev, target_dev, /*local_V=*/8U);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = select_target_logit_reference(logit_t, target_t, 0U, 8U);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/1e-3F, /*atol=*/1e-3F));
}

TEST_F(SelectTargetLogitTest, SmallPartialVocab) {
    using namespace ttml;

    // Simulates TP shard: logit holds only local vocab [4, 8), shape [1, 1, 2, 4]
    // Global columns 4..7 map to local columns 0..3.
    xt::xarray<float> logit_t = {{{{5.F, 6.F, 7.F, 8.F}, {-5.F, -6.F, -7.F, -8.F}}}};
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({1U, 2U});
    target_t(0, 0) = 2U;  // out of shard [4, 8) → output should be 0
    target_t(0, 1) = 5U;  // in shard, local col = 5-4 = 1 → logit[0,0,1,1] = -6

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::select_target_logit(
        logit_dev, target_dev, /*local_V=*/4U, /*cluster_axis=*/std::nullopt, /*first_v=*/4U);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = select_target_logit_reference(logit_t, target_t, 4U, 8U);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/1e-3F, /*atol=*/1e-3F));

    // Explicit checks
    EXPECT_NEAR(result_xt(0, 0, 0, 0), 0.F, 1e-3F);   // out of shard → 0
    EXPECT_NEAR(result_xt(0, 0, 1, 0), -6.F, 1e-3F);  // local col 1 → -6
}

TEST_F(SelectTargetLogitTest, BatchedNonAlignedShape) {
    using namespace ttml;

    // Non-tile-aligned S and V — mirrors cross_entropy_fw_op_test pattern
    const uint32_t N = 2U, S = 91U, V = 157U;

    std::mt19937 gen(42);
    xt::xarray<float> logit_t = xt::empty<float>({N, 1U, S, V});
    std::uniform_real_distribution<float> float_dist(-5.F, 5.F);
    for (auto& v : logit_t) {
        v = float_dist(gen);
    }

    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, S});
    std::uniform_int_distribution<uint32_t> idx_dist(0U, V - 1U);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < S; ++s) {
            target_t(n, s) = idx_dist(gen);
        }
    }

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::select_target_logit(logit_dev, target_dev, /*local_V=*/V);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = select_target_logit_reference(logit_t, target_t, 0U, V);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SelectTargetLogitTest, BatchedPartialVocabShard) {
    // Simulates a TP shard: logit holds only the second half of the vocab [64, 128).
    using namespace ttml;

    const uint32_t N = 2U, S = 64U, global_V = 128U;
    const uint32_t first_v = global_V / 2;      // 64
    const uint32_t last_v = global_V;           // 128
    const uint32_t local_V = last_v - first_v;  // 64

    std::mt19937 gen(7);
    xt::xarray<float> logit_t = xt::empty<float>({N, 1U, S, local_V});
    std::uniform_real_distribution<float> float_dist(-3.F, 3.F);
    for (auto& v : logit_t) {
        v = float_dist(gen);
    }

    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, S});
    // Half the targets fall in [0, 64), half in [64, 128)
    std::uniform_int_distribution<uint32_t> idx_dist(0U, global_V - 1U);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < S; ++s) {
            target_t(n, s) = idx_dist(gen);
        }
    }

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::select_target_logit(
        logit_dev, target_dev, /*local_V=*/local_V, /*cluster_axis=*/std::nullopt, /*first_v=*/first_v);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = select_target_logit_reference(logit_t, target_t, first_v, last_v);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SelectTargetLogitTest, LargeVocab) {
    using namespace ttml;

    const uint32_t N = 1U, S = 1U, V = 32768U;

    std::mt19937 gen(99);
    xt::xarray<float> logit_t = xt::empty<float>({N, 1U, S, V});
    std::uniform_real_distribution<float> float_dist(-10.F, 10.F);
    for (auto& v : logit_t) {
        v = float_dist(gen);
    }

    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, S});
    target_t(0, 0) = 12345U;

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::select_target_logit(logit_dev, target_dev, /*local_V=*/V);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = select_target_logit_reference(logit_t, target_t, 0U, V);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SelectTargetLogitTest, NIGHTLY_LargeBatchLargeVocab) {
    using namespace ttml;

    const uint32_t N = 64U, S = 32U, V = 128000U;

    std::mt19937 gen(42);
    xt::xarray<float> logit_t = xt::empty<float>({N, 1U, S, V});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{logit_t.data(), logit_t.size()},
        []() { return std::uniform_real_distribution<float>(-10.F, 10.F); },
        seed);

    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, S});
    std::uniform_int_distribution<uint32_t> idx_dist(0U, V - 1U);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < S; ++s) {
            target_t(n, s) = idx_dist(gen);
        }
    }

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::select_target_logit(logit_dev, target_dev, /*local_V=*/V);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = select_target_logit_reference(logit_t, target_t, 0U, V);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

namespace {

using ttml::test_utils::partial_grid_seq_len;
using ttml::test_utils::two_core_group_seq_len;

constexpr float kRtol = 3e-2F;
constexpr float kAtol = 1e-2F;

// The vocabulary window [first_v, first_v + local_V) of a launch. The program cache key leaves it out, so a cache hit
// must re-apply it.
struct VocabWindow {
    uint32_t first_v = 0U;
    uint32_t local_V = 0U;

    bool operator==(const VocabWindow&) const = default;
};

// Logit width and target range of the multi-core tests: targets fall inside and outside both windows.
constexpr uint32_t kV = 157U;
constexpr uint32_t kTargetBound = 200U;
constexpr VocabWindow kFullWindow{0U, kV};
// Narrower than the logit and at another offset.
constexpr VocabWindow kNarrowWindow{37U, 100U};

struct LaunchInputs {
    ttnn::Tensor logit;
    ttnn::Tensor target;
    xt::xarray<float> logit_values;  // bf16-rounded, as the device holds them
    xt::xarray<uint32_t> target_values;
};

// logit [N, 1, S, V] in [-5, 5]; targets in [0, target_bound), so they fall inside and outside every window used here.
LaunchInputs make_inputs(uint32_t N, uint32_t S, uint32_t V, uint32_t target_bound, uint32_t seed) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    auto logit = core::from_xtensor(
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, 1U, S, V}, -5.F, 5.F, seed), device);
    auto target_values =
        test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, S}, 0U, target_bound - 1U, seed + 1U);
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(target_values, device, ttnn::Layout::ROW_MAJOR);
    auto logit_values = core::to_xtensor(logit);
    return {std::move(logit), std::move(target), std::move(logit_values), std::move(target_values)};
}

xt::xarray<float> expected_output(const LaunchInputs& in, const VocabWindow& window) {
    return select_target_logit_reference(
        in.logit_values, in.target_values, window.first_v, window.first_v + window.local_V);
}

// The data, and the windows when they differ, must tell launch B from launch A: else stale addresses or a stale window
// would pass.
void assert_launches_distinguishable(
    const LaunchInputs& a, const VocabWindow& window_a, const LaunchInputs& b, const VocabWindow& window_b) {
    const auto expected_b = expected_output(b, window_b);
    ASSERT_FALSE(xt::allclose(expected_output(a, window_a), expected_b, kRtol, kAtol));
    if (window_b != window_a) {
        ASSERT_FALSE(xt::allclose(expected_output(b, window_a), expected_b, kRtol, kAtol));
    }
}

// Launches on A from an empty program cache (a miss), then on fresh B with the same specs while every tensor of A stays
// alive, so each buffer of B is at a new address. B must reuse A's program, yet read its own tensors with its own
// window and write its own output, and leave A's output untouched.
void expect_cache_hit_with_fresh_tensors(
    uint32_t N,
    uint32_t S,
    uint32_t V,
    uint32_t target_bound,
    const VocabWindow& window_a,
    const VocabWindow& window_b,
    std::optional<uint32_t> cluster_axis = std::nullopt) {
    using namespace ttml;
    auto& device = test_utils::device_with_empty_program_cache();

    const auto a = make_inputs(N, S, V, target_bound, /*seed=*/101U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto result_a =
        metal::select_target_logit(a.logit, a.target, window_a.local_V, cluster_axis, window_a.first_v);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch did not build a program";
    const auto result_a_values = core::to_xtensor(result_a);
    EXPECT_TRUE(xt::allclose(result_a_values, expected_output(a, window_a), kRtol, kAtol)) << "first launch";

    const auto b = make_inputs(N, S, V, target_bound, /*seed=*/202U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto result_b =
        metal::select_target_logit(b.logit, b.target, window_b.local_V, cluster_axis, window_b.first_v);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "the second launch was not a cache hit";
    test_utils::expect_new_buffers({a.logit, a.target, result_a}, {b.logit, b.target, result_b}, "the second launch");

    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(a, window_a, b, window_b));
    EXPECT_TRUE(xt::allclose(core::to_xtensor(result_b), expected_output(b, window_b), kRtol, kAtol)) << "cache hit";
    EXPECT_TRUE(core::to_xtensor(result_a) == result_a_values) << "the cache hit wrote into the first launch's output";
}

}  // namespace

// A cache hit must re-patch the logit, target and output addresses on every core of both core groups.
TEST_F(SelectTargetLogitTest, CacheHitFreshTensors_TwoCoreGroups) {
    expect_cache_hit_with_fresh_tensors(2U, two_core_group_seq_len(5U), kV, kTargetBound, kFullWindow, kFullWindow);
}

// One tile-row: a single core and an empty second core group.
TEST_F(SelectTargetLogitTest, CacheHitFreshTensors_SingleCore) {
    const VocabWindow full{0U, 40U};
    expect_cache_hit_with_fresh_tensors(1U, 17U, 40U, /*target_bound=*/50U, full, full);
}

// first_v and local_V are not part of the program cache key, so a launch that changes only them must hit and still
// select with its own window: here narrower than the logit and at another offset. Runs without a cluster_axis and with
// one, since the factory derives the device's TP rank differently for each.
TEST_F(SelectTargetLogitTest, CacheHitChangedWindow_TwoCoreGroups) {
    for (const auto cluster_axis : {std::optional<uint32_t>{}, std::optional<uint32_t>{1U}}) {
        SCOPED_TRACE(cluster_axis.has_value() ? "cluster_axis = 1" : "no cluster_axis");
        expect_cache_hit_with_fresh_tensors(
            2U, two_core_group_seq_len(5U), kV, kTargetBound, kFullWindow, kNarrowWindow, cluster_axis);
    }
}

// The same on part of the grid: the hit must patch exactly the cores that got work.
TEST_F(SelectTargetLogitTest, CacheHitChangedWindow_PartialGrid) {
    for (const auto cluster_axis : {std::optional<uint32_t>{}, std::optional<uint32_t>{1U}}) {
        SCOPED_TRACE(cluster_axis.has_value() ? "cluster_axis = 1" : "no cluster_axis");
        expect_cache_hit_with_fresh_tensors(
            2U, partial_grid_seq_len(5U), kV, kTargetBound, kFullWindow, kNarrowWindow, cluster_axis);
    }
}

// The prim takes an optional preallocated output; production and the public wrapper never pass one. Launch without
// one, with one, with a fresh one and without again, keeping every tensor alive and alternating the window: the last
// two launches are cache hits whether or not the key depends on the output's presence, and each runs with another
// window than the launch that built its program. Each launch must write its own output with its own window and nothing
// else. Preallocated outputs start at 7, which no selected logit (in [-5, 5]) can be.
TEST_F(SelectTargetLogitTest, CacheHitPreallocatedOutput_TwoCoreGroups) {
    using namespace ttml;
    auto& device = test_utils::device_with_empty_program_cache();

    const uint32_t N = 2U, S = two_core_group_seq_len(5U);
    const auto make_output = [&]() { return test_utils::filled_on_device({N, 1U, S, 1U}, 7.F); };
    const auto launch =
        [](const LaunchInputs& in, const VocabWindow& window, const std::optional<ttnn::Tensor>& output) {
            return ttnn::prim::ttml_select_target_logit(
                in.logit, in.target, window.local_V, /*cluster_axis=*/std::nullopt, window.first_v, output);
        };

    const auto a = make_inputs(N, S, kV, kTargetBound, /*seed=*/301U);
    auto entries_before = device.num_program_cache_entries();
    const auto result_a = launch(a, kFullWindow, std::nullopt);
    ASSERT_GT(device.num_program_cache_entries(), entries_before) << "the first launch did not build a program";

    // Either a hit on A's program or a miss that builds one with the narrow window.
    const auto b = make_inputs(N, S, kV, kTargetBound, /*seed=*/302U);
    const auto output_b = make_output();
    const auto result_b = launch(b, kNarrowWindow, output_b);

    const auto c = make_inputs(N, S, kV, kTargetBound, /*seed=*/303U);
    const auto output_c = make_output();
    entries_before = device.num_program_cache_entries();
    const auto result_c = launch(c, kFullWindow, output_c);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before) << "the preallocated launch was not a cache hit";

    const auto d = make_inputs(N, S, kV, kTargetBound, /*seed=*/304U);
    entries_before = device.num_program_cache_entries();
    const auto result_d = launch(d, kNarrowWindow, std::nullopt);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before) << "the launch without output was not a cache hit";

    EXPECT_EQ(result_b.buffer()->address(), output_b.buffer()->address());
    EXPECT_EQ(result_c.buffer()->address(), output_c.buffer()->address());
    EXPECT_NE(result_d.buffer()->address(), result_a.buffer()->address());

    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(a, kFullWindow, b, kNarrowWindow));
    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(b, kNarrowWindow, c, kFullWindow));
    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(c, kFullWindow, d, kNarrowWindow));
    // Read after the last launch, from the tensors the caller holds: each must hold its own launch's result.
    EXPECT_TRUE(xt::allclose(core::to_xtensor(result_a), expected_output(a, kFullWindow), kRtol, kAtol))
        << "without output (miss)";
    EXPECT_TRUE(xt::allclose(core::to_xtensor(output_b), expected_output(b, kNarrowWindow), kRtol, kAtol))
        << "into a preallocated output";
    EXPECT_TRUE(xt::allclose(core::to_xtensor(output_c), expected_output(c, kFullWindow), kRtol, kAtol))
        << "into a fresh preallocated output (hit)";
    EXPECT_TRUE(xt::allclose(core::to_xtensor(result_d), expected_output(d, kNarrowWindow), kRtol, kAtol))
        << "without output (hit)";
}
