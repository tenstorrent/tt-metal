// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <random>
#include <utility>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/subtract_at_target/device/subtract_at_target_device_operation.hpp"
#include "test_utils/program_cache.hpp"

class SubtractAtTargetTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

static xt::xarray<float> make_random_4d(uint32_t N, uint32_t S, uint32_t V, float low, float high, std::mt19937& gen) {
    xt::xarray<float> t = xt::empty<float>({N, 1U, S, V});
    std::uniform_real_distribution<float> float_dist(low, high);
    for (auto& v : t) {
        v = float_dist(gen);
    }
    return t;
}

static xt::xarray<uint32_t> make_random_targets(uint32_t N, uint32_t S, uint32_t vocab_size, std::mt19937& gen) {
    std::uniform_int_distribution<uint32_t> idx_dist(0U, vocab_size - 1U);
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, S});
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < S; ++s) {
            target_t(n, s) = idx_dist(gen);
        }
    }
    return target_t;
}

static xt::xarray<float> subtract_at_target_reference(
    const xt::xarray<float>& input,      // [N, 1, S, local_V]
    const xt::xarray<uint32_t>& target,  // [N, S]
    uint32_t first_v,
    uint32_t last_v,
    float subtract_value = 1.0F) {
    xt::xarray<float> result = input;
    const uint32_t N = static_cast<uint32_t>(target.shape(0));
    const uint32_t S = static_cast<uint32_t>(target.shape(1));
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t c = target(n, s);
            if (c >= first_v && c < last_v) {
                result(n, 0U, s, c - first_v) -= subtract_value;
            }
        }
    }
    return result;
}

TEST_F(SubtractAtTargetTest, SmallFullVocab) {
    using namespace ttml;

    xt::xarray<float> input_t = {{{{0.1F, 0.2F, 0.3F, 0.4F, 0.0F, 0.0F, 0.0F, 0.0F}}}};
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({1U, 1U});
    target_t(0, 0) = 3U;

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(input_dev, target_dev, /*local_V=*/8U);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, 0U, 8U);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
    EXPECT_NEAR(result_xt(0, 0, 0, 3), 0.4F - 1.0F, 1e-2F);
    EXPECT_NEAR(result_xt(0, 0, 0, 0), 0.1F, 1e-2F);
}

TEST_F(SubtractAtTargetTest, SmallPartialVocab) {
    using namespace ttml;

    xt::xarray<float> input_t = {{{{0.5F, 0.6F, 0.7F, 0.8F}, {-0.5F, -0.6F, -0.7F, -0.8F}}}};
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({1U, 2U});
    target_t(0, 0) = 2U;  // out of shard [4, 8) → no modification
    target_t(0, 1) = 5U;  // in shard, local col = 5-4 = 1

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result =
        metal::subtract_at_target(input_dev, target_dev, /*local_V=*/4U, /*cluster_axis=*/std::nullopt, /*first_v=*/4U);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, 4U, 8U);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));

    EXPECT_NEAR(result_xt(0, 0, 0, 0), 0.5F, 1e-2F);          // row 0: target out of shard, unchanged
    EXPECT_NEAR(result_xt(0, 0, 1, 1), -0.6F - 1.0F, 1e-2F);  // row 1: local col 1 modified
}

TEST_F(SubtractAtTargetTest, BatchedNonAlignedShape) {
    using namespace ttml;

    const uint32_t N = 2U, S = 91U, V = 157U;

    std::mt19937 gen(42);
    auto input_t = make_random_4d(N, S, V, -1.F, 1.F, gen);
    auto target_t = make_random_targets(N, S, V, gen);

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(input_dev, target_dev, /*local_V=*/V);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, 0U, V);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SubtractAtTargetTest, BatchedPartialVocabShard) {
    using namespace ttml;

    const uint32_t N = 2U, S = 64U, global_V = 128U;
    const uint32_t first_v = global_V / 2;
    const uint32_t last_v = global_V;
    const uint32_t local_V = last_v - first_v;

    std::mt19937 gen(7);
    auto input_t = make_random_4d(N, S, local_V, -1.F, 1.F, gen);
    auto target_t = make_random_targets(N, S, global_V, gen);

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(
        input_dev, target_dev, /*local_V=*/local_V, /*cluster_axis=*/std::nullopt, /*first_v=*/first_v);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, first_v, last_v);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SubtractAtTargetTest, CustomSubtractValue) {
    using namespace ttml;

    const uint32_t N = 2U, S = 64U, V = 128U;
    const float subtract_value = 1.0F / 128.0F;

    std::mt19937 gen(99);
    auto input_t = make_random_4d(N, S, V, 0.F, 0.01F, gen);
    auto target_t = make_random_targets(N, S, V, gen);

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(
        input_dev,
        target_dev,
        /*local_V=*/V,
        /*cluster_axis=*/std::nullopt,
        /*first_v=*/0U,
        subtract_value);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, 0U, V, subtract_value);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SubtractAtTargetTest, CustomSubtractValuePartialVocab) {
    using namespace ttml;

    const uint32_t N = 2U, S = 64U, global_V = 256U;
    const uint32_t first_v = 64U;
    const uint32_t last_v = 192U;
    const uint32_t local_V = last_v - first_v;
    const float subtract_value = 1.0F / static_cast<float>(N * S);

    std::mt19937 gen(123);
    auto input_t = make_random_4d(N, S, local_V, 0.F, 0.01F, gen);
    auto target_t = make_random_targets(N, S, global_V, gen);

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(
        input_dev, target_dev, /*local_V=*/local_V, /*cluster_axis=*/std::nullopt, /*first_v=*/first_v, subtract_value);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, first_v, last_v, subtract_value);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

namespace {

using ttml::test_utils::partial_grid_seq_len;
using ttml::test_utils::two_core_group_seq_len;

constexpr float kRtol = 3e-2F;
constexpr float kAtol = 1e-2F;

// What a launch passes besides its tensors: the vocabulary window [first_v, first_v + local_V) and subtract_value. The
// program cache key leaves all three out, so a cache hit must re-apply them.
struct CallScalars {
    uint32_t first_v = 0U;
    uint32_t local_V = 0U;
    float subtract_value = 1.0F;

    bool operator==(const CallScalars&) const = default;
};

// Input width and target range of the multi-core tests: targets fall inside and outside both windows.
constexpr uint32_t kV = 157U;
constexpr uint32_t kTargetBound = 200U;
constexpr CallScalars kFullScalars{.local_V = kV};
// A window narrower than the input at another offset, and MEAN's subtract_value over four tokens.
constexpr CallScalars kChangedScalars{.first_v = 37U, .local_V = 100U, .subtract_value = 0.25F};

struct LaunchInputs {
    ttnn::Tensor input;
    ttnn::Tensor target;
    xt::xarray<float> input_values;  // bf16-rounded, as the device holds them before any launch
    xt::xarray<uint32_t> target_values;
};

// input [N, 1, S, V] in [-1, 1]; targets in [0, target_bound), so they fall inside and outside every window used here.
LaunchInputs make_inputs(uint32_t N, uint32_t S, uint32_t V, uint32_t target_bound, uint32_t seed) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    std::mt19937 gen(seed);
    auto input = core::from_xtensor(make_random_4d(N, S, V, -1.F, 1.F, gen), device);
    auto target_values = make_random_targets(N, S, target_bound, gen);
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(target_values, device, ttnn::Layout::ROW_MAJOR);
    auto input_values = core::to_xtensor(input);
    return {std::move(input), std::move(target), std::move(input_values), std::move(target_values)};
}

xt::xarray<float> expected_output(const LaunchInputs& in, const CallScalars& s) {
    return subtract_at_target_reference(
        in.input_values, in.target_values, s.first_v, s.first_v + s.local_V, s.subtract_value);
}

// The data, and the scalars when they differ, must tell launch B from launch A: else stale addresses or stale scalars
// would pass.
void assert_launches_distinguishable(
    const LaunchInputs& a, const CallScalars& scalars_a, const LaunchInputs& b, const CallScalars& scalars_b) {
    const auto expected_b = expected_output(b, scalars_b);
    ASSERT_FALSE(xt::allclose(expected_output(a, scalars_a), expected_b, kRtol, kAtol));
    if (scalars_b != scalars_a) {
        ASSERT_FALSE(xt::allclose(expected_output(b, scalars_a), expected_b, kRtol, kAtol));
    }
}

// Launches on A from an empty program cache (a miss), then on fresh B with the same specs while every tensor of A stays
// alive, so each buffer of B is at a new address. B must reuse A's program, yet read its own tensors with its own
// scalars and write its own output, and leave A's output untouched.
void expect_cache_hit_with_fresh_tensors(
    uint32_t N,
    uint32_t S,
    uint32_t V,
    uint32_t target_bound,
    const CallScalars& scalars_a,
    const CallScalars& scalars_b,
    std::optional<uint32_t> cluster_axis = std::nullopt) {
    using namespace ttml;
    auto& device = test_utils::device_with_empty_program_cache();
    const auto launch = [&](const LaunchInputs& in, const CallScalars& s) {
        return metal::subtract_at_target(in.input, in.target, s.local_V, cluster_axis, s.first_v, s.subtract_value);
    };

    const auto a = make_inputs(N, S, V, target_bound, /*seed=*/101U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto result_a = launch(a, scalars_a);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch did not build a program";
    const auto result_a_values = core::to_xtensor(result_a);
    EXPECT_TRUE(xt::allclose(result_a_values, expected_output(a, scalars_a), kRtol, kAtol)) << "first launch";

    const auto b = make_inputs(N, S, V, target_bound, /*seed=*/202U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto result_b = launch(b, scalars_b);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "the second launch was not a cache hit";
    test_utils::expect_new_buffers({a.input, a.target, result_a}, {b.input, b.target, result_b}, "the second launch");

    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(a, scalars_a, b, scalars_b));
    EXPECT_TRUE(xt::allclose(core::to_xtensor(result_b), expected_output(b, scalars_b), kRtol, kAtol)) << "cache hit";
    EXPECT_TRUE(core::to_xtensor(result_a) == result_a_values) << "the cache hit wrote into the first launch's output";
}

// The vocab-parallel cross-entropy backward calls the prim in place (preallocated_output = input). In place on A, then
// in place on fresh B with other scalars while A stays alive: B must hit, read and write its own tensor with its own
// scalars, and leave A's tensor as A's launch left it.
void expect_in_place_cache_hit(uint32_t N, uint32_t S) {
    using namespace ttml;
    auto& device = test_utils::device_with_empty_program_cache();
    const auto launch_in_place = [](const LaunchInputs& in, const CallScalars& s) {
        return ttnn::prim::ttml_subtract_at_target(
            in.input, in.target, s.local_V, /*cluster_axis=*/std::nullopt, s.first_v, in.input, s.subtract_value);
    };

    const auto a = make_inputs(N, S, kV, kTargetBound, /*seed=*/301U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto result_a = launch_in_place(a, kFullScalars);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch did not build a program";
    EXPECT_EQ(result_a.buffer()->address(), a.input.buffer()->address()) << "the first launch did not run in place";
    const auto a_after_launch = core::to_xtensor(a.input);
    EXPECT_TRUE(xt::allclose(a_after_launch, expected_output(a, kFullScalars), kRtol, kAtol)) << "first launch";

    const auto b = make_inputs(N, S, kV, kTargetBound, /*seed=*/302U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto result_b = launch_in_place(b, kChangedScalars);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "the second launch was not a cache hit";
    EXPECT_EQ(result_b.buffer()->address(), b.input.buffer()->address()) << "the cache hit did not run in place";
    test_utils::expect_new_buffers({a.input, a.target}, {b.input, b.target}, "the second launch");

    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(a, kFullScalars, b, kChangedScalars));
    EXPECT_TRUE(xt::allclose(core::to_xtensor(b.input), expected_output(b, kChangedScalars), kRtol, kAtol))
        << "cache hit";
    EXPECT_TRUE(core::to_xtensor(a.input) == a_after_launch) << "the cache hit wrote into the first launch's tensor";
}

}  // namespace

// A cache hit must re-patch the input, target and output addresses on every core of both core groups.
TEST_F(SubtractAtTargetTest, CacheHitFreshTensors_TwoCoreGroups) {
    expect_cache_hit_with_fresh_tensors(2U, two_core_group_seq_len(5U), kV, kTargetBound, kFullScalars, kFullScalars);
}

// One tile-row: a single core and an empty second core group.
TEST_F(SubtractAtTargetTest, CacheHitFreshTensors_SingleCore) {
    const CallScalars full{.local_V = 40U};
    expect_cache_hit_with_fresh_tensors(1U, 17U, 40U, /*target_bound=*/50U, full, full);
}

// first_v, local_V and subtract_value are not part of the program cache key, so a launch that changes only them must
// hit and still use its own: a narrower window at another offset, and 0.25 where the first launch subtracted 1 (MEAN
// and NONE losses on the same shapes). Runs without a cluster_axis and with one, since the factory derives the
// device's TP rank differently for each.
TEST_F(SubtractAtTargetTest, CacheHitChangedScalars_TwoCoreGroups) {
    for (const auto cluster_axis : {std::optional<uint32_t>{}, std::optional<uint32_t>{1U}}) {
        SCOPED_TRACE(cluster_axis.has_value() ? "cluster_axis = 1" : "no cluster_axis");
        expect_cache_hit_with_fresh_tensors(
            2U, two_core_group_seq_len(5U), kV, kTargetBound, kFullScalars, kChangedScalars, cluster_axis);
    }
}

// The same on part of the grid: the hit must patch exactly the cores that got work.
TEST_F(SubtractAtTargetTest, CacheHitChangedScalars_PartialGrid) {
    for (const auto cluster_axis : {std::optional<uint32_t>{}, std::optional<uint32_t>{1U}}) {
        SCOPED_TRACE(cluster_axis.has_value() ? "cluster_axis = 1" : "no cluster_axis");
        expect_cache_hit_with_fresh_tensors(
            2U, partial_grid_seq_len(5U), kV, kTargetBound, kFullScalars, kChangedScalars, cluster_axis);
    }
}

// Production's in-place call, on both core groups and on part of the grid.
TEST_F(SubtractAtTargetTest, CacheHitInPlace_TwoCoreGroups) {
    expect_in_place_cache_hit(2U, two_core_group_seq_len(5U));
}

TEST_F(SubtractAtTargetTest, CacheHitInPlace_PartialGrid) {
    expect_in_place_cache_hit(2U, partial_grid_seq_len(5U));
}

// The prim also takes a preallocated output that is not the input. Launch without one, with one, with a fresh one and
// without again, keeping every tensor alive and alternating the scalars: the last two launches are cache hits whether
// or not the key depends on the output's presence, and each runs with other scalars than the launch that built its
// program. Each launch must write its own output with its own scalars and nothing else. Preallocated outputs start at
// 7, which no result (in [-2, 1]) can be.
TEST_F(SubtractAtTargetTest, CacheHitPreallocatedOutput_TwoCoreGroups) {
    using namespace ttml;
    auto& device = test_utils::device_with_empty_program_cache();

    const uint32_t N = 2U, S = two_core_group_seq_len(5U);
    const auto make_output = [&]() { return test_utils::filled_on_device({N, 1U, S, kV}, 7.F); };
    const auto launch = [](const LaunchInputs& in, const CallScalars& s, const std::optional<ttnn::Tensor>& output) {
        return ttnn::prim::ttml_subtract_at_target(
            in.input, in.target, s.local_V, /*cluster_axis=*/std::nullopt, s.first_v, output, s.subtract_value);
    };

    const auto a = make_inputs(N, S, kV, kTargetBound, /*seed=*/401U);
    auto entries_before = device.num_program_cache_entries();
    const auto result_a = launch(a, kFullScalars, std::nullopt);
    ASSERT_GT(device.num_program_cache_entries(), entries_before) << "the first launch did not build a program";

    // Either a hit on A's program or a miss that builds one with the changed scalars.
    const auto b = make_inputs(N, S, kV, kTargetBound, /*seed=*/402U);
    const auto output_b = make_output();
    const auto result_b = launch(b, kChangedScalars, output_b);

    const auto c = make_inputs(N, S, kV, kTargetBound, /*seed=*/403U);
    const auto output_c = make_output();
    entries_before = device.num_program_cache_entries();
    const auto result_c = launch(c, kFullScalars, output_c);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before) << "the preallocated launch was not a cache hit";

    const auto d = make_inputs(N, S, kV, kTargetBound, /*seed=*/404U);
    entries_before = device.num_program_cache_entries();
    const auto result_d = launch(d, kChangedScalars, std::nullopt);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before) << "the launch without output was not a cache hit";

    EXPECT_EQ(result_b.buffer()->address(), output_b.buffer()->address());
    EXPECT_EQ(result_c.buffer()->address(), output_c.buffer()->address());
    EXPECT_NE(result_d.buffer()->address(), result_a.buffer()->address());

    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(a, kFullScalars, b, kChangedScalars));
    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(b, kChangedScalars, c, kFullScalars));
    ASSERT_NO_FATAL_FAILURE(assert_launches_distinguishable(c, kFullScalars, d, kChangedScalars));
    // Read after the last launch, from the tensors the caller holds: each must hold its own launch's result, and the
    // inputs must be untouched.
    EXPECT_TRUE(xt::allclose(core::to_xtensor(result_a), expected_output(a, kFullScalars), kRtol, kAtol))
        << "without output (miss)";
    EXPECT_TRUE(xt::allclose(core::to_xtensor(output_b), expected_output(b, kChangedScalars), kRtol, kAtol))
        << "into a preallocated output";
    EXPECT_TRUE(xt::allclose(core::to_xtensor(output_c), expected_output(c, kFullScalars), kRtol, kAtol))
        << "into a fresh preallocated output (hit)";
    EXPECT_TRUE(xt::allclose(core::to_xtensor(result_d), expected_output(d, kChangedScalars), kRtol, kAtol))
        << "without output (hit)";
    for (const auto* in : {&a, &b, &c, &d}) {
        EXPECT_TRUE(core::to_xtensor(in->input) == in->input_values) << "a launch wrote into its input";
    }
}
