// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/profiler_no_op/device/profiler_no_op_device_operation.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

class ProfilerNoOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ProfilerNoOpTest, ProfilerNoOpTest_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 187U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::profiler_no_op(input, "identifier");

    // NOTE: ProfilerNoOp does not change the input, so we just check that the operation completed successfully.
}

// ── Program-cache hits ──────────────────────────────────────────────────────────────────────────────
// A launch whose specs and identifier match a cached program reuses it, and only the buffer addresses of its
// tensors are patched in: a hit must write its own output, whether or not it passes the optional preallocated
// output.

class ProfilerNoOpCacheHitTest : public ttml::test_utils::DeviceSuiteTest {};

namespace {

// Initial value of every output the cache-hit tests check: an element that its launch does not write keeps it.
constexpr float kNotWritten = -1000.F;
// The identifier is part of the program hash.
constexpr auto kCacheHitIdentifier = "cache_hit";

// The op splits N * C * Ht tile rows over the cores. Ht = 2 * cores + 3 is more than the core count and not a
// multiple of it, so split_work_to_cores hands out two core groups on any grid. The last tile row and the last
// tile column are partial.
ttnn::Shape two_core_group_shape() {
    return ttnn::Shape({1U, 1U, ttml::test_utils::two_core_group_units(3U) * 32U - 5U, 147U});
}

// A tensor of the op's output spec for `input` (its shape and memory config, bf16, TILE) filled with kNotWritten.
// It is tilized on the host, so creating it allocates no device buffer besides its own.
ttnn::Tensor make_not_written(const ttnn::Tensor& input) {
    const tt::tt_metal::TensorSpec spec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, input.memory_config()));
    const std::vector<float> values(input.logical_shape().volume(), kNotWritten);
    return ttnn::Tensor::from_vector(values, spec, &ttml::autograd::ctx().get_device());
}

// Bit-exact: the output may hold bf16 denormals.
void expect_same_bits(const xt::xarray<float>& actual, const xt::xarray<float>& expected, const std::string& label) {
    ASSERT_EQ(actual.shape(), expected.shape()) << label;
    EXPECT_EQ(ttml::test_utils::count_bit_mismatches(expected, actual), 0U)
        << label << ": elements that differ from what a cache miss writes";
}

// Launches the op once per entry of `preallocated` on new inputs of one spec, every earlier tensor kept alive.
// Launch i writes into a preallocated output (ttnn::prim::ttml_profiler_no_op) if preallocated[i], else into one the
// op allocates (ttml::metal::profiler_no_op). Every output starts out as kNotWritten: a preallocated one is filled
// with it, and an allocated one lands on the pages of a kNotWritten tensor of its spec freed just before the launch
// (the DRAM allocator is first-fit). Each output must then hold, bit for bit, what a cache miss writes for the same
// input. Launch 0 is the miss, and a launch that repeats an earlier presence pattern must hit, whether or not the
// key includes the preallocated output.
void run_cache_hits(const std::vector<bool>& preallocated) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto shape = two_core_group_shape();

    struct Launch {
        std::string label;
        ttnn::Tensor input;
        xt::xarray<float> input_values;
        xt::xarray<float> expected;
        ttnn::Tensor output;
    };
    std::vector<Launch> launches(preallocated.size());
    for (uint32_t i = 0U; i < launches.size(); ++i) {
        auto& launch = launches[i];
        launch.label = "launch " + std::to_string(i) + (preallocated[i] ? " (preallocated output)" : " (op output)");
        launch.input = ttml::core::from_xtensor(
            ttml::test_utils::make_uniform_xarray<float>(shape, -10.0F, 10.0F, 500U + i),
            device,
            ttnn::Layout::ROW_MAJOR);
        launch.input_values = ttml::core::to_xtensor(launch.input);
        device->clear_program_cache();
        launch.expected = ttml::core::to_xtensor(
            ttnn::prim::ttml_profiler_no_op(launch.input, kCacheHitIdentifier, make_not_written(launch.input)));
    }

    device->clear_program_cache();
    std::array<bool, 2> pattern_launched{};
    for (uint32_t i = 0U; i < launches.size(); ++i) {
        auto& launch = launches[i];
        auto not_written = make_not_written(launch.input);
        std::optional<ttnn::Tensor> preallocated_output;
        if (preallocated[i]) {
            preallocated_output = not_written;
        } else {
            not_written.deallocate();
        }
        const bool repeats_pattern = pattern_launched[preallocated[i] ? 1U : 0U];
        pattern_launched[preallocated[i] ? 1U : 0U] = true;

        const auto entries_before = device->num_program_cache_entries();
        launch.output = preallocated[i]
                            ? ttnn::prim::ttml_profiler_no_op(launch.input, kCacheHitIdentifier, preallocated_output)
                            : ttml::metal::profiler_no_op(launch.input, kCacheHitIdentifier);
        const auto entries_after = device->num_program_cache_entries();
        if (i == 0U) {
            ASSERT_GT(entries_after, entries_before) << launch.label << " must compile a program";
        } else if (repeats_pattern) {
            EXPECT_EQ(entries_after, entries_before) << launch.label << " must reuse the program of its pattern";
        }
        if (preallocated_output.has_value()) {
            EXPECT_EQ(launch.output.buffer()->address(), preallocated_output->buffer()->address())
                << launch.label << " must return its preallocated output";
        }
        for (uint32_t j = 0U; j < i; ++j) {
            EXPECT_NE(launch.input.buffer()->address(), launches[j].input.buffer()->address())
                << launch.label << " input vs " << launches[j].label;
            EXPECT_NE(launch.output.buffer()->address(), launches[j].output.buffer()->address())
                << launch.label << " output vs " << launches[j].label;
        }
        expect_same_bits(ttml::core::to_xtensor(launch.output), launch.expected, launch.label);
    }

    // No launch may have written into the tensors of another one.
    for (const auto& launch : launches) {
        expect_same_bits(ttml::core::to_xtensor(launch.output), launch.expected, launch.label + " after all launches");
        EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(launch.input), launch.input_values)))
            << launch.label << " input changed";
    }
}

}  // namespace

// The miss allocates its output; then the preallocated output appears and disappears again.
TEST_F(ProfilerNoOpCacheHitTest, CacheHitWritesItsOwnOutput_OpOutputFirst_TwoCoreGroups) {
    run_cache_hits({false, true, true, false});
}

// The miss writes a preallocated output; then it disappears and appears again.
TEST_F(ProfilerNoOpCacheHitTest, CacheHitWritesItsOwnOutput_PreallocatedOutputFirst_TwoCoreGroups) {
    run_cache_hits({true, false, false, true});
}
