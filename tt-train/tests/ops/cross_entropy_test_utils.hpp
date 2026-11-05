// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

// Launch tensors and program-cache scenarios shared by the cross_entropy_fw and cross_entropy_bw tests.

namespace ttml::test_utils::cross_entropy {

struct LaunchTensors {
    ttnn::Tensor input;
    ttnn::Tensor target;
    xt::xarray<float> input_values;  // bf16-rounded, as the device holds them
    xt::xarray<uint32_t> target_values;
};

// Same specs for every call with the same (N, H, W); the values come from `seed`.
inline LaunchTensors make_inputs(uint32_t N, uint32_t H, uint32_t W, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto target_values = make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    auto input = ttml::core::from_xtensor(
        make_uniform_xarray<float>(std::array<std::size_t, 4>{N, 1U, H, W}, -10.0F, 10.0F, seed), device);
    auto target =
        ttml::core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(target_values, device, ttnn::Layout::ROW_MAJOR);
    auto input_values = ttml::core::to_xtensor(input);
    return {std::move(input), std::move(target), std::move(input_values), std::move(target_values)};
}

using Launch = std::function<ttnn::Tensor(const LaunchTensors& in)>;
using LaunchWithOutput =
    std::function<ttnn::Tensor(const LaunchTensors& in, const std::optional<ttnn::Tensor>& preallocated_output)>;
// Checks a launch's result (or the output it wrote) against the reference for its inputs.
using ExpectMatches =
    std::function<void(const ttnn::Tensor& result, const LaunchTensors& in, const std::string& label)>;

// A program-cache hit re-patches only the input, target and output addresses. The first launch's tensors stay alive,
// so every buffer of the second launch is at a new address: a stale one reads the first launch's data or overwrites
// its result. Inputs are [2, 1, H, W], H on both core groups.
inline void run_cache_hit_fresh_tensors(uint32_t W, const Launch& launch, const ExpectMatches& expect_matches) {
    auto& device = device_with_empty_program_cache();
    const uint32_t N = 2U, H = two_core_group_seq_len(7U);
    const uint32_t seed = ttml::autograd::ctx().get_generator()();

    const auto a = make_inputs(N, H, W, seed);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto result_a = launch(a);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "first launch did not populate the cache";
    expect_matches(result_a, a, "first launch");

    const auto b = make_inputs(N, H, W, seed + 2U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto result_b = launch(b);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "second launch was not a cache hit";
    expect_new_buffers({a.input, a.target, result_a}, {b.input, b.target, result_b}, "cache hit");

    expect_matches(result_b, b, "cache hit");
    expect_matches(result_a, a, "first launch, re-read after the cache hit");
}

// The prim takes an optional preallocated output; the public wrapper never passes one. Launch without one, with one,
// with a fresh one and without again, keeping every tensor alive: the last two launches are cache hits whether or not
// the key depends on the output's presence, and each launch must write its own output. Then the other order on an
// empty cache: build the program with a preallocated output, launch without one, then with a fresh one (a hit).
// Inputs are [2, 1, H, W], H on both core groups; preallocated outputs are [2, 1, H, output_width] and start at
// `sentinel`, which no result can be.
inline void run_preallocated_output_cache_hits(
    uint32_t W,
    uint32_t output_width,
    float sentinel,
    const LaunchWithOutput& launch,
    const ExpectMatches& expect_matches) {
    auto& device = device_with_empty_program_cache();
    const uint32_t N = 2U, H = two_core_group_seq_len(7U);
    const uint32_t seed = ttml::autograd::ctx().get_generator()();
    const auto make_output = [&]() { return filled_on_device({N, 1U, H, output_width}, sentinel); };

    const auto a = make_inputs(N, H, W, seed);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto result_a = launch(a, std::nullopt);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "first launch did not populate the cache";

    const auto b = make_inputs(N, H, W, seed + 2U);
    const auto output_b = make_output();
    const auto result_b = launch(b, output_b);

    const auto c = make_inputs(N, H, W, seed + 4U);
    const auto output_c = make_output();
    auto entries_before_hit = device.num_program_cache_entries();
    const auto result_c = launch(c, output_c);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "preallocated launch was not a cache hit";

    const auto d = make_inputs(N, H, W, seed + 6U);
    entries_before_hit = device.num_program_cache_entries();
    const auto result_d = launch(d, std::nullopt);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch without output was not a cache hit";

    EXPECT_EQ(result_b.buffer()->address(), output_b.buffer()->address());
    EXPECT_EQ(result_c.buffer()->address(), output_c.buffer()->address());
    EXPECT_NE(result_d.buffer()->address(), result_a.buffer()->address());

    expect_matches(result_a, a, "without preallocated output (miss)");
    expect_matches(output_b, b, "into preallocated output");
    expect_matches(output_c, c, "into a fresh preallocated output (hit)");
    expect_matches(result_d, d, "without preallocated output (hit)");

    device.clear_program_cache();
    const auto e = make_inputs(N, H, W, seed + 8U);
    const auto output_e = make_output();
    const auto entries_before_preallocated_miss = device.num_program_cache_entries();
    const auto result_e = launch(e, output_e);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_preallocated_miss)
        << "first preallocated launch did not populate the cache";

    const auto f = make_inputs(N, H, W, seed + 10U);
    const auto result_f = launch(f, std::nullopt);

    const auto g = make_inputs(N, H, W, seed + 12U);
    const auto output_g = make_output();
    entries_before_hit = device.num_program_cache_entries();
    const auto result_g = launch(g, output_g);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit)
        << "second preallocated launch was not a cache hit";

    EXPECT_EQ(result_e.buffer()->address(), output_e.buffer()->address());
    EXPECT_EQ(result_g.buffer()->address(), output_g.buffer()->address());

    expect_matches(output_e, e, "into preallocated output, on an empty cache (miss)");
    expect_matches(result_f, f, "without preallocated output, after a preallocated launch");
    expect_matches(output_g, g, "into a fresh preallocated output, after a launch without one (hit)");
}

}  // namespace ttml::test_utils::cross_entropy
