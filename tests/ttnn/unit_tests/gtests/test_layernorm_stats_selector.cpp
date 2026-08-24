// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <string_view>

#include "ttnn/operations/normalization/layernorm/device/layernorm_stats_selector.hpp"

namespace {

using ttnn::prim::layernorm::BlackholeStatsSelectorParams;
using ttnn::prim::layernorm::select_interleaved_statistics_backend;
using ttnn::prim::layernorm::select_sharded_statistics_backend;
using ttnn::prim::layernorm::StatisticsBackend;
using ttnn::prim::layernorm::use_blackhole_sfpu_stats;

constexpr BlackholeStatsSelectorParams default_params() {
    return {
        .input_format = tt::DataFormat::Float32,
        .padded_width = 2880,
        .fuse_pre_add = false,
        .has_gamma = true,
        .has_beta = true,
        .num_tile_rows = 512,
        .active_cores = 110,
        .available_cores = 110,
        .compact_two_pass_fits_in_l1 = true,
    };
}

TEST(LayerNormStatsSelector, BlackholeCalibratedBoundaries) {
    auto fp32_affine_below = default_params();
    fp32_affine_below.padded_width = 2879;

    auto fp32_plain = default_params();
    fp32_plain.has_gamma = false;
    fp32_plain.has_beta = false;

    auto fp32_residual = default_params();
    fp32_residual.fuse_pre_add = true;
    fp32_residual.padded_width = 2048;

    auto fp32_residual_below = fp32_residual;
    fp32_residual_below.padded_width = 2047;

    auto bf16_affine = default_params();
    bf16_affine.input_format = tt::DataFormat::Float16_b;
    bf16_affine.padded_width = 32;

    auto bf16_residual_middle = bf16_affine;
    bf16_residual_middle.fuse_pre_add = true;
    bf16_residual_middle.padded_width = 512;

    auto bf16_residual_narrow = bf16_residual_middle;
    bf16_residual_narrow.padded_width = 256;

    auto bf16_residual_wide = bf16_residual_middle;
    bf16_residual_wide.padded_width = 2880;

    auto bf16_plain_below = bf16_affine;
    bf16_plain_below.has_gamma = false;
    bf16_plain_below.has_beta = false;
    bf16_plain_below.padded_width = 3231;

    auto bf16_plain_at = bf16_plain_below;
    bf16_plain_at.padded_width = 3232;

    auto bf16_gamma_only = bf16_plain_at;
    bf16_gamma_only.has_gamma = true;

    auto bfp8_affine = default_params();
    bfp8_affine.input_format = tt::DataFormat::Bfp8_b;

    auto bfp8_residual = bfp8_affine;
    bfp8_residual.fuse_pre_add = true;

    auto bfp8_residual_below = bfp8_residual;
    bfp8_residual_below.padded_width = 2879;

    auto fp32_underutilized = default_params();
    fp32_underutilized.num_tile_rows = 32;
    fp32_underutilized.active_cores = 32;

    auto fp32_residual_replay = fp32_underutilized;
    fp32_residual_replay.fuse_pre_add = true;
    fp32_residual_replay.compact_two_pass_fits_in_l1 = false;

    auto fp32_restricted_grid = default_params();
    fp32_restricted_grid.active_cores = 64;
    fp32_restricted_grid.available_cores = 64;

    auto two_pass_does_not_fit = bf16_affine;
    two_pass_does_not_fit.compact_two_pass_fits_in_l1 = false;

    struct TestCase {
        std::string_view name;
        BlackholeStatsSelectorParams params;
        bool expected;
    };
    const TestCase cases[] = {
        {"fp32 affine at crossover", default_params(), true},
        {"fp32 affine below crossover", fp32_affine_below, false},
        {"fp32 parameter-free", fp32_plain, false},
        {"fp32 residual affine", fp32_residual, true},
        {"fp32 residual affine below crossover", fp32_residual_below, false},
        {"bf16 affine", bf16_affine, true},
        {"bf16 residual narrow", bf16_residual_narrow, true},
        {"bf16 residual middle", bf16_residual_middle, false},
        {"bf16 residual wide", bf16_residual_wide, true},
        {"bf16 parameter-free below crossover", bf16_plain_below, false},
        {"bf16 parameter-free at crossover", bf16_plain_at, true},
        {"bf16 gamma-only at parameter-free crossover", bf16_gamma_only, true},
        {"bfp8 affine", bfp8_affine, false},
        {"bfp8 residual affine", bfp8_residual, true},
        {"bfp8 residual affine below crossover", bfp8_residual_below, false},
        {"fp32 underutilized", fp32_underutilized, false},
        {"fp32 residual replay", fp32_residual_replay, true},
        {"fp32 restricted grid", fp32_restricted_grid, false},
        {"two-pass compact allocation does not fit", two_pass_does_not_fit, false},
    };

    for (const auto& test_case : cases) {
        EXPECT_EQ(use_blackhole_sfpu_stats(test_case.params), test_case.expected) << test_case.name;
    }
}

TEST(LayerNormStatsSelector, InterleavedArchitectureAndLayoutPolicy) {
    const auto params = default_params();
    EXPECT_EQ(
        select_interleaved_statistics_backend(true, tt::ARCH::BLACKHOLE, false, false, true, params),
        StatisticsBackend::SFPU_TWO_PASS);
    EXPECT_EQ(
        select_interleaved_statistics_backend(true, tt::ARCH::BLACKHOLE, false, false, false, params),
        StatisticsBackend::TILE_REDUCTION);
    EXPECT_EQ(
        select_interleaved_statistics_backend(true, tt::ARCH::WORMHOLE_B0, false, false, true, params),
        StatisticsBackend::SFPU_TWO_PASS);
    EXPECT_EQ(
        select_interleaved_statistics_backend(true, tt::ARCH::BLACKHOLE, false, true, true, params),
        StatisticsBackend::SFPU_TWO_PASS);
    EXPECT_EQ(
        select_interleaved_statistics_backend(false, tt::ARCH::BLACKHOLE, false, false, true, params),
        StatisticsBackend::TILE_REDUCTION);
    EXPECT_EQ(
        select_interleaved_statistics_backend(true, tt::ARCH::BLACKHOLE, true, false, true, params),
        StatisticsBackend::SFPU_TWO_PASS);
}

TEST(LayerNormStatsSelector, ShardedDistributedPolicy) {
    EXPECT_EQ(
        select_sharded_statistics_backend(true, tt::ARCH::BLACKHOLE, false, false), StatisticsBackend::TILE_REDUCTION);
    EXPECT_EQ(
        select_sharded_statistics_backend(true, tt::ARCH::BLACKHOLE, true, false), StatisticsBackend::SFPU_TWO_PASS);
    EXPECT_EQ(
        select_sharded_statistics_backend(true, tt::ARCH::WORMHOLE_B0, false, false), StatisticsBackend::SFPU_TWO_PASS);
    EXPECT_EQ(
        select_sharded_statistics_backend(false, tt::ARCH::WORMHOLE_B0, false, false),
        StatisticsBackend::TILE_REDUCTION);
}

}  // namespace
