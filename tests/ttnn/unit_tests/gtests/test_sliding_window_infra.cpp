// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <string>
#include <vector>
#include <algorithm>

#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/api/tt-metalium/core_coord.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::sliding_window::test {

using namespace tt::tt_metal;

class SlidingWindowTestFixture : public testing::TestWithParam<SlidingWindowConfig> {};

TEST_P(SlidingWindowTestFixture, SlidingWindowHash) {
    const auto& sliding_window_a = GetParam();

    // start of same input
    auto sliding_window_b = sliding_window_a;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_EQ(sliding_window_a.get_hash(), sliding_window_b.get_hash());

    // flip snap_to_tile
    sliding_window_b.snap_to_tile = !sliding_window_a.snap_to_tile;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.snap_to_tile = !sliding_window_a.snap_to_tile;

    // flip is_bilinear
    sliding_window_b.is_bilinear = !sliding_window_a.is_bilinear;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.is_bilinear = !sliding_window_a.is_bilinear;

    // flip is_transpose
    sliding_window_b.is_transpose = !sliding_window_a.is_transpose;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.is_transpose = !sliding_window_a.is_transpose;

    // flip ceil_mode
    sliding_window_b.ceil_mode = !sliding_window_a.ceil_mode;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.ceil_mode = !sliding_window_a.ceil_mode;
}

INSTANTIATE_TEST_SUITE_P(
    SlidingWindowHashTests,
    SlidingWindowTestFixture,
    ::testing::Values(SlidingWindowConfig{
        .batch_size = 1,
        .input_hw = {32, 32},
        .window_hw = {3, 3},
        .stride_hw = {1, 1},
        .padding = {1, 1, 1, 1},
        .output_pad_hw = {0, 0},
        .dilation_hw = {1, 1},
        .num_cores_nhw = 1,
        .num_cores_c = 1,
        .core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {7, 7})),
        .snap_to_tile = false,
        .is_bilinear = false,
        .is_transpose = false,
        .ceil_mode = false}));

// -----------------------------------------------------------------------------
// In-Place Halo L1 economics probe (analysis, not a correctness gate).
// For each MaxPool height-sharded shape, compute the width-independent win/lose:
//   saved   = in_nsticks_per_core  (the input-shard buffer in-place avoids)
//   added   = max_ref_size          (outbound-halo sticks -> the remote-temp CB)
// Both scale by the same stick width, so the verdict is purely saved vs added.
// This is the confirm-or-refute experiment for "can in-place halo net-save L1".
// -----------------------------------------------------------------------------
TEST(InPlaceHaloEconomics, MaxPoolHeightSharded) {
    struct Spec {
        uint32_t n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw;
        bool ceil_mode;
        const char* note;
    };
    // From the archived in-place maxpool coverage (height_shard_tests).
    const std::vector<Spec> specs = {
        {1, 128, 150, 150, 2, 2, 2, 2, 0, 0, 1, 1, false, "resnet-like 2x2s2"},
        {1, 16, 25, 23, 2, 2, 2, 2, 0, 0, 1, 1, false, "C=16"},
        {1, 480, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, true, "3x3s2 pad1 ceil"},
        {1, 7, 24, 24, 3, 3, 1, 1, 0, 0, 2, 2, false, "dilation C=7"},
        {1, 1, 59, 59, 3, 5, 4, 2, 1, 1, 5, 4, true, "dilation ceil C=1"},
        {1, 64, 400, 544, 3, 3, 2, 2, 1, 1, 1, 1, false, "massive NHW 3x3s2"},
        {1, 832, 14, 14, 4, 4, 2, 2, 0, 0, 1, 1, true, ">800ch 4x4"},
        {1, 160, 30, 30, 15, 15, 1, 1, 7, 5, 1, 1, false, "15x15 kernel"},
        {1, 224, 20, 20, 8, 8, 6, 6, 2, 4, 1, 1, false, "8x8s6"},
        {1, 320, 48, 48, 36, 36, 1, 1, 0, 0, 1, 1, false, "36x36 kernel wide"},
        {1, 32, 6, 6, 3, 3, 1, 1, 1, 1, 1, 1, false, "tiny 6x6"},
        {1, 32, 13, 8, 4, 3, 6, 5, 2, 1, 1, 1, true, "ceil edge"},
        {8, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, true, "N=8 112x112 (fwd+rev reads)"},
        {32, 32, 264, 40, 5, 5, 2, 2, 2, 2, 1, 1, true, "N=32 264x40 5x5 (fwd+rev reads)"},
    };

    constexpr uint32_t kMaxCores = 64;  // WH 8x8 worker grid
    log_info(
        tt::LogTest,
        "{:<32} {:>7} {:>9} {:>9} {:>8} {:>7}  verdict",
        "shape",
        "cores",
        "in/core",
        "maxref",
        "ratio%",
        "out/core");
    for (const auto& s : specs) {
        auto make_config = [&](uint32_t ncores) {
            return SlidingWindowConfig{
                .batch_size = s.n,
                .channels = s.c,
                .input_hw = {s.h, s.w},
                .window_hw = {s.kh, s.kw},
                .stride_hw = {s.sh, s.sw},
                .padding = {s.ph, s.ph, s.pw, s.pw},
                .dilation_hw = {s.dh, s.dw},
                .num_cores_nhw = ncores,
                .num_cores_c = 1,
                .core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {7, 7})),
                .snap_to_tile = false,
                .ceil_mode = s.ceil_mode,
            };
        };
        // output nhw (channel excluded) -> pick a realistic core count.
        // out = (in + pad_total - dilation*(k-1) - 1) / stride + 1  (ceil variant when ceil_mode)
        auto out_dim = [&](uint32_t in, uint32_t pad_lo, uint32_t pad_hi, uint32_t k, uint32_t st, uint32_t dil) {
            const int eff = static_cast<int>(dil * (k - 1) + 1);
            const int num = static_cast<int>(in + pad_lo + pad_hi) - eff;
            if (num < 0) {
                return 1u;
            }
            const uint32_t base = s.ceil_mode ? ((num + st - 1) / st) : (num / st);
            return base + 1u;
        };
        const uint32_t out_h = out_dim(s.h, s.ph, s.ph, s.kh, s.sh, s.dh);
        const uint32_t out_w = out_dim(s.w, s.pw, s.pw, s.kw, s.sw, s.dw);
        const uint32_t output_nhw = s.n * out_h * out_w;
        const uint32_t num_cores = std::max<uint32_t>(1, std::min(kMaxCores, output_nhw));
        const auto config = make_config(num_cores);

        const auto pad_meta = generate_pad_metadata(config);
        const auto shard_bounds = generate_shard_boundaries(config);
        const uint32_t input_nhw = s.n * s.h * s.w;
        const uint32_t in_nsticks_per_core = (input_nhw + num_cores - 1) / num_cores;
        const auto tensor_meta = generate_tensor_metadata(pad_meta, config, in_nsticks_per_core);
        const uint32_t max_out = generate_max_out_nsticks_per_core(shard_bounds);

        // Exercise the real shipping functions (the ones the op/factory/callers use).
        const uint32_t max_ref_size = compute_max_outbound_halo_sticks(tensor_meta, shard_bounds, num_cores);
        const bool activates =
            should_halo_be_in_place(config, in_nsticks_per_core, /*is_height_sharded*/ true, /*is_in_tiled*/ false);

        const double ratio = in_nsticks_per_core ? (100.0 * max_ref_size / in_nsticks_per_core) : 0.0;
        const char* verdict = (max_ref_size < in_nsticks_per_core) ? "SAVE" : "LOSE";
        log_info(
            tt::LogTest,
            "{:<32} {:>7} {:>9} {:>9} {:>7.1f} {:>7}  {} activates={} ({})",
            s.note,
            num_cores,
            in_nsticks_per_core,
            max_ref_size,
            ratio,
            max_out,
            verdict,
            activates,
            s.note);
        EXPECT_GT(in_nsticks_per_core, 0u);
        // The gate uses a 0.75 net-L1 margin, so activation implies a clear per-buffer save.
        if (activates) {
            EXPECT_LT(max_ref_size, in_nsticks_per_core);
        }
    }
}

}  // namespace ttnn::operations::sliding_window::test
