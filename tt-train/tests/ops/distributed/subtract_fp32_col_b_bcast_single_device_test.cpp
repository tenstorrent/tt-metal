// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Single-device companion to subtract_fp32_col_b_bcast_test.cpp.
//
// That sibling file reproduces a device-side hang in
//   ttnn::subtract(bf16_lhs, bf16_rhs, output_dtype=FLOAT32)
// with COL_B broadcast (rhs has W=1), BF16 inputs forced to FP32 output, and
// the LHS sharded across a 1x4 multi-device mesh. Hang is non-monotonic in
// W_tiles_per_shard: W ∈ {3, 5, 7, ≥8} hangs, W ∈ {1, 2, 4, 6} passes.
//
// This file runs the same op on a SINGLE chip at the exact local shape that
// each chip sees in the multi-device sharded test. It bisects the question:
//
//   Is sharding/multi-device actually a precondition, or is this a single-chip
//   kernel bug that just happens to be reachable via the sharded code path?
//
// Outcomes:
//   - Hangs at the same W values as the multi-device test → bug is purely in
//     the eltwise-binary kernel / program factory. Sharding is a red herring.
//   - Passes at all W → something specific to the sharded mesh-storage path
//     (tensor metadata, kernel selection, dispatch) is required to trigger it.
//
// W is read from the same env var as the sibling file (`REPRO_W_TILES_PER_SHARD`,
// default 2). No MGD env var needed here — uses the default single-device path.
//
// Run e.g. via:
//   REPRO_W_TILES_PER_SHARD=3 ../build_Release/tt-train/tests/ttml_tests \
//     --gtest_filter='SubtractFp32ColBBcastSingleDeviceTest.*'

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tt-metalium/distributed.hpp>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

class SubtractFp32ColBBcastSingleDeviceTest : public ::testing::Test {
protected:
    static constexpr uint32_t kTileW = 32U;
    static constexpr uint32_t kDefaultWTilesPerShard = 2U;

    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }

    // Read REPRO_W_TILES_PER_SHARD from the environment (shared with the
    // multi-device test so a single sweep step probes both at the same width).
    static uint32_t w_tiles_per_shard() {
        const char* w_env = std::getenv("REPRO_W_TILES_PER_SHARD");
        if (w_env == nullptr) {
            return kDefaultWTilesPerShard;
        }
        const auto w = static_cast<uint32_t>(std::stoul(w_env));
        return (w == 0U) ? kDefaultWTilesPerShard : w;
    }

    // Print + Synchronize so on hang we can tell host-side (dispatch never
    // returned) from device-side (dispatch returned but Synchronize blocks).
    static void sync_mesh(ttnn::distributed::MeshDevice* device, const char* tag) {
        std::fprintf(stderr, "[repro-1d] sync BEFORE %s\n", tag);
        std::fflush(stderr);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>{});
        std::fprintf(stderr, "[repro-1d] sync AFTER  %s\n", tag);
        std::fflush(stderr);
    }
};

// Single-device mirror of ColBBroadcast_Fp32Output_ShardedLhs_Hangs.
// Local shape matches what each chip sees in the sharded multi-device test:
// lhs = [B=5, 1, S=256, V=W*32] BF16, rhs = [B=5, 1, S=256, 1] BF16, out = FP32.
TEST_F(SubtractFp32ColBBcastSingleDeviceTest, ColBBroadcast_Fp32Output_Hangs) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t W = w_tiles_per_shard();
    const uint32_t B = 5U, S = 256U;
    const uint32_t V = W * kTileW;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, 1U});

    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE);
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE);

    std::fprintf(
        stderr,
        "[repro-1d] single-device lhs.shape=%s rhs.shape=%s W_tiles=%u (V=%u)\n",
        fmt::format("{}", lhs.logical_shape()).c_str(),
        fmt::format("{}", rhs.logical_shape()).c_str(),
        W,
        V);
    std::fflush(stderr);

    sync_mesh(device, "before subtract (1-device, FP32 out, COL_B bcast)");
    auto out = ttnn::subtract(lhs, rhs, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro-1d] dispatch returned (W_tiles=%u)\n", W);
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (1-device, FP32 out, COL_B bcast)");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}
