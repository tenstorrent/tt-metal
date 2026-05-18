// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal multi-device reproducer for the hang observed when
// `ttnn::subtract(bf16_lhs, bf16_rhs, output_dtype=FLOAT32)` is invoked with a
// COL_B broadcast (rhs has W=1) in vocab_parallel_cross_entropy_loss.
//
// Originally documented in `ops/distributed/losses.cpp`:
//   "Requesting an FP32 output here causes binary_ng to auto-inject a
//    TYPECAST(BF16,FP32) into the post_activations chain. Combined with COL_B
//    broadcast (b's W=1) and the multi-device TP setup that
//    vocab_parallel_cross_entropy uses, that path hangs."
//
// Empirical findings from the parametric test below (W = LHS tiles per shard
// on the innermost dim; LHS sharded across a 1x4 mesh, BF16 inputs, FP32 out,
// rhs.W = 1):
//
//     W  result   W  result
//     1  PASS     5  HANG
//     2  PASS     6  PASS
//     3  HANG     7  HANG
//     4  PASS     8  HANG  (and HANG for all larger W tested up to 256)
//
// I.e. the hang is **non-monotonic in W** in the 3..7 range. Smallest known
// hanging configuration is W=3 (96 columns per shard).
//
// ALL FOUR TESTS in this file read W from the same env var
// (`REPRO_W_TILES_PER_SHARD`, default 2), so a single sweep step covers
// {BF16+COL_B+replicated, FP32+no-bcast+replicated, FP32+COL_B+replicated,
// FP32+COL_B+sharded} at the same per-device width. The three controls being
// independent of the bug means they double as smoke checks: if a control fails
// or hangs, the parametric outcome that follows is suspect (likely a wedged
// device, not a real bug repro).
//
// Each test prints + Synchronizes between dispatch and the next host call so
// that on hang we can tell whether the freeze is host-side (dispatch never
// returned) or device-side (dispatch returned but Synchronize blocks forever).
//
// Run e.g. via:
//   TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.textproto
//   \
//     ../build_Release/tt-train/tests/ttml_tests \
//     --gtest_filter='SubtractFp32ColBBcastTest.*'

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

// Number of devices we want to open. Matches the user's training setup on a
// bh_qb p150_x4 box (4 chips arranged as 1x4 per the locally-edited
// p150_x4_mesh_graph_descriptor.textproto).
//
// NOTE: We can't use a (1, 2) sub-mesh of the 2x2 physical topology because
// autodiscovery may pick diagonally-opposite chips that have no direct ethernet
// link, which makes fabric router sync time out in SetUp().
//
// Requires `TT_MESH_GRAPH_DESC_PATH` to point at a matching `[1, 4]` MGD before
// running. Example:
//   export TT_METAL_HOME=/localdev/bklockiewicz/tt-metal
//   export
//   TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.textproto
constexpr uint32_t kMeshRows = 1U;
constexpr uint32_t kMeshCols = 4U;
constexpr uint32_t kMeshDevices = kMeshRows * kMeshCols;

}  // namespace

class SubtractFp32ColBBcastTest : public ::testing::Test {
protected:
    static constexpr uint32_t kTileW = 32U;
    static constexpr uint32_t kDefaultWTilesPerShard = 2U;

    void SetUp() override {
        const auto cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
        if (cluster_desc->get_number_of_chips() < kMeshDevices) {
            GTEST_SKIP() << "Need at least " << kMeshDevices << " chips for this test";
        }
        ttml::ttnn_fixed::distributed::enable_fabric(kMeshDevices);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(kMeshRows, kMeshCols));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }

    // Read REPRO_W_TILES_PER_SHARD from the environment, defaulting to a
    // known-passing value if unset. All tests use this so a single sweep step
    // exercises controls + parametric with the same per-device width.
    static uint32_t w_tiles_per_shard() {
        const char* w_env = std::getenv("REPRO_W_TILES_PER_SHARD");
        if (w_env == nullptr) {
            return kDefaultWTilesPerShard;
        }
        const auto w = static_cast<uint32_t>(std::stoul(w_env));
        return (w == 0U) ? kDefaultWTilesPerShard : w;
    }

    // Synchronize then print, so we know exactly which dispatched op has finished.
    static void sync_mesh(ttnn::distributed::MeshDevice* device, const char* tag) {
        std::fprintf(stderr, "[repro] sync BEFORE %s\n", tag);
        std::fflush(stderr);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>{});
        std::fprintf(stderr, "[repro] sync AFTER  %s\n", tag);
        std::fflush(stderr);
    }
};

// ─── Control 1 ───────────────────────────────────────────────────────────────
// COL_B broadcast (rhs W=1) but NO output_dtype override → keeps BF16 throughout.
// This is the path the production code uses when not in diagnostic mode and is
// expected to PASS at all W (BF16 path doesn't have the typecast injection).
TEST_F(SubtractFp32ColBBcastTest, ColBBroadcast_DefaultDtype_NoHang) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t W = w_tiles_per_shard();
    const uint32_t B = 2U, S = 32U, V = W * kTileW;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, 1U});

    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, replicate.get());
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE, replicate.get());

    std::fprintf(stderr, "[repro] replicated default-dtype W_tiles=%u (V=%u)\n", W, V);
    std::fflush(stderr);

    sync_mesh(device, "before subtract (default dtype, COL_B bcast)");
    auto out = ttnn::subtract(lhs, rhs);  // no output_dtype override
    std::fprintf(stderr, "[repro] dispatch returned (default dtype)\n");
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (default dtype, COL_B bcast)");

    EXPECT_EQ(out.dtype(), ttnn::DataType::BFLOAT16);
}

// ─── Control 2 ───────────────────────────────────────────────────────────────
// FP32 output but NO broadcast (rhs has same shape as lhs). If this passes and
// the broadcast test below hangs, the bug is specifically in the FP32 + COL_B
// broadcast combination, not in the FP32 output itself.
TEST_F(SubtractFp32ColBBcastTest, NoBroadcast_Fp32Output_NoHang) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t W = w_tiles_per_shard();
    const uint32_t B = 2U, S = 32U, V = W * kTileW;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, V});

    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, replicate.get());
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE, replicate.get());

    std::fprintf(stderr, "[repro] replicated FP32-no-bcast W_tiles=%u (V=%u)\n", W, V);
    std::fflush(stderr);

    sync_mesh(device, "before subtract (FP32 out, no bcast)");
    auto out = ttnn::subtract(lhs, rhs, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro] dispatch returned (FP32 out, no bcast)\n");
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (FP32 out, no bcast)");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}

// ─── Replicated FP32 + COL_B broadcast ──────────────────────────────────────
// COL_B broadcast (rhs W=1) AND FP32 output override, BOTH operands replicated.
// Originally written as the suspected hang case; at W=4 it empirically PASSES,
// so the bug needs more than just dtype + broadcast. Parametrizing by W lets us
// check whether the sharded test's non-monotonic hang pattern (3,5,7,8 …) also
// appears with replicated tensors. If this test starts hanging at the same Ws
// as the sharded one, sharding isn't actually a precondition.
TEST_F(SubtractFp32ColBBcastTest, ColBBroadcast_Fp32Output_ReplicatedLhs_NoHang) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t W = w_tiles_per_shard();
    const uint32_t B = 2U, S = 32U, V = W * kTileW;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, 1U});

    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, replicate.get());
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE, replicate.get());

    std::fprintf(stderr, "[repro] replicated FP32+COL_B W_tiles=%u (V=%u)\n", W, V);
    std::fflush(stderr);

    sync_mesh(device, "before subtract (FP32 out, COL_B bcast, replicated lhs)");
    auto out = ttnn::subtract(lhs, rhs, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro] dispatch returned (FP32 out, COL_B bcast, replicated lhs)\n");
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (FP32 out, COL_B bcast, replicated lhs)");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}

// ─── Canonical hang repro / W sweep harness ─────────────────────────────────
// Sharded LHS (TP) + FP32 output + COL_B broadcast. The number of LHS tiles
// per shard along the innermost dim (W) is tunable via the env var
// `REPRO_W_TILES_PER_SHARD`. Default = 2 — a known-passing value so the test
// is a no-op smoke check in CI. Set `REPRO_W_TILES_PER_SHARD=3` (or 5, 7, 8…)
// to exercise the hang.
//
// Sweep example (this is what produced the table in the file header):
//
//   for w in 1 2 3 4 5 6 7 8; do
//       tt-smi -r
//       echo "== W_tiles=$w =="
//       timeout 60 env REPRO_W_TILES_PER_SHARD=$w \
//           TT_MESH_GRAPH_DESC_PATH=... \
//           ../build_Release/tt-train/tests/ttml_tests \
//           --gtest_filter='SubtractFp32ColBBcastTest.ColBBroadcast_Fp32Output_ShardedLhs_Hangs'
//       echo "== W_tiles=$w exit=$? =="
//   done
//
// Exit 0 → kernel completed (PASS). Exit 124 → timed out (HANG).
//
// NOTE: a hang leaves the cluster in a wedged state — every subsequent
// Synchronize on the same chips will block on the upload sync (pre-op) rather
// than the post-op sync. Always `tt-smi -r` between probes if you care about
// the result.
TEST_F(SubtractFp32ColBBcastTest, ColBBroadcast_Fp32Output_ShardedLhs_Hangs) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t W = w_tiles_per_shard();
    const uint32_t B = 5U, S = 256U;
    const uint32_t V_per_shard = W * kTileW;
    const uint32_t V_total = V_per_shard * kMeshDevices;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V_total});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, 1U});

    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/3);
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, shard.get());
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE, replicate.get());

    std::fprintf(
        stderr,
        "[repro] sharded lhs.shape=%s rhs.shape=%s W_tiles_per_shard=%u (V_per_shard=%u)\n",
        fmt::format("{}", lhs.logical_shape()).c_str(),
        fmt::format("{}", rhs.logical_shape()).c_str(),
        W,
        V_per_shard);
    std::fflush(stderr);

    sync_mesh(device, "before subtract (FP32 out, COL_B bcast, sharded)");
    auto out = ttnn::subtract(lhs, rhs, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro] dispatch returned (W_tiles=%u)\n", W);
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (FP32 out, COL_B bcast, sharded)");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}
