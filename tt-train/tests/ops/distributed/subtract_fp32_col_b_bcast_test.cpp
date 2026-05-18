// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal multi-device reproducer for the hang observed when
// `ttnn::subtract(bf16_lhs, bf16_rhs, output_dtype=FLOAT32)` is invoked with a
// COL_B broadcast (rhs has W=1) in vocab_parallel_cross_entropy_loss.
//
// The bug is documented in `ops/distributed/losses.cpp`:
//   "Requesting an FP32 output here causes binary_ng to auto-inject a
//    TYPECAST(BF16,FP32) into the post_activations chain. Combined with COL_B
//    broadcast (b's W=1) and the multi-device TP setup that
//    vocab_parallel_cross_entropy uses, that path hangs."
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
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

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
// expected to PASS.
TEST_F(SubtractFp32ColBBcastTest, ColBBroadcast_DefaultDtype_NoHang) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 2U, S = 32U, V = 128U;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, 1U});

    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, replicate.get());
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE, replicate.get());

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
// broadcast combination.
TEST_F(SubtractFp32ColBBcastTest, NoBroadcast_Fp32Output_NoHang) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 2U, S = 32U, V = 128U;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, V});

    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, replicate.get());
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE, replicate.get());

    sync_mesh(device, "before subtract (FP32 out, no bcast)");
    auto out = ttnn::subtract(lhs, rhs, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro] dispatch returned (FP32 out, no bcast)\n");
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (FP32 out, no bcast)");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}

// ─── Replicated FP32 + COL_B broadcast ──────────────────────────────────────
// COL_B broadcast (rhs W=1) AND FP32 output override, BOTH operands replicated.
// Originally written as the suspected hang case; empirically PASSES on a 1x4
// p150_x4 mesh, so the bug needs more than just dtype + broadcast.
TEST_F(SubtractFp32ColBBcastTest, ColBBroadcast_Fp32Output_ReplicatedLhs_NoHang) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 2U, S = 32U, V = 128U;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, 1U});

    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, replicate.get());
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE, replicate.get());

    sync_mesh(device, "before subtract (FP32 out, COL_B bcast, replicated lhs)");
    auto out = ttnn::subtract(lhs, rhs, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro] dispatch returned (FP32 out, COL_B bcast, replicated lhs)\n");
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (FP32 out, COL_B bcast, replicated lhs)");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}

// ─── Production-sized shards / sweep harness ─────────────────────────────────
// Same setup as ShardedLhs_Hangs but with W tiles per shard tunable via the
// env var `REPRO_W_TILES_PER_SHARD`. Default = 256 (8192 / 32), which matches
// the TinyLlama TP path post-LM-head and is the known-hanging configuration.
//
// Use to binary-search the smallest W_tiles_per_shard that still hangs:
//
//   tt-smi -r
//   for w in 256 128 64 32 16 8 4 2 1; do
//       echo "== W_tiles=$w =="
//       timeout 60 env REPRO_W_TILES_PER_SHARD=$w \
//           TT_METAL_HOME=... TT_METAL_RUNTIME_ROOT=... TT_MESH_GRAPH_DESC_PATH=... \
//           ../build_Release/tt-train/tests/ttml_tests \
//           --gtest_filter='SubtractFp32ColBBcastTest.ColBBroadcast_Fp32Output_ShardedLhs_BigW_Hangs'
//       rc=$?
//       echo "== W_tiles=$w exit=$rc =="
//       tt-smi -r
//   done
//
// Exit 0 → kernel completed (PASS). Exit 124 → timed out (HANG).
TEST_F(SubtractFp32ColBBcastTest, ColBBroadcast_Fp32Output_ShardedLhs_BigW_Hangs) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    constexpr uint32_t kTileW = 32U;
    constexpr uint32_t kDefaultWTilesPerShard = 256U;
    const char* w_env = std::getenv("REPRO_W_TILES_PER_SHARD");
    const uint32_t W_tiles_per_shard =
        (w_env != nullptr) ? static_cast<uint32_t>(std::stoul(w_env)) : kDefaultWTilesPerShard;
    ASSERT_GE(W_tiles_per_shard, 1U);

    const uint32_t B = 5U, S = 256U;
    const uint32_t V_per_shard = W_tiles_per_shard * kTileW;
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
        W_tiles_per_shard,
        V_per_shard);
    std::fflush(stderr);

    sync_mesh(device, "before subtract (FP32 out, COL_B bcast, sharded)");
    auto out = ttnn::subtract(lhs, rhs, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro] dispatch returned (W_tiles=%u)\n", W_tiles_per_shard);
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (FP32 out, COL_B bcast, sharded)");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}

// ─── Production-faithful rhs (built from all_gather + max) ───────────────────
// In production `global_max` is the output of `max → all_gather → max` over
// the sharded logits, not a freshly created TILE tensor. The collective leaves
// the tensor with a specific MemoryConfig that may matter to binary_ng's
// kernel-selection logic. This test rebuilds rhs that way.
TEST_F(SubtractFp32ColBBcastTest, ColBBroadcast_Fp32Output_RhsFromAllGather_Hangs) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 2U, S = 32U;
    const uint32_t V_per_shard = 32U;
    const uint32_t V_total = V_per_shard * kMeshDevices;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V_total});

    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/3);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, shard.get());

    // Mirror losses.cpp steps 1+2: local max → all_gather → global max.
    auto local_max = ttnn::max(lhs, 3, /*keepdim=*/true);
    auto all_max_val = ttml::ttnn_fixed::distributed::all_gather(local_max, /*dim=*/3, /*cluster_axis=*/std::nullopt);
    auto global_max = ttnn::max(all_max_val, 3, /*keepdim=*/true);

    std::fprintf(
        stderr,
        "[repro] all-gather rhs lhs.shape=%s global_max.shape=%s\n",
        fmt::format("{}", lhs.logical_shape()).c_str(),
        fmt::format("{}", global_max.logical_shape()).c_str());
    std::fflush(stderr);

    sync_mesh(device, "before subtract (FP32 out, COL_B bcast, rhs=allgather+max) [EXPECTED HANG CASE]");
    auto out = ttnn::subtract(lhs, global_max, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro] dispatch returned (rhs=allgather+max)\n");
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (FP32 out, COL_B bcast, rhs=allgather+max) [EXPECTED HANG CASE]");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}

// ─── The original hang case (matches production) ────────────────────────────
// Same as the previous test but with `lhs` SHARDED on dim 3 across the TP mesh
// — i.e. exactly what `vocab_parallel_cross_entropy_loss` does with the logits.
// This is the configuration that hangs the production training run.
//
// EXPECTATION (before fix): host print "[repro] dispatch returned (sharded
// lhs)" appears, then "sync BEFORE after subtract (sharded lhs)" appears, then
// `Synchronize` never returns — the device kernel hangs. Use a timeout when
// running this test:
//   timeout 60 ../build_Release/tt-train/tests/ttml_tests \
//       --gtest_filter='SubtractFp32ColBBcastTest.ColBBroadcast_Fp32Output_ShardedLhs_Hangs'
TEST_F(SubtractFp32ColBBcastTest, ColBBroadcast_Fp32Output_ShardedLhs_Hangs) {
    SKIP_FOR_WATCHER();
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    const uint32_t B = 2U, S = 32U;
    // V_total must be a multiple of kMeshDevices AND divisible into tile-aligned
    // shards (32 cols each here) — same arithmetic as the production TP loss path.
    const uint32_t V_per_shard = 32U;
    const uint32_t V_total = V_per_shard * kMeshDevices;

    xt::xarray<float> lhs_xt = xt::ones<float>({B, 1U, S, V_total});
    xt::xarray<float> rhs_xt = xt::ones<float>({B, 1U, S, 1U});

    // Match production: shard lhs on dim 3 (vocab), replicate rhs (global max).
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/3);
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto lhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(lhs_xt, device, ttnn::Layout::TILE, shard.get());
    auto rhs = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(rhs_xt, device, ttnn::Layout::TILE, replicate.get());

    std::fprintf(
        stderr,
        "[repro] sharded lhs.shape=%s rhs.shape=%s\n",
        fmt::format("{}", lhs.logical_shape()).c_str(),
        fmt::format("{}", rhs.logical_shape()).c_str());
    std::fflush(stderr);

    sync_mesh(device, "before subtract (FP32 out, COL_B bcast, sharded lhs) [EXPECTED HANG CASE]");
    auto out = ttnn::subtract(lhs, rhs, ttnn::DataType::FLOAT32);
    std::fprintf(stderr, "[repro] dispatch returned (sharded lhs)\n");
    std::fflush(stderr);
    sync_mesh(device, "after  subtract (FP32 out, COL_B bcast, sharded lhs) [EXPECTED HANG CASE]");

    EXPECT_EQ(out.dtype(), ttnn::DataType::FLOAT32);
}
