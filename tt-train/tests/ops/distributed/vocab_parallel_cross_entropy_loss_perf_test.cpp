// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Performance / profiling harness for vocab_parallel_cross_entropy_loss vs the
// standard cross_entropy_loss at production-scale shapes.
//
// These tests are DISABLED by default — they require an 8-device mesh
// (T3K / 1×8 galaxy partition) and are intended to be run manually under tracy:
//
//   TT_METAL_PROFILER_MID_RUN_DUMP=1 python -m tracy -r -v -p \
//     "build/tt-train/tests/ttml_tests \
//      --gtest_also_run_disabled_tests \
//      --gtest_filter=*VocabParallelCETrainStepLoop*"
//
// Replace the filter with *StandardCETrainStepLoop* to profile the baseline.
//
// Shapes match configs/training_configs/training_shakespeare_tinyllama_tp_galaxy_new.yaml
// + configs/model_configs/tinyllama_bpe.yaml:
//   B = 1, S = 2048 (= max_sequence_length), V = 32000, mesh = [1, 8]

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <random>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/distributed/losses.hpp"
#include "ops/losses.hpp"
#include "tools/profiler/op_profiler_serialize.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace {

constexpr uint32_t kTpSize = 8U;  // mesh_shape: [1, 8] (matches t3k_mesh_graph_descriptor.textproto)

// Production shapes — match training_shakespeare_tinyllama_tp_galaxy_new.yaml +
// configs/model_configs/tinyllama_bpe.yaml.  Per-device local vocab = 4000 (tile-aligned).
// kProdSeqLen pinned to TinyLlama's max_sequence_length (2048) to capture the
// worst-case loss cost the production training step pays.
constexpr uint32_t kProdBatchSize = 1U;
constexpr uint32_t kProdSeqLen = 2048U;
constexpr uint32_t kProdVocabSize = 32000U;

// Smoke shapes — same code path, tiny tiles.  Used to verify the op chain
// completes on 1×8 before paying the cold-compile cost at production size.
constexpr uint32_t kSmokeBatchSize = 1U;
constexpr uint32_t kSmokeSeqLen = 32U;
constexpr uint32_t kSmokeVocabSize = 256U;  // local 32 = one tile per device

constexpr uint32_t kIterations = 5U;
constexpr uint32_t kSmokeIterations = 2U;

// Use UMD directly — calling tt_metal::GetNumAvailableDevices() implicitly initializes
// MetalContext, which would lock in auto-discovery topology *before* enable_fabric has
// a chance to set TT_MESH_GRAPH_DESC_PATH.  The MGD path is what the production binary
// uses (1×8 line topology); without it CCL takes a different physical-mesh code path
// and has been observed to wedge inside the composite all-reduce semaphore init.
[[nodiscard]] bool has_enough_devices_for_t3k() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_number_of_chips() >= kTpSize;
}

}  // namespace

class ShardedCrossEntropyLossPerfT3KTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!has_enough_devices_for_t3k()) {
            GTEST_SKIP() << "Skipping perf test: requires >= " << kTpSize << " devices.";
        }
        // IMPORTANT: enable_fabric must run before anything that initializes MetalContext.
        // It sets TT_MESH_GRAPH_DESC_PATH, which is read once during MetalContext init;
        // calling it after the first MetalContext touch silently leaves the process on the
        // auto-discovery topology and causes CCL deadlocks on 1×8 t3k.
        ttml::ttnn_fixed::distributed::enable_fabric(kTpSize);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, kTpSize));
        ttml::autograd::ctx().set_seed(42);
        // Mirror nano_gpt main.cpp: when TP is enabled the binary always initializes the
        // parallelism context before any model dispatch.
        ttml::autograd::ctx().initialize_parallelism_context({.enable_tp = true});
    }

    void TearDown() override {
        if (has_enough_devices_for_t3k()) {
            ttml::autograd::ctx().close_device();
        }
    }
};

namespace {

// Generate the same `(logits, targets)` pair used by both perf tests so the
// two runs are directly comparable on identical data.
struct HostInputs {
    xt::xarray<float> logits;
    xt::xarray<uint32_t> targets;
};

[[nodiscard]] HostInputs make_host_inputs(uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size) {
    HostInputs inputs;
    inputs.logits = xt::empty<float>({batch_size, 1U, seq_len, vocab_size});
    inputs.targets = xt::zeros<uint32_t>({batch_size, seq_len});

    std::mt19937 gen(0xC0FFEEU);
    std::uniform_real_distribution<float> dist(-3.0F, 3.0F);
    for (auto& v : inputs.logits) {
        v = dist(gen);
    }

    std::uniform_int_distribution<uint32_t> idx_dist(0U, vocab_size - 1U);
    for (uint32_t b = 0U; b < batch_size; ++b) {
        for (uint32_t s = 0U; s < seq_len; ++s) {
            inputs.targets(b, s) = idx_dist(gen);
        }
    }
    return inputs;
}

[[nodiscard]] double elapsed_ms_since(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now() - t0).count();
}

// Emit a Tracy signpost in the exact framing that tools/tracy/process_ops_logs.py
// scans for (it greps the captured Tracy stream for `TT_SIGNPOST` and uses each
// hit as a boundary in the generated ops CSV).  Surrounding backticks match the
// Python `tracy.signpost` helper so the CSV exporter doesn't break the row.
void perf_signpost(std::string_view label) {
    tt::tt_metal::op_profiler::tracy_message(fmt::format("`TT_SIGNPOST: {}`", label));
}

void run_vocab_parallel_loop(uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size, uint32_t iterations) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();

    fmt::println(
        "[perf-test] vocab-parallel CE: B={} S={} V={} iters={} (V/tp_size={})",
        batch_size,
        seq_len,
        vocab_size,
        iterations,
        vocab_size / kTpSize);

    const auto host = make_host_inputs(batch_size, seq_len, vocab_size);

    // 2D placement layout matching the production TP layout (mesh_shape=[1,8]):
    //   axis 0 (DP / replicate-only on a 1×N mesh) -> Replicate
    //   axis 1 (TP, cluster_axis=1)                -> Shard along dim 3 (vocab)
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;

    ttsl::SmallVector<Placement> tp_logits_placements{Replicate{}, Shard{3}};
    ttsl::SmallVector<Placement> tp_targets_placements{Replicate{}, Replicate{}};
    auto tp_logits_mapper =
        ttnn::distributed::TensorToMesh::create(*device, {.placements = std::move(tp_logits_placements)});
    auto tp_targets_mapper =
        ttnn::distributed::TensorToMesh::create(*device, {.placements = std::move(tp_targets_placements)});
    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);

    auto targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        host.targets, device, ttnn::Layout::ROW_MAJOR, &tp_targets_mapper);
    auto targets_ptr = autograd::create_tensor(targets_dev, false);

    xt::xarray<float> grad_ones = xt::ones<float>({1U, 1U, 1U, 1U});
    auto grad_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        grad_ones, device, ttnn::Layout::TILE, replicate_mapper.get());

    // Warm-up: prior to the model forward the binary issues many TP all-reduces
    // (one per ColumnParallelLinear layer), which causes prim::all_broadcast and
    // its GlobalSemaphores to be created and cached well before the loss runs.
    // The test goes straight to vocab_parallel_cross_entropy_loss on a cold mesh,
    // and the very first composite all-reduce has been observed to wedge on the
    // GlobalSemaphore-init mesh write.  Pre-issue one all_reduce on a representative
    // shape to warm the workload+semaphore caches before timing begins.
    {
        fmt::println("[perf-test] warm-up: dispatching one all_reduce to seed CCL caches");
        const auto t_warm = std::chrono::steady_clock::now();
        auto warm_logits_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            host.logits, device, ttnn::Layout::TILE, &tp_logits_mapper);
        auto sum_per_pos = ttnn::sum(warm_logits_dev, 3, /*keepdim=*/true);
        (void)ttnn_fixed::distributed::all_reduce(sum_per_pos, /*cluster_axis=*/1U);
        autograd::ctx().reset_graph();
        fmt::println("[perf-test] warm-up done in {:.1f} ms", elapsed_ms_since(t_warm));
    }

    // Upload logits once and reuse the device tensor across iterations so the
    // tilize / memcpy kernels from from_xtensor don't pollute the per-iteration
    // signpost slice.  Data values are irrelevant for perf timings, and
    // reset_graph() only clears autograd state, not the underlying ttnn tensor.
    auto logits_dev =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(host.logits, device, ttnn::Layout::TILE, &tp_logits_mapper);

    for (uint32_t it = 0U; it < iterations; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        fmt::println("[perf-test] iter {} start", it);

        auto logits_ptr = autograd::create_tensor(logits_dev, true);

        const auto t_fwd_start = std::chrono::steady_clock::now();
        perf_signpost(fmt::format("vp_ce_fwd_begin iter={}", it));
        auto loss = ops::distributed::vocab_parallel_cross_entropy_loss(logits_ptr, targets_ptr, /*cluster_axis=*/1U);
        perf_signpost(fmt::format("vp_ce_fwd_end iter={}", it));
        const auto t_fwd_done = std::chrono::steady_clock::now();
        fmt::println("[perf-test] iter {} forward dispatched in {:.1f} ms", it, elapsed_ms_since(t_fwd_start));

        loss->set_grad(grad_dev);
        perf_signpost(fmt::format("vp_ce_bwd_begin iter={}", it));
        loss->backward();
        perf_signpost(fmt::format("vp_ce_bwd_end iter={}", it));
        const auto t_bwd_done = std::chrono::steady_clock::now();
        fmt::println(
            "[perf-test] iter {} backward dispatched in {:.1f} ms",
            it,
            std::chrono::duration<double, std::milli>(t_bwd_done - t_fwd_done).count());

        autograd::ctx().reset_graph();
        fmt::println("[perf-test] iter {} done in {:.1f} ms", it, elapsed_ms_since(t0));
    }
}

void run_standard_ce_loop(uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size, uint32_t iterations) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();

    fmt::println(
        "[perf-test] standard CE (TP-sharded input + all_gather): B={} S={} V={} iters={} (V/tp_size={})",
        batch_size,
        seq_len,
        vocab_size,
        iterations,
        vocab_size / kTpSize);

    const auto host = make_host_inputs(batch_size, seq_len, vocab_size);

    // Mirror the production setup: a TP model with `gather_output=true` on the classifier head
    // produces sharded logits `[B, 1, S, V/tp]` per device, then runs `ops::distributed::all_gather`
    // (dim=last, GradOutputType::REPLICATED) immediately before the loss.  See
    // tt-train/sources/ttml/modules/distributed/linear.cpp:114-118.
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;

    ttsl::SmallVector<Placement> tp_logits_placements{Replicate{}, Shard{3}};
    auto tp_logits_mapper =
        ttnn::distributed::TensorToMesh::create(*device, {.placements = std::move(tp_logits_placements)});
    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);

    auto targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        host.targets, device, ttnn::Layout::ROW_MAJOR, replicate_mapper.get());
    auto targets_ptr = autograd::create_tensor(targets_dev, false);

    xt::xarray<float> grad_ones = xt::ones<float>({1U, 1U, 1U, 1U});
    auto grad_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        grad_ones, device, ttnn::Layout::TILE, replicate_mapper.get());

    // Same warm-up as the VP CE loop: pre-issue an all_reduce on a representative shape so that
    // the very first composite-CCL doesn't pay the GlobalSemaphore-init wedge cost.  Use the full
    // V replicated (representative of what the loss receives post-gather) to seed the right caches.
    {
        fmt::println("[perf-test] warm-up: dispatching one all_reduce to seed CCL caches");
        const auto t_warm = std::chrono::steady_clock::now();
        auto warm_logits_dev = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            host.logits, device, ttnn::Layout::TILE, replicate_mapper.get());
        auto sum_per_pos = ttnn::sum(warm_logits_dev, 3, /*keepdim=*/true);
        (void)ttnn_fixed::distributed::all_reduce(sum_per_pos, /*cluster_axis=*/1U);
        autograd::ctx().reset_graph();
        fmt::println("[perf-test] warm-up done in {:.1f} ms", elapsed_ms_since(t_warm));
    }

    // Upload sharded logits once, reuse across iterations.  Each device holds [B, 1, S, V/tp]
    // in BF16 at this point — the per-iter all_gather inside the signposts is what materializes
    // the full V on every device, just like the production training loop.
    auto logits_dev =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(host.logits, device, ttnn::Layout::TILE, &tp_logits_mapper);

    for (uint32_t it = 0U; it < iterations; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        fmt::println("[perf-test] iter {} start", it);

        auto logits_ptr = autograd::create_tensor(logits_dev, true);

        const auto t_fwd_start = std::chrono::steady_clock::now();
        perf_signpost(fmt::format("std_ce_fwd_begin iter={}", it));
        // Production-shaped path: gather the sharded TP output along the vocab dim, replicate the
        // grad on backward (matching ColumnParallelLinear with gather_output=true).
        auto gathered_logits = ops::distributed::all_gather(
            logits_ptr, /*dim=*/3, /*cluster_axis=*/1U, ops::distributed::GradOutputType::REPLICATED);
        auto loss = ops::cross_entropy_loss(gathered_logits, targets_ptr, ops::ReduceType::MEAN);
        perf_signpost(fmt::format("std_ce_fwd_end iter={}", it));
        const auto t_fwd_done = std::chrono::steady_clock::now();
        fmt::println("[perf-test] iter {} forward dispatched in {:.1f} ms", it, elapsed_ms_since(t_fwd_start));

        loss->set_grad(grad_dev);
        perf_signpost(fmt::format("std_ce_bwd_begin iter={}", it));
        loss->backward();
        perf_signpost(fmt::format("std_ce_bwd_end iter={}", it));
        const auto t_bwd_done = std::chrono::steady_clock::now();
        fmt::println(
            "[perf-test] iter {} backward dispatched in {:.1f} ms",
            it,
            std::chrono::duration<double, std::milli>(t_bwd_done - t_fwd_done).count());

        autograd::ctx().reset_graph();
        fmt::println("[perf-test] iter {} done in {:.1f} ms", it, elapsed_ms_since(t0));
    }
}

}  // namespace

// Smoke test: vocab-parallel CE at tiny shapes on the 1×8 mesh.  Cheap to
// compile and run.  If this hangs, the issue is in the op itself on 1×8 (which
// has not been exercised before — existing tests are 1×2 only); if this passes
// quickly, the production-shape variant below is just compile-bound.
TEST_F(ShardedCrossEntropyLossPerfT3KTest, DISABLED_VocabParallelCESmoke) {
    run_vocab_parallel_loop(kSmokeBatchSize, kSmokeSeqLen, kSmokeVocabSize, kSmokeIterations);
}

// Vocab-parallel CE at production shapes.  Mirrors the call site in
// nano_gpt main.cpp:
//     vocab_parallel_cross_entropy_loss(output, target, /*cluster_axis=*/1U).
TEST_F(ShardedCrossEntropyLossPerfT3KTest, DISABLED_VocabParallelCETrainStepLoop) {
    run_vocab_parallel_loop(kProdBatchSize, kProdSeqLen, kProdVocabSize, kIterations);
}

// Standard (non-vocab-parallel) CE on the same production shapes, logits
// replicated across all 8 devices.  Apples-to-apples baseline: every device
// materializes the full V=32000 vocab and runs the loss redundantly.
TEST_F(ShardedCrossEntropyLossPerfT3KTest, DISABLED_StandardCETrainStepLoop) {
    run_standard_ce_loop(kProdBatchSize, kProdSeqLen, kProdVocabSize, kIterations);
}
