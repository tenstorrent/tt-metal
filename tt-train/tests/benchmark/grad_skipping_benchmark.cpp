// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Frozen-weight gradient skipping benchmark
//
// Uses Google Benchmark to measure the average time and DRAM peak of one forward + synthetic-grad backward step of
// the projections in Qwen3.8-27B's decoder layers, its fused SwiGLU MLP and lm_head, with the weights trainable and
// with them frozen. The LoRA run freezes all of them: q/k/v/o_proj, in_proj_qkv, in_proj_z and out_proj sit under
// LoRA adapters, while in_proj_a/b, the MLP and lm_head carry none. The input always requires grad, as it does in
// every block under the run's block recompute.
//
// Shapes are what one chip computes in the Qwen3.8-27B LoRA fine-tune (umales/qwen38, finetune_govreport.py): TP=4
// with sequence parallelism on a 1x4 mesh, S=8192, batch 1. Column-parallel layers hold a quarter of the output
// features and row-parallel ones a quarter of the input features; sequence parallelism gathers the full sequence
// before the token mixer, the MLP and lm_head, so every input has all 8192 tokens. Every fourth layer is gated full
// attention (24 query and 4 KV heads of 256; q_proj also produces the output gate), the rest are gated DeltaNet linear
// attention (16 key and 48 value heads of 128). Rows are named after the checkpoint's modules; a row whose shape
// several modules share stands for all of them. The DeltaNet recurrence, conv1d and the attention itself are not
// benchmarked.
//
// Both arms run in the same build. Before frozen grads were skipped, the frozen arm computed the weight grads and
// add_grad dropped them; this file only uses ops that exist there too, so building it at such a commit gives the
// "before" numbers for the frozen arm.
//
// Timed regions enqueue several steps and synchronize once at the end.
// Results are printed as time and DRAM change of the frozen arm relative to the trainable one.
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <cstdint>
#include <exception>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "ops/linear_op.hpp"
#include "ops/swiglu_op.hpp"
#include "ttnn/operations/rand/rand.hpp"

namespace {

constexpr uint32_t kNumWarmup = 2U;
constexpr uint32_t kNumMeasure = 5U;
constexpr uint32_t kBatchSize = 1U;
constexpr uint32_t kSequenceLength = 8192U;
constexpr double kBytesPerMb = 1024.0 * 1024.0;

enum class OpKind { Linear, SwiGLU };

// Linear: input [B, 1, S, in_features], weight [1, 1, out_features, in_features], no bias (as in Qwen3.8).
// SwiGLU: input [B, 1, S, in_features]; w1 and w3 [1, 1, out_features, in_features], w2 [1, 1, in_features,
// out_features].
struct OpShape {
    std::string name;
    OpKind kind = OpKind::Linear;
    uint32_t in_features = 0;
    uint32_t out_features = 0;
};

// The comments give each layer's full width; a chip holds a quarter of it.
const std::vector<OpShape>& all_ops() {
    static const std::vector<OpShape> ops = {
        // Full attention layers. q_proj is 2 x 24 x 256 wide (query and output gate), k_proj and v_proj 4 x 256
        // (they share this shape), o_proj takes 24 x 256.
        {"attn.q_proj", OpKind::Linear, 5120U, 3072U},
        {"attn.k_proj", OpKind::Linear, 5120U, 256U},
        {"attn.o_proj", OpKind::Linear, 1536U, 5120U},
        // Linear attention layers. in_proj_qkv is 2 x 16 x 128 + 48 x 128 wide, in_proj_z 48 x 128, in_proj_b and
        // in_proj_a 48 (they share this shape), out_proj takes 48 x 128.
        {"linear.in_proj_qkv", OpKind::Linear, 5120U, 2560U},
        {"linear.in_proj_z", OpKind::Linear, 5120U, 1536U},
        {"linear.in_proj_ba", OpKind::Linear, 5120U, 12U},
        {"linear.out_proj", OpKind::Linear, 1536U, 5120U},
        // Every layer: the fused swiglu op with gate_proj, down_proj and up_proj (intermediate 17408).
        {"mlp.swiglu", OpKind::SwiGLU, 5120U, 4352U},
        // Vocabulary 248320, left sharded for the vocab-parallel loss.
        {"lm_head", OpKind::Linear, 5120U, 62080U},
    };
    return ops;
}

std::vector<ttnn::Shape> weight_shapes(const OpShape& op) {
    const ttnn::Shape in_to_out({1U, 1U, op.out_features, op.in_features});
    if (op.kind == OpKind::Linear) {
        return {in_to_out};
    }
    // w1, w2, w3 in swiglu's argument order.
    return {in_to_out, ttnn::Shape({1U, 1U, op.in_features, op.out_features}), in_to_out};
}

uint32_t output_features(const OpShape& op) {
    return op.kind == OpKind::Linear ? op.out_features : op.in_features;
}

std::string weight_label(const OpShape& op) {
    if (op.kind == OpKind::Linear) {
        return fmt::format("[{}, {}]", op.out_features, op.in_features);
    }
    return fmt::format("w1,w3 [{0}, {1}], w2 [{1}, {0}]", op.out_features, op.in_features);
}

struct ArmResult {
    double step_ms = 0.0;
    double dram_peak_mb = 0.0;
};

struct CaseResult {
    ArmResult trainable;
    ArmResult frozen;
};

ArmResult run_arm(
    const OpShape& op,
    const ttnn::Tensor& input_value,
    const std::vector<ttnn::Tensor>& weight_values,
    const ttnn::Tensor& grad_out_value,
    const bool weights_require_grad) {
    auto* const device = &ttml::autograd::ctx().get_device();

    const auto run_step = [&]() {
        const auto input = ttml::autograd::create_tensor(input_value, /*requires_grad=*/true);
        std::vector<ttml::autograd::TensorPtr> weights;
        weights.reserve(weight_values.size());
        for (const auto& weight_value : weight_values) {
            weights.push_back(ttml::autograd::create_tensor(weight_value, weights_require_grad));
        }
        auto out = op.kind == OpKind::Linear ? ttml::ops::linear_op(input, weights[0], nullptr)
                                             : ttml::ops::swiglu(input, weights[0], weights[1], weights[2]);
        out->set_grad(grad_out_value);
        out->backward();
        ttml::autograd::ctx().reset_graph();
    };

    const std::string name = fmt::format("{}_{}", op.name, weights_require_grad ? "trainable" : "frozen");
    ArmResult result{};
    result.step_ms = ttml::benchmark_utils::time_device_avg_us(*device, kNumWarmup, kNumMeasure, run_step) / 1000.0;
    result.dram_peak_mb = static_cast<double>(ttml::benchmark_utils::capture_dram_peak(name, run_step)) / kBytesPerMb;
    return result;
}

CaseResult run_case(const OpShape& op) {
    const tt::tt_metal::distributed::MeshShape mesh(1, 1);
    ttml::autograd::ctx().open_device(mesh);
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const uint32_t seed = ttml::benchmark_utils::seed_from_name(op.name);
    // Values don't affect the timing. Drawn on device directly in BF16, so lm_head's 318M-element weight never goes
    // through a host vector or ops::randn's FP32 intermediates.
    const auto uniform = [&](const ttnn::Shape& shape, const float from, const float to, const uint32_t shape_seed) {
        return ttnn::rand(
            shape,
            *device,
            ttnn::DataType::BFLOAT16,
            ttnn::Layout::TILE,
            ttnn::DRAM_MEMORY_CONFIG,
            from,
            to,
            shape_seed);
    };
    const auto input_value = uniform(ttnn::Shape({kBatchSize, 1U, kSequenceLength, op.in_features}), -1.0F, 1.0F, seed);
    std::vector<ttnn::Tensor> weight_values;
    for (const auto& shape : weight_shapes(op)) {
        weight_values.push_back(uniform(shape, -0.02F, 0.02F, seed + 1U + static_cast<uint32_t>(weight_values.size())));
    }
    const auto grad_out_value =
        uniform(ttnn::Shape({kBatchSize, 1U, kSequenceLength, output_features(op)}), -1.0F, 1.0F, seed + 4U);
    tt::tt_metal::distributed::Synchronize(*device, std::nullopt);

    CaseResult result{
        .trainable = run_arm(op, input_value, weight_values, grad_out_value, /*weights_require_grad=*/true),
        .frozen = run_arm(op, input_value, weight_values, grad_out_value, /*weights_require_grad=*/false),
    };

    ttml::autograd::ctx().reset_graph();
    tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
    ttml::autograd::ctx().close_device();
    return result;
}

struct RowSummary {
    std::string op_name;
    std::string input_shape;
    std::string weight_shape;
    CaseResult result;
};

void print_table(const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "Frozen vs trainable weights (input requires grad in both), one FW+BW step\n"
        "| Op                 | Input                | Weight                              | Trainable ms | Frozen ms |"
        " Saved ms | Speedup | Reduction | DRAM trainable MB | DRAM frozen MB | DRAM reduction |\n");
    fmt::print(
        "|--------------------|----------------------|-------------------------------------|--------------|-----------|"
        "----------|---------|-----------|-------------------|----------------|----------------|\n");
    for (const auto& row : rows) {
        const auto& trainable = row.result.trainable;
        const auto& frozen = row.result.frozen;
        fmt::print(
            "| {:<18} | {:<20} | {:<35} | {:>12.2f} | {:>9.2f} | {:>8.2f} | {:>6.2f}x | {:>8.2f}% | {:>17.2f} | "
            "{:>14.2f} | {:>13.2f}% |\n",
            row.op_name,
            row.input_shape,
            row.weight_shape,
            trainable.step_ms,
            frozen.step_ms,
            trainable.step_ms - frozen.step_ms,
            ttml::benchmark_utils::speedup_x(trainable.step_ms, frozen.step_ms),
            ttml::benchmark_utils::reduction_pct(trainable.step_ms, frozen.step_ms),
            trainable.dram_peak_mb,
            frozen.dram_peak_mb,
            ttml::benchmark_utils::reduction_pct(trainable.dram_peak_mb, frozen.dram_peak_mb));
    }
}

std::vector<RowSummary> g_rows;

void BM_GradSkipping(benchmark::State& state) {
    const auto& op = all_ops().at(static_cast<size_t>(state.range(0)));

    for ([[maybe_unused]] auto _ : state) {
        const auto result = run_case(op);
        g_rows.push_back(RowSummary{
            .op_name = op.name,
            .input_shape = fmt::format("[{}, 1, {}, {}]", kBatchSize, kSequenceLength, op.in_features),
            .weight_shape = weight_label(op),
            .result = result,
        });

        state.SetIterationTime(result.frozen.step_ms / 1000.0);
        state.SetLabel(fmt::format("{} B={} S={}", op.name, kBatchSize, kSequenceLength));
        state.counters["Trainable_ms"] = result.trainable.step_ms;
        state.counters["Frozen_ms"] = result.frozen.step_ms;
        state.counters["DRAM_trainable_MB"] = result.trainable.dram_peak_mb;
        state.counters["DRAM_frozen_MB"] = result.frozen.dram_peak_mb;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }

        fmt::print(
            "Gradient skipping op-level benchmark (frozen vs trainable weights, per-chip Qwen3.8-27B shapes at "
            "TP=4)\n");
        fmt::print("batch={} seq_len={} warmup={} measure={}\n", kBatchSize, kSequenceLength, kNumWarmup, kNumMeasure);
        fmt::print(
            "Runs op-level timings on BF16 tensors with a synthetic uniform upstream gradient; no loss/optimizer.\n"
            "Each measured sample enqueues multiple FW+BW steps and synchronizes once at the end.\n"
            "DRAM columns are the peak of one captured FW+BW step.\n");

        benchmark::RegisterBenchmark("GradSkipping", BM_GradSkipping)
            ->DenseRange(0, static_cast<int>(all_ops().size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();

        print_table(g_rows);
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "grad_skipping_benchmark failed: {}\n", e.what());
        return 1;
    }
}
