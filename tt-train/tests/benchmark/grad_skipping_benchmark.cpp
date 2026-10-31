// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Frozen-weight gradient skipping benchmark
//
// Uses Google Benchmark to measure the average time and DRAM peak of one forward +
// synthetic-grad backward step of the linear ops in Qwen3.8-27B's decoder layers, plus lm_head,
// with the weight trainable and with it frozen (LoRA base weights).
// The input always requires grad, as it does in every layer above the first.
//
// Shapes come from the Qwen/Qwen3.8-27B checkpoint (hidden 5120, intermediate 17408, vocab 248320)
// at full size, which is what every chip computes under FSDP. Every fourth layer is gated full
// attention (24 query and 4 KV heads of 256; q_proj also produces the output gate), the rest are
// gated DeltaNet linear attention (16 key and 48 value heads of 128). Projection names follow the
// checkpoint, with K and V fused into kv_proj as in ttml's Qwen3. The DeltaNet recurrence, conv1d
// and the attention itself are not benchmarked.
//
// Both arms run in the same build. Before frozen grads were skipped, the frozen arm computed the
// weight grad and add_grad dropped it; this file only uses ops that exist there too, so building
// it at such a commit gives the "before" numbers for the frozen arm.
//
// Timed regions enqueue several steps and synchronize once at the end.
// Results are printed as time and DRAM change of the frozen arm relative to the trainable one.
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "ops/linear_op.hpp"
#include "ttnn/operations/rand/rand.hpp"

namespace {

constexpr uint32_t kNumWarmup = 2U;
constexpr uint32_t kNumMeasure = 5U;
constexpr uint32_t kSequenceLength = 2048U;
constexpr double kBytesPerMb = 1024.0 * 1024.0;

const std::vector<uint32_t> kBatchSizes = {1U, 4U};

// Input [B, 1, S, in_features], weight [1, 1, out_features, in_features], no bias (as in Qwen3).
struct OpShape {
    std::string name;
    uint32_t in_features;
    uint32_t out_features;
};

const std::vector<OpShape>& all_ops() {
    static const std::vector<OpShape> ops = {
        // Full attention layers.
        {"attn.q_proj", 5120U, 12288U},
        {"attn.kv_proj", 5120U, 2048U},
        {"attn.o_proj", 6144U, 5120U},
        // Linear attention layers. in_proj_b and in_proj_a share this shape.
        {"linear.in_proj_qkv", 5120U, 10240U},
        {"linear.in_proj_z", 5120U, 6144U},
        {"linear.in_proj_ba", 5120U, 48U},
        {"linear.out_proj", 6144U, 5120U},
        // Every layer. gate_proj and up_proj share this shape.
        {"mlp.gate_up_proj", 5120U, 17408U},
        {"mlp.down_proj", 17408U, 5120U},
        {"lm_head", 5120U, 248320U},
    };
    return ops;
}

struct BenchmarkCase {
    uint32_t op_index = 0;
    uint32_t batch_size = 0;
};

std::vector<BenchmarkCase> make_benchmark_cases() {
    std::vector<BenchmarkCase> cases;
    const auto& ops = all_ops();
    cases.reserve(ops.size() * kBatchSizes.size());
    for (uint32_t op_index = 0; op_index < ops.size(); ++op_index) {
        for (const auto batch_size : kBatchSizes) {
            cases.push_back(BenchmarkCase{.op_index = op_index, .batch_size = batch_size});
        }
    }
    return cases;
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
    const ttnn::Tensor& weight_value,
    const ttnn::Tensor& grad_out_value,
    const bool weight_requires_grad) {
    auto* const device = &ttml::autograd::ctx().get_device();

    const auto run_step = [&]() {
        const auto input = ttml::autograd::create_tensor(input_value, /*requires_grad=*/true);
        const auto weight = ttml::autograd::create_tensor(weight_value, weight_requires_grad);
        auto out = ttml::ops::linear_op(input, weight, nullptr);
        out->set_grad(grad_out_value);
        out->backward();
        ttml::autograd::ctx().reset_graph();
    };

    const std::string name = fmt::format("{}_{}", op.name, weight_requires_grad ? "trainable" : "frozen");
    ArmResult result{};
    result.step_ms = ttml::benchmark_utils::time_device_avg_us(*device, kNumWarmup, kNumMeasure, run_step) / 1000.0;
    result.dram_peak_mb = static_cast<double>(ttml::benchmark_utils::capture_dram_peak(name, run_step)) / kBytesPerMb;
    return result;
}

CaseResult run_case(const OpShape& op, const uint32_t batch_size) {
    const tt::tt_metal::distributed::MeshShape mesh(1, 1);
    ttml::autograd::ctx().open_device(mesh);
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const uint32_t seed = ttml::benchmark_utils::seed_from_name(op.name) + batch_size;
    const ttnn::Shape input_shape({batch_size, 1U, kSequenceLength, op.in_features});
    const ttnn::Shape output_shape({batch_size, 1U, kSequenceLength, op.out_features});
    const ttnn::Shape weight_shape({1U, 1U, op.out_features, op.in_features});
    // Values don't affect the timing. Drawn on device directly in BF16: lm_head's weight is 778M elements, too many
    // for a host vector or for ops::randn's FP32 intermediates.
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
    const auto input_value = uniform(input_shape, -1.0F, 1.0F, seed);
    const auto weight_value = uniform(weight_shape, -0.02F, 0.02F, seed + 1U);
    const auto grad_out_value = uniform(output_shape, -1.0F, 1.0F, seed + 2U);
    tt::tt_metal::distributed::Synchronize(*device, std::nullopt);

    CaseResult result{
        .trainable = run_arm(op, input_value, weight_value, grad_out_value, /*weight_requires_grad=*/true),
        .frozen = run_arm(op, input_value, weight_value, grad_out_value, /*weight_requires_grad=*/false),
    };

    ttml::autograd::ctx().reset_graph();
    tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
    ttml::autograd::ctx().close_device();
    return result;
}

struct RowSummary {
    std::string op_name;
    uint32_t batch_size = 0;
    std::string input_shape;
    std::string weight_shape;
    CaseResult result;
};

void print_table(const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "Frozen vs trainable weight (input requires grad in both), one FW+BW step\n"
        "| Op                 | B | Input               | Weight            | Trainable ms | Frozen ms | Saved ms |"
        " Speedup | Reduction | DRAM trainable MB | DRAM frozen MB | DRAM reduction |\n");
    fmt::print(
        "|--------------------|---|---------------------|-------------------|--------------|-----------|----------|----"
        "-----|"
        "-----------|-------------------|----------------|----------------|\n");
    for (const auto& row : rows) {
        const auto& trainable = row.result.trainable;
        const auto& frozen = row.result.frozen;
        fmt::print(
            "| {:<18} | {} | {:<19} | {:<17} | {:>12.2f} | {:>9.2f} | {:>8.2f} | {:>6.2f}x | {:>8.2f}% | {:>17.2f} | "
            "{:>14.2f} | {:>13.2f}% |\n",
            row.op_name,
            row.batch_size,
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

std::vector<BenchmarkCase> g_cases;
std::vector<RowSummary> g_rows;

void BM_GradSkipping(benchmark::State& state) {
    const auto& bench_case = g_cases.at(static_cast<size_t>(state.range(0)));
    const auto& op = all_ops().at(bench_case.op_index);
    const uint32_t batch_size = bench_case.batch_size;

    for ([[maybe_unused]] auto _ : state) {
        const auto result = run_case(op, batch_size);
        g_rows.push_back(RowSummary{
            .op_name = op.name,
            .batch_size = batch_size,
            .input_shape = fmt::format("[{}, 1, {}, {}]", batch_size, kSequenceLength, op.in_features),
            .weight_shape = fmt::format("[{}, {}]", op.out_features, op.in_features),
            .result = result,
        });

        state.SetIterationTime(result.frozen.step_ms / 1000.0);
        state.SetLabel(fmt::format("{} B={} S={}", op.name, batch_size, kSequenceLength));
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

        g_cases = make_benchmark_cases();
        if (g_cases.empty()) {
            throw std::invalid_argument("No grad skipping benchmark cases selected.");
        }

        fmt::print("Gradient skipping op-level benchmark (frozen vs trainable weight, Qwen3.8-27B shapes)\n");
        fmt::print("batches=1,4 seq_len={} warmup={} measure={}\n", kSequenceLength, kNumWarmup, kNumMeasure);
        fmt::print(
            "Runs op-level timings on BF16 tensors with a synthetic uniform upstream gradient; no loss/optimizer.\n"
            "Each measured sample enqueues multiple FW+BW steps and synchronizes once at the end.\n"
            "DRAM columns are the peak of one captured FW+BW step.\n");

        benchmark::RegisterBenchmark("GradSkipping", BM_GradSkipping)
            ->DenseRange(0, static_cast<int>(g_cases.size()) - 1, 1)
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
