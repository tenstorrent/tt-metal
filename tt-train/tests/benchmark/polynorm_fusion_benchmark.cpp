// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// PolyNorm3 op-level fusion benchmark
//
// Uses Google Benchmark to measure the average wall-clock time and DRAM peak memory
// of a PolyNorm3 forward + synthetic-grad backward step for two model-shaped hidden
// dimensions (tinyllama, undisclosed-model).
//
// Four configurations are compared:
//   baseline       — composite forward + composite backward
//   fw ON, bw OFF  — fused forward   + composite backward
//   fw OFF, bw ON  — composite forward + fused backward
//   fw ON, bw ON   — fused forward   + fused backward
//
// Timed regions enqueue multiple steps and synchronize once at the end; this amortizes
// dispatch/sync latency and is closer to how model-level runs queue TTNN operations.
// Results are printed as % time and DRAM change relative to the baseline.
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/polynorm_op.hpp"
#include "test_utils/random_data.hpp"
#include "utils/memory_utils.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> batch_sizes = {1, 2, 4, 8, 16};
    uint32_t num_warmup = 2;
    uint32_t num_measure = 5;
};

struct ModelShape {
    std::string name;
    uint32_t embedding_dim;
    uint32_t sequence_length;
};

struct RunResult {
    double step_ms = 0.0;
    double dram_peak_mb = 0.0;
};

struct BenchmarkCase {
    uint32_t model_index = 0;
    uint32_t batch_size = 0;
};

const std::vector<ModelShape>& all_models() {
    static const std::vector<ModelShape> models = {
        // TinyLlama hidden dim and max sequence length.
        {"tinyllama", 5632, 2048},
        // Client model config: embedding_dim=4096, max_sequence_length=4096.
        {"undisclosed-model", 4096, 4096},
    };
    return models;
}

SweepConfig load_sweep_config() {
    return SweepConfig{};
}

std::vector<BenchmarkCase> make_benchmark_cases(const SweepConfig& cfg) {
    std::vector<BenchmarkCase> cases;
    const auto& models = all_models();
    cases.reserve(models.size() * cfg.batch_sizes.size());
    for (uint32_t model_index = 0; model_index < models.size(); ++model_index) {
        for (const auto batch_size : cfg.batch_sizes) {
            cases.push_back(BenchmarkCase{.model_index = model_index, .batch_size = batch_size});
        }
    }
    return cases;
}

RunResult run_single_mode(
    const ModelShape& shape,
    const SweepConfig& cfg,
    const uint32_t batch_size,
    const uint32_t sequence_length,
    const ttml::ops::PolyNorm3ForwardVariant polynorm3_forward_variant,
    const ttml::ops::PolyNorm3BackwardVariant polynorm3_backward_variant,
    const tt::tt_metal::distributed::MeshShape& mesh) {
    ttml::autograd::ctx().open_device(mesh);
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const ttnn::Shape x_shape({batch_size, 1, sequence_length, shape.embedding_dim});
    const ttnn::Shape w_shape({1, 1, 1, 3});
    const ttnn::Shape b_shape({1, 1, 1, 1});
    const size_t x_count = static_cast<size_t>(batch_size) * sequence_length * shape.embedding_dim;
    const auto x_host =
        ttml::test_utils::make_uniform_vector<float>(x_count, -1.0F, 1.0F, 2026U + batch_size + shape.embedding_dim);
    const auto w_host = std::vector<float>{0.2F, 0.3F, 0.5F};
    const auto b_host = std::vector<float>{0.1F};
    constexpr float epsilon = 1e-5F;

    const auto input_value =
        ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(x_host, x_shape, device, ttnn::Layout::TILE);
    const auto weight_value =
        ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(w_host, w_shape, device, ttnn::Layout::TILE);
    const auto bias_value =
        ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(b_host, b_shape, device, ttnn::Layout::TILE);
    const auto grad_out_value = ttml::core::ones_like(input_value);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    const auto make_step_tensors = [&]() {
        const auto input = ttml::autograd::create_tensor(input_value, /*requires_grad=*/true);
        const auto weight = ttml::autograd::create_tensor(weight_value, /*requires_grad=*/true);
        const auto bias = ttml::autograd::create_tensor(bias_value, /*requires_grad=*/true);
        return std::tuple{input, weight, bias};
    };

    const auto run_step = [&]() {
        auto [input, weight, bias] = make_step_tensors();
        auto out =
            ttml::ops::polynorm3(input, weight, bias, epsilon, polynorm3_forward_variant, polynorm3_backward_variant);
        out->set_grad(grad_out_value);
        out->backward();
        ttml::autograd::ctx().reset_graph();
    };

    RunResult result{};

    for (uint32_t step = 0; step < cfg.num_warmup; ++step) {
        run_step();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t step = 0; step < cfg.num_measure; ++step) {
        run_step();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    const auto t1 = std::chrono::high_resolution_clock::now();
    result.step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cfg.num_measure;

    {
        auto memory_guard = ttml::utils::MemoryUsageTracker::begin_capture();
        (void)run_step();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        ttml::utils::MemoryUsageTracker::snapshot("STEP_CAPTURE");
        const auto dram_usage = ttml::utils::MemoryUsageTracker::get_dram_usage("STEP_CAPTURE");
        constexpr double kBytesPerMb = 1024.0 * 1024.0;
        result.dram_peak_mb = static_cast<double>(dram_usage.peak) / kBytesPerMb;
    }

    ttml::autograd::ctx().reset_graph();
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    ttml::autograd::ctx().close_device();
    return result;
}

struct RowSummary {
    std::string model_name;
    uint32_t batch_size = 0;
    uint32_t sequence_length = 0;
    uint32_t embedding_dim = 0;

    double total_composite_ms = 0.0;
    double fw_fused_total_ms = 0.0;
    double bw_fused_total_ms = 0.0;
    double total_fused_ms = 0.0;

    double dram_composite_mb = 0.0;
    double dram_fused_mb = 0.0;
};

void print_forward_table(const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "Forward-fusion impact (backward stays composite)\n"
        "| Model      | B     | S     | C     | Baseline ms | FW fused ms | Saved ms | Speedup | Reduction |\n");
    fmt::print("|------------|-------|-------|-------|--------------|----------|----------|---------|-----------|\n");
    for (const auto& row : rows) {
        const double saved_ms = row.total_composite_ms - row.fw_fused_total_ms;
        fmt::print(
            "| {:<10} | {:>5} | {:>5} | {:>5} | {:>12.2f} | {:>8.2f} | {:>8.2f} | {:>6.2f}x | {:>8.2f}% |\n",
            row.model_name,
            row.batch_size,
            row.sequence_length,
            row.embedding_dim,
            row.total_composite_ms,
            row.fw_fused_total_ms,
            saved_ms,
            ttml::benchmark_utils::speedup_x(row.total_composite_ms, row.fw_fused_total_ms),
            ttml::benchmark_utils::reduction_pct(row.total_composite_ms, row.fw_fused_total_ms));
    }
}

void print_backward_table(const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "Backward-fusion impact (forward stays composite)\n"
        "| Model      | B     | S     | C     | Baseline ms | BW fused ms | Saved ms | Speedup | Reduction |\n");
    fmt::print("|------------|-------|-------|-------|--------------|----------|----------|---------|-----------|\n");
    for (const auto& row : rows) {
        const double saved_ms = row.total_composite_ms - row.bw_fused_total_ms;
        fmt::print(
            "| {:<10} | {:>5} | {:>5} | {:>5} | {:>12.2f} | {:>8.2f} | {:>8.2f} | {:>6.2f}x | {:>8.2f}% |\n",
            row.model_name,
            row.batch_size,
            row.sequence_length,
            row.embedding_dim,
            row.total_composite_ms,
            row.bw_fused_total_ms,
            saved_ms,
            ttml::benchmark_utils::speedup_x(row.total_composite_ms, row.bw_fused_total_ms),
            ttml::benchmark_utils::reduction_pct(row.total_composite_ms, row.bw_fused_total_ms));
    }
}

void print_total_table(const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "Total (FW+BW fused-on-on path) + DRAM\n"
        "| Model      | B     | S     | C     | Composite ms | Fused ms | Saved ms | Speedup | Reduction | "
        "DRAM comp MB | DRAM fused MB | DRAM reduction |\n");
    fmt::print(
        "|------------|-------|-------|-------|--------------|----------|----------|---------|-----------|"
        "--------------|---------------|----------------|\n");
    for (const auto& row : rows) {
        const double saved_ms = row.total_composite_ms - row.total_fused_ms;
        fmt::print(
            "| {:<10} | {:>5} | {:>5} | {:>5} | {:>12.2f} | {:>8.2f} | {:>8.2f} | {:>6.2f}x | {:>8.2f}% | "
            "{:>12.2f} | {:>13.2f} | {:>13.2f}% |\n",
            row.model_name,
            row.batch_size,
            row.sequence_length,
            row.embedding_dim,
            row.total_composite_ms,
            row.total_fused_ms,
            saved_ms,
            ttml::benchmark_utils::speedup_x(row.total_composite_ms, row.total_fused_ms),
            ttml::benchmark_utils::reduction_pct(row.total_composite_ms, row.total_fused_ms),
            row.dram_composite_mb,
            row.dram_fused_mb,
            ttml::benchmark_utils::reduction_pct(row.dram_composite_mb, row.dram_fused_mb));
    }
}

SweepConfig g_sweep_cfg;
std::vector<BenchmarkCase> g_cases;
std::vector<RowSummary> g_rows;

void BM_PolyNormFusion(benchmark::State& state) {
    const auto case_index = static_cast<size_t>(state.range(0));
    const auto& bench_case = g_cases.at(case_index);
    const auto& model = all_models().at(bench_case.model_index);
    const uint32_t batch_size = bench_case.batch_size;
    const uint32_t sequence_length = model.sequence_length;
    const tt::tt_metal::distributed::MeshShape mesh(1, 1);

    for ([[maybe_unused]] auto _ : state) {
        using FV = ttml::ops::PolyNorm3ForwardVariant;
        using BV = ttml::ops::PolyNorm3BackwardVariant;
        const auto baseline = run_single_mode(
            model,
            g_sweep_cfg,
            batch_size,
            sequence_length,
            FV::CompositeComparisonOnly,
            BV::CompositeComparisonOnly,
            mesh);
        const auto fw_on_bw_off = run_single_mode(
            model, g_sweep_cfg, batch_size, sequence_length, FV::Fused, BV::CompositeComparisonOnly, mesh);
        const auto fw_off_bw_on = run_single_mode(
            model, g_sweep_cfg, batch_size, sequence_length, FV::CompositeComparisonOnly, BV::Fused, mesh);
        const auto fw_on_bw_on =
            run_single_mode(model, g_sweep_cfg, batch_size, sequence_length, FV::Fused, BV::Fused, mesh);

        g_rows.push_back(RowSummary{
            .model_name = model.name,
            .batch_size = batch_size,
            .sequence_length = sequence_length,
            .embedding_dim = model.embedding_dim,
            .total_composite_ms = baseline.step_ms,
            .fw_fused_total_ms = fw_on_bw_off.step_ms,
            .bw_fused_total_ms = fw_off_bw_on.step_ms,
            .total_fused_ms = fw_on_bw_on.step_ms,
            .dram_composite_mb = baseline.dram_peak_mb,
            .dram_fused_mb = fw_on_bw_on.dram_peak_mb,
        });

        state.SetIterationTime(fw_on_bw_on.step_ms / 1000.0);
        state.SetLabel(fmt::format("{} B={} S={} C={}", model.name, batch_size, sequence_length, model.embedding_dim));
        state.counters["Baseline_ms"] = baseline.step_ms;
        state.counters["FW_fused_ms"] = fw_on_bw_off.step_ms;
        state.counters["BW_fused_ms"] = fw_off_bw_on.step_ms;
        state.counters["Both_fused_ms"] = fw_on_bw_on.step_ms;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }

        g_sweep_cfg = load_sweep_config();
        g_cases = make_benchmark_cases(g_sweep_cfg);
        if (g_cases.empty()) {
            throw std::invalid_argument("No PolyNorm benchmark cases selected.");
        }

        fmt::print("PolyNorm op-level benchmark (forward/backward fused ON/OFF vs composite)\n");
        fmt::print(
            "preset models=tinyllama,undisclosed-model batches={} warmup={} measure={} "
            "seq_lens=tinyllama:2048,undisclosed-model:4096\n",
            "1,2,4,8,16",
            g_sweep_cfg.num_warmup,
            g_sweep_cfg.num_measure);
        fmt::print(
            "Runs op-level timings on BF16 tensors with synthetic upstream gradient; no loss/optimizer.\n"
            "Each measured sample enqueues multiple FW+BW steps and synchronizes once at the end.\n"
            "Baseline uses composite PolyNorm3 forward and backward.\n");

        benchmark::RegisterBenchmark("PolyNormFusion", BM_PolyNormFusion)
            ->DenseRange(0, static_cast<int>(g_cases.size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();

        print_backward_table(g_rows);
        print_forward_table(g_rows);
        print_total_table(g_rows);
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "polynorm_fusion_benchmark failed: {}\n", e.what());
        return 1;
    }
}
