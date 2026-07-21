// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Q RoPE op-level fusion benchmark
//
// Measures average wall-clock time for Q RoPE + head-split:
//   composite - reshape/transpose + slice + rotary_embedding_llama + concat
//   fused     - ttml::metal::mla_q_rope (packed q_pre -> head-major q_roped)
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "benchmark_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/rope_op.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> batch_sizes = {16, 32};
    std::vector<uint32_t> sequence_lengths = {128, 256, 512};
    uint32_t num_warmup = 5;
    uint32_t num_measure = 20;
};

struct ModelShape {
    std::string name;
    uint32_t n_heads;
    uint32_t qk_nope_dim;
    uint32_t qk_rope_dim;
};

struct BenchmarkCase {
    uint32_t model_index = 0;
    uint32_t batch_size = 0;
    uint32_t sequence_length = 0;
};

const std::vector<ModelShape>& all_models() {
    static const std::vector<ModelShape> models = {
        {"deepseek-mla", 16U, 128U, 64U},
        {"nano-mla", 8U, 64U, 32U},
        {"mla_like", 8U, 128U, 64U},
        {"asym_rope2", 4U, 128U, 64U},
    };
    return models;
}

bool can_benchmark_fused(const ModelShape& shape, uint32_t seq_len) {
    if (shape.qk_nope_dim % 32U != 0U || shape.qk_rope_dim % 32U != 0U) {
        return false;
    }
    if (seq_len % 32U != 0U) {
        return false;
    }
    if (shape.qk_rope_dim > 128U) {
        return false;
    }
    const uint32_t Tr = shape.qk_rope_dim / 32U;
    if (Tr + 1U > 8U) {
        return false;
    }
    return true;
}

SweepConfig load_sweep_config() {
    return SweepConfig{};
}

std::vector<BenchmarkCase> make_benchmark_cases(const SweepConfig& cfg) {
    std::vector<BenchmarkCase> cases;
    const auto& models = all_models();
    cases.reserve(models.size() * cfg.batch_sizes.size() * cfg.sequence_lengths.size());
    for (uint32_t model_index = 0; model_index < models.size(); ++model_index) {
        for (const auto batch_size : cfg.batch_sizes) {
            for (const auto sequence_length : cfg.sequence_lengths) {
                if (!can_benchmark_fused(models[model_index], sequence_length)) {
                    continue;
                }
                cases.push_back(BenchmarkCase{
                    .model_index = model_index, .batch_size = batch_size, .sequence_length = sequence_length});
            }
        }
    }
    return cases;
}

ttnn::Tensor make_q_input(uint32_t batch, uint32_t n_heads, uint32_t seq_len, uint32_t qk_head, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t count = static_cast<size_t>(batch) * n_heads * seq_len * qk_head;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(
        host, ttnn::Shape{batch, 1U, seq_len, n_heads * qk_head}, device, ttnn::Layout::TILE);
}

ttnn::Tensor composite_q_rope(
    const ttnn::Tensor& q_pre,
    const ttml::ops::RotaryEmbeddingParams& params,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim) {
    const auto shape = q_pre.logical_shape();
    const uint32_t B = shape[0];
    const uint32_t S = shape[2];
    const uint32_t qk_head = qk_nope_dim + qk_rope_dim;

    auto q_in = ttnn::transpose(ttnn::reshape(q_pre, ttnn::Shape({B, S, n_heads, qk_head})), 1, 2);

    ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
    auto q_nope = ttnn::slice(
        q_in, ttsl::SmallVector<uint32_t>{0, 0, 0, 0}, ttsl::SmallVector<uint32_t>{B, n_heads, S, qk_nope_dim}, step);
    auto q_pe = ttnn::slice(
        q_in,
        ttsl::SmallVector<uint32_t>{0, 0, 0, qk_nope_dim},
        ttsl::SmallVector<uint32_t>{B, n_heads, S, qk_head},
        step);

    auto q_pe_rot = ttnn::experimental::rotary_embedding_llama(
        q_pe,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        /*is_decode_mode=*/false,
        /*memory_config=*/std::nullopt,
        ttml::core::ComputeKernelConfig::precise());

    return ttnn::concat(std::vector<ttnn::Tensor>{q_nope, q_pe_rot}, /*dim=*/3);
}

double run_single(const ModelShape& shape, const SweepConfig& cfg, uint32_t batch, uint32_t seq_len, bool use_fused) {
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const auto q_pre = make_q_input(batch, shape.n_heads, seq_len, qk_head, 1001U);
    const auto params = ttml::ops::build_rope_params(seq_len, shape.qk_rope_dim, /*theta=*/10000.0F);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    const auto run_step = [&]() {
        if (use_fused) {
            auto out = ttml::metal::mla_q_rope(
                q_pre,
                params.cos_cache,
                params.sin_cache,
                params.trans_mat,
                shape.qk_nope_dim,
                shape.qk_rope_dim,
                /*packed_input=*/true);
            benchmark::DoNotOptimize(out);
        } else {
            auto out = composite_q_rope(q_pre, params, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim);
            benchmark::DoNotOptimize(out);
        }
    };

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

    ttml::autograd::ctx().reset_graph();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / cfg.num_measure;
}

struct RowSummary {
    std::string model_name;
    uint32_t batch_size = 0;
    uint32_t sequence_length = 0;
    uint32_t n_heads = 0;
    uint32_t qk_nope_dim = 0;
    uint32_t qk_rope_dim = 0;
    double composite_ms = 0.0;
    double fused_ms = 0.0;
};

void print_table(const std::string& title, const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "{}\n"
        "| Model        | B     | S     | H     | Nope | Rope | Composite ms | Fused ms | Saved ms | Speedup | "
        "Reduction |\n",
        title);
    fmt::print(
        "|--------------|-------|-------|-------|------|------|--------------|----------|----------|---------|----"
        "-------|\n");
    for (const auto& row : rows) {
        const double composite_ms = row.composite_ms;
        const double fused_ms = row.fused_ms;
        const double saved_ms = composite_ms - fused_ms;
        fmt::print(
            "| {:<12} | {:>5} | {:>5} | {:>5} | {:>4} | {:>4} | {:>12.4f} | {:>8.4f} | {:>8.4f} | {:>6.2f}x | "
            "{:>8.2f}% |\n",
            row.model_name,
            row.batch_size,
            row.sequence_length,
            row.n_heads,
            row.qk_nope_dim,
            row.qk_rope_dim,
            composite_ms,
            fused_ms,
            saved_ms,
            ttml::benchmark_utils::speedup_x(composite_ms, fused_ms),
            ttml::benchmark_utils::reduction_pct(composite_ms, fused_ms));
    }
}

SweepConfig g_sweep_cfg;
std::vector<BenchmarkCase> g_cases;
std::vector<RowSummary> g_rows;

void BM_MLA_QRope(benchmark::State& state) {
    const auto case_index = static_cast<size_t>(state.range(0));
    const auto& bench_case = g_cases.at(case_index);
    const auto& model = all_models().at(bench_case.model_index);
    const uint32_t batch_size = bench_case.batch_size;
    const uint32_t sequence_length = bench_case.sequence_length;

    for ([[maybe_unused]] auto _ : state) {
        const double composite_ms = run_single(model, g_sweep_cfg, batch_size, sequence_length, /*use_fused=*/false);
        const double fused_ms = run_single(model, g_sweep_cfg, batch_size, sequence_length, /*use_fused=*/true);

        g_rows.push_back(RowSummary{
            .model_name = model.name,
            .batch_size = batch_size,
            .sequence_length = sequence_length,
            .n_heads = model.n_heads,
            .qk_nope_dim = model.qk_nope_dim,
            .qk_rope_dim = model.qk_rope_dim,
            .composite_ms = composite_ms,
            .fused_ms = fused_ms,
        });

        state.SetIterationTime(fused_ms / 1000.0);
        state.SetLabel(fmt::format(
            "{} B={} S={} H={} nope={} rope={}",
            model.name,
            batch_size,
            sequence_length,
            model.n_heads,
            model.qk_nope_dim,
            model.qk_rope_dim));
        state.counters["Composite_ms"] = composite_ms;
        state.counters["Fused_ms"] = fused_ms;
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
            throw std::runtime_error("No benchmark cases: check model dims and sweep batch/sequence lengths.");
        }

        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);

        fmt::print("Q RoPE op-level benchmark (composite vs fused)\n");
        fmt::print(
            "preset models=deepseek-mla,nano-mla,mla_like,asym_rope2 batches=16,32 seq_lens=128,256,512 warmup={} "
            "measure={}\n",
            g_sweep_cfg.num_warmup,
            g_sweep_cfg.num_measure);
        fmt::print(
            "Composite: reshape/transpose + slice + rotary_embedding_llama (precise) + concat.\n"
            "Fused: mla_q_rope packed q_pre -> head-major (fp32 dest acc, qk_rope_dim <= 128).\n");

        benchmark::RegisterBenchmark("MLA_QRope", BM_MLA_QRope)
            ->DenseRange(0, static_cast<int>(g_cases.size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();

        print_table("Q RoPE (composite vs fused)", g_rows);

        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "mla_q_rope_benchmark failed: {}\n", e.what());
        return 1;
    }
}
