// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// MLA QKV assemble op-level fusion benchmark
//
// Uses Google Benchmark to measure average wall-clock time of the DeepSeek MLA
// QKV assemble forward for two paths:
//   composite — split_heads (reshape + transpose) + slice + concat + k_pe broadcast
//   fused     — the single ttml::metal::mla_qkv_assemble_fw data-movement op
//
// This op is forward-only (backward lands in a follow-up), so only the forward
// assemble is timed. Timed regions enqueue multiple steps and synchronize once
// at the end to amortize dispatch latency.
//
// Environment variables:
//   TTML_MLA_ASSEMBLE_BENCH_WARMUP   — warmup iterations  (default 5)
//   TTML_MLA_ASSEMBLE_BENCH_MEASURE  — measured iterations (default 20)
//   TTML_MLA_ASSEMBLE_BENCH_BATCHES  — comma-separated batch sizes (default 16,32)
//   TTML_MLA_ASSEMBLE_BENCH_SEQS     — comma-separated sequence lengths (default 128,256,512)
//   TTML_MLA_ASSEMBLE_BENCH_MODELS   — comma-separated model names to include
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> batch_sizes = {16, 32};
    std::vector<uint32_t> sequence_lengths = {128, 256, 512};
    uint32_t num_warmup = 5;
    uint32_t num_measure = 20;
    std::vector<std::string> model_filter;
};

// MLA per-head geometry. All channel dims are multiples of TILE_WIDTH (32).
struct ModelShape {
    std::string name;
    uint32_t n_heads;
    uint32_t qk_nope_dim;
    uint32_t qk_rope_dim;
    uint32_t v_dim;
};

struct BenchmarkCase {
    uint32_t model_index = 0;
    uint32_t batch_size = 0;
    uint32_t sequence_length = 0;
};

const std::vector<ModelShape>& all_models() {
    static const std::vector<ModelShape> models = {
        // DeepSeek-V3 MLA per-head dims (qk_nope=128, qk_rope=64, v=128), reduced head count for N300.
        {"deepseek-mla", 16U, 128U, 64U, 128U},
        // Smaller nano-style head geometry.
        {"nano-mla", 8U, 64U, 32U, 64U},
    };
    return models;
}

double reduction_pct(double baseline, double fused) {
    if (baseline == 0.0) {
        return 0.0;
    }
    return (baseline - fused) / baseline * 100.0;
}

double speedup_x(double baseline, double fused) {
    if (fused == 0.0) {
        return 0.0;
    }
    return baseline / fused;
}

std::string join_u32_csv(const std::vector<uint32_t>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += std::to_string(values[i]);
    }
    return out;
}

std::vector<uint32_t> parse_u32_csv(const std::string& csv) {
    std::vector<uint32_t> out;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        out.push_back(static_cast<uint32_t>(std::stoul(token)));
    }
    return out;
}

std::vector<std::string> parse_string_csv(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        out.push_back(token);
    }
    return out;
}

bool model_is_enabled(const SweepConfig& cfg, const std::string& name) {
    if (cfg.model_filter.empty()) {
        return true;
    }
    return std::find(cfg.model_filter.begin(), cfg.model_filter.end(), name) != cfg.model_filter.end();
}

SweepConfig load_sweep_config_from_env() {
    SweepConfig sweep_cfg{};
    if (const char* env_warmup = std::getenv("TTML_MLA_ASSEMBLE_BENCH_WARMUP")) {
        sweep_cfg.num_warmup = static_cast<uint32_t>(std::stoul(env_warmup));
    }
    if (const char* env_measure = std::getenv("TTML_MLA_ASSEMBLE_BENCH_MEASURE")) {
        sweep_cfg.num_measure = static_cast<uint32_t>(std::stoul(env_measure));
    }
    if (const char* env_batches = std::getenv("TTML_MLA_ASSEMBLE_BENCH_BATCHES")) {
        auto parsed = parse_u32_csv(env_batches);
        if (!parsed.empty()) {
            sweep_cfg.batch_sizes = std::move(parsed);
        }
    }
    if (const char* env_seqs = std::getenv("TTML_MLA_ASSEMBLE_BENCH_SEQS")) {
        auto parsed = parse_u32_csv(env_seqs);
        if (!parsed.empty()) {
            sweep_cfg.sequence_lengths = std::move(parsed);
        }
    }
    if (const char* env_models = std::getenv("TTML_MLA_ASSEMBLE_BENCH_MODELS")) {
        sweep_cfg.model_filter = parse_string_csv(env_models);
    }
    if (sweep_cfg.num_measure == 0U) {
        throw std::invalid_argument("TTML_MLA_ASSEMBLE_BENCH_MEASURE must be greater than zero.");
    }
    return sweep_cfg;
}

std::vector<BenchmarkCase> make_benchmark_cases(const SweepConfig& cfg) {
    std::vector<BenchmarkCase> cases;
    const auto& models = all_models();
    cases.reserve(models.size() * cfg.batch_sizes.size() * cfg.sequence_lengths.size());
    for (uint32_t model_index = 0; model_index < models.size(); ++model_index) {
        if (!model_is_enabled(cfg, models[model_index].name)) {
            continue;
        }
        for (const auto batch_size : cfg.batch_sizes) {
            for (const auto sequence_length : cfg.sequence_lengths) {
                cases.push_back(BenchmarkCase{
                    .model_index = model_index, .batch_size = batch_size, .sequence_length = sequence_length});
            }
        }
    }
    return cases;
}

// Composite assemble: the pre-fusion MLA path (split_heads + slice + concat + k_pe broadcast).
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> composite_assemble(
    const ttnn::Tensor& q_pre,
    const ttnn::Tensor& kv_up,
    const ttnn::Tensor& k_pe,
    uint32_t batch,
    uint32_t seq_len,
    const ModelShape& shape) {
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const uint32_t kv_w = shape.qk_nope_dim + shape.v_dim;

    // split_heads: [B, 1, S, H*W] -> [B, S, H, W] -> [B, H, S, W]
    const auto q = ttnn::transpose(ttnn::reshape(q_pre, ttnn::Shape({batch, seq_len, shape.n_heads, qk_head})), 1, 2);
    const auto kv = ttnn::transpose(ttnn::reshape(kv_up, ttnn::Shape({batch, seq_len, shape.n_heads, kv_w})), 1, 2);

    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    const auto k_nope = ttnn::slice(
        kv,
        ttsl::SmallVector<uint32_t>{0U, 0U, 0U, 0U},
        ttsl::SmallVector<uint32_t>{batch, shape.n_heads, seq_len, shape.qk_nope_dim},
        step);
    const auto v = ttnn::slice(
        kv,
        ttsl::SmallVector<uint32_t>{0U, 0U, 0U, shape.qk_nope_dim},
        ttsl::SmallVector<uint32_t>{batch, shape.n_heads, seq_len, kv_w},
        step);

    // Broadcast shared k_pe [B, 1, S, qk_rope] across heads -> [B, H, S, qk_rope].
    const std::vector<ttnn::Tensor> kpe_copies(shape.n_heads, k_pe);
    const auto k_pe_b = ttnn::concat(kpe_copies, /*dim=*/1);
    const auto k = ttnn::concat(std::vector<ttnn::Tensor>{k_nope, k_pe_b}, /*dim=*/3);
    return {q, k, v};
}

ttnn::Tensor make_input(uint32_t batch, uint32_t seq_len, uint32_t width, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t count = static_cast<size_t>(batch) * seq_len * width;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    const ttnn::Shape shape({batch, 1U, seq_len, width});
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(host, shape, device, ttnn::Layout::TILE);
}

double run_single(const ModelShape& shape, const SweepConfig& cfg, uint32_t batch, uint32_t seq_len, bool use_fused) {
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const auto q_pre = make_input(batch, seq_len, shape.n_heads * qk_head, 1001U);
    const auto kv_up = make_input(batch, seq_len, shape.n_heads * (shape.qk_nope_dim + shape.v_dim), 2002U);
    const auto k_pe = make_input(batch, seq_len, shape.qk_rope_dim, 3003U);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    const auto run_step = [&]() {
        if (use_fused) {
            auto out = ttml::metal::mla_qkv_assemble_fw(
                q_pre, kv_up, k_pe, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);
            benchmark::DoNotOptimize(out);
        } else {
            auto out = composite_assemble(q_pre, kv_up, k_pe, batch, seq_len, shape);
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
    double composite_ms = 0.0;
    double fused_ms = 0.0;
};

void print_table(const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "MLA QKV assemble forward (composite vs fused)\n"
        "| Model        | B     | S     | H     | Composite ms | Fused ms | Saved ms | Speedup | Reduction |\n");
    fmt::print("|--------------|-------|-------|-------|--------------|----------|----------|---------|-----------|\n");
    for (const auto& row : rows) {
        const double saved_ms = row.composite_ms - row.fused_ms;
        fmt::print(
            "| {:<12} | {:>5} | {:>5} | {:>5} | {:>12.4f} | {:>8.4f} | {:>8.4f} | {:>6.2f}x | {:>8.2f}% |\n",
            row.model_name,
            row.batch_size,
            row.sequence_length,
            row.n_heads,
            row.composite_ms,
            row.fused_ms,
            saved_ms,
            speedup_x(row.composite_ms, row.fused_ms),
            reduction_pct(row.composite_ms, row.fused_ms));
    }
}

SweepConfig g_sweep_cfg;
std::vector<BenchmarkCase> g_cases;
std::vector<RowSummary> g_rows;

void BM_MLAQKVAssemble(benchmark::State& state) {
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
            .composite_ms = composite_ms,
            .fused_ms = fused_ms,
        });

        state.SetIterationTime(fused_ms / 1000.0);
        state.SetLabel(fmt::format("{} B={} S={} H={}", model.name, batch_size, sequence_length, model.n_heads));
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

        g_sweep_cfg = load_sweep_config_from_env();
        g_cases = make_benchmark_cases(g_sweep_cfg);
        if (g_cases.empty()) {
            throw std::invalid_argument("No MLA QKV assemble benchmark cases selected.");
        }

        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);

        fmt::print("MLA QKV assemble op-level benchmark (composite vs fused, forward only)\n");
        fmt::print(
            "preset models=deepseek-mla,nano-mla batches={} seq_lens={} warmup={} measure={}\n",
            join_u32_csv(g_sweep_cfg.batch_sizes),
            join_u32_csv(g_sweep_cfg.sequence_lengths),
            g_sweep_cfg.num_warmup,
            g_sweep_cfg.num_measure);
        fmt::print(
            "Composite path: split_heads (reshape + transpose) + slice + concat + k_pe broadcast.\n"
            "Each measured sample enqueues multiple forward assembles and synchronizes once at the end.\n");

        benchmark::RegisterBenchmark("MLAQKVAssemble", BM_MLAQKVAssemble)
            ->DenseRange(0, static_cast<int>(g_cases.size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();

        print_table(g_rows);

        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "mla_qkv_assemble_benchmark failed: {}\n", e.what());
        return 1;
    }
}
