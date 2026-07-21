// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// MLA KV assemble op-level fusion benchmark
//
// Uses Google Benchmark to measure average wall-clock time of the DeepSeek MLA KV assemble, forward
// and backward, each for two paths:
//   composite — split_heads (reshape + transpose) + slice + concat + k_pe broadcast (and its reverse)
//   fused     — the single ttml::metal::mla_kv_assemble_fw / mla_kv_assemble_bw data-movement op
//
// Each measured sample enqueues several assembles and synchronizes once at the end to amortize
// dispatch latency. Results are printed as two tables (forward, backward).
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "benchmark_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> batch_sizes = {16, 32};
    std::vector<uint32_t> sequence_lengths = {128, 256, 512};
    uint32_t num_warmup = 5;
    uint32_t num_measure = 20;
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
                cases.push_back(BenchmarkCase{
                    .model_index = model_index, .batch_size = batch_size, .sequence_length = sequence_length});
            }
        }
    }
    return cases;
}

// Composite assemble: the pre-fusion MLA KV path (split_heads + slice + concat + k_pe broadcast).
std::tuple<ttnn::Tensor, ttnn::Tensor> composite_assemble(
    const ttnn::Tensor& kv_up, const ttnn::Tensor& k_pe, uint32_t batch, uint32_t seq_len, const ModelShape& shape) {
    const uint32_t kv_w = shape.qk_nope_dim + shape.v_dim;

    // split_heads: [B, 1, S, H*W] -> [B, S, H, W] -> [B, H, S, W]
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
    return {k, v};
}

// Composite backward: the pre-fusion gradient path (reverse head-split + slice/concat + head-axis sum).
std::tuple<ttnn::Tensor, ttnn::Tensor> composite_assemble_bw(
    const ttnn::Tensor& dK, const ttnn::Tensor& dV, uint32_t batch, uint32_t seq_len, const ModelShape& shape) {
    const uint32_t H = shape.n_heads;
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const uint32_t kv_w = shape.qk_nope_dim + shape.v_dim;
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};

    // dkv_up: concat [dK_nope | dV] per head, then reverse head-split.
    const auto dk_nope = ttnn::slice(
        dK,
        ttsl::SmallVector<uint32_t>{0U, 0U, 0U, 0U},
        ttsl::SmallVector<uint32_t>{batch, H, seq_len, shape.qk_nope_dim},
        step);
    const auto kv_head = ttnn::concat(std::vector<ttnn::Tensor>{dk_nope, dV}, /*dim=*/3);
    const auto dkv_up = ttnn::reshape(ttnn::transpose(kv_head, 1, 2), ttnn::Shape({batch, 1U, seq_len, H * kv_w}));

    // dk_pe: sum dK's rope suffix over the head axis -> [B, 1, S, qk_rope].
    const auto dk_rope = ttnn::slice(
        dK,
        ttsl::SmallVector<uint32_t>{0U, 0U, 0U, shape.qk_nope_dim},
        ttsl::SmallVector<uint32_t>{batch, H, seq_len, qk_head},
        step);
    const auto dk_pe =
        ttnn::sum(dk_rope, /*dim=*/1, /*keepdim=*/true, std::nullopt, ttml::core::ComputeKernelConfig::precise());
    return {dkv_up, dk_pe};
}

ttnn::Tensor make_input(uint32_t batch, uint32_t seq_len, uint32_t width, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t count = static_cast<size_t>(batch) * seq_len * width;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    const ttnn::Shape shape({batch, 1U, seq_len, width});
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(host, shape, device, ttnn::Layout::TILE);
}

// Head-major [B, H, S, W] input for the backward grads dK / dV.
ttnn::Tensor make_input_hm(uint32_t batch, uint32_t n_heads, uint32_t seq_len, uint32_t width, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t count = static_cast<size_t>(batch) * n_heads * seq_len * width;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    const ttnn::Shape shape({batch, n_heads, seq_len, width});
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(host, shape, device, ttnn::Layout::TILE);
}

double run_single(const ModelShape& shape, const SweepConfig& cfg, uint32_t batch, uint32_t seq_len, bool use_fused) {
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const auto kv_up = make_input(batch, seq_len, shape.n_heads * (shape.qk_nope_dim + shape.v_dim), 2002U);
    const auto k_pe = make_input(batch, seq_len, shape.qk_rope_dim, 3003U);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    const auto run_step = [&]() {
        if (use_fused) {
            auto out = ttml::metal::mla_kv_assemble_fw(
                kv_up, k_pe, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);
            benchmark::DoNotOptimize(out);
        } else {
            auto out = composite_assemble(kv_up, k_pe, batch, seq_len, shape);
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

double run_single_bw(
    const ModelShape& shape, const SweepConfig& cfg, uint32_t batch, uint32_t seq_len, bool use_fused) {
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const auto dK = make_input_hm(batch, shape.n_heads, seq_len, qk_head, 5005U);
    const auto dV = make_input_hm(batch, shape.n_heads, seq_len, shape.v_dim, 6006U);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    const auto run_step = [&]() {
        if (use_fused) {
            auto out = ttml::metal::mla_kv_assemble_bw(
                dK, dV, shape.n_heads, shape.qk_nope_dim, shape.qk_rope_dim, shape.v_dim);
            benchmark::DoNotOptimize(out);
        } else {
            auto out = composite_assemble_bw(dK, dV, batch, seq_len, shape);
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
    double bw_composite_ms = 0.0;
    double bw_fused_ms = 0.0;
};

void print_table(const std::string& title, const std::vector<RowSummary>& rows, bool backward) {
    fmt::print("\n");
    fmt::print(
        "{}\n"
        "| Model        | B     | S     | H     | Composite ms | Fused ms | Saved ms | Speedup | Reduction |\n",
        title);
    fmt::print("|--------------|-------|-------|-------|--------------|----------|----------|---------|-----------|\n");
    for (const auto& row : rows) {
        const double composite_ms = backward ? row.bw_composite_ms : row.composite_ms;
        const double fused_ms = backward ? row.bw_fused_ms : row.fused_ms;
        const double saved_ms = composite_ms - fused_ms;
        fmt::print(
            "| {:<12} | {:>5} | {:>5} | {:>5} | {:>12.4f} | {:>8.4f} | {:>8.4f} | {:>6.2f}x | {:>8.2f}% |\n",
            row.model_name,
            row.batch_size,
            row.sequence_length,
            row.n_heads,
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

void BM_MLAKVAssemble(benchmark::State& state) {
    const auto case_index = static_cast<size_t>(state.range(0));
    const auto& bench_case = g_cases.at(case_index);
    const auto& model = all_models().at(bench_case.model_index);
    const uint32_t batch_size = bench_case.batch_size;
    const uint32_t sequence_length = bench_case.sequence_length;

    for ([[maybe_unused]] auto _ : state) {
        const double composite_ms = run_single(model, g_sweep_cfg, batch_size, sequence_length, /*use_fused=*/false);
        const double fused_ms = run_single(model, g_sweep_cfg, batch_size, sequence_length, /*use_fused=*/true);
        const double bw_composite_ms =
            run_single_bw(model, g_sweep_cfg, batch_size, sequence_length, /*use_fused=*/false);
        const double bw_fused_ms = run_single_bw(model, g_sweep_cfg, batch_size, sequence_length, /*use_fused=*/true);

        g_rows.push_back(RowSummary{
            .model_name = model.name,
            .batch_size = batch_size,
            .sequence_length = sequence_length,
            .n_heads = model.n_heads,
            .composite_ms = composite_ms,
            .fused_ms = fused_ms,
            .bw_composite_ms = bw_composite_ms,
            .bw_fused_ms = bw_fused_ms,
        });

        state.SetIterationTime(fused_ms / 1000.0);
        state.SetLabel(fmt::format("{} B={} S={} H={}", model.name, batch_size, sequence_length, model.n_heads));
        state.counters["Composite_ms"] = composite_ms;
        state.counters["Fused_ms"] = fused_ms;
        state.counters["BW_composite_ms"] = bw_composite_ms;
        state.counters["BW_fused_ms"] = bw_fused_ms;
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

        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);

        fmt::print("MLA QKV assemble op-level benchmark (composite vs fused, forward + backward)\n");
        fmt::print(
            "preset models=deepseek-mla,nano-mla batches=16,32 seq_lens=128,256,512 warmup={} measure={}\n",
            g_sweep_cfg.num_warmup,
            g_sweep_cfg.num_measure);
        fmt::print(
            "Composite path: split_heads (reshape + transpose) + slice + concat + k_pe broadcast.\n"
            "Each measured sample enqueues multiple assembles and synchronizes once at the end.\n");

        benchmark::RegisterBenchmark("MLAKVAssemble", BM_MLAKVAssemble)
            ->DenseRange(0, static_cast<int>(g_cases.size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();

        print_table("MLA QKV assemble forward (composite vs fused)", g_rows, /*backward=*/false);
        print_table("MLA QKV assemble backward (composite vs fused)", g_rows, /*backward=*/true);

        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "mla_kv_assemble_benchmark failed: {}\n", e.what());
        return 1;
    }
}
