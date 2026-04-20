// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// PolyNorm3 op-level fusion benchmark
//
// Measures the wall-clock time and DRAM peak memory of a single
// forward + backward pass (PolyNorm3 forward + synthetic-grad backward) for two
// model-shaped hidden dimensions (tinyllama, undisclosed-model).
//
// Four configurations are compared:
//   baseline       — composite forward + composite backward
//   fw ON, bw OFF  — fused forward   + composite backward
//   fw OFF, bw ON  — composite forward + fused backward
//   fw ON, bw ON   — fused forward   + fused backward
//
// Results are printed as % time and DRAM change relative to the baseline.
//
// Environment variables:
//   TTML_POLYNORM_BENCH_WARMUP   — warmup iterations  (default 2)
//   TTML_POLYNORM_BENCH_MEASURE  — measured iterations (default 5)
//   TTML_POLYNORM_BENCH_BATCHES  — comma-separated batch sizes (default 1,2,4,8,16)
//   TTML_POLYNORM_BENCH_MODELS   — comma-separated model names to include
// ============================================================================

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/polynorm_op.hpp"
#include "utils/memory_utils.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> batch_sizes = {1, 2, 4, 8, 16};
    uint32_t num_warmup = 2;
    uint32_t num_measure = 5;
    std::vector<std::string> model_filter;
};

struct ModelShape {
    std::string name;
    uint32_t embedding_dim;
    uint32_t sequence_length;
};

struct RunResult {
    std::vector<double> forward_times;
    std::vector<double> backward_times;
    double dram_peak_mb = 0.0;
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

double avg(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("avg() requires at least one sample; ensure TTML_POLYNORM_BENCH_MEASURE > 0.");
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
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

std::vector<float> make_random_values(size_t count, uint32_t seed, float low = -1.0F, float high = 1.0F) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> values(count);
    std::generate(values.begin(), values.end(), [&]() { return dist(gen); });
    return values;
}

RunResult run_single_mode(
    const ModelShape& shape,
    const SweepConfig& cfg,
    uint32_t batch_size,
    uint32_t sequence_length,
    ttml::ops::PolyNorm3ForwardVariant polynorm3_forward_variant,
    ttml::ops::PolyNorm3BackwardVariant polynorm3_backward_variant,
    const tt::tt_metal::distributed::MeshShape& mesh) {
    ttml::autograd::ctx().open_device(mesh);
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const ttnn::Shape x_shape({batch_size, 1, sequence_length, shape.embedding_dim});
    const ttnn::Shape w_shape({1, 1, 1, 3});
    const ttnn::Shape b_shape({1, 1, 1, 1});
    const size_t x_count = static_cast<size_t>(batch_size) * sequence_length * shape.embedding_dim;
    const auto x_host = make_random_values(x_count, 2026U + batch_size + shape.embedding_dim);
    const auto w_host = std::vector<float>{0.2F, 0.3F, 0.5F};
    const auto b_host = std::vector<float>{0.1F};
    constexpr float epsilon = 1e-5F;

    const auto make_step_tensors = [&]() {
        const auto input = ttml::autograd::create_tensor(
            ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(x_host, x_shape, device, ttnn::Layout::TILE),
            /*requires_grad=*/true);
        const auto weight = ttml::autograd::create_tensor(
            ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(w_host, w_shape, device, ttnn::Layout::TILE),
            /*requires_grad=*/true);
        const auto bias = ttml::autograd::create_tensor(
            ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(b_host, b_shape, device, ttnn::Layout::TILE),
            /*requires_grad=*/true);
        return std::tuple{input, weight, bias};
    };

    const auto run_step = [&]() -> std::pair<double, double> {
        auto [input, weight, bias] = make_step_tensors();
        const auto t0_fw = std::chrono::high_resolution_clock::now();
        auto out =
            ttml::ops::polynorm3(input, weight, bias, epsilon, polynorm3_forward_variant, polynorm3_backward_variant);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        const auto t1_fw = std::chrono::high_resolution_clock::now();

        // Backward-only timing uses synthetic upstream gradient to avoid loss/optimizer overhead.
        auto grad_out = ttml::core::ones_like(out->get_value());
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        out->set_grad(grad_out);

        const auto t0_bw = std::chrono::high_resolution_clock::now();
        out->backward();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        const auto t1_bw = std::chrono::high_resolution_clock::now();

        ttml::autograd::ctx().reset_graph();
        return {
            std::chrono::duration<double, std::milli>(t1_fw - t0_fw).count(),
            std::chrono::duration<double, std::milli>(t1_bw - t0_bw).count()};
    };

    RunResult result{};
    const uint32_t max_steps = cfg.num_warmup + cfg.num_measure;
    for (uint32_t step = 0; step < max_steps; ++step) {
        const bool is_measured = step >= cfg.num_warmup;
        const auto [fw_ms, bw_ms] = run_step();
        if (is_measured) {
            result.forward_times.push_back(fw_ms);
            result.backward_times.push_back(bw_ms);
        }
    }

    {
        auto memory_guard = ttml::utils::MemoryUsageTracker::begin_capture();
        (void)run_step();
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

    double fw_composite_ms = 0.0;
    double fw_fused_ms = 0.0;
    double bw_composite_ms = 0.0;
    double bw_fused_ms = 0.0;
    double total_composite_ms = 0.0;
    double total_fused_ms = 0.0;

    double dram_composite_mb = 0.0;
    double dram_fused_mb = 0.0;
};

void print_forward_table(const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "Forward (composite vs fused-forward)\n"
        "| Model      | B     | S     | C     | Composite ms | Fused ms | Saved ms | Speedup | Reduction |\n");
    fmt::print("|------------|-------|-------|-------|--------------|----------|----------|---------|-----------|\n");
    for (const auto& row : rows) {
        const double saved_ms = row.fw_composite_ms - row.fw_fused_ms;
        fmt::print(
            "| {:<10} | {:>5} | {:>5} | {:>5} | {:>12.2f} | {:>8.2f} | {:>8.2f} | {:>6.2f}x | {:>8.2f}% |\n",
            row.model_name,
            row.batch_size,
            row.sequence_length,
            row.embedding_dim,
            row.fw_composite_ms,
            row.fw_fused_ms,
            saved_ms,
            speedup_x(row.fw_composite_ms, row.fw_fused_ms),
            reduction_pct(row.fw_composite_ms, row.fw_fused_ms));
    }
}

void print_backward_table(const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "Backward (composite vs fused-backward)\n"
        "| Model      | B     | S     | C     | Composite ms | Fused ms | Saved ms | Speedup | Reduction |\n");
    fmt::print("|------------|-------|-------|-------|--------------|----------|----------|---------|-----------|\n");
    for (const auto& row : rows) {
        const double saved_ms = row.bw_composite_ms - row.bw_fused_ms;
        fmt::print(
            "| {:<10} | {:>5} | {:>5} | {:>5} | {:>12.2f} | {:>8.2f} | {:>8.2f} | {:>6.2f}x | {:>8.2f}% |\n",
            row.model_name,
            row.batch_size,
            row.sequence_length,
            row.embedding_dim,
            row.bw_composite_ms,
            row.bw_fused_ms,
            saved_ms,
            speedup_x(row.bw_composite_ms, row.bw_fused_ms),
            reduction_pct(row.bw_composite_ms, row.bw_fused_ms));
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
            speedup_x(row.total_composite_ms, row.total_fused_ms),
            reduction_pct(row.total_composite_ms, row.total_fused_ms),
            row.dram_composite_mb,
            row.dram_fused_mb,
            reduction_pct(row.dram_composite_mb, row.dram_fused_mb));
    }
}

}  // namespace

int main() {
    try {
        SweepConfig sweep_cfg{};
        if (const char* env_warmup = std::getenv("TTML_POLYNORM_BENCH_WARMUP")) {
            sweep_cfg.num_warmup = static_cast<uint32_t>(std::stoul(env_warmup));
        }
        if (const char* env_measure = std::getenv("TTML_POLYNORM_BENCH_MEASURE")) {
            sweep_cfg.num_measure = static_cast<uint32_t>(std::stoul(env_measure));
        }
        if (const char* env_batches = std::getenv("TTML_POLYNORM_BENCH_BATCHES")) {
            auto parsed = parse_u32_csv(env_batches);
            if (!parsed.empty()) {
                sweep_cfg.batch_sizes = std::move(parsed);
            }
        }
        if (const char* env_models = std::getenv("TTML_POLYNORM_BENCH_MODELS")) {
            sweep_cfg.model_filter = parse_string_csv(env_models);
        }
        if (sweep_cfg.num_measure == 0U) {
            throw std::invalid_argument("TTML_POLYNORM_BENCH_MEASURE must be greater than zero.");
        }
        const auto& models = all_models();
        const tt::tt_metal::distributed::MeshShape mesh(1, 1);

        fmt::print("PolyNorm op-level benchmark (forward/backward fused ON/OFF vs composite)\n");
        fmt::print(
            "preset models=tinyllama,undisclosed-model batches={} warmup={} measure={} "
            "seq_lens=tinyllama:2048,undisclosed-model:4096\n",
            join_u32_csv(sweep_cfg.batch_sizes),
            sweep_cfg.num_warmup,
            sweep_cfg.num_measure);
        fmt::print(
            "Runs op-level timings: PolyNorm3 forward and backward separately on BF16 tensors.\n"
            "Backward uses synthetic upstream grad (ones_like(out)); no loss/optimizer in timed region.\n"
            "Baseline uses composite PolyNorm3 forward and backward.\n");

        std::vector<RowSummary> all_rows;
        all_rows.reserve(models.size() * sweep_cfg.batch_sizes.size());
        for (const auto& model : models) {
            if (!model_is_enabled(sweep_cfg, model.name)) {
                continue;
            }
            const uint32_t sequence_length = model.sequence_length;
            for (const auto batch_size : sweep_cfg.batch_sizes) {
                using FV = ttml::ops::PolyNorm3ForwardVariant;
                using BV = ttml::ops::PolyNorm3BackwardVariant;
                const auto baseline = run_single_mode(
                    model,
                    sweep_cfg,
                    batch_size,
                    sequence_length,
                    FV::CompositeComparisonOnly,
                    BV::CompositeComparisonOnly,
                    mesh);
                const auto fw_on_bw_off = run_single_mode(
                    model, sweep_cfg, batch_size, sequence_length, FV::Fused, BV::CompositeComparisonOnly, mesh);
                const auto fw_off_bw_on = run_single_mode(
                    model, sweep_cfg, batch_size, sequence_length, FV::CompositeComparisonOnly, BV::Fused, mesh);
                const auto fw_on_bw_on =
                    run_single_mode(model, sweep_cfg, batch_size, sequence_length, FV::Fused, BV::Fused, mesh);

                const double baseline_forward = avg(baseline.forward_times);
                const double fused_forward = avg(fw_on_bw_off.forward_times);
                const double baseline_backward = avg(baseline.backward_times);
                const double fused_backward = avg(fw_off_bw_on.backward_times);
                const double baseline_total = baseline_forward + baseline_backward;
                const double fused_total = avg(fw_on_bw_on.forward_times) + avg(fw_on_bw_on.backward_times);
                all_rows.push_back(RowSummary{
                    .model_name = model.name,
                    .batch_size = batch_size,
                    .sequence_length = sequence_length,
                    .embedding_dim = model.embedding_dim,
                    .fw_composite_ms = baseline_forward,
                    .fw_fused_ms = fused_forward,
                    .bw_composite_ms = baseline_backward,
                    .bw_fused_ms = fused_backward,
                    .total_composite_ms = baseline_total,
                    .total_fused_ms = fused_total,
                    .dram_composite_mb = baseline.dram_peak_mb,
                    .dram_fused_mb = fw_on_bw_on.dram_peak_mb,
                });
            }
        }
        print_backward_table(all_rows);
        print_forward_table(all_rows);
        print_total_table(all_rows);
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "polynorm_fusion_benchmark failed: {}\n", e.what());
        return 1;
    }
}
