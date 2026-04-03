// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/llama.hpp"
#include "ops/losses.hpp"
#include "optimizers/adamw.hpp"
#include "utils/memory_utils.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> batch_sizes = {4, 16, 32, 64};
    uint32_t num_warmup = 2;
    uint32_t num_measure = 5;
    uint32_t sequence_length = 256;
    // 0 means: use model.num_blocks from the selected preset.
    uint32_t stacked_blocks = 0;
    std::vector<std::string> model_filter;
};

struct ModelShape {
    std::string name;
    uint32_t num_blocks;
    uint32_t embedding_dim;
    uint32_t hidden_dim;
    uint32_t num_heads;
    uint32_t num_groups;
    ttml::models::llama::WeightTyingType weight_tying;
};

struct RunResult {
    std::vector<double> step_times;
    size_t dram_peak = 0;
};

const std::vector<ModelShape>& all_models() {
    static const std::vector<ModelShape> models = {
        // Reduced block counts as used in N300 comparisons.
        {"nanollama3", 6, 384, 1024, 6, 3, ttml::models::llama::WeightTyingType::Enabled},
        {"tinyllama", 4, 2048, 5632, 32, 4, ttml::models::llama::WeightTyingType::Disabled},
        {"llama1b", 4, 2048, 8192, 32, 8, ttml::models::llama::WeightTyingType::Enabled},
    };
    return models;
}

double avg(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

size_t cumulative_peak_from_captured_traces() {
    long long cumulative_current = 0;
    long long cumulative_peak = 0;
    for (const auto& trace_name : ttml::utils::MemoryUsageTracker::get_trace_names()) {
        const auto usage = ttml::utils::MemoryUsageTracker::get_dram_usage(trace_name);
        const long long segment_abs_peak = cumulative_current + static_cast<long long>(usage.peak);
        cumulative_peak = std::max(cumulative_peak, segment_abs_peak);
        cumulative_current +=
            static_cast<long long>(usage.total_allocations) - static_cast<long long>(usage.total_deallocations);
    }
    return static_cast<size_t>(std::max(0LL, cumulative_peak));
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

std::vector<uint32_t> make_random_tokens(size_t count, uint32_t vocab_size, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0U, vocab_size - 1U);
    std::vector<uint32_t> values(count);
    std::generate(values.begin(), values.end(), [&]() { return dist(gen); });
    return values;
}

RunResult run_single(const ModelShape& shape, const SweepConfig& cfg, uint32_t batch_size, bool use_fused) {
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    if (use_fused) {
        ::unsetenv("TTML_SWIGLU_FORCE_COMPOSITE");
    } else {
        ::setenv("TTML_SWIGLU_FORCE_COMPOSITE", "1", 1);
    }

    const uint32_t num_blocks = (cfg.stacked_blocks == 0U) ? shape.num_blocks : cfg.stacked_blocks;
    constexpr uint32_t kSyntheticVocabSize = 96U;
    ttml::models::llama::LlamaConfig model_cfg{
        .num_heads = shape.num_heads,
        .num_groups = shape.num_groups,
        .embedding_dim = shape.embedding_dim,
        .intermediate_dim = shape.hidden_dim,
        .dropout_prob = 0.0F,
        .theta = 10000.0F,
        .num_blocks = num_blocks,
        .vocab_size = kSyntheticVocabSize,
        .max_sequence_length = cfg.sequence_length,
        .runner_type = ttml::models::llama::RunnerType::Default,
        .weight_tying = shape.weight_tying,
    };

    auto optimizer_cfg = ttml::optimizers::AdamWConfig{};
    optimizer_cfg.lr = 3e-4F;
    optimizer_cfg.weight_decay = 1e-2F;

    constexpr uint32_t kFeaturesSeedBase = 2026U;
    constexpr uint32_t kTargetsSeedBase = 3039U;
    const size_t token_count = static_cast<size_t>(batch_size) * cfg.sequence_length;
    const auto features_host =
        make_random_tokens(token_count, model_cfg.vocab_size, kFeaturesSeedBase + batch_size + num_blocks);
    const auto targets_host =
        make_random_tokens(token_count, model_cfg.vocab_size, kTargetsSeedBase + batch_size + num_blocks);
    const ttnn::Shape features_shape({batch_size, 1, 1, cfg.sequence_length});
    const ttnn::Shape targets_shape({batch_size, cfg.sequence_length});

    const auto make_batch = [&]() {
        auto* const dev = &ttml::autograd::ctx().get_device();
        const auto features = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            features_host, features_shape, dev, ttnn::Layout::ROW_MAJOR));
        const auto targets = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            targets_host, targets_shape, dev, ttnn::Layout::ROW_MAJOR));
        return std::pair{features, targets};
    };

    RunResult result;

    // Memory pass: mirror nano_gpt first-iteration summary semantics.
    ttml::utils::MemoryUsageTracker::clear();
    {
        const auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
        (void)guard;

        auto model_mem = ttml::models::llama::create(model_cfg);
        ttml::utils::MemoryUsageTracker::snapshot("MODEL_CREATION");

        auto optimizer_mem = std::make_shared<ttml::optimizers::AdamW>(model_mem->parameters(), optimizer_cfg);
        ttml::utils::MemoryUsageTracker::snapshot("OPTIMIZER_CREATION");

        const auto run_step_with_snapshots = [&](double& step_ms) {
            auto [features, targets] = make_batch();
            auto* const dev = &ttml::autograd::ctx().get_device();
            const auto t0 = std::chrono::high_resolution_clock::now();
            optimizer_mem->zero_grad();
            auto logits = (*model_mem)(features, std::nullopt);
            ttml::utils::MemoryUsageTracker::snapshot("FORWARD_PASS");
            auto loss = ttml::ops::cross_entropy_loss(logits, targets);
            loss->backward();
            ttml::utils::MemoryUsageTracker::snapshot("BACKWARD_PASS");
            optimizer_mem->step();
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
            const auto t1 = std::chrono::high_resolution_clock::now();
            ttml::autograd::ctx().reset_graph();
            step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        double ignored_ms = 0.0;
        run_step_with_snapshots(ignored_ms);
        ttml::utils::MemoryUsageTracker::end_capture("FIRST_ITERATION_COMPLETE");
        if (std::getenv("TTML_SWIGLU_BENCH_PRINT_MEMORY_SUMMARY") != nullptr) {
            ttml::utils::MemoryUsageTracker::print_memory_usage();
        }
        result.dram_peak = cumulative_peak_from_captured_traces();
        ttml::utils::MemoryUsageTracker::clear();
    }

    // Timing pass: use a fresh model, but reuse compiled programs from memory pass.
    {
        auto model_timed = ttml::models::llama::create(model_cfg);
        auto optimizer_timed = std::make_shared<ttml::optimizers::AdamW>(model_timed->parameters(), optimizer_cfg);

        const auto run_step_timed = [&]() -> double {
            auto [features, targets] = make_batch();  // mimic dataloader/collate outside timing region
            auto* const dev = &ttml::autograd::ctx().get_device();
            const auto t0 = std::chrono::high_resolution_clock::now();
            optimizer_timed->zero_grad();
            auto logits = (*model_timed)(features, std::nullopt);
            auto loss = ttml::ops::cross_entropy_loss(logits, targets);
            loss->backward();
            optimizer_timed->step();
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
            const auto t1 = std::chrono::high_resolution_clock::now();
            ttml::autograd::ctx().reset_graph();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        const uint32_t max_steps = cfg.num_warmup + cfg.num_measure;
        for (uint32_t step = 0; step < max_steps; ++step) {
            const bool is_measured = step >= cfg.num_warmup;
            const double step_ms = run_step_timed();
            if (is_measured) {
                result.step_times.push_back(step_ms);
            }
        }

        optimizer_timed.reset();
        model_timed.reset();
        ttml::autograd::ctx().reset_graph();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    }

    return result;
}

double pct(double fused, double baseline) {
    if (baseline == 0.0) {
        return 0.0;
    }
    return (fused - baseline) / baseline * 100.0;
}

struct RowSummary {
    uint32_t batch = 0;
    double baseline_total_ms = 0.0;
    double fused_total_ms = 0.0;
    double total_pct = 0.0;
    double baseline_dram_kb = 0.0;
    double fused_dram_kb = 0.0;
    double dram_pct = 0.0;
};

void print_model_table(const ModelShape& model, const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "Model: {} (blocks={}, D={}, H={})\n", model.name, model.num_blocks, model.embedding_dim, model.hidden_dim);
    fmt::print("+-------+--------------+--------------+-----------+--------------+--------------+-----------+\n");
    fmt::print("| Batch | Baseline ms  | Fused ms     | Delta %   | Baseline KB  | Fused KB     | Delta %   |\n");
    fmt::print("+-------+--------------+--------------+-----------+--------------+--------------+-----------+\n");
    for (const auto& row : rows) {
        fmt::print(
            "| {:>5} | {:>12.1f} | {:>12.1f} | {:>+9.2f} | {:>12.0f} | {:>12.0f} | {:>+9.2f} |\n",
            row.batch,
            row.baseline_total_ms,
            row.fused_total_ms,
            row.total_pct,
            row.baseline_dram_kb,
            row.fused_dram_kb,
            row.dram_pct);
    }
    fmt::print("+-------+--------------+--------------+-----------+--------------+--------------+-----------+\n");
}

}  // namespace

int main() {
    try {
        SweepConfig sweep_cfg{};
        if (const char* env_blocks = std::getenv("TTML_SWIGLU_BENCH_STACKED_BLOCKS")) {
            const int parsed = std::atoi(env_blocks);
            if (parsed > 0) {
                sweep_cfg.stacked_blocks = static_cast<uint32_t>(parsed);
            }
        }
        if (const char* env_warmup = std::getenv("TTML_SWIGLU_BENCH_WARMUP")) {
            sweep_cfg.num_warmup = static_cast<uint32_t>(std::stoul(env_warmup));
        }
        if (const char* env_measure = std::getenv("TTML_SWIGLU_BENCH_MEASURE")) {
            sweep_cfg.num_measure = static_cast<uint32_t>(std::stoul(env_measure));
        }
        if (const char* env_batches = std::getenv("TTML_SWIGLU_BENCH_BATCHES")) {
            auto parsed = parse_u32_csv(env_batches);
            if (!parsed.empty()) {
                sweep_cfg.batch_sizes = std::move(parsed);
            }
        }
        if (const char* env_models = std::getenv("TTML_SWIGLU_BENCH_MODELS")) {
            sweep_cfg.model_filter = parse_string_csv(env_models);
        }
        const auto& models = all_models();

        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);

        fmt::print("SwiGLU training-like benchmark (composite baseline vs fused)\n");
        fmt::print(
            "preset models=nanollama3,tinyllama,llama1b batches=4,16,32,64 warmup={} measure={} seq_len={} "
            "stacked_blocks={}\n",
            sweep_cfg.num_warmup,
            sweep_cfg.num_measure,
            sweep_cfg.sequence_length,
            (sweep_cfg.stacked_blocks == 0U) ? std::string("model.num_blocks")
                                             : std::to_string(sweep_cfg.stacked_blocks));
        fmt::print("Runs full training-like step: model forward + CE loss + backward + AdamW step.\n");

        for (const auto& model : models) {
            if (!model_is_enabled(sweep_cfg, model.name)) {
                continue;
            }
            std::vector<RowSummary> rows;
            rows.reserve(sweep_cfg.batch_sizes.size());
            for (const auto batch_size : sweep_cfg.batch_sizes) {
                const auto baseline = run_single(model, sweep_cfg, batch_size, /*use_fused=*/false);
                const auto fused = run_single(model, sweep_cfg, batch_size, /*use_fused=*/true);

                const double b_total = avg(baseline.step_times);
                const double f_total = avg(fused.step_times);

                rows.push_back(RowSummary{
                    .batch = batch_size,
                    .baseline_total_ms = b_total,
                    .fused_total_ms = f_total,
                    .total_pct = pct(f_total, b_total),
                    .baseline_dram_kb = static_cast<double>(baseline.dram_peak) / 1024.0,
                    .fused_dram_kb = static_cast<double>(fused.dram_peak) / 1024.0,
                    .dram_pct = pct(static_cast<double>(fused.dram_peak), static_cast<double>(baseline.dram_peak)),
                });
            }
            print_model_table(model, rows);
        }

        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "swiglu_fusion_benchmark failed: {}\n", e.what());
        return 1;
    }
}
