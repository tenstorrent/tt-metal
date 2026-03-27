// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
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
#include "ops/losses.hpp"
#include "ops/polynorm_op.hpp"
#include "utils/memory_utils.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> batch_sizes = {4, 16, 32, 64};
    uint32_t num_warmup = 2;
    uint32_t num_measure = 5;
    uint32_t sequence_length = 256;
    std::vector<std::string> model_filter;
};

struct ModelShape {
    std::string name;
    uint32_t embedding_dim;
};

struct RunResult {
    std::vector<double> step_times;
    double dram_peak_mb = 0.0;
};

enum class ForwardMode {
    Composite,
    Fused,
};

const std::vector<ModelShape>& all_models() {
    static const std::vector<ModelShape> models = {
        {"nanollama3", 384},
        {"tinyllama", 2048},
        {"llama1b", 2048},
    };
    return models;
}

double avg(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double pct(double value, double baseline) {
    if (baseline == 0.0) {
        return 0.0;
    }
    return (value - baseline) / baseline * 100.0;
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
    ForwardMode mode,
    const tt::tt_metal::distributed::MeshShape& mesh) {
    ttml::autograd::ctx().open_device(mesh);
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const ttnn::Shape x_shape({batch_size, 1, cfg.sequence_length, shape.embedding_dim});
    const ttnn::Shape w_shape({1, 1, 1, 3});
    const ttnn::Shape b_shape({1, 1, 1, 1});
    const size_t x_count = static_cast<size_t>(batch_size) * cfg.sequence_length * shape.embedding_dim;
    const auto x_host = make_random_values(x_count, 2026U + batch_size + shape.embedding_dim);
    const auto target_host = make_random_values(x_count, 3039U + batch_size + shape.embedding_dim);
    const auto w_host = std::vector<float>{0.2F, 0.3F, 0.5F};
    const auto b_host = std::vector<float>{0.1F};

    const auto run_step = [&]() -> double {
        auto x =
            ttml::autograd::create_tensor(ttml::core::from_vector(x_host, x_shape, device), /*requires_grad=*/true);
        auto w =
            ttml::autograd::create_tensor(ttml::core::from_vector(w_host, w_shape, device), /*requires_grad=*/true);
        auto b =
            ttml::autograd::create_tensor(ttml::core::from_vector(b_host, b_shape, device), /*requires_grad=*/true);
        auto target = ttml::autograd::create_tensor(
            ttml::core::from_vector(target_host, x_shape, device), /*requires_grad=*/false);

        const auto t0 = std::chrono::high_resolution_clock::now();
        auto out =
            (mode == ForwardMode::Composite) ? ttml::ops::polynorm3_composite(x, w, b) : ttml::ops::polynorm3(x, w, b);
        auto loss = ttml::ops::mse_loss(out, target);
        loss->backward();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        const auto t1 = std::chrono::high_resolution_clock::now();
        ttml::autograd::ctx().reset_graph();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    RunResult result{};
    const uint32_t max_steps = cfg.num_warmup + cfg.num_measure;
    for (uint32_t step = 0; step < max_steps; ++step) {
        const bool is_measured = step >= cfg.num_warmup;
        const double step_ms = run_step();
        if (is_measured) {
            result.step_times.push_back(step_ms);
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
        const auto& models = all_models();
        const tt::tt_metal::distributed::MeshShape mesh(1, 1);

        fmt::print("PolyNorm model-level benchmark (composite vs fused forward)\n");
        fmt::print(
            "preset models=nanollama3,tinyllama,llama1b batches=4,16,32,64 warmup={} measure={} seq_len={}\n",
            sweep_cfg.num_warmup,
            sweep_cfg.num_measure,
            sweep_cfg.sequence_length);
        fmt::print("| Model      | Batch | ΔDRAM FW | ΔStep FW |\n");
        fmt::print("|------------|-------|---------:|---------:|\n");

        for (const auto& model : models) {
            if (!model_is_enabled(sweep_cfg, model.name)) {
                continue;
            }
            for (const auto batch_size : sweep_cfg.batch_sizes) {
                const auto composite = run_single_mode(model, sweep_cfg, batch_size, ForwardMode::Composite, mesh);
                const auto fused = run_single_mode(model, sweep_cfg, batch_size, ForwardMode::Fused, mesh);

                fmt::print(
                    "| {:<10} | {:>5} | {:>+8.2f}% | {:>+8.2f}% |\n",
                    model.name,
                    batch_size,
                    pct(fused.dram_peak_mb, composite.dram_peak_mb),
                    pct(avg(fused.step_times), avg(composite.step_times)));
            }
        }

        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "polynorm_fusion_benchmark failed: {}\n", e.what());
        return 1;
    }
}
