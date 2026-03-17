// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/swiglu_op.hpp"
#include "utils/memory_utils.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> batch_sizes = {4, 16, 32, 64};
    uint32_t num_warmup = 2;
    uint32_t num_measure = 5;
    uint32_t sequence_length = 256;
};

struct ModelShape {
    std::string name;
    uint32_t num_blocks;
    uint32_t embedding_dim;
    uint32_t hidden_dim;
};

struct RunResult {
    std::vector<double> step_times;
    std::vector<double> fwd_times;
    std::vector<double> bwd_times;
    size_t dram_peak = 0;
};

const std::vector<ModelShape>& all_models() {
    static const std::vector<ModelShape> models = {
        // Reduced block counts as used in N300 comparisons.
        {"nanollama3", 6, 384, 1024},
        {"tinyllama", 4, 2048, 5632},
        {"llama1b", 4, 2048, 8192},
    };
    return models;
}

double avg(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

std::vector<float> make_random_vector(size_t size, float stddev, uint32_t seed) {
    std::vector<float> data(size);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0F, stddev);
    for (auto& v : data) {
        v = dist(gen);
    }
    return data;
}

RunResult run_single(const ModelShape& shape, const SweepConfig& cfg, uint32_t batch_size, bool use_fused) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();

    const uint32_t d = shape.embedding_dim;
    const uint32_t h = shape.hidden_dim;
    const uint32_t s = cfg.sequence_length;
    const ttnn::Shape x_shape({batch_size, 1, s, d});
    const ttnn::Shape w1_shape({h, d});
    const ttnn::Shape w2_shape({d, h});
    const ttnn::Shape w3_shape({h, d});

    const float act_std = 1.0F;
    const float w13_std = 1.0F / std::sqrt(static_cast<float>(d));
    const float w2_std = 1.0F / std::sqrt(static_cast<float>(h));

    auto x_data = make_random_vector(x_shape.volume(), act_std, 101U + batch_size + d);
    auto w1_data = make_random_vector(w1_shape.volume(), w13_std, 202U + batch_size + h);
    auto w2_data = make_random_vector(w2_shape.volume(), w2_std, 303U + batch_size + h);
    auto w3_data = make_random_vector(w3_shape.volume(), w13_std, 404U + batch_size + d);

    auto x_value = ttml::core::from_vector(x_data, x_shape, device);
    auto w1_value = ttml::core::from_vector(w1_data, w1_shape, device);
    auto w2_value = ttml::core::from_vector(w2_data, w2_shape, device);
    auto w3_value = ttml::core::from_vector(w3_data, w3_shape, device);

    auto op = use_fused ? &ttml::ops::swiglu : &ttml::ops::swiglu_composite;
    RunResult result;
    const uint32_t max_steps = cfg.num_warmup + cfg.num_measure;
    for (uint32_t step = 0; step < max_steps; ++step) {
        bool is_measured = step >= cfg.num_warmup;
        bool capture_memory = is_measured && result.dram_peak == 0;

        auto run_step = [&](double& fwd_ms, double& bwd_ms, bool with_capture) {
            auto* dev = &ttml::autograd::ctx().get_device();
            auto x = ttml::autograd::create_tensor(x_value, /*requires_grad=*/true);
            auto w1 = ttml::autograd::create_tensor(w1_value, /*requires_grad=*/true);
            auto w2 = ttml::autograd::create_tensor(w2_value, /*requires_grad=*/true);
            auto w3 = ttml::autograd::create_tensor(w3_value, /*requires_grad=*/true);

            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = (*op)(x, w1, w2, w3, /*dropout_prob=*/0.0F);
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
            auto t1 = std::chrono::high_resolution_clock::now();

            if (with_capture) {
                ttml::utils::MemoryUsageTracker::snapshot("forward");
            }

            out->set_grad(ttml::core::ones_like(out->get_value()));
            out->backward();
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
            auto t2 = std::chrono::high_resolution_clock::now();
            ttml::autograd::ctx().reset_graph();

            fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            bwd_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        };

        double fwd_ms = 0.0;
        double bwd_ms = 0.0;
        if (capture_memory) {
            auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
            (void)guard;
            run_step(fwd_ms, bwd_ms, /*with_capture=*/true);
            ttml::utils::MemoryUsageTracker::end_capture("backward");

            const auto fwd_usage = ttml::utils::MemoryUsageTracker::get_dram_usage("forward");
            const auto bwd_usage = ttml::utils::MemoryUsageTracker::get_dram_usage("backward");

            const long long fwd_current = static_cast<long long>(fwd_usage.total_allocations) -
                                          static_cast<long long>(fwd_usage.total_deallocations);
            const long long fwd_abs_peak = static_cast<long long>(fwd_usage.peak);
            const long long bwd_abs_peak = fwd_current + static_cast<long long>(bwd_usage.peak);
            const long long cumulative_abs_peak = std::max(fwd_abs_peak, bwd_abs_peak);
            result.dram_peak = static_cast<size_t>(std::max(0LL, cumulative_abs_peak));
            ttml::utils::MemoryUsageTracker::clear();
        } else {
            run_step(fwd_ms, bwd_ms, /*with_capture=*/false);
        }

        if (is_measured) {
            result.step_times.push_back(fwd_ms + bwd_ms);
            result.fwd_times.push_back(fwd_ms);
            result.bwd_times.push_back(bwd_ms);
        }
    }
    return result;
}

double pct(double fused, double baseline) {
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
        const auto& models = all_models();

        auto mesh = tt::tt_metal::distributed::MeshShape(1, 1);
        ttml::autograd::ctx().open_device(mesh);

        fmt::print("SwiGLU isolated op benchmark (composite baseline vs fused)\n");
        fmt::print(
            "preset models=nanollama3,tinyllama,llama1b batches=4,16,32,64 warmup={} measure={} seq_len={}\n",
            sweep_cfg.num_warmup,
            sweep_cfg.num_measure,
            sweep_cfg.sequence_length);
        fmt::print("Runs isolated SwiGLU forward+backward only (no model/dataloader/optimizer loop).\n");

        for (const auto& model : models) {
            std::vector<RowSummary> rows;
            rows.reserve(sweep_cfg.batch_sizes.size());
            for (auto batch_size : sweep_cfg.batch_sizes) {
                auto baseline = run_single(model, sweep_cfg, batch_size, /*use_fused=*/false);
                auto fused = run_single(model, sweep_cfg, batch_size, /*use_fused=*/true);

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
