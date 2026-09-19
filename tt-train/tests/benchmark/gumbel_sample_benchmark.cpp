// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

struct BenchConfig {
    std::string name;
    uint32_t batch = 1;
    uint32_t tokens = 1;
    uint32_t vocab = 1;
    float temperature = 0.7F;
    bool with_mask = false;
    bool with_positions = false;
    uint32_t warmup = 5;
    uint32_t iters = 50;
};

// GRPO-representative shapes. decode: one token row per entry, full-row mode.
// prefill: per-row positions so only the prompt-end tiles are read.
// pos_large: batch large enough that positions staging and merge traffic dominate.
const std::vector<BenchConfig>& all_configs() {
    static const std::vector<BenchConfig> configs = {
        {.name = "decode_b8_v151936", .batch = 8, .tokens = 1, .vocab = 151936, .warmup = 10, .iters = 100},
        // Diagnostics, not part of the headline geomean: greedy compiles out the noise chain
        // (noise compiled out) — the gap to decode_b8_v151936 is the chain's device cost;
        // the small-vocab run checks whether time scales with Wt (device-bound signature).
        {.name = "greedy_b8_v151936",
         .batch = 8,
         .tokens = 1,
         .vocab = 151936,
         .temperature = 0.0F,
         .warmup = 10,
         .iters = 100},
        {.name = "decode_b8_v16000", .batch = 8, .tokens = 1, .vocab = 16000, .warmup = 10, .iters = 100},
        {.name = "decode_b8_v151936_mask",
         .batch = 8,
         .tokens = 1,
         .vocab = 151936,
         .with_mask = true,
         .warmup = 10,
         .iters = 100},
        {.name = "prefill_pos_b8_t256_v151936",
         .batch = 8,
         .tokens = 256,
         .vocab = 151936,
         .with_positions = true,
         .warmup = 5,
         .iters = 30},
        {.name = "pos_b512_t64_v16000",
         .batch = 512,
         .tokens = 64,
         .vocab = 16000,
         .with_positions = true,
         .warmup = 3,
         .iters = 20},
    };
    return configs;
}

std::vector<float> make_logits(const BenchConfig& cfg) {
    const size_t count = static_cast<size_t>(cfg.batch) * cfg.tokens * cfg.vocab;
    std::vector<float> data(count);
    uint32_t state = 0x12345678U;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        data[i] = static_cast<float>(state >> 8U) * (1.0F / 16777216.0F) * 8.0F - 4.0F;
    }
    return data;
}

ttnn::Tensor make_positions_tensor(const BenchConfig& cfg) {
    std::vector<uint32_t> positions(cfg.batch);
    for (uint32_t b = 0; b < cfg.batch; ++b) {
        positions[b] = (b * 7U + 3U) % cfg.tokens;
    }
    return ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        positions,
        ttnn::Shape({cfg.batch, 1U, 1U, 1U}),
        &ttml::autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
}

ttnn::Tensor make_mask_tensor(const BenchConfig& cfg) {
    std::vector<float> mask(cfg.vocab, 0.0F);
    for (uint32_t v = cfg.vocab - 64U; v < cfg.vocab; ++v) {
        mask[v] = 1e9F;
    }
    return ttml::core::from_vector(
        mask, ttnn::Shape({1U, 1U, 1U, cfg.vocab}), &ttml::autograd::ctx().get_device());
}

double percentile(std::vector<double> sorted, double p) {
    const size_t idx = static_cast<size_t>(p * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
}

void run_config(const BenchConfig& cfg) {
    auto* device = &ttml::autograd::ctx().get_device();

    const auto logits_data = make_logits(cfg);
    auto logits = ttml::core::from_vector(
        logits_data, ttnn::Shape({cfg.batch, 1U, cfg.tokens, cfg.vocab}), device);

    std::optional<ttnn::Tensor> mask;
    if (cfg.with_mask) {
        mask = make_mask_tensor(cfg);
    }
    std::optional<ttnn::Tensor> positions;
    if (cfg.with_positions) {
        positions = make_positions_tensor(cfg);
    }

    // Timed region includes the token-id readback: GRPO consumes the sampled ids on host
    // every step, so enqueue-only timing would hide the cost the training loop actually pays.
    auto step = [&](uint32_t seed) {
        auto out = ttml::ttnn_fixed::sample(logits, cfg.temperature, seed, mask, std::nullopt, positions);
        return ttml::core::to_vector<uint32_t>(out);
    };

    for (uint32_t i = 0; i < cfg.warmup; ++i) {
        step(1000U + i);
    }

    std::vector<double> times_ms;
    times_ms.reserve(cfg.iters);
    for (uint32_t i = 0; i < cfg.iters; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        auto ids = step(2000U + i);
        const auto end = std::chrono::high_resolution_clock::now();
        if (ids.empty()) {
            fmt::print("ERROR: empty output for {}\n", cfg.name);
            return;
        }
        times_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(times_ms.begin(), times_ms.end());
    fmt::print(
        "GUMBEL_BENCH_JSON {{\"config\":\"{}\",\"iters\":{},\"min_ms\":{:.4f},\"median_ms\":{:.4f},"
        "\"p90_ms\":{:.4f},\"mean_ms\":{:.4f}}}\n",
        cfg.name,
        cfg.iters,
        times_ms.front(),
        percentile(times_ms, 0.5),
        percentile(times_ms, 0.9),
        ttml::benchmark_utils::average(times_ms));

    // Pipelined mode: N async enqueues, one readback sync at the end. Per-call cost here is
    // max(enqueue floor, device time) — the gap vs the synced number above isolates how much of
    // the per-step latency is the host sync, and exposes device-time deltas the sync hides.
    const uint32_t pipelined_n = cfg.iters;
    ttnn::Tensor last;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < pipelined_n; ++i) {
        last = ttml::ttnn_fixed::sample(logits, cfg.temperature, 3000U + i, mask, std::nullopt, positions);
    }
    auto ids = ttml::core::to_vector<uint32_t>(last);
    const auto end = std::chrono::high_resolution_clock::now();
    if (ids.empty()) {
        fmt::print("ERROR: empty pipelined output for {}\n", cfg.name);
        return;
    }
    const double per_call_ms = std::chrono::duration<double, std::milli>(end - start).count() / pipelined_n;
    fmt::print(
        "GUMBEL_BENCH_JSON {{\"config\":\"{}_pipelined\",\"iters\":{},\"median_ms\":{:.4f}}}\n",
        cfg.name,
        pipelined_n,
        per_call_ms);
}

}  // namespace

int main() {
    ttml::autograd::ctx().open_device();
    for (const auto& cfg : all_configs()) {
        run_config(cfg);
    }
    ttml::autograd::ctx().close_device();
    return 0;
}
