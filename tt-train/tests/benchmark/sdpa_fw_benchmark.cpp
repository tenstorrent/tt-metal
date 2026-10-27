// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SDPA forward on the consumer shapes: the program factory's default chunking against the fixed
// chunking main used (one query tile-row, four K/V tiles per step), plus any config given on the
// command line. Prints device microseconds per launch and the speedup.
//
// Usage: sdpa_fw_benchmark [scenario-name-substring] [q_chunk_size,k_chunk_size,heads_per_item ...]
// Env:   TTML_SDPA_BENCH_WARMUP (default 2), TTML_SDPA_BENCH_MEASURE (default 10)

#include <fmt/format.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"
#include "xtensor/generators/xbuilder.hpp"

namespace {

struct Scenario {
    std::string name;
    uint32_t batch;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t seq_len;
    uint32_t head_dim;
    ttml::metal::AttentionMaskType mask_type;
    bool gate = false;
};

// TinyLlama 1.1B (32/4 heads, D=64), Llama-3-8B style (32/8, D=128), then the other modes of the shared
// kernel on the TinyLlama shape: no mask, an explicit (causal) mask tensor, and the output gate.
const std::vector<Scenario> kScenarios = {
    {"tinyllama_s2048_causal", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s4096_causal", 1U, 32U, 4U, 4096U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"llama8b_s2048_causal", 1U, 32U, 8U, 2048U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s2048_nomask", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::None},
    {"tinyllama_s2048_arbitrary", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Arbitrary},
    {"tinyllama_s2048_causal_gate", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal, true},
};

// What main ran: one query tile-row per item, four K/V tiles per step, no head stacking.
const ttml::metal::SDPAForwardProgramConfig kMainConfig{
    .q_chunk_size = 32U, .k_chunk_size = 128U, .heads_per_item = 1U};

struct Row {
    std::string label;
    double device_us = 0.0;
    double speedup = 0.0;  // main_us / device_us
    float max_abs_diff = 0.0F;
    std::string skipped;
};

uint32_t env_or(const char* name, uint32_t fallback) {
    if (const char* value = std::getenv(name)) {
        const int parsed = std::atoi(value);
        if (parsed > 0) {
            return static_cast<uint32_t>(parsed);
        }
    }
    return fallback;
}

void run_scenario(
    const Scenario& sc,
    const std::vector<ttml::metal::SDPAForwardProgramConfig>& extra_configs,
    uint32_t num_warmup,
    uint32_t num_measure) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();

    const std::array<std::size_t, 4> q_shape{sc.batch, sc.q_heads, sc.seq_len, sc.head_dim};
    const std::array<std::size_t, 4> kv_shape{sc.batch, sc.kv_heads, sc.seq_len, sc.head_dim};
    const uint32_t seed = benchmark_utils::seed_from_name(sc.name);
    auto query = core::from_xtensor(test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, seed), &device);
    auto key = core::from_xtensor(test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, seed + 1U), &device);
    auto value = core::from_xtensor(test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, seed + 2U), &device);

    std::optional<ttnn::Tensor> mask;
    if (sc.mask_type == metal::AttentionMaskType::Arbitrary) {
        xt::xarray<float> causal = xt::tril(xt::ones<float>({sc.seq_len, sc.seq_len}));
        causal.reshape({1UL, 1UL, sc.seq_len, sc.seq_len});
        mask = core::from_xtensor(causal, &device);
    }
    std::optional<ttnn::Tensor> gate;
    if (sc.gate) {
        gate = core::from_xtensor(test_utils::make_uniform_xarray<float>(q_shape, -4.0F, 4.0F, seed + 3U), &device);
    }
    auto run = [&](const metal::SDPAForwardProgramConfig& cfg) {
        return metal::sdpa_fw(query, key, value, sc.mask_type, mask, 0.0F, /*return_intermediates=*/true, gate, cfg)[0]
            .value();
    };
    auto time_cfg = [&](const metal::SDPAForwardProgramConfig& cfg) {
        return benchmark_utils::time_device_avg_us(device, num_warmup, num_measure, [&]() { (void)run(cfg); });
    };

    const double main_us = time_cfg(kMainConfig);
    const xt::xarray<float> main_out = core::to_xtensor(run(kMainConfig));

    std::vector<std::pair<std::string, metal::SDPAForwardProgramConfig>> configs = {{"default (picked)", {}}};
    for (const auto& cfg : extra_configs) {
        configs.emplace_back(fmt::format("q{} k{} h{}", cfg.q_chunk_size, cfg.k_chunk_size, cfg.heads_per_item), cfg);
    }
    std::vector<Row> rows;
    for (const auto& [label, cfg] : configs) {
        Row row{.label = label};
        try {
            row.device_us = time_cfg(cfg);
            row.speedup = benchmark_utils::speedup_x(main_us, row.device_us);
            const xt::xarray<float> out = core::to_xtensor(run(cfg));
            row.max_abs_diff = static_cast<float>(xt::amax(xt::abs(out - main_out))());
        } catch (const std::exception& e) {
            std::string_view what = e.what();  // first line only: TT_FATAL messages carry a backtrace
            row.skipped = std::string(what.substr(0, what.find('\n')));
        }
        rows.push_back(std::move(row));
    }

    fmt::print(
        "\n== {}: B={} Hq={} Hkv={} S={} D={} mask={} ==\n",
        sc.name,
        sc.batch,
        sc.q_heads,
        sc.kv_heads,
        sc.seq_len,
        sc.head_dim,
        sc.mask_type == metal::AttentionMaskType::Causal      ? (sc.gate ? "causal+gate" : "causal")
        : sc.mask_type == metal::AttentionMaskType::Arbitrary ? "arbitrary"
                                                              : "none");
    fmt::print("main chunking (q=32, k=128, h=1): {:.1f} us\n", main_us);
    fmt::print("{:<20} {:>12} {:>9} {:>12}\n", "config", "device_us", "speedup", "max_abs_diff");
    for (const Row& row : rows) {
        if (row.skipped.empty()) {
            fmt::print(
                "{:<20} {:>12.1f} {:>8.3f}x {:>12.3e}\n", row.label, row.device_us, row.speedup, row.max_abs_diff);
        } else {
            fmt::print("{:<20} {:>12} {:>9} {:>12}  skipped: {}\n", row.label, "-", "-", "-", row.skipped);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string filter;
    std::vector<ttml::metal::SDPAForwardProgramConfig> extra_configs;
    for (int i = 1; i < argc; ++i) {
        ttml::metal::SDPAForwardProgramConfig cfg{};
        if (std::sscanf(argv[i], "%u,%u,%u", &cfg.q_chunk_size, &cfg.k_chunk_size, &cfg.heads_per_item) == 3) {
            extra_configs.push_back(cfg);
        } else {
            filter = argv[i];
        }
    }
    const uint32_t num_warmup = env_or("TTML_SDPA_BENCH_WARMUP", 2U);
    const uint32_t num_measure = env_or("TTML_SDPA_BENCH_MEASURE", 10U);
    try {
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 1));
        ttml::autograd::ctx().get_device().enable_program_cache();
        fmt::print(
            "SDPA forward benchmark (warmup={} measure={}, device us per launch, one sync after the timed loop)\n",
            num_warmup,
            num_measure);
        for (const auto& sc : kScenarios) {
            if (!filter.empty() && sc.name.find(filter) == std::string::npos) {
                continue;
            }
            run_scenario(sc, extra_configs, num_warmup, num_measure);
        }
        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "sdpa_fw_benchmark failed: {}\n", e.what());
        return 1;
    }
}
