// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Brute-force sweep of the SDPA forward program-config knobs (q_chunk_size, k_chunk_size)
// on training-like attention shapes. For every scenario it times the default configuration,
// then every valid knob combination, checks each one's output against the default's, and
// prints a table sorted by device time. Pure measurement tool: it changes nothing about how
// the kernel picks its defaults.
//
// Usage: sdpa_fw_chunk_sweep_benchmark [scenario-name-substring]
// Env:   TTML_SDPA_SWEEP_WARMUP (default 2), TTML_SDPA_SWEEP_MEASURE (default 10)

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
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

namespace {

struct Scenario {
    std::string name;
    uint32_t batch;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t seq_len;
    uint32_t head_dim;
    ttml::metal::AttentionMaskType mask_type;
};

// Shapes tt-train actually trains: TinyLlama 1.1B (32 q heads / 4 kv heads / 64 dim) and a
// Llama-3-8B style block (32 / 8 / 128). Causal is the training case; the no-mask row shows
// the knobs without the triangular work pattern.
const std::vector<Scenario> kScenarios = {
    {"tinyllama_s2048_causal", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s4096_causal", 1U, 32U, 4U, 4096U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"llama8b_s2048_causal", 1U, 32U, 8U, 2048U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s2048_nomask", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::None},
};

const std::vector<uint32_t> kQChunkSizes = {32U, 64U, 128U, 256U};
const std::vector<uint32_t> kKChunkSizes = {32U, 64U, 128U, 256U, 512U};

struct Row {
    uint32_t q_chunk_size;
    uint32_t k_chunk_size;
    double device_us;
    double speedup;       // default_us / device_us
    float max_abs_diff;   // vs the default configuration's output
    std::string skipped;  // non-empty when the configuration was refused
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

void run_scenario(const Scenario& sc, uint32_t num_warmup, uint32_t num_measure) {
    using namespace ttml;
    auto& device = autograd::ctx().get_device();

    const std::array<std::size_t, 4> q_shape{sc.batch, sc.q_heads, sc.seq_len, sc.head_dim};
    const std::array<std::size_t, 4> kv_shape{sc.batch, sc.kv_heads, sc.seq_len, sc.head_dim};
    const uint32_t seed = benchmark_utils::seed_from_name(sc.name);
    auto query = core::from_xtensor(test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, seed), &device);
    auto key = core::from_xtensor(test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, seed + 1U), &device);
    auto value =
        core::from_xtensor(test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, seed + 2U), &device);

    auto run = [&](const metal::SDPAForwardProgramConfig& cfg) {
        return metal::sdpa_fw(
            query, key, value, sc.mask_type, std::nullopt, 0.0F, /*return_intermediates=*/true, std::nullopt, cfg)[0]
            .value();
    };
    auto time_cfg = [&](const metal::SDPAForwardProgramConfig& cfg) {
        return benchmark_utils::time_device_avg_us(device, num_warmup, num_measure, [&]() { (void)run(cfg); });
    };

    const metal::SDPAForwardProgramConfig default_cfg{};
    const double default_us = time_cfg(default_cfg);
    const xt::xarray<float> default_out = core::to_xtensor(run(default_cfg));

    std::vector<Row> rows;
    for (const uint32_t q : kQChunkSizes) {
        for (const uint32_t k : kKChunkSizes) {
            const metal::SDPAForwardProgramConfig cfg{.q_chunk_size = q, .k_chunk_size = k};
            Row row{.q_chunk_size = q, .k_chunk_size = k, .device_us = 0.0, .speedup = 0.0, .max_abs_diff = 0.0F};
            try {
                row.device_us = time_cfg(cfg);
                row.speedup = benchmark_utils::speedup_x(default_us, row.device_us);
                const xt::xarray<float> out = core::to_xtensor(run(cfg));
                row.max_abs_diff = static_cast<float>(xt::amax(xt::abs(out - default_out))());
            } catch (const std::exception& e) {
                // First line of the error only: TT_FATAL messages carry a backtrace.
                std::string_view what = e.what();
                row.skipped = std::string(what.substr(0, what.find('\n')));
            }
            rows.push_back(std::move(row));
        }
    }

    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.skipped.empty() != b.skipped.empty()) {
            return a.skipped.empty();
        }
        return a.device_us < b.device_us;
    });

    fmt::print(
        "\n== {}: B={} Hq={} Hkv={} S={} D={} mask={} ==\n",
        sc.name,
        sc.batch,
        sc.q_heads,
        sc.kv_heads,
        sc.seq_len,
        sc.head_dim,
        sc.mask_type == metal::AttentionMaskType::Causal ? "causal" : "none");
    fmt::print("default (q=0,k=0): {:.1f} us\n", default_us);
    fmt::print("{:>6} {:>6} {:>12} {:>9} {:>12}  {}\n", "q", "k", "device_us", "speedup", "max_abs_diff", "note");
    for (const Row& row : rows) {
        if (row.skipped.empty()) {
            fmt::print(
                "{:>6} {:>6} {:>12.1f} {:>8.3f}x {:>12.3e}\n",
                row.q_chunk_size,
                row.k_chunk_size,
                row.device_us,
                row.speedup,
                row.max_abs_diff);
        } else {
            fmt::print("{:>6} {:>6} {:>12} {:>9} {:>12}  skipped: {}\n", row.q_chunk_size, row.k_chunk_size, "-", "-", "-", row.skipped);
        }
    }
    if (!rows.empty() && rows.front().skipped.empty()) {
        fmt::print(
            "best: q_chunk_size={} k_chunk_size={} -> {:.1f} us ({:.3f}x vs default)\n",
            rows.front().q_chunk_size,
            rows.front().k_chunk_size,
            rows.front().device_us,
            rows.front().speedup);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    const uint32_t num_warmup = env_or("TTML_SDPA_SWEEP_WARMUP", 2U);
    const uint32_t num_measure = env_or("TTML_SDPA_SWEEP_MEASURE", 10U);
    try {
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 1));
        ttml::autograd::ctx().get_device().enable_program_cache();
        fmt::print(
            "SDPA forward q_chunk_size x k_chunk_size sweep (warmup={} measure={}, times are host us per launch "
            "between device syncs)\n",
            num_warmup,
            num_measure);
        for (const auto& sc : kScenarios) {
            if (!filter.empty() && sc.name.find(filter) == std::string::npos) {
                continue;
            }
            run_scenario(sc, num_warmup, num_measure);
        }
        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "sdpa_fw_chunk_sweep_benchmark failed: {}\n", e.what());
        return 1;
    }
}
