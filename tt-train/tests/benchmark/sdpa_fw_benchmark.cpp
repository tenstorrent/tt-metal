// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SDPA forward on the consumer shapes: the program factory's default chunking and schedule against this
// same kernel pinned to the chunking main used (one query tile-row, four K/V tiles per step, no cuts), plus
// any config given on the command line. Prints host microseconds per launch (the timed loop sits between
// two device syncs), the speedup, the DRAM peak of the first launch (the split units' scratch is allocated
// then and lives with the cached program) and how many sender / merger units the schedule has, so a row
// that is meant to split can be seen to split. main's own kernel is not timed here: build the merge-base
// to compare.
//
// Usage: sdpa_fw_benchmark [scenario-name-substring] [q_chunk_size,k_chunk_size,heads_per_item[,max_pieces[,num_cores]]
// ...] Env:   TTML_SDPA_BENCH_WARMUP (default 2), TTML_SDPA_BENCH_MEASURE (default 10)

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
#include "metal/common/program_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_config.hpp"
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
// kernel on the TinyLlama shape: no mask, an explicit (causal) mask tensor, and the output gate. Then the
// per-chip shapes of tensor-parallel runs: few heads leave the schedule fewer items than cores, which is
// where cutting the heavy causal items pays (pass 0,0,0,1 to time the same rows uncut). Then the model
// table.
const std::vector<Scenario> kScenarios = {
    {"tinyllama_s2048_causal", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s4096_causal", 1U, 32U, 4U, 4096U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"llama8b_s2048_causal", 1U, 32U, 8U, 2048U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s2048_nomask", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::None},
    {"tinyllama_s2048_arbitrary", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Arbitrary},
    {"tinyllama_s2048_causal_gate", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal, true},
    {"llama8b_tp8_s4096_causal", 1U, 4U, 1U, 4096U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama8b_tp4_s2048_causal", 1U, 8U, 2U, 2048U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama8b_tp8_s8192_causal", 1U, 4U, 1U, 8192U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_tp8_s4096_causal", 1U, 4U, 1U, 4096U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"fewhead_2h_s2048_causal", 1U, 2U, 2U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal},
    // One chip's attention of the models tt-train trains or is asked about, at S=2k, 8k and 16k; the
    // table in the program factory's PR is this list. None of these cuts an item: with 32 heads there are
    // more items than cores at every length, and the 6 to 8-head shapes only at S=16k.
    {"llama3_8b_h32kv8d128_s2k", 1U, 32U, 8U, 2048U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama3_8b_h32kv8d128_s8k", 1U, 32U, 8U, 8192U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama3_8b_h32kv8d128_s16k", 1U, 32U, 8U, 16384U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama3_8b_tp4_h8kv2d128_s8k", 1U, 8U, 2U, 8192U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama3_8b_tp4_h8kv2d128_s16k", 1U, 8U, 2U, 16384U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama70b_tp8_h8kv1d128_s16k", 1U, 8U, 1U, 16384U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama2_7b_h32d128_s8k", 1U, 32U, 32U, 8192U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama32_1b_h32kv8d64_s8k", 1U, 32U, 8U, 8192U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"llama32_1b_h32kv8d64_s16k", 1U, 32U, 8U, 16384U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"llama32_3b_h24kv8d128_s8k", 1U, 24U, 8U, 8192U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"qwen25_7b_h28kv4d128_s8k", 1U, 28U, 4U, 8192U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"qwen38_h6kv1d256_s16k", 1U, 6U, 1U, 16384U, 256U, ttml::metal::AttentionMaskType::Causal},
    {"gemma7b_h16d256_s8k", 1U, 16U, 16U, 8192U, 256U, ttml::metal::AttentionMaskType::Causal},
    {"gpt2s_h12d64_s8k", 1U, 12U, 12U, 8192U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"gpt2s_h12d64_s16k", 1U, 12U, 12U, 16384U, 64U, ttml::metal::AttentionMaskType::Causal},
};

// Why a config was refused: a TT_FATAL's what() is "TT_FATAL @ file:line: condition", then "info:" and the
// message on the next line, then a backtrace; the message is the line that names the knob and its limit.
std::string reason_of(const std::exception& e) {
    const std::string_view what = e.what();
    const std::string_view info_tag = "info:\n";
    std::size_t from = what.find(info_tag);
    from = from == std::string_view::npos ? 0U : from + info_tag.size();
    return std::string(what.substr(from, what.find('\n', from) - from));
}

// The chunking main ran: one query tile-row per item, four K/V tiles per step, no head stacking, no cuts.
const ttml::metal::SDPAForwardProgramConfig kMainConfig{
    .q_chunk_size = 32U, .k_chunk_size = 128U, .heads_per_item = 1U, .max_pieces = 1U};

struct Row {
    std::string label;
    double device_us = 0.0;
    double speedup = 0.0;  // main_us / device_us, both this kernel
    double dram_peak_mib = 0.0;
    float max_abs_diff = 0.0F;
    std::string units;  // "<senders>S/<mergers>M" of the schedule
    std::string skipped;
};

// The schedule the program factory builds for cfg on sc, summarised as sender / merger counts.
std::string schedule_units(const Scenario& sc, const ttml::metal::SDPAForwardProgramConfig& cfg) {
    using namespace ttml::metal::ops::sdpa_fw::device;
    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const SDPAForwardShape shape{
        .batch = sc.batch,
        .q_heads = sc.q_heads,
        .kv_heads = sc.kv_heads,
        .St = sc.seq_len / tt::constants::TILE_HEIGHT,
        .qWt = sc.head_dim / tt::constants::TILE_WIDTH,
        .vWt = sc.head_dim / tt::constants::TILE_WIDTH,
        .mask_type = sc.mask_type,
        .has_gate = sc.gate,
        .return_intermediates = true};
    const SDPAForwardChunks chunks = resolve_chunks(cfg, shape, ttml::metal::available_l1_bytes(*device), grid_cores);
    uint32_t senders = 0;
    uint32_t mergers = 0;
    for (const auto& units : schedule_for(cfg, shape, chunks, grid_cores).per_core) {
        for (const auto& unit : units) {
            senders += unit.role == UnitRole::Sender ? 1U : 0U;
            mergers += unit.role == UnitRole::Merger ? 1U : 0U;
        }
    }
    return fmt::format("{}S/{}M", senders, mergers);
}

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

    auto measure = [&](const std::string& label, const metal::SDPAForwardProgramConfig& cfg) {
        Row row{.label = label};
        row.units = schedule_units(sc, cfg);
        row.dram_peak_mib =
            static_cast<double>(benchmark_utils::capture_dram_peak(sc.name + "_" + label, [&]() { (void)run(cfg); })) /
            (1024.0 * 1024.0);
        row.device_us = time_cfg(cfg);
        row.speedup = benchmark_utils::speedup_x(main_us, row.device_us);
        const xt::xarray<float> out = core::to_xtensor(run(cfg));
        row.max_abs_diff = static_cast<float>(xt::amax(xt::abs(out - main_out))());
        return row;
    };

    // The default pick is what the model runs, so its failure fails the benchmark; the command-line
    // configs are optional and a config the shape cannot take is reported as skipped.
    std::vector<Row> rows;
    rows.push_back(measure("default (picked)", {}));
    for (const auto& cfg : extra_configs) {
        const std::string label = fmt::format(
            "q{} k{} h{} p{} c{}",
            cfg.q_chunk_size,
            cfg.k_chunk_size,
            cfg.heads_per_item,
            cfg.max_pieces,
            cfg.num_cores);
        try {
            rows.push_back(measure(label, cfg));
        } catch (const std::exception& e) {
            rows.push_back(Row{.label = label, .skipped = reason_of(e)});
        }
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
    fmt::print("this kernel at main's chunking (q=32, k=128, h=1, no cuts): {:.1f} us\n", main_us);
    fmt::print(
        "{:<24} {:>12} {:>9} {:>13} {:>12} {:>10}\n",
        "config",
        "device_us",
        "speedup",
        "dram_peak_MiB",
        "max_abs_diff",
        "units");
    for (const Row& row : rows) {
        if (row.skipped.empty()) {
            fmt::print(
                "{:<24} {:>12.1f} {:>8.3f}x {:>13.1f} {:>12.3e} {:>10}\n",
                row.label,
                row.device_us,
                row.speedup,
                row.dram_peak_mib,
                row.max_abs_diff,
                row.units);
        } else {
            fmt::print(
                "{:<24} {:>12} {:>9} {:>13} {:>12} {:>10}  skipped: {}\n",
                row.label,
                "-",
                "-",
                "-",
                "-",
                "-",
                row.skipped);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string filter;
    std::vector<ttml::metal::SDPAForwardProgramConfig> extra_configs;
    for (int i = 1; i < argc; ++i) {
        ttml::metal::SDPAForwardProgramConfig cfg{};
        if (std::sscanf(
                argv[i],
                "%u,%u,%u,%u,%u",
                &cfg.q_chunk_size,
                &cfg.k_chunk_size,
                &cfg.heads_per_item,
                &cfg.max_pieces,
                &cfg.num_cores) >= 3) {
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
            "SDPA forward benchmark (warmup={} measure={}, host us per launch between two device syncs)\n",
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
