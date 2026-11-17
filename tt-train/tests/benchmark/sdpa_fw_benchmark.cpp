// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SDPA forward on the consumer shapes: the program factory's default chunking and schedule against this
// same kernel pinned to the chunking main used (one query tile-row, four K/V tiles per step, no cuts), plus
// any config given on the command line. Prints host microseconds per launch (the timed loop sits between
// two device syncs), the speedup, the DRAM the first launch allocates (the outputs, the logsumexp and, for a
// schedule that cuts, the split units' scratch, which lives on with the cached program) and how many sender
// / merger units the schedule has, so a row that is meant to split can be seen to split. main's own kernel
// is not timed here: build this file on the merge-base with run() calling
// sdpa_fw(query, key, value, sc.mask_type, mask, 0.0F, true, gate) to compare. Measured that way on
// Blackhole p150b (merge-base 0fff850f3c6 vs the chunk-knob branch, us per launch), this kernel at main's
// chunking matches main's kernel within noise, so the speedups below are the chunking (and, with a mask
// tensor, the head stacking) and the schedule, not a kernel change:
//
//   scenario                    main's kernel   this kernel, main's chunking   default pick (no cuts)
//   tinyllama_s2048_causal             1656.4                         1603.8          666.6
//   tinyllama_s4096_causal             6263.0                         6051.3         2192.1
//   llama8b_s2048_causal               3149.2                         3081.7          972.0
//   tinyllama_s2048_nomask             3038.2                         3007.2         1220.8
//   tinyllama_s2048_arbitrary          4103.9                         3825.9         1369.6
//   llama8b_s2048_arbitrary          not timed                         6638.3         2083.7
//   tinyllama_s2048_causal_gate        1704.1                         1669.0          717.5
//
// The per-chip tensor-parallel scenarios at this head, p150b with a 110-core compute grid (warmup 3,
// measured 20). "main pairs" is the program main ran; "default pairs" is the pick's chunking at main's
// placement, never cut, which is what the chunk-only branch (PR #61884) ran; "picked" is this PR. So the
// last two columns separate what #61884 already had from what this PR's placement and cut add:
//
//   scenario                     main pairs   default pairs   picked   picked units   this PR
//   qwen3_8_tp4_s1024_causal          375.7           375.7    248.6        60S/60M     1.51x
//   qwen3_8_tp4_s8192_causal        16786.1          5676.8   5872.2          0S/0M     0.97x
//   llama70b_tp8_s2048_causal         832.1           740.2    402.2        48S/48M     1.84x
//   llama70b_tp8_s8192_causal       11470.5          4509.9   3646.5          0S/0M     1.24x
//   llama8b_tp8_s4096_causal         1590.9          1454.9    697.5        52S/52M     2.09x
//   llama8b_tp4_s2048_causal          831.3           738.0    405.2        48S/48M     1.82x
//   llama8b_tp8_s8192_causal         5706.8          3023.7   1993.6          0S/0M     1.52x
//   tinyllama_tp8_s4096_causal        905.3           943.2    446.3        56S/56M     2.11x
//   llama3_8b_tp4_h8kv2d128_s8k     11465.8          4505.6   3643.0          0S/0M     1.24x
//   llama3_8b_tp4_h8kv2d128_s16k    44879.3         14638.3  13846.4          0S/0M     1.06x
//   llama70b_tp8_h8kv1d128_s16k     44849.4         14657.9  13852.8          0S/0M     1.06x
//
// Every row's output matches main's program's. The shapes that cut are the short ones, where the items
// are too few to fill 110 cores; the long ones gain from longest-first placement alone. One row is a
// loss: qwen3_8_tp4_s8192 runs 3% slower placed longest first than in main's pairs at the same chunking
// (5872.2 against 5676.8), its 384 items spreading over the cores where the pairs sit in runs. These are
// single-chip attention timings, not distributed training.
//
// Long-context training on the same device: NanoLlama3 char, B=1, S=8192, six blocks, Hq/Hkv=6/3,
// D=64, AdamW, seed 5489. nano_gpt's reported median for steps 4..23: 135.8 ms with sdpa_fw pinned
// to q32/k128/h1 at the autograd call, 119.3 ms with the default pick. All 23 logged losses match.
// Both runs use the chunk-only branch kernel; this is not a comparison against a build of main. Reproduce with
// training_shakespeare_nanollama3_char.yaml: batch_size=1, max_steps=23, model_save_interval=0,
// and nanollama3_char.yaml: max_sequence_length=8192. No other model/optimizer settings changed.
//
// Usage: sdpa_fw_benchmark [scenario-name-substring] [shape=B,Hq,Hkv,S,D[,causal|none|arbitrary[,gate]] ...]
//        [q_chunk_size,k_chunk_size,heads_per_item[,max_pieces[,num_cores]] ...]
//        Every scenario measures three rows by default (kDefaultRows): the pick, main's program, and the
//        pick's chunking at main's placement without cuts (PR #61884's program). Configs given on the
//        command line are measured after them.
// Env:   TTML_SDPA_BENCH_WARMUP (default 2), TTML_SDPA_BENCH_MEASURE (default 10)

#include <fmt/format.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
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
// kernel on the TinyLlama shape: no mask, an explicit (causal) mask tensor (also on the Llama-3-8B shape),
// and the output gate. Then the per-chip shapes of tensor-parallel runs: few heads leave the schedule fewer
// items than cores, which is where cutting the heavy causal items pays (pass 0,0,0,1 to time the same rows
// uncut). Then the model table.
const std::vector<Scenario> kScenarios = {
    // Per-chip full-attention shapes: Qwen3.8 TP4 and Llama-70B TP8.
    {"qwen3_8_tp4_s1024_causal", 1U, 6U, 1U, 1024U, 256U, ttml::metal::AttentionMaskType::Causal},
    {"qwen3_8_tp4_s8192_causal", 1U, 6U, 1U, 8192U, 256U, ttml::metal::AttentionMaskType::Causal},
    {"llama70b_tp8_s2048_causal", 1U, 8U, 1U, 2048U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama70b_tp8_s8192_causal", 1U, 8U, 1U, 8192U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s2048_causal", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s4096_causal", 1U, 32U, 4U, 4096U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"llama8b_s2048_causal", 1U, 32U, 8U, 2048U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_s2048_nomask", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::None},
    {"tinyllama_s2048_arbitrary", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Arbitrary},
    {"llama8b_s2048_arbitrary", 1U, 32U, 8U, 2048U, 128U, ttml::metal::AttentionMaskType::Arbitrary},
    {"tinyllama_s2048_causal_gate", 1U, 32U, 4U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal, true},
    {"llama8b_tp8_s4096_causal", 1U, 4U, 1U, 4096U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama8b_tp4_s2048_causal", 1U, 8U, 2U, 2048U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"llama8b_tp8_s8192_causal", 1U, 4U, 1U, 8192U, 128U, ttml::metal::AttentionMaskType::Causal},
    {"tinyllama_tp8_s4096_causal", 1U, 4U, 1U, 4096U, 64U, ttml::metal::AttentionMaskType::Causal},
    {"fewhead_2h_s2048_causal", 1U, 2U, 2U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal},
    // 192 four-row items on 120 to 130 cores: one to two items per core, where cutting the heavy ones
    // puts the mergers on cores the whole items already fill and the scheduler must keep the uncut
    // schedule (it did not, once).
    {"gpt2s_h12d64_s2k", 1U, 12U, 12U, 2048U, 64U, ttml::metal::AttentionMaskType::Causal},
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

// The program main ran: its chunking (one query tile-row per item, four K/V tiles per step, no head
// stacking, no cuts) at its placement (heavy/light causal pairs). The baseline every row is measured
// against, so the speedups are against what main actually ran, not against main's chunking placed some
// other way.
const ttml::metal::SDPAForwardProgramConfig kMainConfig{
    .q_chunk_size = 32U,
    .k_chunk_size = 128U,
    .heads_per_item = 1U,
    .max_pieces = 1U,
    .num_cores = 0U,
    .causal_pairs = true};

// The rows every scenario measures, in order. "default (picked)" is this PR: the pick's chunking, placed
// longest first, cut where the model rates the cut faster. The next two take that apart -- "main pairs" is
// main's program, "default pairs" is the pick's chunking at main's placement and never cut, which is what
// PR #61884 (the chunk-only branch) ran. So the gap between the last two is this PR's placement and cut,
// and the gap between "default pairs" and "main pairs" is the chunking #61884 already had.
const std::vector<std::pair<const char*, ttml::metal::SDPAForwardProgramConfig>> kDefaultRows = {
    {"default (picked)", {}},
    {"main pairs", kMainConfig},
    {"default pairs", {.max_pieces = 1U, .causal_pairs = true}},
};

struct Row {
    std::string label;
    double device_us = 0.0;
    double predicted_us = 0.0;    // what the cost model rated this schedule
    double speedup = 0.0;         // main_chunking_us / device_us, both this kernel
    double dram_alloc_mib = 0.0;  // allocated by the first launch: outputs, logsumexp, scratch
    float max_abs_diff = 0.0F;
    std::string units;  // "<senders>S/<mergers>M" of the schedule
    std::string skipped;
};

struct ScheduleSummary {
    std::string units;         // "<senders>S/<mergers>M"
    double predicted_us = 0.0;  // the model's rating, the number the pick compares candidates on
};

// The schedule the program factory builds for cfg on this shape: sender / merger counts and the model's
// rating. The shape comes from the run's own tensors through shape_of, the same call the factory makes,
// rather than being rebuilt from the scenario here.
ScheduleSummary schedule_summary(
    const ttml::metal::ops::sdpa_fw::device::SDPAForwardShape& shape,
    const ttml::metal::SDPAForwardProgramConfig& cfg) {
    using namespace ttml::metal::ops::sdpa_fw::device;
    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const auto [chunks, schedule] = resolve_plan(cfg, shape, ttml::metal::available_l1_bytes(*device), grid_cores);
    uint32_t senders = 0;
    uint32_t mergers = 0;
    for (const auto& units : schedule.per_core) {
        for (const auto& unit : units) {
            senders += unit.role == UnitRole::Sender ? 1U : 0U;
            mergers += unit.role == UnitRole::Merger ? 1U : 0U;
        }
    }
    return {fmt::format("{}S/{}M", senders, mergers), schedule.predicted_us};
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
    // The shape the factory will resolve, from the tensors the runs use.
    const auto shape = metal::ops::sdpa_fw::device::shape_of(
        query, key, value, sc.mask_type, sc.gate, /* return_intermediates */ true);
    auto time_cfg = [&](const metal::SDPAForwardProgramConfig& cfg) {
        return benchmark_utils::time_device_avg_us(device, num_warmup, num_measure, [&]() { (void)run(cfg); });
    };

    const double main_us = time_cfg(kMainConfig);  // this kernel at main's program, not main's kernel
    const xt::xarray<float> main_out = core::to_xtensor(run(kMainConfig));

    auto measure = [&](const std::string& label, const metal::SDPAForwardProgramConfig& cfg) {
        Row row{.label = label};
        const ScheduleSummary summary = schedule_summary(shape, cfg);
        row.units = summary.units;
        row.predicted_us = summary.predicted_us;
        row.dram_alloc_mib =
            static_cast<double>(benchmark_utils::capture_dram_peak(sc.name + "_" + label, [&]() { (void)run(cfg); })) /
            (1024.0 * 1024.0);
        row.device_us = time_cfg(cfg);
        row.speedup = benchmark_utils::speedup_x(main_us, row.device_us);
        const xt::xarray<float> out = core::to_xtensor(run(cfg));
        row.max_abs_diff = static_cast<float>(xt::amax(xt::abs(out - main_out))());
        return row;
    };

    // The default rows are what the comparison is about, so a failure there fails the benchmark; the
    // command-line configs are optional and a config the shape cannot take is reported as skipped.
    std::vector<Row> rows;
    for (const auto& [label, cfg] : kDefaultRows) {
        rows.push_back(measure(label, cfg));
    }
    for (const auto& cfg : extra_configs) {
        const std::string label = fmt::format(
            "q{} k{} h{} p{} c{}{}",
            cfg.q_chunk_size,
            cfg.k_chunk_size,
            cfg.heads_per_item,
            cfg.max_pieces,
            cfg.num_cores,
            cfg.causal_pairs ? " pairs" : "");
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
    const auto grid = device.compute_with_storage_grid_size();
    const auto picked =
        metal::ops::sdpa_fw::device::resolve_chunks({}, shape, metal::available_l1_bytes(device), grid.x * grid.y);
    fmt::print(
        "default chunks: q={} k={} h={} on {} cores\n",
        picked.Sq_chunk_t * tt::constants::TILE_HEIGHT,
        picked.Sk_chunk_t * tt::constants::TILE_HEIGHT,
        picked.heads_per_item,
        grid.x * grid.y);
    fmt::print("this kernel at main's program (q=32, k=128, h=1, pairs, no cuts): {:.1f} us\n", main_us);
    // predicted_us is what the pick compares its candidates on; pred/meas next to it is the model's drift,
    // so a schedule the model rates well and the device does not shows up here (the cost model's constants
    // are calibrated on p150b and sum the reader's DRAM time and the compute passes, which assumes the two
    // do not overlap; a ratio that moves with the K/V step is where that assumption breaks).
    fmt::print(
        "{:<24} {:>12} {:>12} {:>10} {:>9} {:>13} {:>12} {:>10}\n",
        "config",
        "device_us",
        "predicted_us",
        "pred/meas",
        "speedup",
        "dram_alloc_MiB",
        "max_abs_diff",
        "units");
    for (const Row& row : rows) {
        if (row.skipped.empty()) {
            fmt::print(
                "{:<24} {:>12.1f} {:>12.1f} {:>10.3f} {:>8.3f}x {:>13.1f} {:>12.3e} {:>10}\n",
                row.label,
                row.device_us,
                row.predicted_us,
                row.device_us > 0.0 ? row.predicted_us / row.device_us : 0.0,
                row.speedup,
                row.dram_alloc_mib,
                row.max_abs_diff,
                row.units);
        } else {
            fmt::print(
                "{:<24} {:>12} {:>12} {:>10} {:>9} {:>13} {:>12} {:>10}  skipped: {}\n",
                row.label,
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                row.skipped);
        }
    }
}

// shape=B,Hq,Hkv,S,D[,causal|none|arbitrary[,gate]]: a scenario that is not in the table.
std::optional<Scenario> scenario_of(const char* arg) {
    Scenario sc;
    char mask[16] = "causal";
    char gate[8] = "";
    if (std::sscanf(
            arg,
            "shape=%u,%u,%u,%u,%u,%15[a-z],%7[a-z]",
            &sc.batch,
            &sc.q_heads,
            &sc.kv_heads,
            &sc.seq_len,
            &sc.head_dim,
            mask,
            gate) < 5) {
        return std::nullopt;
    }
    const std::string_view mask_name = mask;
    sc.mask_type = mask_name == "none"        ? ttml::metal::AttentionMaskType::None
                   : mask_name == "arbitrary" ? ttml::metal::AttentionMaskType::Arbitrary
                                              : ttml::metal::AttentionMaskType::Causal;
    sc.gate = std::string_view(gate) == "gate";
    sc.name = fmt::format(
        "shape_b{}_h{}_kv{}_s{}_d{}_{}{}",
        sc.batch,
        sc.q_heads,
        sc.kv_heads,
        sc.seq_len,
        sc.head_dim,
        mask_name,
        sc.gate ? "_gate" : "");
    return sc;
}

}  // namespace

int main(int argc, char** argv) {
    std::string filter;
    std::vector<ttml::metal::SDPAForwardProgramConfig> extra_configs;
    std::vector<Scenario> scenarios;
    for (int i = 1; i < argc; ++i) {
        ttml::metal::SDPAForwardProgramConfig cfg{};
        if (const auto sc = scenario_of(argv[i])) {
            scenarios.push_back(*sc);
        } else if (
            std::sscanf(
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
        if (scenarios.empty()) {
            scenarios = kScenarios;
        }
        for (const auto& sc : scenarios) {
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
