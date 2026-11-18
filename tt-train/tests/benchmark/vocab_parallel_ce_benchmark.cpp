// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Vocabulary-parallel cross-entropy op-level benchmark: the tensor-parallel training loss,
// ops::distributed::vocab_parallel_cross_entropy_loss with MEAN reduction
//
// The mesh is [1, tp] with tp = 8, 4 or 2, the largest the system holds (nano_gpt's
// training_shakespeare_llama3_tensor_parallel.yaml uses [1, 8]). Logits are bf16 in [-8, 8], sharded on the
// vocabulary along mesh axis 1 (cluster_axis = 1, as nano_gpt calls it); W is the full vocabulary and each device
// holds W / tp. Targets are uniform and replicated.
//
// Shapes: NanoLlama3's full-batch logits [64, 1, 256, 32000], TinyLlama's [5, 1, 2048, 32000], smaller batches,
// and few rows of a 32000 and a Llama-3-sized 128256 vocabulary.
//
// Each direction is host time per launch over a case's measured launches between two mesh syncs, the median of
// kNumRepeats samples. Backward replays the graph of an untimed forward. DRAM is per device, the peak of one
// forward + backward step, excluding the logits and targets. Build it on two revisions to compare implementations.
// Run it pinned -- `taskset -c N ./vocab_parallel_ce_benchmark`.
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <tt-metalium/system_mesh.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/losses.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

using ttml::autograd::TensorPtr;

constexpr uint32_t kNumRepeats = 5U;
constexpr double kBytesPerMb = 1024.0 * 1024.0;

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

uint32_t tp_size() {
    const size_t devices = tt::tt_metal::distributed::SystemMesh::instance().shape().mesh_size();
    return devices >= 8U ? 8U : (devices >= 4U ? 4U : 2U);
}

struct Case {
    std::array<uint32_t, 4> shape;
    uint32_t warmup = 3U;
    uint32_t measure = 20U;
    std::string label;
};

const std::vector<Case>& all_cases() {
    static const std::vector<Case> cases = {
        // training_shakespeare_llama3_tensor_parallel.yaml: batch 64; llama3.yaml: T = 256, V = 32000.
        {.shape = {64, 1, 256, 32000}, .warmup = 2U, .measure = 3U, .label = "NanoLlama3"},
        // training_shakespeare_tinyllama.yaml under TP: batch 5; tinyllama_loudbox_tp.yaml: T = 2048, V = 32000.
        {.shape = {5, 1, 2048, 32000}, .warmup = 2U, .measure = 3U, .label = "TinyLlama"},
        {.shape = {8, 1, 256, 32000}, .measure = 10U},
        {.shape = {1, 1, 256, 32000}},
        {.shape = {1, 1, 32, 32000}},
        {.shape = {1, 1, 32, 128256}},
    };
    return cases;
}

struct Row {
    Case bench_case;
    double forward_us = 0.0;
    double backward_us = 0.0;
    double dram_peak_mb = 0.0;
};

TensorPtr loss_of(const TensorPtr& logits, const TensorPtr& targets) {
    return ttml::ops::distributed::vocab_parallel_cross_entropy_loss(
        logits, targets, /* cluster_axis */ 1U, ttml::ops::ReduceType::MEAN);
}

Row run_case(const Case& c) {
    using namespace ttml::benchmark_utils;
    using ttml::test_utils::make_uniform_vector;
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    const ttnn::Shape shape(std::vector<uint32_t>(c.shape.begin(), c.shape.end()));

    auto logits_mapper = ttnn::distributed::TensorToMesh::create(
        *device, {.placements = ttsl::SmallVector<Placement>{Replicate{}, Shard{3}}});
    auto replicate_mapper = ttnn::distributed::TensorToMesh::create(
        *device, {.placements = ttsl::SmallVector<Placement>{Replicate{}, Replicate{}}});
    xt::xarray<float> logits_xt = xt::adapt(
        make_uniform_vector<float>(shape.volume(), -8.F, 8.F, 2026U),
        std::vector<size_t>(c.shape.begin(), c.shape.end()));
    const auto logits = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        logits_xt, device, ttnn::Layout::TILE, &logits_mapper);
    std::vector<uint32_t> target_values(static_cast<size_t>(shape[0]) * shape[2]);
    const auto random_targets =
        make_uniform_vector<float>(target_values.size(), 0.F, static_cast<float>(shape[3]) - 1.F, 2027U);
    std::transform(random_targets.begin(), random_targets.end(), target_values.begin(), [](const float target) {
        return static_cast<uint32_t>(target);
    });
    xt::xarray<uint32_t> targets_xt = xt::adapt(target_values, std::vector<size_t>{shape[0], shape[2]});
    const auto targets = ttml::autograd::create_tensor(
        ttml::core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
            targets_xt, device, ttnn::Layout::ROW_MAJOR, &replicate_mapper),
        /* requires_grad */ false);
    xt::xarray<float> one_xt = xt::ones<float>({1U, 1U, 1U, 1U});
    const auto one = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        one_xt, device, ttnn::Layout::TILE, &replicate_mapper);

    std::vector<double> forward_us;
    std::vector<double> backward_us;
    for (uint32_t repeat = 0; repeat < kNumRepeats; ++repeat) {
        forward_us.push_back(time_device_avg_us(*device, c.warmup, c.measure, [&]() {
            auto input = ttml::autograd::create_tensor(logits, /* requires_grad */ true);
            (void)loss_of(input, targets);
            ttml::autograd::ctx().reset_graph();
        }));
        auto input = ttml::autograd::create_tensor(logits, /* requires_grad */ true);
        auto loss = loss_of(input, targets);
        loss->set_grad(one);
        backward_us.push_back(time_device_avg_us(*device, c.warmup, c.measure, [&]() {
            input->set_grad(ttnn::Tensor());
            loss->backward(/* retain_graph */ true);
        }));
        ttml::autograd::ctx().reset_graph();
    }

    const auto dram_peak = capture_dram_peak("vocab_parallel_ce_step", [&]() {
        auto input = ttml::autograd::create_tensor(logits, /* requires_grad */ true);
        auto loss = loss_of(input, targets);
        loss->set_grad(one);
        loss->backward();
        ttml::autograd::ctx().reset_graph();
    });
    return Row{
        .bench_case = c,
        .forward_us = median(forward_us),
        .backward_us = median(backward_us),
        .dram_peak_mb = static_cast<double>(dram_peak) / kBytesPerMb,
    };
}

std::string case_label(const Case& c) {
    auto label = fmt::format("[{}, {}, {}, {}]", c.shape[0], c.shape[1], c.shape[2], c.shape[3]);
    return c.label.empty() ? label : fmt::format("{} {}", label, c.label);
}

void print_table(const std::vector<Row>& rows) {
    fmt::print(
        "\nVocabulary-parallel cross-entropy, TP{}, MEAN reduction (µs per launch) + DRAM peak of one step per device "
        "(MB)\n"
        "| Shape                         | Forward   | Backward  | Forward + backward | DRAM peak |\n"
        "|:------------------------------|----------:|----------:|-------------------:|----------:|\n",
        tp_size());
    for (const auto& row : rows) {
        fmt::print(
            "| {:<29} | {:>9.1f} | {:>9.1f} | {:>18.1f} | {:>9.1f} |\n",
            case_label(row.bench_case),
            row.forward_us,
            row.backward_us,
            row.forward_us + row.backward_us,
            row.dram_peak_mb);
    }
}

std::vector<Row> g_rows;

void BM_VocabParallelCrossEntropy(benchmark::State& state) {
    const auto& c = all_cases().at(static_cast<size_t>(state.range(0)));
    for ([[maybe_unused]] auto _ : state) {
        const auto row = run_case(c);
        g_rows.push_back(row);
        state.SetIterationTime((row.forward_us + row.backward_us) / 1e6);
        state.SetLabel(case_label(c));
        state.counters["Forward_us"] = row.forward_us;
        state.counters["Backward_us"] = row.backward_us;
        state.counters["DramPeak_MB"] = row.dram_peak_mb;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }

        const uint32_t tp = tp_size();
        fmt::print(
            "Vocabulary-parallel cross-entropy op-level benchmark "
            "(ops::distributed::vocab_parallel_cross_entropy_loss, "
            "MEAN)\n");
        fmt::print("[1, {}] mesh; bf16 logits sharded on the vocabulary; repeats={} (median).\n", tp, kNumRepeats);

        ttml::ttnn_fixed::distributed::enable_fabric(tp);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, tp));
        benchmark::RegisterBenchmark("VocabParallelCrossEntropy", BM_VocabParallelCrossEntropy)
            ->DenseRange(0, static_cast<int>(all_cases().size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        ttml::autograd::ctx().close_device();

        print_table(g_rows);
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "vocab_parallel_ce_benchmark failed: {}\n", e.what());
        return 1;
    }
}
