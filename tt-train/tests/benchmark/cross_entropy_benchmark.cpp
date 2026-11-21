// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Cross-entropy op-level benchmark: the training loss path, ops::cross_entropy_loss with MEAN reduction
//
// The forward is the fused cross_entropy_fw and a moreh_mean; the backward is the fused cross_entropy_bw with the
// upstream gradient folded in. Both use the collapse-resistant row-softmax program shared with the fused
// log_softmax (https://arxiv.org/abs/2610.31642, Section 2). Logits are bf16 in [-8, 8], targets uniform.
//
// Shapes: NanoLlama3's full-batch logits [64, 1, 256, 32000] and smaller batches of it, few rows of a 32000 and a
// Llama-3-sized 128256 vocabulary (rows split across cores), and a narrow 4096 vocabulary.
//
// Each direction is host time per launch over a case's measured launches between two device syncs, the median
// of kNumRepeats samples. Backward replays the graph of an untimed forward. DRAM is the peak of one forward +
// backward step, excluding the logits and targets. Build it on two revisions to compare implementations.
// Run it pinned -- `taskset -c N ./cross_entropy_benchmark`.
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"
#include "test_utils/random_data.hpp"

namespace {

using ttml::autograd::TensorPtr;

constexpr uint32_t kNumRepeats = 5U;
constexpr double kBytesPerMb = 1024.0 * 1024.0;

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct Case {
    std::array<uint32_t, 4> shape;
    uint32_t warmup = 3U;
    uint32_t measure = 20U;
    std::string label;
};

const std::vector<Case>& all_cases() {
    static const std::vector<Case> cases = {
        // training_shakespeare_nanollama3.yaml: batch 64, T = 256, V = 32000.
        {.shape = {64, 1, 256, 32000}, .warmup = 2U, .measure = 3U, .label = "NanoLlama3"},
        {.shape = {8, 1, 256, 32000}, .measure = 10U},
        {.shape = {1, 1, 256, 32000}},
        {.shape = {1, 1, 32, 32000}},
        {.shape = {1, 1, 32, 128256}},
        {.shape = {1, 1, 1024, 4096}},
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
    return ttml::ops::cross_entropy_loss(logits, targets, ttml::ops::ReduceType::MEAN);
}

Row run_case(const Case& c) {
    using namespace ttml::benchmark_utils;
    using ttml::test_utils::make_uniform_vector;
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    const ttnn::Shape shape(std::vector<uint32_t>(c.shape.begin(), c.shape.end()));
    const auto logits =
        ttml::core::from_vector(make_uniform_vector<float>(shape.volume(), -8.F, 8.F, 2026U), shape, device);
    std::vector<uint32_t> target_values(static_cast<size_t>(shape[0]) * shape[2]);
    const auto random_targets =
        make_uniform_vector<float>(target_values.size(), 0.F, static_cast<float>(shape[3]) - 1.F, 2027U);
    std::transform(random_targets.begin(), random_targets.end(), target_values.begin(), [](const float target) {
        return static_cast<uint32_t>(target);
    });
    const auto targets = ttml::autograd::create_tensor(
        ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            target_values, ttnn::Shape({shape[0], shape[2]}), device, ttnn::Layout::ROW_MAJOR),
        /* requires_grad */ false);
    const auto one = ttml::core::ones(ttnn::Shape({1, 1, 1, 1}), device);

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

    const auto dram_peak = capture_dram_peak("cross_entropy_step", [&]() {
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
        "\nCross-entropy, MEAN reduction (µs per launch) + DRAM peak of one step (MB)\n"
        "| Shape                         | Forward   | Backward  | Forward + backward | DRAM peak |\n"
        "|:------------------------------|----------:|----------:|-------------------:|----------:|\n");
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

void BM_CrossEntropy(benchmark::State& state) {
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

        fmt::print("Cross-entropy op-level benchmark (ops::cross_entropy_loss, MEAN)\n");
        fmt::print("bf16 logits; repeats={} (median).\n", kNumRepeats);

        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);
        benchmark::RegisterBenchmark("CrossEntropy", BM_CrossEntropy)
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
        fmt::print(stderr, "cross_entropy_benchmark failed: {}\n", e.what());
        return 1;
    }
}
