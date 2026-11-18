// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// log_softmax op-level benchmark: the fused, collapse-resistant ops::log_softmax against the composites
//
// Arms, all autograd ops on the same bf16 input and upstream gradient:
//   original   the pre-fix composite (max, exp, sum, approximate log), frozen here as the baseline. It has the
//              softmax collapse of https://arxiv.org/abs/2610.31642 (Section 2).
//   composite  ops::log_softmax_composite, the corrected composite (one selected maximum, log1p of the tail).
//   fused      ops::log_softmax, one device program forward and one backward.
//
// Shapes cover small widths, NanoLlama3's full-batch logits [64, 1, 256, 32000], few rows of a 32000 and a
// Llama-3-sized 128256 vocabulary (where the fused op splits rows across cores), and a reduction over dim 2.
//
// Each direction is host time per launch over a case's measured launches between two device syncs, the median
// of kNumRepeats samples taken round-robin over the arms. Backward replays the graph of an untimed forward. DRAM is
// the peak of one forward + backward step, excluding the input and the upstream gradient.
// Run it pinned -- `taskset -c N ./log_softmax_benchmark`.
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

using ttml::autograd::TensorPtr;

constexpr uint32_t kNumRepeats = 5U;
constexpr double kBytesPerMb = 1024.0 * 1024.0;

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// The composite as it was before the collapse fix, with its bf16 intermediates and approximate log.
TensorPtr original_log_softmax(const TensorPtr& input, int dim) {
    auto maximum = ttnn::max(input->get_value(), dim, /* keepdim */ true);
    auto shifted = ttnn::subtract(input->get_value(), maximum);
    auto denominator = ttml::ttnn_fixed::sum_over_dim(ttnn::exp(shifted), dim);
    auto log_denominator = ttnn::log(denominator, /* fast_and_approximate_mode */ true);
    auto output = ttml::autograd::create_tensor(ttnn::subtract(shifted, log_denominator));
    ttml::autograd::GradFunction backward = [input, output, dim]() {
        auto probabilities = ttnn::exp(output->get_value());
        auto sum_grad = ttml::ttnn_fixed::sum_over_dim(output->get_grad(), dim);
        input->add_grad(ttnn::subtract(output->get_grad(), ttnn::multiply(probabilities, sum_grad)));
    };
    output->set_node(ttml::autograd::add_backward_node(std::move(backward), output, input));
    return output;
}

using LogSoftmax = TensorPtr (*)(const TensorPtr&, int);

enum class Arm { Original, Composite, Fused };
constexpr std::array<Arm, 3> kArms{Arm::Original, Arm::Composite, Arm::Fused};

LogSoftmax arm_op(const Arm arm) {
    switch (arm) {
        case Arm::Original: return original_log_softmax;
        case Arm::Composite: return ttml::ops::log_softmax_composite;
        case Arm::Fused: return ttml::ops::log_softmax;
    }
    return nullptr;
}

struct Case {
    std::array<uint32_t, 4> shape;
    int dim = 3;
    uint32_t warmup = 3U;
    uint32_t measure = 20U;
    std::string label;
};

const std::vector<Case>& all_cases() {
    static const std::vector<Case> cases = {
        {.shape = {1, 1, 32, 128}},
        {.shape = {1, 1, 128, 1024}},
        {.shape = {1, 1, 1024, 4096}},
        {.shape = {1, 1, 1024, 4096}, .dim = 2},
        {.shape = {1, 1, 32, 32000}},
        {.shape = {1, 1, 256, 32000}},
        {.shape = {1, 1, 32, 128256}},
        {.shape = {1, 1, 512, 128256}, .warmup = 2U, .measure = 5U},
        // training_shakespeare_nanollama3.yaml: batch 64, T = 256, V = 32000. The composite takes ~0.4 s per step.
        {.shape = {64, 1, 256, 32000}, .warmup = 1U, .measure = 2U, .label = "NanoLlama3"},
    };
    return cases;
}

struct ArmResult {
    // False when the arm ran out of DRAM (the composite at NanoLlama3 needs ~15 GB, more than a Wormhole chip).
    bool fits = true;
    double forward_us = 0.0;
    double backward_us = 0.0;
    double dram_peak_mb = 0.0;

    double total_us() const {
        return forward_us + backward_us;
    }
};

using ArmResults = std::array<ArmResult, kArms.size()>;

struct Row {
    Case bench_case;
    ArmResults arms;
};

std::string case_label(const Case& c) {
    auto label = fmt::format("[{}, {}, {}, {}] dim {}", c.shape[0], c.shape[1], c.shape[2], c.shape[3], c.dim);
    return c.label.empty() ? label : fmt::format("{} {}", label, c.label);
}

void forward_step(const LogSoftmax op, const ttnn::Tensor& x, const int dim) {
    auto input = ttml::autograd::create_tensor(x, /* requires_grad */ true);
    (void)op(input, dim);
    ttml::autograd::ctx().reset_graph();
}

void forward_backward_step(const LogSoftmax op, const ttnn::Tensor& x, const ttnn::Tensor& dy, const int dim) {
    auto input = ttml::autograd::create_tensor(x, /* requires_grad */ true);
    auto output = op(input, dim);
    output->set_grad(dy);
    output->backward();
    ttml::autograd::ctx().reset_graph();
}

Row run_case(const Case& c) {
    using namespace ttml::benchmark_utils;
    using ttml::test_utils::make_uniform_vector;
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    const ttnn::Shape shape(std::vector<uint32_t>(c.shape.begin(), c.shape.end()));
    const auto x = ttml::core::from_vector(make_uniform_vector<float>(shape.volume(), -1.F, 1.F, 2026U), shape, device);
    const auto dy =
        ttml::core::from_vector(make_uniform_vector<float>(shape.volume(), -1.F, 1.F, 2027U), shape, device);

    Row row{.bench_case = c};
    std::array<std::vector<double>, kArms.size()> forward_us;
    std::array<std::vector<double>, kArms.size()> backward_us;
    for (uint32_t repeat = 0; repeat < kNumRepeats; ++repeat) {
        for (size_t i = 0; i < kArms.size(); ++i) {
            if (!row.arms[i].fits) {
                continue;
            }
            const auto op = arm_op(kArms[i]);
            try {
                forward_us[i].push_back(
                    time_device_avg_us(*device, c.warmup, c.measure, [&]() { forward_step(op, x, c.dim); }));
                // The backward replays the graph of one untimed forward.
                auto input = ttml::autograd::create_tensor(x, /* requires_grad */ true);
                auto output = op(input, c.dim);
                output->set_grad(dy);
                backward_us[i].push_back(time_device_avg_us(*device, c.warmup, c.measure, [&]() {
                    input->set_grad(ttnn::Tensor());
                    output->backward(/* retain_graph */ true);
                }));
            } catch (const std::exception& e) {
                fmt::print("{}: arm {} skipped: {}\n", case_label(c), i, e.what());
                row.arms[i].fits = false;
            }
            ttml::autograd::ctx().reset_graph();
        }
    }

    for (size_t i = 0; i < kArms.size(); ++i) {
        if (!row.arms[i].fits) {
            continue;
        }
        const auto op = arm_op(kArms[i]);
        const auto dram_peak =
            capture_dram_peak("log_softmax_step", [&]() { forward_backward_step(op, x, dy, c.dim); });
        row.arms[i].forward_us = median(forward_us[i]);
        row.arms[i].backward_us = median(backward_us[i]);
        row.arms[i].dram_peak_mb = static_cast<double>(dram_peak) / kBytesPerMb;
    }
    return row;
}

// A table cell: the value, or "n/a" for an arm that did not fit in DRAM.
std::string cell(const bool fits, const double value) {
    return fits ? fmt::format("{:.1f}", value) : std::string("n/a");
}

std::string speedup_cell(const ArmResult& baseline, const double baseline_value, const double fused_value) {
    return baseline.fits ? fmt::format("{:.2f}x", ttml::benchmark_utils::speedup_x(baseline_value, fused_value))
                         : std::string("n/a");
}

void print_direction_table(const std::vector<Row>& rows, const bool forward) {
    const auto pick = [forward](const ArmResult& r) { return forward ? r.forward_us : r.backward_us; };
    fmt::print(
        "\n{} (µs per launch)\n"
        "| Shape                                | Original  | Composite  | Fused     | vs original | vs composite |\n"
        "|:-------------------------------------|----------:|-----------:|----------:|------------:|-------------:|\n",
        forward ? "Forward" : "Backward");
    for (const auto& row : rows) {
        const auto& [original, composite, fused] = row.arms;
        fmt::print(
            "| {:<36} | {:>9} | {:>10} | {:>9} | {:>11} | {:>12} |\n",
            case_label(row.bench_case),
            cell(original.fits, pick(original)),
            cell(composite.fits, pick(composite)),
            cell(fused.fits, pick(fused)),
            speedup_cell(original, pick(original), pick(fused)),
            speedup_cell(composite, pick(composite), pick(fused)));
    }
}

void print_total_table(const std::vector<Row>& rows) {
    fmt::print(
        "\nForward + backward (µs per step) + DRAM peak of one step (MB)\n"
        "| Shape                                | Original  | Composite  | Fused     | vs original | vs composite | "
        "DRAM orig | DRAM comp | DRAM fused |\n"
        "|:-------------------------------------|----------:|-----------:|----------:|------------:|-------------:|"
        "----------:|----------:|-----------:|\n");
    for (const auto& row : rows) {
        const auto& [original, composite, fused] = row.arms;
        fmt::print(
            "| {:<36} | {:>9} | {:>10} | {:>9} | {:>11} | {:>12} | {:>9} | {:>9} | {:>10} |\n",
            case_label(row.bench_case),
            cell(original.fits, original.total_us()),
            cell(composite.fits, composite.total_us()),
            cell(fused.fits, fused.total_us()),
            speedup_cell(original, original.total_us(), fused.total_us()),
            speedup_cell(composite, composite.total_us(), fused.total_us()),
            cell(original.fits, original.dram_peak_mb),
            cell(composite.fits, composite.dram_peak_mb),
            cell(fused.fits, fused.dram_peak_mb));
    }
}

std::vector<Row> g_rows;

void BM_LogSoftmax(benchmark::State& state) {
    const auto& c = all_cases().at(static_cast<size_t>(state.range(0)));
    for ([[maybe_unused]] auto _ : state) {
        const auto row = run_case(c);
        g_rows.push_back(row);
        state.SetIterationTime(row.arms[2].total_us() / 1e6);
        state.SetLabel(case_label(c));
        state.counters["Original_us"] = row.arms[0].total_us();
        state.counters["Composite_us"] = row.arms[1].total_us();
        state.counters["Fused_us"] = row.arms[2].total_us();
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }

        fmt::print("log_softmax op-level benchmark (original composite, corrected composite, fused)\n");
        fmt::print(
            "bf16 input and upstream gradient; repeats={} (median); speedups are fused against each arm.\n",
            kNumRepeats);

        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);
        benchmark::RegisterBenchmark("LogSoftmax", BM_LogSoftmax)
            ->DenseRange(0, static_cast<int>(all_cases().size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        ttml::autograd::ctx().close_device();

        print_direction_table(g_rows, /* forward */ true);
        print_direction_table(g_rows, /* forward */ false);
        print_total_table(g_rows);
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "log_softmax_benchmark failed: {}\n", e.what());
        return 1;
    }
}
