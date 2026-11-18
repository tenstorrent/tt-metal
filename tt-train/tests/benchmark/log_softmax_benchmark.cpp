// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// log_softmax op-level benchmark: the two paths of ops::log_softmax
//
// Both run ops::log_softmax on the same bf16 logits and upstream gradient:
//   composite  its fallback (one selected maximum, log1p of the tail), reached by giving it rank-3 views of the
//              logits and the upstream gradient (dims 0 and 1 merged, no copy), made once outside autograd, so its
//              graph holds no extra nodes.
//   fused      one device program forward and one backward.
//
// Shapes cover small widths, NanoLlama3's full-batch logits [64, 1, 256, 32000], few rows of a 32000 and a
// Llama-3-sized 128256 vocabulary (where the fused op splits rows across cores), and a reduction over dim 2.
//
// Each direction is host time per launch over a case's measured launches between two device syncs, the median
// of kNumRepeats samples taken round-robin over the paths. Backward replays the graph of an untimed forward. DRAM is
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
#include <string_view>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace {

constexpr uint32_t kNumRepeats = 5U;
constexpr double kBytesPerMb = 1024.0 * 1024.0;

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// The path of ops::log_softmax a run takes.
enum class Path { Composite, Fused };
constexpr std::array<Path, 2> kPaths{Path::Composite, Path::Fused};

// A path's logits, upstream gradient and reduction dim. The composite path takes rank-3 views (dims 0 and 1
// merged), which the fused op does not support, so ops::log_softmax runs its fallback. `dim` is 2 or 3.
struct PathInputs {
    ttnn::Tensor x;
    ttnn::Tensor dy;
    int dim;
};

PathInputs path_inputs(const Path path, const ttnn::Tensor& x, const ttnn::Tensor& dy, const int dim) {
    if (path == Path::Fused) {
        return {x, dy, dim};
    }
    const auto& shape = x.logical_shape();
    const ttnn::Shape merged({shape[0] * shape[1], shape[2], shape[3]});
    return {ttnn::reshape(x, merged), ttnn::reshape(dy, merged), dim - 1};
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

struct PathResult {
    // False when the path ran out of DRAM (the composite at NanoLlama3 needs ~15 GB, more than a Wormhole chip).
    bool fits = true;
    double forward_us = 0.0;
    double backward_us = 0.0;
    double dram_peak_mb = 0.0;

    double total_us() const {
        return forward_us + backward_us;
    }
};

using PathResults = std::array<PathResult, kPaths.size()>;

struct Row {
    Case bench_case;
    PathResults paths;
};

std::string case_label(const Case& c) {
    auto label = fmt::format("[{}, {}, {}, {}] dim {}", c.shape[0], c.shape[1], c.shape[2], c.shape[3], c.dim);
    return c.label.empty() ? label : fmt::format("{} {}", label, c.label);
}

void forward_step(const ttnn::Tensor& x, const int dim) {
    auto input = ttml::autograd::create_tensor(x, /* requires_grad */ true);
    (void)ttml::ops::log_softmax(input, dim);
    ttml::autograd::ctx().reset_graph();
}

void forward_backward_step(const ttnn::Tensor& x, const ttnn::Tensor& dy, const int dim) {
    auto input = ttml::autograd::create_tensor(x, /* requires_grad */ true);
    auto output = ttml::ops::log_softmax(input, dim);
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
    std::array<std::vector<double>, kPaths.size()> forward_us;
    std::array<std::vector<double>, kPaths.size()> backward_us;
    for (uint32_t repeat = 0; repeat < kNumRepeats; ++repeat) {
        for (size_t i = 0; i < kPaths.size(); ++i) {
            if (!row.paths[i].fits) {
                continue;
            }
            const auto in = path_inputs(kPaths[i], x, dy, c.dim);
            try {
                forward_us[i].push_back(
                    time_device_avg_us(*device, c.warmup, c.measure, [&]() { forward_step(in.x, in.dim); }));
                // The backward replays the graph of one untimed forward. The graph's only tensors are the leaf, whose
                // gradient each replay clears, and the root, whose gradient is the upstream one.
                auto input = ttml::autograd::create_tensor(in.x, /* requires_grad */ true);
                auto output = ttml::ops::log_softmax(input, in.dim);
                output->set_grad(in.dy);
                backward_us[i].push_back(time_device_avg_us(*device, c.warmup, c.measure, [&]() {
                    input->set_grad(ttnn::Tensor());
                    output->backward(/* retain_graph */ true);
                }));
            } catch (const std::exception& e) {
                // Only the composite may run out of DRAM (it needs ~15 GB at NanoLlama3); anything else fails the
                // benchmark.
                if (kPaths[i] == Path::Fused ||
                    std::string_view(e.what()).find("Out of Memory") == std::string_view::npos) {
                    throw;
                }
                fmt::print("{}: composite skipped, out of DRAM: {}\n", case_label(c), e.what());
                row.paths[i].fits = false;
            }
            ttml::autograd::ctx().reset_graph();
        }
    }

    for (size_t i = 0; i < kPaths.size(); ++i) {
        if (!row.paths[i].fits) {
            continue;
        }
        const auto in = path_inputs(kPaths[i], x, dy, c.dim);
        const auto dram_peak =
            capture_dram_peak("log_softmax_step", [&]() { forward_backward_step(in.x, in.dy, in.dim); });
        row.paths[i].forward_us = median(forward_us[i]);
        row.paths[i].backward_us = median(backward_us[i]);
        row.paths[i].dram_peak_mb = static_cast<double>(dram_peak) / kBytesPerMb;
    }
    return row;
}

// A table cell: the value, or "n/a" for a path that did not fit in DRAM.
std::string cell(const bool fits, const double value) {
    return fits ? fmt::format("{:.1f}", value) : std::string("n/a");
}

std::string speedup_cell(const PathResult& composite, const double composite_value, const double fused_value) {
    return composite.fits ? fmt::format("{:.2f}x", ttml::benchmark_utils::speedup_x(composite_value, fused_value))
                          : std::string("n/a");
}

void print_direction_table(const std::vector<Row>& rows, const bool forward) {
    const auto pick = [forward](const PathResult& r) { return forward ? r.forward_us : r.backward_us; };
    fmt::print(
        "\n{} (µs per launch)\n"
        "| Shape                                | Composite  | Fused     | Speedup |\n"
        "|:-------------------------------------|-----------:|----------:|--------:|\n",
        forward ? "Forward" : "Backward");
    for (const auto& row : rows) {
        const auto& [composite, fused] = row.paths;
        fmt::print(
            "| {:<36} | {:>10} | {:>9} | {:>7} |\n",
            case_label(row.bench_case),
            cell(composite.fits, pick(composite)),
            cell(fused.fits, pick(fused)),
            speedup_cell(composite, pick(composite), pick(fused)));
    }
}

void print_total_table(const std::vector<Row>& rows) {
    fmt::print(
        "\nForward + backward (µs per step) + DRAM peak of one step (MB)\n"
        "| Shape                                | Composite  | Fused     | Speedup | DRAM comp | DRAM fused |\n"
        "|:-------------------------------------|-----------:|----------:|--------:|----------:|-----------:|\n");
    for (const auto& row : rows) {
        const auto& [composite, fused] = row.paths;
        fmt::print(
            "| {:<36} | {:>10} | {:>9} | {:>7} | {:>9} | {:>10} |\n",
            case_label(row.bench_case),
            cell(composite.fits, composite.total_us()),
            cell(fused.fits, fused.total_us()),
            speedup_cell(composite, composite.total_us(), fused.total_us()),
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
        state.SetIterationTime(row.paths[1].total_us() / 1e6);
        state.SetLabel(case_label(c));
        state.counters["Composite_us"] = row.paths[0].total_us();
        state.counters["Fused_us"] = row.paths[1].total_us();
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }

        fmt::print("log_softmax op-level benchmark (composite fallback, fused)\n");
        fmt::print(
            "bf16 input and upstream gradient; repeats={} (median); speedup is fused against composite.\n",
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
