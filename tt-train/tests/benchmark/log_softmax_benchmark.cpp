// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
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
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

using ttml::autograd::TensorPtr;
using Forward = TensorPtr (*)(const TensorPtr&, int);
constexpr int kWarmup = 5;
constexpr int kIterations = 20;
constexpr int kSamples = 7;

// Frozen pre-fix composite implementation, including its BF16 intermediates and
// approximate log. Run both versions against the same runtime and input buffers.
TensorPtr baseline_log_softmax(const TensorPtr& input, int dim) {
    auto maximum = ttnn::max(input->get_value(), dim, true);
    auto shifted = ttnn::subtract(input->get_value(), maximum);
    auto exponentials = ttnn::exp(shifted);
    auto denominator = ttml::ttnn_fixed::sum_over_dim(exponentials, dim);
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

struct Case {
    std::array<uint32_t, 4> shape;
    int dim;
};

constexpr std::array<Case, 6> kCases = {{
    {{1, 1, 32, 128}, 3},
    {{1, 1, 128, 1024}, 3},
    {{1, 1, 1024, 4096}, 3},
    {{1, 1, 32, 32000}, 3},
    {{1, 1, 256, 32000}, 3},
    {{1, 1, 1024, 4096}, 2},
}};

enum class Phase { Forward, Backward, ForwardBackward };
constexpr std::array<Phase, 3> kPhases = {Phase::Forward, Phase::Backward, Phase::ForwardBackward};
constexpr std::array<std::string_view, 3> kPhaseNames = {"forward", "backward", "forward_backward"};

void synchronize() {
    tt::tt_metal::distributed::Synchronize(ttml::autograd::ctx().get_device(), std::nullopt);
}

// Median of seven batch averages. Each batch enqueues twenty invocations and
// synchronizes once at its end: host dispatch + device completion, not kernel-only time.
double measure(Forward forward, Phase phase, const ttnn::Tensor& value, const ttnn::Tensor& dy, int dim) {
    auto input = ttml::autograd::create_tensor(value, /* requires_grad */ true);
    TensorPtr saved_output;
    if (phase == Phase::Backward) {
        saved_output = forward(input, dim);
        saved_output->set_grad(dy);
    }
    const auto invoke = [&]() {
        input->set_grad(ttnn::Tensor());
        if (phase == Phase::Backward) {
            saved_output->backward(/* retain_graph */ true);
        } else {
            auto output = forward(input, dim);
            if (phase == Phase::ForwardBackward) {
                output->set_grad(dy);
                output->backward();
            }
            ttml::autograd::ctx().reset_graph();
        }
    };
    for (int i = 0; i < kWarmup; ++i) {
        invoke();
    }
    synchronize();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        invoke();
    }
    synchronize();
    const auto end = std::chrono::steady_clock::now();
    ttml::autograd::ctx().reset_graph();
    return std::chrono::duration<double, std::micro>(end - start).count() / kIterations;
}

struct Memory {
    long long forward_peak;
    long long forward_retained;
    long long step_peak;
};

// Allocation tracing is outside timed regions. Input and upstream buffers are
// allocated before capture, so these are incremental per-op DRAM bytes.
Memory measure_memory(Forward forward, const ttnn::Tensor& value, const ttnn::Tensor& dy, int dim) {
    auto input = ttml::autograd::create_tensor(value, /* requires_grad */ true);
    TensorPtr output;
    Memory result{};
    {
        auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
        output = forward(input, dim);
        synchronize();
        ttml::utils::MemoryUsageTracker::end_capture("forward");
        const auto usage = ttml::utils::MemoryUsageTracker::get_dram_usage("forward");
        result.forward_peak = usage.peak;
        result.forward_retained = usage.total_allocations - usage.total_deallocations;
    }
    ttml::autograd::ctx().reset_graph();
    output.reset();
    {
        auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
        output = forward(input, dim);
        output->set_grad(dy);
        output->backward();
        synchronize();
        ttml::utils::MemoryUsageTracker::end_capture("step");
        result.step_peak = ttml::utils::MemoryUsageTracker::get_dram_usage("step").peak;
    }
    ttml::autograd::ctx().reset_graph();
    return result;
}

void run_case(benchmark::State& state) {
    const auto& test_case = kCases[static_cast<size_t>(state.range(0))];
    const ttnn::Shape shape(std::vector<uint32_t>(test_case.shape.begin(), test_case.shape.end()));
    auto* device = &ttml::autograd::ctx().get_device();
    auto value = ttml::core::from_vector(
        ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.F, 1.F, 2026U), shape, device);
    auto dy = ttml::core::from_vector(
        ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.F, 1.F, 2027U), shape, device);
    constexpr std::array<Forward, 2> forwards = {baseline_log_softmax, ttml::ops::log_softmax};
    constexpr std::array<std::string_view, 2> names = {"before", "after"};
    for ([[maybe_unused]] auto _ : state) {
        for (size_t phase = 0; phase < kPhases.size(); ++phase) {
            std::array<std::vector<double>, 2> samples;
            for (int sample = 0; sample < kSamples; ++sample) {
                // Alternate order to reduce systematic clock/temperature bias.
                for (int order = 0; order < 2; ++order) {
                    const int variant = (sample + order) % 2;
                    samples[variant].push_back(measure(forwards[variant], kPhases[phase], value, dy, test_case.dim));
                }
            }
            for (size_t variant = 0; variant < forwards.size(); ++variant) {
                auto& times = samples[variant];
                std::sort(times.begin(), times.end());
                const double median = times[times.size() / 2];
                state.counters[fmt::format("{}_{}_us", names[variant], kPhaseNames[phase])] = median;
                fmt::print(
                    "LOG_SOFTMAX_TIME,{},{},{},{},{},{},{:.3f},{:.3f},{:.3f}\n",
                    shape[0],
                    shape[1],
                    shape[2],
                    shape[3],
                    test_case.dim,
                    fmt::format("{}_{}", names[variant], kPhaseNames[phase]),
                    median,
                    times.front(),
                    times.back());
                if (phase == 2 && variant == 1) {
                    state.SetIterationTime(median / 1e6);
                }
            }
        }
        for (size_t variant = 0; variant < forwards.size(); ++variant) {
            const auto memory = measure_memory(forwards[variant], value, dy, test_case.dim);
            fmt::print(
                "LOG_SOFTMAX_MEMORY,{},{},{},{},{},{},{},{},{}\n",
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                test_case.dim,
                names[variant],
                memory.forward_peak,
                memory.forward_retained,
                memory.step_peak);
        }
    }
}

// Isolate the new TILE argmax to distinguish its scan cost from the remaining
// normalization, mask construction, dispatch, and autograd overhead.
void run_argmax(benchmark::State& state) {
    const auto& test_case = kCases[static_cast<size_t>(state.range(0))];
    const ttnn::Shape shape(std::vector<uint32_t>(test_case.shape.begin(), test_case.shape.end()));
    auto* device = &ttml::autograd::ctx().get_device();
    auto value = ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(
        ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.F, 1.F, 2026U), shape, device);
    for ([[maybe_unused]] auto _ : state) {
        const double elapsed_us = ttml::benchmark_utils::time_device_avg_us(*device, kWarmup, kIterations, [&]() {
            auto result = ttnn::argmax(value, test_case.dim, /* keepdim */ true);
            benchmark::DoNotOptimize(result);
        });
        state.SetIterationTime(elapsed_us / 1e6);
        state.counters["Time_us"] = elapsed_us;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().get_device().enable_program_cache();
        fmt::print(
            "BF16 eager composite log_softmax; warmup={}, iterations={}, samples={}; timings in us\n",
            kWarmup,
            kIterations,
            kSamples);
        fmt::print("LOG_SOFTMAX_TIME,N,C,H,W,dim,variant_phase,median_us,min_us,max_us\n");
        fmt::print(
            "LOG_SOFTMAX_MEMORY,N,C,H,W,dim,variant,forward_peak_bytes,forward_retained_bytes,step_peak_bytes\n");
        benchmark::RegisterBenchmark("LogSoftmax", run_case)
            ->DenseRange(0, static_cast<int>(kCases.size()) - 1)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RegisterBenchmark("LogSoftmaxArgmax", run_argmax)
            ->Arg(1)
            ->Arg(2)
            ->Arg(4)
            ->UseManualTime()
            ->Iterations(1)
            ->Repetitions(kSamples);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "log_softmax_benchmark failed: {}\n", e.what());
        return 1;
    }
}
