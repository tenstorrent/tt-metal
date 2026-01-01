// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// RMSNorm op-level benchmark: the plain rmsnorm_fw/bw, and their gated mode against a composite gate
//
// Ungated: rmsnorm_fw + rmsnorm_bw on bf16 [1, H, T, C] -- model norms over the hidden size (H = 1, C = 2048, 4096
// and 5120, the Qwen3.8 block norm) and per-head QK-norms (H = 12, C = 128 as in ttml Qwen3, 256 as in Qwen3.8).
// The backward runs with dL/dgamma, which is all main has, and without it (frozen gamma). Apart from the frozen
// backward these are calls main has, so the section can be timed at an older commit too.
//
// Gated: the forward and backward of out = rmsnorm(x) * gamma * silu(gate) at the Qwen3.8 TP=4 head layout (H = 12
// heads of V = 128). x is the chunked delta rule's head-major fp32 output [B*H, 1, T, V]; gate, the upstream gradient
// and out are head-merged [B, 1, T, H*V], the layout the output projection consumes.
//
// Gated arms, each including the relayouts it needs:
//   composite       rmsnorm_fw on x, nlp_concat_heads to the head-merged layout, then ttnn::multiply
//                   with a SILU rhs activation. Backward: dgate and du = dy * silu(gate) from one
//                   swiglu_elemwise_bw (SwiGLU gating with up = y), nlp_create_qkv_heads to bring du back to
//                   head-major, then rmsnorm_bw. Every tensor is freed at its last use.
//   composite+cast  the composite on x typecast to bf16 first, as the RMSNorm on main requires.
//   fused           rmsnorm_fw/rmsnorm_bw in gated mode.
//
// Each direction is host time per launch over kNumMeasure launches between two device syncs, the median
// of kNumRepeats such samples taken round-robin over the arms. DRAM is the peak of one forward + backward
// step, including what the forward keeps for the backward.
// Run it pinned -- `taskset -c N ./rmsnorm_benchmark`.
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "metal/ops/rmsnorm_bw/rmsnorm_bw.hpp"
#include "metal/ops/rmsnorm_fw/rmsnorm_fw.hpp"
#include "metal/ops/swiglu_elemwise_bw/swiglu_elemwise_bw.hpp"
#include "test_utils/random_data.hpp"

namespace {

constexpr uint32_t kNumWarmup = 3U;
constexpr uint32_t kNumMeasure = 20U;
constexpr uint32_t kNumRepeats = 5U;
constexpr float kEpsilon = 1e-6F;
constexpr double kBytesPerMb = 1024.0 * 1024.0;

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// ============================================================================
// Ungated
// ============================================================================

struct UngatedCase {
    uint32_t heads = 1U;  // dim 1 of the input
    uint32_t width = 0U;  // the normalized dim
    uint32_t sequence_length = 0U;

    ttnn::Shape shape() const {
        return ttnn::Shape({1U, heads, sequence_length, width});
    }
};

const std::vector<UngatedCase>& ungated_cases() {
    static const std::vector<UngatedCase> cases = [] {
        std::vector<UngatedCase> result;
        for (const uint32_t sequence_length : {1024U, 4096U, 16384U}) {
            for (const uint32_t width : {2048U, 4096U, 5120U}) {
                result.push_back(UngatedCase{.heads = 1U, .width = width, .sequence_length = sequence_length});
            }
            for (const uint32_t width : {128U, 256U}) {
                result.push_back(UngatedCase{.heads = 12U, .width = width, .sequence_length = sequence_length});
            }
        }
        return result;
    }();
    return cases;
}

struct UngatedRow {
    UngatedCase bench_case;
    double forward_us = 0.0;
    double backward_us = 0.0;
    double backward_frozen_us = 0.0;
};

UngatedRow run_ungated_case(const UngatedCase& c) {
    using namespace ttml::benchmark_utils;
    using ttml::test_utils::make_uniform_vector;
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    const auto shape = c.shape();
    const size_t count = shape.volume();
    const uint32_t seed = 2027U + c.sequence_length + c.width;
    const auto x = ttml::core::from_vector(make_uniform_vector<float>(count, -2.0F, 2.0F, seed), shape, device);
    const auto gamma = ttml::core::from_vector(
        make_uniform_vector<float>(c.width, 0.5F, 1.5F, seed + 1U), ttnn::Shape({1U, 1U, 1U, c.width}), device);
    const auto dL_dout =
        ttml::core::from_vector(make_uniform_vector<float>(count, -1.0F, 1.0F, seed + 2U), shape, device);
    const auto rms = ttml::metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEpsilon)[1].value();

    std::vector<double> forward_us;
    std::vector<double> backward_us;
    std::vector<double> backward_frozen_us;
    for (uint32_t repeat = 0; repeat < kNumRepeats; ++repeat) {
        forward_us.push_back(time_device_avg_us(*device, kNumWarmup, kNumMeasure, [&]() {
            (void)ttml::metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEpsilon);
        }));
        backward_us.push_back(time_device_avg_us(
            *device, kNumWarmup, kNumMeasure, [&]() { (void)ttml::metal::rmsnorm_bw(x, gamma, rms, dL_dout); }));
        backward_frozen_us.push_back(time_device_avg_us(*device, kNumWarmup, kNumMeasure, [&]() {
            (void)ttml::metal::rmsnorm_bw(x, gamma, rms, dL_dout, /* compute_dgamma */ false);
        }));
    }
    return UngatedRow{
        .bench_case = c,
        .forward_us = median(forward_us),
        .backward_us = median(backward_us),
        .backward_frozen_us = median(backward_frozen_us),
    };
}

void print_ungated_table(const std::vector<UngatedRow>& rows) {
    fmt::print(
        "\nUngated, bf16 [1, H, T, C] (µs per launch)\n"
        "| H  | T     | C    | Forward  | Backward | Backward, frozen gamma |\n"
        "|---:|------:|-----:|---------:|---------:|-----------------------:|\n");
    for (const auto& row : rows) {
        const auto& c = row.bench_case;
        fmt::print(
            "| {:>2} | {:>5} | {:>4} | {:>8.1f} | {:>8.1f} | {:>22.1f} |\n",
            c.heads,
            c.sequence_length,
            c.width,
            row.forward_us,
            row.backward_us,
            row.backward_frozen_us);
    }
}

std::vector<UngatedRow> g_ungated_rows;

void BM_RmsNormUngated(benchmark::State& state) {
    const auto& c = ungated_cases().at(static_cast<size_t>(state.range(0)));
    for ([[maybe_unused]] auto _ : state) {
        const auto row = run_ungated_case(c);
        g_ungated_rows.push_back(row);
        state.SetIterationTime((row.forward_us + row.backward_us) / 1e6);
        state.SetLabel(fmt::format("H={} T={} C={}", c.heads, c.sequence_length, c.width));
        state.counters["Forward_us"] = row.forward_us;
        state.counters["Backward_us"] = row.backward_us;
        state.counters["BackwardFrozen_us"] = row.backward_frozen_us;
    }
}

// ============================================================================
// Gated
// ============================================================================

struct Case {
    uint32_t batch = 1U;
    uint32_t heads = 0U;
    uint32_t head_dim = 0U;
    uint32_t sequence_length = 0U;
    bool train_gamma = true;

    ttnn::Shape head_major() const {
        return ttnn::Shape({batch * heads, 1U, sequence_length, head_dim});
    }
    ttnn::Shape head_merged() const {
        return ttnn::Shape({batch, 1U, sequence_length, heads * head_dim});
    }
};

const std::vector<Case>& all_cases() {
    static const std::vector<Case> cases = [] {
        std::vector<Case> result;
        for (const uint32_t sequence_length : {1024U, 4096U, 16384U}) {
            for (const bool train_gamma : {true, false}) {
                result.push_back(Case{
                    .heads = 12U, .head_dim = 128U, .sequence_length = sequence_length, .train_gamma = train_gamma});
            }
        }
        return result;
    }();
    return cases;
}

struct Inputs {
    ttnn::Tensor x;
    ttnn::Tensor gate;
    ttnn::Tensor gamma;
    ttnn::Tensor dL_dout;
};

Inputs make_inputs(const Case& c, ttnn::distributed::MeshDevice* device) {
    using ttml::test_utils::make_uniform_vector;
    const size_t count = static_cast<size_t>(c.batch) * c.heads * c.sequence_length * c.head_dim;
    const uint32_t seed = 2026U + c.sequence_length;
    return Inputs{
        .x = ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(
            make_uniform_vector<float>(count, -2.0F, 2.0F, seed), c.head_major(), device),
        .gate = ttml::core::from_vector(
            make_uniform_vector<float>(count, -20.0F, 20.0F, seed + 1U), c.head_merged(), device),
        .gamma = ttml::core::from_vector(
            make_uniform_vector<float>(c.head_dim, 0.5F, 1.5F, seed + 2U),
            ttnn::Shape({1U, 1U, 1U, c.head_dim}),
            device),
        .dL_dout =
            ttml::core::from_vector(make_uniform_vector<float>(count, -1.0F, 1.0F, seed + 3U), c.head_merged(), device),
    };
}

// What the backward reads besides the inputs: the input the kernels normalized (x or its bf16 cast), rms,
// and for the composite the normalized head-merged y that multiplies silu(gate).
struct Saved {
    ttnn::Tensor x;
    ttnn::Tensor rms;
    std::optional<ttnn::Tensor> y;
};

struct Forward {
    ttnn::Tensor out;
    Saved saved;
};

struct Grads {
    ttnn::Tensor dx;
    ttnn::Tensor dgate;
    std::optional<ttnn::Tensor> dgamma;
};

enum class Arm { Composite, CompositeCast, Fused };

ttnn::Tensor times_silu(const ttnn::Tensor& lhs, const ttnn::Tensor& gate) {
    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
    const ttsl::Span<const EltwiseUnary> no_acts;
    const ttsl::Span<const EltwiseUnary> silu_rhs(&silu_act, 1);
    return ttnn::multiply(lhs, gate, std::nullopt, std::nullopt, std::nullopt, no_acts, no_acts, silu_rhs);
}

Forward composite_forward(const Case& c, const Inputs& in, const bool cast_input) {
    const auto x = cast_input ? ttnn::typecast(in.x, ttnn::DataType::BFLOAT16) : in.x;
    auto fw = ttml::metal::rmsnorm_fw(x, in.gamma, /* return_intermediates */ true, kEpsilon);
    // [B*H, 1, T, V] -> [B, H, T, V] -> [B, 1, T, H*V]
    const auto y = ttnn::experimental::nlp_concat_heads(
        ttnn::reshape(fw[0].value(), ttnn::Shape({c.batch, c.heads, c.sequence_length, c.head_dim})));
    return Forward{.out = times_silu(y, in.gate), .saved = Saved{.x = x, .rms = fw[1].value(), .y = y}};
}

Grads composite_backward(const Case& c, const Inputs& in, Saved saved) {
    // out = y * silu(gate) is SwiGLU gating with up = y: the op's dL_dlinear1 is dgate and its dL_dgate is du.
    auto [dgate, du] = ttml::metal::swiglu_elemwise_bw(in.gate, saved.y.value(), in.dL_dout);
    saved.y.reset();
    // [B, 1, T, H*V] -> [B, H, T, V] -> [B*H, 1, T, V]
    const auto du_heads = std::get<0>(ttnn::experimental::nlp_create_qkv_heads(
        du, std::nullopt, c.heads, /* num_kv_heads */ 0U, /* transpose_k_heads */ false));
    du.deallocate();
    auto grads = ttml::metal::rmsnorm_bw(
        saved.x, in.gamma, saved.rms, ttnn::reshape(du_heads, c.head_major()), /* compute_dgamma */ c.train_gamma);
    return Grads{.dx = grads[0].value(), .dgate = dgate, .dgamma = grads[1]};
}

Forward fused_forward(const Inputs& in) {
    auto fw = ttml::metal::rmsnorm_fw(in.x, in.gamma, /* return_intermediates */ true, kEpsilon, in.gate);
    return Forward{.out = fw[0].value(), .saved = Saved{.x = in.x, .rms = fw[1].value()}};
}

Grads fused_backward(const Case& c, const Inputs& in, const Saved& saved) {
    auto grads =
        ttml::metal::rmsnorm_bw(saved.x, in.gamma, saved.rms, in.dL_dout, /* compute_dgamma */ c.train_gamma, in.gate);
    return Grads{.dx = grads[0].value(), .dgate = grads[2].value(), .dgamma = grads[1]};
}

Forward run_forward(const Arm arm, const Case& c, const Inputs& in) {
    return arm == Arm::Fused ? fused_forward(in) : composite_forward(c, in, arm == Arm::CompositeCast);
}

Grads run_backward(const Arm arm, const Case& c, const Inputs& in, Saved saved) {
    return arm == Arm::Fused ? fused_backward(c, in, saved) : composite_backward(c, in, std::move(saved));
}

struct ArmResult {
    double forward_us = 0.0;
    double backward_us = 0.0;
    double dram_peak_mb = 0.0;

    double total_us() const {
        return forward_us + backward_us;
    }
};

constexpr std::array<Arm, 3> kArms{Arm::Composite, Arm::CompositeCast, Arm::Fused};
using ArmResults = std::array<ArmResult, kArms.size()>;

ArmResults time_arms(const Case& c, const Inputs& in, ttnn::distributed::MeshDevice* device) {
    using namespace ttml::benchmark_utils;
    std::vector<Saved> saved;
    saved.reserve(kArms.size());
    for (const auto arm : kArms) {
        saved.push_back(run_forward(arm, c, in).saved);
    }

    std::array<std::vector<double>, kArms.size()> forward_us;
    std::array<std::vector<double>, kArms.size()> backward_us;
    for (uint32_t repeat = 0; repeat < kNumRepeats; ++repeat) {
        for (size_t i = 0; i < kArms.size(); ++i) {
            const auto arm = kArms[i];
            forward_us[i].push_back(
                time_device_avg_us(*device, kNumWarmup, kNumMeasure, [&]() { (void)run_forward(arm, c, in); }));
            // Each launch gets a copy of the saved tensors, so freeing y in the composite keeps saved[i].y alive.
            backward_us[i].push_back(time_device_avg_us(
                *device, kNumWarmup, kNumMeasure, [&]() { (void)run_backward(arm, c, in, saved[i]); }));
        }
    }

    ArmResults results;
    for (size_t i = 0; i < kArms.size(); ++i) {
        const auto arm = kArms[i];
        const auto dram_peak = capture_dram_peak("rmsnorm_gated_step", [&]() {
            auto forward = run_forward(arm, c, in);
            (void)run_backward(arm, c, in, std::move(forward.saved));
        });
        results[i] = ArmResult{
            .forward_us = median(forward_us[i]),
            .backward_us = median(backward_us[i]),
            .dram_peak_mb = static_cast<double>(dram_peak) / kBytesPerMb,
        };
    }
    return results;
}

// max |got - expected| / max |expected|. This is a sanity bound that the arms compute the same thing, not
// an accuracy test -- accuracy is covered by tests/ops/rmsnorm_op_test.cpp and tests/python/test_rmsnorm.py against
// float64 oracles.
double max_rel_diff(const ttnn::Tensor& got, const ttnn::Tensor& expected) {
    const auto a = ttml::core::to_vector<float>(got);
    const auto b = ttml::core::to_vector<float>(expected);
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    double peak = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(a[i]) - b[i]));
        peak = std::max(peak, std::abs(static_cast<double>(b[i])));
    }
    return peak == 0.0 ? worst : worst / peak;
}

struct RowSummary {
    Case bench_case;
    ArmResult composite;
    ArmResult composite_cast;
    ArmResult fused;
    // Fused against composite.
    double out_diff = 0.0;
    double dx_diff = 0.0;
    double dgate_diff = 0.0;
    std::optional<double> dgamma_diff;
};

RowSummary run_case(const Case& c) {
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    const auto in = make_inputs(c, device);

    RowSummary row{.bench_case = c};
    {
        const auto composite = run_forward(Arm::Composite, c, in);
        const auto fused = run_forward(Arm::Fused, c, in);
        const auto composite_grads = run_backward(Arm::Composite, c, in, composite.saved);
        const auto fused_grads = run_backward(Arm::Fused, c, in, fused.saved);
        row.out_diff = max_rel_diff(fused.out, composite.out);
        row.dx_diff = max_rel_diff(fused_grads.dx, composite_grads.dx);
        row.dgate_diff = max_rel_diff(fused_grads.dgate, composite_grads.dgate);
        if (c.train_gamma) {
            row.dgamma_diff = max_rel_diff(fused_grads.dgamma.value(), composite_grads.dgamma.value());
        }
    }
    const auto arms = time_arms(c, in, device);
    row.composite = arms[0];
    row.composite_cast = arms[1];
    row.fused = arms[2];
    return row;
}

std::string gamma_label(const Case& c) {
    return c.train_gamma ? "trained" : "frozen";
}

void print_direction_table(const std::vector<RowSummary>& rows, const bool forward) {
    const auto pick = [forward](const ArmResult& r) { return forward ? r.forward_us : r.backward_us; };
    fmt::print(
        "\nGated {} (µs per launch)\n"
        "| B | T     | H  | V   | gamma   | Composite | Comp+cast | Fused    | Saved    | Speedup | Reduction |\n"
        "|--:|------:|---:|----:|:--------|----------:|----------:|---------:|---------:|--------:|----------:|\n",
        forward ? "forward" : "backward");
    for (const auto& row : rows) {
        const auto& c = row.bench_case;
        const double composite = pick(row.composite);
        const double fused = pick(row.fused);
        fmt::print(
            "| {} | {:>5} | {:>2} | {:>3} | {:<7} | {:>9.1f} | {:>9.1f} | {:>8.1f} | {:>8.1f} | {:>6.2f}x | {:>8.2f}% "
            "|\n",
            c.batch,
            c.sequence_length,
            c.heads,
            c.head_dim,
            gamma_label(c),
            composite,
            pick(row.composite_cast),
            fused,
            composite - fused,
            ttml::benchmark_utils::speedup_x(composite, fused),
            ttml::benchmark_utils::reduction_pct(composite, fused));
    }
}

void print_total_table(const std::vector<RowSummary>& rows) {
    fmt::print(
        "\nGated total forward + backward (µs per step) + DRAM peak of one step\n"
        "| B | T     | H  | V   | gamma   | Composite | Comp+cast | Fused    | Speedup | Reduction | "
        "DRAM comp MB | DRAM fused MB | DRAM reduction |\n"
        "|--:|------:|---:|----:|:--------|----------:|----------:|---------:|--------:|----------:|"
        "-------------:|--------------:|---------------:|\n");
    for (const auto& row : rows) {
        const auto& c = row.bench_case;
        fmt::print(
            "| {} | {:>5} | {:>2} | {:>3} | {:<7} | {:>9.1f} | {:>9.1f} | {:>8.1f} | {:>6.2f}x | {:>8.2f}% | "
            "{:>12.1f} | {:>13.1f} | {:>13.2f}% |\n",
            c.batch,
            c.sequence_length,
            c.heads,
            c.head_dim,
            gamma_label(c),
            row.composite.total_us(),
            row.composite_cast.total_us(),
            row.fused.total_us(),
            ttml::benchmark_utils::speedup_x(row.composite.total_us(), row.fused.total_us()),
            ttml::benchmark_utils::reduction_pct(row.composite.total_us(), row.fused.total_us()),
            row.composite.dram_peak_mb,
            row.fused.dram_peak_mb,
            ttml::benchmark_utils::reduction_pct(row.composite.dram_peak_mb, row.fused.dram_peak_mb));
    }
}

void print_diff_table(const std::vector<RowSummary>& rows) {
    fmt::print(
        "\nGated fused vs composite, max |diff| / max |composite|\n"
        "| T     | gamma   | out      | dx       | dgate    | dgamma   |\n"
        "|------:|:--------|---------:|---------:|---------:|---------:|\n");
    for (const auto& row : rows) {
        const auto& c = row.bench_case;
        fmt::print(
            "| {:>5} | {:<7} | {:>8.2e} | {:>8.2e} | {:>8.2e} | {:>8} |\n",
            c.sequence_length,
            gamma_label(c),
            row.out_diff,
            row.dx_diff,
            row.dgate_diff,
            row.dgamma_diff ? fmt::format("{:.2e}", *row.dgamma_diff) : std::string("-"));
    }
}

std::vector<RowSummary> g_rows;

void BM_GatedRmsNorm(benchmark::State& state) {
    const auto& c = all_cases().at(static_cast<size_t>(state.range(0)));
    for ([[maybe_unused]] auto _ : state) {
        const auto row = run_case(c);
        g_rows.push_back(row);
        state.SetIterationTime(row.fused.total_us() / 1e6);
        state.SetLabel(fmt::format(
            "B={} T={} H={} V={} gamma={}", c.batch, c.sequence_length, c.heads, c.head_dim, gamma_label(c)));
        state.counters["Composite_us"] = row.composite.total_us();
        state.counters["CompositeCast_us"] = row.composite_cast.total_us();
        state.counters["Fused_us"] = row.fused.total_us();
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }

        fmt::print("RMSNorm op-level benchmark (plain rmsnorm_fw/bw; composite gate vs their gated mode)\n");
        fmt::print(
            "warmup={} measure={} repeats={} (median). Gated: x is FLOAT32 head-major, gate and the upstream gradient "
            "BFLOAT16 head-merged.\n"
            "Composite and Comp+cast include their relayouts; Speedup and Reduction are fused against composite.\n",
            kNumWarmup,
            kNumMeasure,
            kNumRepeats);

        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);
        benchmark::RegisterBenchmark("RmsNormUngated", BM_RmsNormUngated)
            ->DenseRange(0, static_cast<int>(ungated_cases().size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RegisterBenchmark("GatedRmsNorm", BM_GatedRmsNorm)
            ->DenseRange(0, static_cast<int>(all_cases().size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        ttml::autograd::ctx().close_device();

        print_ungated_table(g_ungated_rows);
        print_direction_table(g_rows, /* forward */ true);
        print_direction_table(g_rows, /* forward */ false);
        print_total_table(g_rows);
        print_diff_table(g_rows);
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "rmsnorm_benchmark failed: {}\n", e.what());
        return 1;
    }
}
