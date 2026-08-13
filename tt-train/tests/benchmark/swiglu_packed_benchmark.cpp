// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Op-level A/B for the packed-SwiGLU gating kernels against the separate-tensor path they replace.
//
// Arms:
//   composite  ttnn::multiply(lhs_activation=SILU) + swiglu_elemwise_bw, on separate gate/up
//              tensors. The production pre-fusion path (ops/swiglu_op.cpp:99,161).
//   packed     swiglu_packed_fw/bw over one [.., R, 2I] tensor, called as ops/swiglu_packed_op.cpp.
//   packed+pre swiglu_packed_bw with a preallocated dL_dpacked. Prices the option: ops/swiglu_op.cpp
//              can pass one because linear1 is a dead local, while swiglu_packed_op has no dead 2I
//              buffer to give, so only a fused-MLP caller could claim this saving.
//   slice+comp two ttnn::slice calls to split a packed tensor, then the composite arm. What the
//              fused-weight producer would have to pay to keep using the separate-tensor kernels,
//              i.e. the cost the packed op exists to avoid.
//
// Run it pinned -- `taskset -c N ./swiglu_packed_benchmark`.

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/swiglu_elemwise_bw/swiglu_elemwise_bw.hpp"
#include "metal/ops/swiglu_packed_bw/swiglu_packed_bw.hpp"
#include "metal/ops/swiglu_packed_fw/swiglu_packed_fw.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "utils/memory_utils.hpp"

namespace {

constexpr uint32_t kNumWarmup = 3U;
constexpr uint32_t kNumMeasure = 20U;

// The op sees only batch*sequence flattened.
// `inner` is the intermediate width I; the packed tensor is 2I wide.
struct Case {
    std::string name;
    uint32_t rows;
    uint32_t inner;

    uint32_t tile_rows() const {
        return rows / tt::constants::TILE_HEIGHT;
    }
};

const std::vector<Case>& all_cases() {
    static const std::vector<Case> cases = {
        // Three intermediate widths spanning what ttml's Llama presets use.
        {.name = "i1024_1k_rows", .rows = 1024U, .inner = 1024U},
        {.name = "i1024_4k_rows", .rows = 4096U, .inner = 1024U},
        {.name = "i5632_1k_rows", .rows = 1024U, .inner = 5632U},
        {.name = "i5632_4k_rows", .rows = 4096U, .inner = 5632U},
        {.name = "i14336_1k_rows", .rows = 1024U, .inner = 14336U},
        {.name = "i14336_4k_rows", .rows = 4096U, .inner = 14336U},
        // Same number of tiles as i14336_1k_rows, arranged differently.
        {.name = "i3584_4k_rows", .rows = 4096U, .inner = 3584U},
    };
    return cases;
}

struct ArmResult {
    std::string name;
    double forward_us = 0.0;
    double backward_us = 0.0;
    size_t forward_dram_peak = 0U;
    size_t backward_dram_peak = 0U;
};

// Worst absolute deviation of each direction from its separate-tensor reference.
struct MaxAbsDiff {
    double forward = 0.0;
    double backward = 0.0;
};

struct CaseResult {
    std::vector<ArmResult> arms;
    MaxAbsDiff max_abs_diff;
};

// Inputs shared by every arm: one packed [1,1,R,2I] tensor and its two halves as standalone
// [1,1,R,I] tensors holding the identical values, so the arms differ only in the kernel invoked.
struct Inputs {
    ttnn::Tensor packed;
    ttnn::Tensor gate;
    ttnn::Tensor up;
    ttnn::Tensor dL_dh;
    // Destinations for the backwards that accept one, allocated outside every timing region.
    ttnn::Tensor preallocated_dgate;
    ttnn::Tensor preallocated_dpacked;
};

Inputs make_inputs(const Case& c, ttnn::distributed::MeshDevice* device) {
    const uint32_t rows = c.rows;
    const uint32_t inner = c.inner;
    const uint32_t seed = ttml::benchmark_utils::seed_from_name(c.name);

    const auto gate_host =
        ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(rows) * inner, -4.0F, 4.0F, seed);
    const auto up_host =
        ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(rows) * inner, -2.0F, 2.0F, seed + 1U);
    const auto dh_host =
        ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(rows) * inner, -1.0F, 1.0F, seed + 2U);

    // Concatenate the halves within each row into [gate | up].
    std::vector<float> packed_host(static_cast<size_t>(rows) * 2U * inner);
    for (uint32_t r = 0; r < rows; ++r) {
        const size_t src = static_cast<size_t>(r) * inner;
        const size_t dst = static_cast<size_t>(r) * 2U * inner;
        std::copy_n(gate_host.begin() + src, inner, packed_host.begin() + dst);
        std::copy_n(up_host.begin() + src, inner, packed_host.begin() + dst + inner);
    }

    const ttnn::Shape inner_shape({1U, 1U, rows, inner});
    const ttnn::Shape packed_shape({1U, 1U, rows, 2U * inner});
    const auto to_device = [&](const std::vector<float>& host, const ttnn::Shape& shape) {
        return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(host, shape, device, ttnn::Layout::TILE);
    };

    Inputs in{
        .packed = to_device(packed_host, packed_shape),
        .gate = to_device(gate_host, inner_shape),
        .up = to_device(up_host, inner_shape),
        .dL_dh = to_device(dh_host, inner_shape),
        .preallocated_dgate = to_device(gate_host, inner_shape),
        .preallocated_dpacked = to_device(packed_host, packed_shape),
    };
    return in;
}

ttnn::Tensor composite_forward(const ttnn::Tensor& gate, const ttnn::Tensor& up) {
    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
    const ttsl::Span<const EltwiseUnary> no_acts;
    const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);
    return ttnn::multiply(gate, up, std::nullopt, std::nullopt, std::nullopt, no_acts, silu_lhs);
}

std::pair<ttnn::Tensor, ttnn::Tensor> slice_halves(const ttnn::Tensor& packed) {
    const auto& shape = packed.logical_shape();
    const uint32_t rows = shape[-2];
    const uint32_t inner = shape[-1] / 2U;
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    const ttsl::SmallVector<uint32_t> gate_start = {0U, 0U, 0U, 0U};
    const ttsl::SmallVector<uint32_t> gate_end = {1U, 1U, rows, inner};
    const ttsl::SmallVector<uint32_t> up_start = {0U, 0U, 0U, inner};
    const ttsl::SmallVector<uint32_t> up_end = {1U, 1U, rows, 2U * inner};
    return {ttnn::slice(packed, gate_start, gate_end, step), ttnn::slice(packed, up_start, up_end, step)};
}

template <typename Fn>
double time_avg_us(ttnn::distributed::MeshDevice* device, Fn&& fn) {
    for (uint32_t i = 0; i < kNumWarmup; ++i) {
        fn();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < kNumMeasure; ++i) {
        fn();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(kNumMeasure);
}

template <typename Fn>
size_t capture_dram_peak(const std::string& name, Fn&& fn) {
    ttml::utils::MemoryUsageTracker::clear();
    const auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
    (void)guard;
    fn();
    ttml::utils::MemoryUsageTracker::end_capture(name);
    return static_cast<size_t>(std::max(0LL, ttml::utils::MemoryUsageTracker::get_dram_usage(name).peak));
}

template <typename FwFn, typename BwFn>
ArmResult run_arm(const std::string& name, ttnn::distributed::MeshDevice* device, FwFn&& fw, BwFn&& bw) {
    ArmResult r;
    r.name = name;
    r.forward_us = time_avg_us(device, fw);
    r.backward_us = time_avg_us(device, bw);
    r.forward_dram_peak = capture_dram_peak(name + "_fw", fw);
    r.backward_dram_peak = capture_dram_peak(name + "_bw", bw);
    return r;
}

// This is a sanity bound, not an accuracy test -- accuracy is
// covered by tests/python/test_swiglu_packed.py against a float64 oracle.
double max_abs_diff_fw_vs_composite(const Inputs& in) {
    const auto packed_out = ttml::core::to_vector<float>(ttml::metal::swiglu_packed_fw(in.packed));
    const auto composite_out = ttml::core::to_vector<float>(composite_forward(in.gate, in.up));
    if (packed_out.size() != composite_out.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    for (size_t i = 0; i < packed_out.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(packed_out[i]) - composite_out[i]));
    }
    return worst;
}

// This is a sanity bound, not an accuracy test -- accuracy is
// covered by tests/python/test_swiglu_packed.py against a float64 oracle.
double max_abs_diff_bw_vs_elemwise(const Inputs& in) {
    const auto& packed_shape = in.packed.logical_shape();
    const uint32_t rows = packed_shape[-2];
    const uint32_t inner = packed_shape[-1] / 2U;

    const auto packed_grad = ttml::core::to_vector<float>(ttml::metal::swiglu_packed_bw(in.packed, in.dL_dh));
    const auto ref = ttml::metal::swiglu_elemwise_bw(in.gate, in.up, in.dL_dh);
    const auto dgate_ref = ttml::core::to_vector<float>(ref.dL_dlinear1);
    const auto dup_ref = ttml::core::to_vector<float>(ref.dL_dgate);

    const size_t expected = static_cast<size_t>(rows) * 2U * inner;
    if (packed_grad.size() != expected || dgate_ref.size() != static_cast<size_t>(rows) * inner) {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    for (uint32_t r = 0; r < rows; ++r) {
        const size_t half = static_cast<size_t>(r) * inner;
        const size_t full = static_cast<size_t>(r) * 2U * inner;
        for (uint32_t i = 0; i < inner; ++i) {
            worst = std::max(worst, std::abs(static_cast<double>(packed_grad[full + i]) - dgate_ref[half + i]));
            worst = std::max(worst, std::abs(static_cast<double>(packed_grad[full + inner + i]) - dup_ref[half + i]));
        }
    }
    return worst;
}

CaseResult run_case(const Case& c) {
    auto* const device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    const auto in = make_inputs(c, device);

    CaseResult result;
    result.max_abs_diff = {max_abs_diff_fw_vs_composite(in), max_abs_diff_bw_vs_elemwise(in)};

    result.arms.push_back(run_arm(
        "composite",
        device,
        [&]() { (void)composite_forward(in.gate, in.up); },
        [&]() { (void)ttml::metal::swiglu_elemwise_bw(in.gate, in.up, in.dL_dh, in.preallocated_dgate); }));

    result.arms.push_back(run_arm(
        "packed",
        device,
        [&]() { (void)ttml::metal::swiglu_packed_fw(in.packed); },
        [&]() { (void)ttml::metal::swiglu_packed_bw(in.packed, in.dL_dh); }));

    result.arms.push_back(run_arm(
        "packed+pre",
        device,
        [&]() { (void)ttml::metal::swiglu_packed_fw(in.packed); },
        [&]() { (void)ttml::metal::swiglu_packed_bw(in.packed, in.dL_dh, in.preallocated_dpacked); }));

    result.arms.push_back(run_arm(
        "slice+comp",
        device,
        [&]() {
            const auto [gate, up] = slice_halves(in.packed);
            (void)composite_forward(gate, up);
        },
        [&]() {
            const auto [gate, up] = slice_halves(in.packed);
            (void)ttml::metal::swiglu_elemwise_bw(gate, up, in.dL_dh);
        }));

    return result;
}

void print_case_table(const Case& c, const CaseResult& result) {
    const auto& baseline = result.arms.front();

    fmt::print(
        "\nCase: {}   rows={} ({} tile-rows)   I={}   packed={}\n"
        "  max abs diff vs reference: fw {:g}, bw {:g}\n",
        c.name,
        c.rows,
        c.tile_rows(),
        c.inner,
        2U * c.inner,
        result.max_abs_diff.forward,
        result.max_abs_diff.backward);
    fmt::print(
        "+------------+-----------+-----------+-----------+-----------+-------------+-------------+\n"
        "| Arm        | Fwd µs    | Fwd %     | Bwd µs    | Bwd %     | Fwd DRAM KB | Bwd DRAM KB |\n"
        "+------------+-----------+-----------+-----------+-----------+-------------+-------------+\n");
    for (const auto& r : result.arms) {
        fmt::print(
            "| {:<10} | {:>9.1f} | {:>+9.2f} | {:>9.1f} | {:>+9.2f} | {:>11.1f} | {:>11.1f} |\n",
            r.name,
            r.forward_us,
            ttml::benchmark_utils::relative_change_pct(r.forward_us, baseline.forward_us),
            r.backward_us,
            ttml::benchmark_utils::relative_change_pct(r.backward_us, baseline.backward_us),
            static_cast<double>(r.forward_dram_peak) / 1024.0,
            static_cast<double>(r.backward_dram_peak) / 1024.0);
    }
    fmt::print("+------------+-----------+-----------+-----------+-----------+-------------+-------------+\n");
}

}  // namespace

int main() {
    try {
        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);

        fmt::print("Packed-SwiGLU op-level benchmark (separate-tensor composite baseline vs packed)\n");
        fmt::print(
            "warmup={} measure={}; Fwd %/Bwd % are each direction against the composite arm.\n",
            kNumWarmup,
            kNumMeasure);
        fmt::print("DRAM columns are the peak of one captured invocation of that direction.\n");

        for (const auto& c : all_cases()) {
            print_case_table(c, run_case(c));
        }

        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "swiglu_packed_benchmark failed: {}\n", e.what());
        return 1;
    }
}
