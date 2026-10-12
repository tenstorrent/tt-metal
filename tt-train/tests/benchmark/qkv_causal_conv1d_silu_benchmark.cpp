// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qkv causal conv1d + SiLU benchmark: one layer "model" (Kimi-like widths), forward + backward.
//   composite - backward built from stock ttnn ops (what you get without the fused kernel)
//   fused     - ttml::metal::qkv_causal_conv1d_silu_bw
// Forward is always the fused ttnn kernel. Env: QKV_T, QKV_WIDTH (or QKV_Q/QKV_K/QKV_V), QKV_CHUNKS (csv),
// QKV_SPLITS (csv, 0 = auto).

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/pad/pad.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/qkv_causal_conv1d_silu.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"

namespace {

std::vector<uint32_t> env_list(const char* name, std::vector<uint32_t> fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    std::vector<uint32_t> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return out;
}

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : static_cast<uint32_t>(std::stoul(value));
}

ttnn::Tensor random_device(const ttnn::Shape& shape, uint32_t seed, ttnn::Layout layout = ttnn::Layout::TILE) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto host = ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.0F, 1.0F, seed);
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(host, shape, device, layout);
}

struct Widths {
    uint32_t q, k, v;
    uint32_t c() const {
        return q + k + v;
    }
};

struct Inputs {
    ttnn::Tensor input_rm, history_rm, actual_start;
    std::array<ttnn::Tensor, 4> taps;
    ttnn::Tensor grad_q, grad_k, grad_v;
};

Inputs make_inputs(uint32_t T, Widths w) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto tap = [&](uint32_t j) { return random_device(ttnn::Shape{1, 1, w.c()}, 3 + j); };
    return Inputs{
        random_device(ttnn::Shape{1, T, w.c()}, 1, ttnn::Layout::ROW_MAJOR),
        random_device(ttnn::Shape{1, 3, w.c()}, 2, ttnn::Layout::ROW_MAJOR),
        ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            std::vector<uint32_t>{0}, ttnn::Shape({1}), device, ttnn::Layout::ROW_MAJOR),
        {tap(0), tap(1), tap(2), tap(3)},
        random_device(ttnn::Shape{1, T, w.q}, 10),
        random_device(ttnn::Shape{1, T, w.k}, 11),
        random_device(ttnn::Shape{1, T, w.v}, 12)};
}

ttnn::Tensor shifted_rows(const ttnn::Tensor& window, uint32_t begin, uint32_t T) {
    const auto C = static_cast<uint32_t>(window.logical_shape()[-1]);
    return ttnn::slice(
        window,
        ttsl::SmallVector<uint32_t>{0, begin, 0},
        ttsl::SmallVector<uint32_t>{1, begin + T, C},
        ttsl::SmallVector<uint32_t>{1, 1, 1});
}

// Backward from stock ops: recompute preact, silu', then dtap_j = sum_t dpre * x[t-3+j], dx = reversed conv.
void composite_bw(const Inputs& in, uint32_t T) {
    auto input = ttnn::to_layout(in.input_rm, ttnn::Layout::TILE);
    auto history = ttnn::to_layout(in.history_rm, ttnn::Layout::TILE);
    auto window = ttnn::concat(std::vector<ttnn::Tensor>{history, input}, 1);
    std::array<ttnn::Tensor, 4> views;
    std::optional<ttnn::Tensor> preact;
    for (uint32_t j = 0; j < 4; ++j) {
        views[j] = shifted_rows(window, j, T);
        auto term = ttnn::multiply(views[j], in.taps[j]);
        preact = preact ? ttnn::add(*preact, term) : term;
    }
    auto grad_y = ttnn::concat(std::vector<ttnn::Tensor>{in.grad_q, in.grad_k, in.grad_v}, 2);
    auto sig = ttnn::sigmoid(*preact);
    auto grad_preact = ttnn::multiply(
        grad_y, ttnn::multiply(sig, ttnn::add(ttnn::multiply(*preact, ttnn::subtract(1.0F, sig)), 1.0F)));
    std::array<ttnn::Tensor, 4> grad_taps;
    for (uint32_t j = 0; j < 4; ++j) {
        grad_taps[j] = ttnn::sum(ttnn::multiply(grad_preact, views[j]), 1, true);
        benchmark::DoNotOptimize(grad_taps[j]);
    }
    auto padded = ttnn::pad(grad_preact, ttsl::SmallVector<std::array<uint32_t, 2>>{{0, 0}, {0, 3}, {0, 0}}, 0.0F);
    std::optional<ttnn::Tensor> grad_input;
    for (uint32_t s = 0; s < 4; ++s) {
        auto term = ttnn::multiply(shifted_rows(padded, s, T), in.taps[3 - s]);
        grad_input = grad_input ? ttnn::add(*grad_input, term) : term;
    }
    benchmark::DoNotOptimize(grad_input);
}

void fused_bw(const Inputs& in, Widths w, uint32_t chunk, uint32_t splits) {
    auto out = ttml::metal::qkv_causal_conv1d_silu_bw(
        in.grad_q,
        in.grad_k,
        in.grad_v,
        in.input_rm,
        in.history_rm,
        in.taps[0],
        in.taps[1],
        in.taps[2],
        in.taps[3],
        w.q,
        w.k,
        w.v,
        chunk,
        splits);
    benchmark::DoNotOptimize(out);
}

void fused_fw(const Inputs& in, Widths w, uint32_t chunk) {
    auto out = ttnn::experimental::kda::qkv_causal_conv1d_silu(
        in.input_rm,
        in.history_rm,
        in.taps[0],
        in.taps[1],
        in.taps[2],
        in.taps[3],
        w.q,
        w.k,
        w.v,
        ttnn::experimental::kda::QkvCausalConv1dSiluProgramConfig{chunk},
        in.actual_start,
        in.history_rm);
    benchmark::DoNotOptimize(out);
}

template <typename Fn>
double time_ms(Fn&& fn, uint32_t warmup, uint32_t measure) {
    auto* device = &ttml::autograd::ctx().get_device();
    return ttml::benchmark_utils::time_device_avg_us(*device, warmup, measure, fn) / 1000.0;
}

void BM_QkvCausalConv1dSilu(benchmark::State& state) {
    const uint32_t T = env_u32("QKV_T", 4096);
    const uint32_t W = env_u32("QKV_WIDTH", 4096);
    const Widths w{env_u32("QKV_Q", W), env_u32("QKV_K", W), env_u32("QKV_V", W)};
    const auto chunks = env_list("QKV_CHUNKS", {32, 64, 128, 256});
    const auto splits = env_list("QKV_SPLITS", {0});
    const uint32_t warmup = 3;
    const uint32_t measure = 10;

    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    const auto in = make_inputs(T, w);
    tt::tt_metal::distributed::Synchronize(*device, std::nullopt);

    const double composite_ms = time_ms([&] { composite_bw(in, T); }, warmup, measure);
    fmt::print("T={} Q={} K={} V={} cores={}x{}\n", T, w.q, w.k, w.v, grid.x, grid.y);
    fmt::print("composite bw (stock ttnn ops): {:.3f} ms\n", composite_ms);
    double best_fw = 1e9, best_bw = 1e9;
    for (const auto chunk : chunks) {
        const double fw_ms = time_ms([&] { fused_fw(in, w, chunk); }, warmup, measure);
        best_fw = std::min(best_fw, fw_ms);
        for (const auto split : splits) {
            const double bw_ms = time_ms([&] { fused_bw(in, w, chunk, split); }, warmup, measure);
            best_bw = std::min(best_bw, bw_ms);
            fmt::print(
                "chunk={:4} splits={:3}  fused fw {:.3f} ms  fused bw {:.3f} ms  bw speedup vs composite {:.2f}x\n",
                chunk,
                split,
                fw_ms,
                bw_ms,
                composite_ms / bw_ms);
        }
    }
    fmt::print(
        "step (fw+bw): composite {:.3f} ms  fused {:.3f} ms  speedup {:.2f}x\n",
        best_fw + composite_ms,
        best_fw + best_bw,
        (best_fw + composite_ms) / (best_fw + best_bw));
    for (auto _ : state) {
        fused_bw(in, w, chunks.front(), splits.front());
    }
    ttml::autograd::ctx().reset_graph();
}

BENCHMARK(BM_QkvCausalConv1dSilu)->Unit(benchmark::kMillisecond)->Iterations(1);

}  // namespace

int main(int argc, char** argv) {
    ttml::autograd::ctx().open_device();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    ttml::autograd::ctx().close_device();
    return 0;
}
