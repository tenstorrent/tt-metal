// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qkv causal conv1d + SiLU benchmark, one layer, forward + backward.
//   composite - backward built from stock ttnn ops (what you get without the fused kernel)
//   fused     - ttml::metal::qkv_causal_conv1d_silu_bw, per chunk / splits, with and without tap gradients
//   autograd  - ops::qkv_causal_conv1d_silu forward + backward from a TILE input with frozen taps: the call the
//               model makes, layout conversions and tap post-processing included
// Defaults are the Qwen3.8-27B per-chip shape (TP=4): T=1024, Q=K=512, V=1536. Env: TTML_QKV_CONV_BENCH_T,
// _WIDTH (sets Q=K=V) or _Q/_K/_V, _CHUNKS (csv, default 32,64,128,256), _SPLITS (csv, 0 = auto).

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
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/program_utils.hpp"
#include "metal/operations.hpp"
#include "ops/qkv_causal_conv1d_silu_op.hpp"
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
    ttnn::Tensor input_tile, history_rm, actual_start;
    std::array<ttnn::Tensor, 4> taps;
    ttnn::Tensor grad_q, grad_k, grad_v;
};

Inputs make_inputs(uint32_t T, Widths w) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto tap = [&](uint32_t j) { return random_device(ttnn::Shape{1, 1, w.c()}, 3 + j); };
    return Inputs{
        random_device(ttnn::Shape{1, T, w.c()}, 1),
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
    auto history = ttnn::to_layout(in.history_rm, ttnn::Layout::TILE);
    auto window = ttnn::concat(std::vector<ttnn::Tensor>{history, in.input_tile}, 1);
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

void fused_bw(const Inputs& in, Widths w, uint32_t chunk, uint32_t splits, bool tap_grads) {
    auto out = ttml::metal::qkv_causal_conv1d_silu_bw(
        in.grad_q,
        in.grad_k,
        in.grad_v,
        in.input_tile,
        in.history_rm,
        in.taps[0],
        in.taps[1],
        in.taps[2],
        in.taps[3],
        w.q,
        w.k,
        w.v,
        chunk,
        splits,
        tap_grads);
    benchmark::DoNotOptimize(out);
}

void fused_fw(const Inputs& in, Widths w, uint32_t chunk) {
    auto out = ttnn::experimental::kda::qkv_causal_conv1d_silu(
        in.input_tile,
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

// The model's call: autograd op forward + backward from a TILE input, taps frozen.
void autograd_fw_bw(const Inputs& in, Widths w) {
    using namespace ttml;
    auto input = autograd::create_tensor(in.input_tile, /* requires_grad */ true);
    std::array<autograd::TensorPtr, 4> taps;
    for (size_t j = 0; j < 4; ++j) {
        taps[j] = autograd::create_tensor(in.taps[j], /* requires_grad */ false);
    }
    auto [q, k, v] =
        ops::qkv_causal_conv1d_silu(input, in.history_rm, taps[0], taps[1], taps[2], taps[3], w.q, w.k, w.v);
    q->set_grad(in.grad_q);
    k->set_grad(in.grad_k);
    v->set_grad(in.grad_v);
    q->backward();
    benchmark::DoNotOptimize(input->get_grad());
    autograd::ctx().reset_graph();
}

template <typename Fn>
double time_ms(Fn&& fn, uint32_t warmup, uint32_t measure) {
    auto* device = &ttml::autograd::ctx().get_device();
    return ttml::benchmark_utils::time_device_avg_us(*device, warmup, measure, fn) / 1000.0;
}

void BM_QkvCausalConv1dSilu(benchmark::State& state) {
    const uint32_t T = env_u32("TTML_QKV_CONV_BENCH_T", 1024);
    const uint32_t W = env_u32("TTML_QKV_CONV_BENCH_WIDTH", 0);
    const Widths w{
        env_u32("TTML_QKV_CONV_BENCH_Q", W != 0 ? W : 512),
        env_u32("TTML_QKV_CONV_BENCH_K", W != 0 ? W : 512),
        env_u32("TTML_QKV_CONV_BENCH_V", W != 0 ? W : 1536)};
    const auto chunks = env_list("TTML_QKV_CONV_BENCH_CHUNKS", {32, 64, 128, 256});
    const auto splits = env_list("TTML_QKV_CONV_BENCH_SPLITS", {0});
    const uint32_t warmup = 3;
    const uint32_t measure = 10;

    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    const auto in = make_inputs(T, w);
    tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
    fmt::print("T={} Q={} K={} V={} cores={}x{}\n", T, w.q, w.k, w.v, grid.x, grid.y);

    const double composite_ms = time_ms([&] { composite_bw(in, T); }, warmup, measure);
    fmt::print("composite bw (stock ttnn ops): {:.3f} ms\n", composite_ms);
    struct Best {
        double ms = 1e9;
        uint32_t chunk = 0, splits = 0;
        void offer(double candidate_ms, uint32_t candidate_chunk, uint32_t candidate_splits) {
            if (candidate_ms < ms) {
                ms = candidate_ms;
                chunk = candidate_chunk;
                splits = candidate_splits;
            }
        }
    };
    double best_fw = 1e9;
    Best best_bw, best_bw_frozen;
    for (const auto chunk : chunks) {
        const double fw_ms = time_ms([&] { fused_fw(in, w, chunk); }, warmup, measure);
        best_fw = std::min(best_fw, fw_ms);
        for (const auto split : splits) {
            const double bw_ms = time_ms([&] { fused_bw(in, w, chunk, split, true); }, warmup, measure);
            const double bw_frozen_ms = time_ms([&] { fused_bw(in, w, chunk, split, false); }, warmup, measure);
            best_bw.offer(bw_ms, chunk, split);
            best_bw_frozen.offer(bw_frozen_ms, chunk, split);
            fmt::print(
                "chunk={:4} splits={:3}  fused fw {:.3f} ms  fused bw {:.3f} ms  frozen taps {:.3f} ms  bw speedup vs "
                "composite {:.2f}x\n",
                chunk,
                split,
                fw_ms,
                bw_ms,
                bw_frozen_ms,
                composite_ms / bw_ms);
        }
    }
    // What the cost model picks on this grid, next to the best of the rows above when they enumerate splits (with
    // the default splits 0 every row already is the model's pick for its chunk).
    const uint64_t l1 = ttml::metal::available_l1_bytes(*device);
    const bool enumerated_splits = splits.size() > 1 || splits.front() != 0;
    for (const auto& [name, tap_grads, best] :
         {std::tuple{"bw", true, best_bw}, std::tuple{"frozen taps", false, best_bw_frozen}}) {
        const auto pick =
            ttml::metal::qkv_causal_conv1d_silu_bw_work_split(w.c(), T, grid.x * grid.y, l1, 0, 0, tap_grads);
        fmt::print("{}: auto pick chunk={} splits={}", name, pick.channel_chunk_size, pick.row_splits);
        if (enumerated_splits) {
            fmt::print("; best measured chunk={} splits={} at {:.3f} ms", best.chunk, best.splits, best.ms);
        }
        fmt::print("\n");
    }
    const double autograd_ms = time_ms([&] { autograd_fw_bw(in, w); }, warmup, measure);
    fmt::print("autograd op, TILE input, frozen taps (fw + bw, auto chunk / splits): {:.3f} ms\n", autograd_ms);

    for (auto _ : state) {
        state.SetIterationTime(autograd_ms / 1000.0);
    }
    state.counters["Composite_bw_ms"] = composite_ms;
    state.counters["Fused_fw_ms"] = best_fw;
    state.counters["Fused_bw_ms"] = best_bw.ms;
    state.counters["Fused_bw_frozen_taps_ms"] = best_bw_frozen.ms;
    state.counters["Autograd_fw_bw_ms"] = autograd_ms;
    ttml::autograd::ctx().reset_graph();
}

BENCHMARK(BM_QkvCausalConv1dSilu)->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(1);

}  // namespace

int main(int argc, char** argv) {
    ttml::autograd::ctx().open_device();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    ttml::autograd::ctx().close_device();
    return 0;
}
