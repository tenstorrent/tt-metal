// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <tt-metalium/distributed.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/swiglu_op.hpp"
#include "utils/memory_utils.hpp"

class SwiGLUPerfTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

struct PerfResult {
    std::string label;
    double fwd_time_ms = 0.0;
    double bwd_time_ms = 0.0;
    double total_time_ms = 0.0;
    size_t fwd_dram_peak = 0;
    size_t bwd_dram_peak = 0;
};

struct TestShape {
    std::vector<uint32_t> input_shape;
    uint32_t hidden_dim;
    std::string name;
};

void create_random_data(xt::xarray<float>& out, uint32_t seed) {
    const float bound = 0.5f;
    ttml::core::parallel_generate<float>(
        out, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, seed);
}

PerfResult run_swiglu_variant(
    const std::string& label, const TestShape& shape, bool use_optimized, int warmup_iters, int measure_iters) {
    using namespace ttml;

    auto& rng = autograd::ctx().get_generator();
    auto* device = &autograd::ctx().get_device();

    const auto& input_shape = shape.input_shape;
    const uint32_t input_dim = input_shape.back();
    const uint32_t H = shape.hidden_dim;

    // Generate weights in LinearLayer convention [out, in]
    std::vector<uint32_t> w13_lin_shape = {H, input_dim};
    std::vector<uint32_t> w2_lin_shape = {input_dim, H};

    xt::xarray<float> input_data = xt::empty<float>(input_shape);
    create_random_data(input_data, rng());
    xt::xarray<float> w1_lin = xt::empty<float>(w13_lin_shape);
    create_random_data(w1_lin, rng());
    xt::xarray<float> w2_lin = xt::empty<float>(w2_lin_shape);
    create_random_data(w2_lin, rng());
    xt::xarray<float> w3_lin = xt::empty<float>(w13_lin_shape);
    create_random_data(w3_lin, rng());

    // Pre-compute rank-4 transposed weights for baseline
    auto to_rank4 = [](const xt::xarray<float>& w, uint32_t r, uint32_t c) {
        return xt::xarray<float>(xt::reshape_view(xt::transpose(w), {1U, 1U, r, c}));
    };
    auto w1_r4 = to_rank4(w1_lin, input_dim, H);
    auto w2_r4 = to_rank4(w2_lin, H, input_dim);
    auto w3_r4 = to_rank4(w3_lin, input_dim, H);

    auto run_fwd_bwd_timed = [&](PerfResult& result) {
        auto input_tensor = autograd::create_tensor(core::from_xtensor(input_data, device));

        auto t0 = std::chrono::high_resolution_clock::now();
        autograd::TensorPtr out;
        if (use_optimized) {
            auto w1 = autograd::create_tensor(core::from_xtensor(w1_lin, device));
            auto w2 = autograd::create_tensor(core::from_xtensor(w2_lin, device));
            auto w3 = autograd::create_tensor(core::from_xtensor(w3_lin, device));
            out = ops::swiglu_optimized(input_tensor, w1, w2, w3);
        } else {
            auto w1 = autograd::create_tensor(core::from_xtensor(w1_r4, device));
            auto w2 = autograd::create_tensor(core::from_xtensor(w2_r4, device));
            auto w3 = autograd::create_tensor(core::from_xtensor(w3_r4, device));
            out = ops::swiglu(input_tensor, w1, w2, w3);
        }
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        auto t1 = std::chrono::high_resolution_clock::now();

        out->set_grad(core::ones_like(out->get_value()));
        out->backward();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        auto t2 = std::chrono::high_resolution_clock::now();

        result.fwd_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.bwd_time_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();

        autograd::ctx().reset_graph();
    };

    auto run_fwd_bwd_memory = [&](PerfResult& result) {
        auto input_tensor = autograd::create_tensor(core::from_xtensor(input_data, device));

        auto guard = utils::MemoryUsageTracker::begin_capture();

        autograd::TensorPtr out;
        if (use_optimized) {
            auto w1 = autograd::create_tensor(core::from_xtensor(w1_lin, device));
            auto w2 = autograd::create_tensor(core::from_xtensor(w2_lin, device));
            auto w3 = autograd::create_tensor(core::from_xtensor(w3_lin, device));
            out = ops::swiglu_optimized(input_tensor, w1, w2, w3);
        } else {
            auto w1 = autograd::create_tensor(core::from_xtensor(w1_r4, device));
            auto w2 = autograd::create_tensor(core::from_xtensor(w2_r4, device));
            auto w3 = autograd::create_tensor(core::from_xtensor(w3_r4, device));
            out = ops::swiglu(input_tensor, w1, w2, w3);
        }
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        utils::MemoryUsageTracker::snapshot("forward");

        out->set_grad(core::ones_like(out->get_value()));
        out->backward();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        utils::MemoryUsageTracker::end_capture("backward");

        result.fwd_dram_peak = utils::MemoryUsageTracker::get_dram_usage("forward").peak;
        result.bwd_dram_peak = utils::MemoryUsageTracker::get_dram_usage("backward").peak;
        utils::MemoryUsageTracker::clear();

        autograd::ctx().reset_graph();
    };

    PerfResult warmup_result;
    for (int i = 0; i < warmup_iters; ++i) {
        run_fwd_bwd_timed(warmup_result);
    }

    PerfResult result;
    result.label = label;

    // Capture memory on a dedicated run (graph tracing adds overhead, keep it separate)
    run_fwd_bwd_memory(result);

    for (int i = 0; i < measure_iters; ++i) {
        run_fwd_bwd_timed(result);
    }

    result.fwd_time_ms /= measure_iters;
    result.bwd_time_ms /= measure_iters;
    result.total_time_ms = result.fwd_time_ms + result.bwd_time_ms;

    return result;
}

void print_comparison(const TestShape& shape, const PerfResult& baseline, const PerfResult& optimized) {
    auto kb = [](size_t bytes) { return static_cast<double>(bytes) / 1024.0; };
    auto pct = [](double a, double b) -> std::string {
        if (b == 0) {
            return "N/A";
        }
        double change = (a - b) / b * 100.0;
        std::ostringstream oss;
        oss << std::showpos << std::fixed << std::setprecision(1) << change << "%";
        return oss.str();
    };

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Shape: " << std::left << std::setw(70) << shape.name << " ║\n";
    std::cout << "╠═════════════════════╦═════════════╦═════════════╦═════════════╦══════════════╣\n";
    std::cout << "║ Variant             ║  Fwd (ms)   ║  Bwd (ms)   ║ Total (ms)  ║ DRAM Pk (KB) ║\n";
    std::cout << "╠═════════════════════╬═════════════╬═════════════╬═════════════╬══════════════╣\n";

    auto print_row = [&](const PerfResult& r) {
        std::cout << "║ " << std::setw(19) << std::left << r.label << " ║ " << std::setw(11) << std::right << std::fixed
                  << std::setprecision(2) << r.fwd_time_ms << " ║ " << std::setw(11) << std::right << std::fixed
                  << std::setprecision(2) << r.bwd_time_ms << " ║ " << std::setw(11) << std::right << std::fixed
                  << std::setprecision(2) << r.total_time_ms << " ║ " << std::setw(12) << std::right << std::fixed
                  << std::setprecision(1) << kb(r.bwd_dram_peak) << " ║\n";
    };

    print_row(baseline);
    print_row(optimized);

    std::cout << "╠═════════════════════╬═════════════╬═════════════╬═════════════╬══════════════╣\n";
    std::cout << "║ Delta (opt vs base) ║ " << std::setw(11) << std::right
              << pct(optimized.fwd_time_ms, baseline.fwd_time_ms) << " ║ " << std::setw(11) << std::right
              << pct(optimized.bwd_time_ms, baseline.bwd_time_ms) << " ║ " << std::setw(11) << std::right
              << pct(optimized.total_time_ms, baseline.total_time_ms) << " ║ " << std::setw(12) << std::right
              << pct(static_cast<double>(optimized.bwd_dram_peak), static_cast<double>(baseline.bwd_dram_peak))
              << " ║\n";

    std::cout << "╚═════════════════════╩═════════════╩═════════════╩═════════════╩══════════════╝\n";
}

void RunComparison(const TestShape& shape, int warmup = 2, int measure = 5) {
    auto baseline = run_swiglu_variant("swiglu", shape, false, warmup, measure);
    auto optimized = run_swiglu_variant("swiglu_optimized", shape, true, warmup, measure);

    print_comparison(shape, baseline, optimized);

    // The optimized backward should use less or equal DRAM than the baseline
    EXPECT_LE(optimized.bwd_dram_peak, baseline.bwd_dram_peak)
        << "Optimized backward DRAM peak (" << optimized.bwd_dram_peak << ") should not exceed baseline ("
        << baseline.bwd_dram_peak << ")";
}

}  // namespace

TEST_F(SwiGLUPerfTest, NIGHTLY_Small_4x1x64x128) {
    RunComparison({.input_shape = {4, 1, 64, 128}, .hidden_dim = 128, .name = "4x1x64x128, H=128"});
}

TEST_F(SwiGLUPerfTest, NIGHTLY_Medium_2x1x128x256) {
    RunComparison({.input_shape = {2, 1, 128, 256}, .hidden_dim = 512, .name = "2x1x128x256, H=512"});
}

TEST_F(SwiGLUPerfTest, NIGHTLY_Large_4x1x256x384) {
    RunComparison({.input_shape = {4, 1, 256, 384}, .hidden_dim = 1024, .name = "4x1x256x384, H=1024 (NanoGPT-like)"});
}

TEST_F(SwiGLUPerfTest, NIGHTLY_XLarge_1x1x1024x1024) {
    RunComparison({.input_shape = {1, 1, 1024, 1024}, .hidden_dim = 2048, .name = "1x1x1024x1024, H=2048"});
}

// TinyLlama 1.1B: hidden_size=2048, intermediate_size=5632
// Expected to fail: swiglu_fw kernel circular buffers exceed L1 capacity at these dimensions
TEST_F(SwiGLUPerfTest, DISABLED_NIGHTLY_TinyLlama_1x1x256x2048) {
    RunComparison(
        {.input_shape = {1, 1, 256, 2048}, .hidden_dim = 5632, .name = "1x1x256x2048, H=5632 (TinyLlama, seq=256)"});
}

TEST_F(SwiGLUPerfTest, DISABLED_NIGHTLY_TinyLlama_1x1x1024x2048) {
    RunComparison(
        {.input_shape = {1, 1, 1024, 2048}, .hidden_dim = 5632, .name = "1x1x1024x2048, H=5632 (TinyLlama, seq=1024)"});
}

TEST_F(SwiGLUPerfTest, DISABLED_NIGHTLY_TinyLlama_4x1x512x2048) {
    RunComparison(
        {.input_shape = {4, 1, 512, 2048},
         .hidden_dim = 5632,
         .name = "4x1x512x2048, H=5632 (TinyLlama, B=4 seq=512)"});
}

// Llama 3.2 1B: hidden_size=2048, intermediate_size=8192
TEST_F(SwiGLUPerfTest, NIGHTLY_Llama1B_1x1x256x2048) {
    RunComparison(
        {.input_shape = {1, 1, 256, 2048}, .hidden_dim = 8192, .name = "1x1x256x2048, H=8192 (Llama-1B, seq=256)"});
}

TEST_F(SwiGLUPerfTest, NIGHTLY_Llama1B_1x1x1024x2048) {
    RunComparison(
        {.input_shape = {1, 1, 1024, 2048}, .hidden_dim = 8192, .name = "1x1x1024x2048, H=8192 (Llama-1B, seq=1024)"});
}

TEST_F(SwiGLUPerfTest, NIGHTLY_Llama1B_4x1x512x2048) {
    RunComparison(
        {.input_shape = {4, 1, 512, 2048}, .hidden_dim = 8192, .name = "4x1x512x2048, H=8192 (Llama-1B, B=4 seq=512)"});
}
