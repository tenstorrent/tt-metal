// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <iomanip>
#include <iostream>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/operations.hpp"
#include "ops/newton_schulz_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn_fixed/matmuls.hpp"

class KSplitGramMatmulBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

ttnn::Tensor make_test_tensor(uint32_t M_tiles, uint32_t K_dim = 0) {
    auto* device = &ttml::autograd::ctx().get_device();
    uint32_t M = M_tiles * 32;
    uint32_t K = (K_dim > 0) ? K_dim : M;
    std::vector<float> data(M * K);
    std::generate(data.begin(), data.end(), []() {
        static std::mt19937 gen(42);
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        return dist(gen);
    });
    auto shape = ttnn::Shape({1, 1, M, K});
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

// Newton-Schulz with k-split gram matmul replacing ttnn::matmul for X @ X^T
ttnn::Tensor newtonschulz5_gram(const ttnn::Tensor& G, int steps, float eps = 1e-7f) {
    float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    ttnn::Tensor X = G;

    ttnn::Tensor squares = ttnn::square(X);
    ttnn::Tensor sum_squares = ttnn::sum(
        squares, ttsl::SmallVector<int>{-2, -1}, true, std::nullopt, ttml::core::ComputeKernelConfig::precise());
    ttnn::Tensor norm_tensor = ttnn::sqrt(sum_squares);
    ttnn::Tensor norm_plus_eps = ttnn::add(norm_tensor, eps);
    X = ttnn::divide(X, norm_plus_eps);

    auto shape = X.logical_shape();
    const uint32_t m = shape[-2];
    const uint32_t n = shape[-1];
    const bool needs_transpose = (m > n);

    if (needs_transpose) {
        X = ttnn::transpose(X, -2, -1);
        shape = X.logical_shape();
    }

    auto* device = X.device();
    const auto rank = shape.rank();
    std::vector<uint32_t> mm_dims;
    for (uint32_t i = 0; i < rank; ++i) mm_dims.push_back(shape[i]);
    mm_dims.back() = mm_dims[rank - 2];
    ttnn::Shape mm_shape(mm_dims);

    auto buf_A = ttnn::empty(mm_shape, X.dtype(), X.layout(), device, X.memory_config());
    auto buf_A2 = ttnn::empty(mm_shape, X.dtype(), X.layout(), device, X.memory_config());
    auto buf_BX = ttnn::empty(shape, X.dtype(), X.layout(), device, X.memory_config());

    for (int iter = 0; iter < steps; ++iter) {
        buf_A = ttml::metal::gram_matmul(X, ttml::metal::OutputMode::Full);

        ttml::ttnn_fixed::matmul(buf_A, buf_A, false, false, buf_A2);

        ttnn::multiply(buf_A, b, std::nullopt, std::nullopt, buf_A);
        ttnn::addalpha(buf_A, buf_A2, c, std::nullopt, buf_A);

        ttml::ttnn_fixed::matmul(buf_A, X, false, false, buf_BX);
        ttnn::addalpha(buf_BX, X, a, std::nullopt, X);
    }

    if (needs_transpose) {
        X = ttnn::transpose(X, -2, -1);
    }
    return X;
}

}  // namespace

TEST_F(KSplitGramMatmulBenchmark, MatmulComparison) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto device_grid = device->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(device_grid.x, device_grid.y);

    struct Shape {
        uint32_t M_tiles, K_dim;
        const char* label;
    };
    Shape shapes[] = {
        {64, 2048, "2048x2048"},
        {64, 5632, "2048x5632"},
        {128, 4096, "4096x4096"},
        {128, 11008, "4096x11008"},
        {256, 8192, "8192x8192"},
    };

    constexpr int warmup = 3;
    constexpr int iters = 10;

    auto bench = [&](auto fn) {
        for (int i = 0; i < warmup; i++) {
            fn();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            fn();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    };

    constexpr MathFidelity bench_fidelity = MathFidelity::HiFi4;
    auto compute_kernel_config = ttml::core::ComputeKernelConfig::matmul();

    // Utilization parameters
    constexpr int cycles_per_tile = (bench_fidelity == MathFidelity::HiFi4) ? 64 : 32;
    constexpr int num_cores = 110;
    constexpr double freq_ghz = 1.35;

    for (auto& s : shapes) {
        auto input = make_test_tensor(s.M_tiles, s.K_dim);
        auto input_t = ttnn::transpose(input, -2, -1);

        uint64_t M = s.M_tiles * 32;
        uint64_t K = s.K_dim;
        double num_tile_ops = static_cast<double>(M) * M * K / (32.0 * 32.0 * 32.0);
        double ideal_us = num_tile_ops * cycles_per_tile / num_cores / (freq_ghz * 1e3);
        auto util = [&](double us) { return 100.0 * ideal_us / us; };

        double t_gram = bench([&]() {
            auto out = ttml::metal::gram_matmul(input, ttml::metal::OutputMode::UpperTriangle, bench_fidelity);
            out.deallocate();
        });

        double t_minimal = bench([&]() {
            auto out = ttnn::experimental::minimal_matmul(
                input,
                input_t,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                compute_kernel_config);
            out.deallocate();
        });

        double t_ttnn = bench([&]() {
            auto out = ttnn::matmul(
                input,
                input_t,
                false,
                false,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                compute_kernel_config,
                core_grid,
                std::nullopt);
            out.deallocate();
        });

        std::cout << "\n  " << s.label << ":\n" << std::flush;
        std::cout << "    gram_matmul:      " << std::fixed << std::setprecision(1) << t_gram << " us  "
                  << std::setprecision(1) << util(t_gram) << "%\n";
        std::cout << "    minimal_matmul:   " << std::setprecision(1) << t_minimal << " us  " << std::setprecision(1)
                  << util(t_minimal) << "%\n";
        std::cout << "    ttnn::matmul:     " << std::setprecision(1) << t_ttnn << " us  " << std::setprecision(1)
                  << util(t_ttnn) << "%\n";
        std::cout << "    vs minimal:       " << std::setprecision(2) << t_minimal / t_gram << "x\n";
        std::cout << "    vs ttnn:          " << std::setprecision(2) << t_ttnn / t_gram << "x\n" << std::flush;

        input.deallocate();
        input_t.deallocate();
    }
    SUCCEED();
}

TEST_F(KSplitGramMatmulBenchmark, NewtonSchulz) {
    auto* device = &ttml::autograd::ctx().get_device();

    struct Shape {
        uint32_t M, K;
        const char* label;
    };
    Shape shapes[] = {
        {2048, 2048, "2048x2048"},
        {2048, 5632, "2048x5632"},
    };

    constexpr int warmup = 3;
    constexpr int iters = 10;
    constexpr int ns_steps = 5;

    auto bench = [&](auto fn) {
        for (int i = 0; i < warmup; i++) {
            fn();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            fn();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    };

    for (auto& s : shapes) {
        auto input = make_test_tensor(s.M / 32, s.K);

        double t_current = bench([&]() {
            auto out = ttml::ops::newtonschulz5(input, ns_steps);
            out.deallocate();
        });

        double t_gram = bench([&]() {
            auto out = newtonschulz5_gram(input, ns_steps);
            out.deallocate();
        });

        std::cout << "\n  Newton-Schulz (" << s.label << ", " << ns_steps << " steps):\n";
        std::cout << "    current (ttnn::matmul):   " << std::fixed << std::setprecision(1) << t_current << " us\n";
        std::cout << "    k-split (gram_matmul):    " << std::setprecision(1) << t_gram << " us\n";
        std::cout << "    speedup:                  " << std::setprecision(2) << t_current / t_gram << "x\n"
                  << std::flush;

        input.deallocate();
    }
    SUCCEED();
}
