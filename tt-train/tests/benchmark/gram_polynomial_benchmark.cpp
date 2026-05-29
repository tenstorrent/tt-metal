// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "metal/operations.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace {

struct Shape {
    uint32_t M;
    std::string name;
};

const int num_warmup = 3;
const int num_iters = 10;

const float b = -4.775f;
const float c = 2.0315f;

// NS sees G of size [min(rows, cols), min(rows, cols)] after transpose-if-tall.
// TinyLlama [5632,2048] and [2048,2048] → G = 2048².
// Llama 70B [28672,8192] and [8192,8192] → G = 8192².
const std::vector<Shape> shapes = {
    {2048, "2048 (TinyLlama)"},
    {4096, "4096"},
    {8192, "8192 (Llama 70B)"},
};

ttnn::Tensor make_random_tensor(uint32_t M, ttnn::distributed::MeshDevice* device, uint32_t seed) {
    std::vector<float> data(M * M);
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);
    auto shape = ttnn::Shape({1, 1, M, M});
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

double bench_op(auto fn, ttnn::distributed::MeshDevice* device) {
    for (int i = 0; i < num_warmup; ++i) {
        fn();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        fn();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / num_iters;
}

void BM_GramPolynomial(benchmark::State& state) {
    auto device = ttnn::device::open_mesh_device(0);
    device->enable_program_cache();

    auto compute_kernel_config = ttml::core::ComputeKernelConfig::matmul();
    auto device_grid = device->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(device_grid.x, device_grid.y);

    struct Result {
        std::string name;
        double fused_us, composite_us;
        double fused_tf, composite_tf;
    };
    std::vector<Result> results;

    for (const auto& s : shapes) {
        const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(s.name));
        auto G = make_random_tensor(s.M, device.get(), seed);
        auto G_t = ttnn::transpose(G, -2, -1);

        double flops = 2.0 * s.M * s.M * s.M;
        auto tflops = [&](double us) { return flops / (us * 1e-6) / 1e12; };

        double fused_us = bench_op(
            [&]() {
                auto out = ttml::metal::gram_polynomial(G, b, c);
                out.deallocate();
            },
            device.get());

        // Composite: c * ttnn::matmul(G, G^T) + b * G
        double composite_us = bench_op(
            [&]() {
                auto g2 = ttnn::matmul(
                    G,
                    G_t,
                    false,
                    false,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    compute_kernel_config,
                    core_grid,
                    std::nullopt);
                auto cg2 = ttnn::multiply(g2, c);
                g2.deallocate();
                auto bg = ttnn::multiply(G, b);
                auto out = ttnn::add(cg2, bg);
                cg2.deallocate();
                bg.deallocate();
                out.deallocate();
            },
            device.get());

        results.push_back({s.name, fused_us, composite_us, tflops(fused_us), tflops(composite_us)});

        G.deallocate();
        G_t.deallocate();
    }

    // Print results table
    std::cout << "\n  b=" << b << ", c=" << c << "\n";
    std::cout << "  ┌──────────────────┬──────────────────┬──────────────────┬─────────────┐\n";
    std::cout << "  │ Shape            │ gram_polynomial  │ composite (ttnn) │ speedup     │\n";
    std::cout << "  ├──────────────────┼──────────────────┼──────────────────┼─────────────┤\n";
    for (const auto& r : results) {
        char line[256];
        std::snprintf(
            line,
            sizeof(line),
            "  │ %-16s │ %6.0f us %3.0f TF │ %6.0f us %3.0f TF │      %5.2fx │",
            r.name.c_str(),
            r.fused_us,
            r.fused_tf,
            r.composite_us,
            r.composite_tf,
            r.composite_us / r.fused_us);
        std::cout << line << "\n";
    }
    std::cout << "  └──────────────────┴──────────────────┴──────────────────┴─────────────┘\n";
    std::cout << std::flush;

    state.SetIterationTime(results.back().fused_us * 1e-6);

    device->close();
}

}  // namespace

BENCHMARK(BM_GramPolynomial)->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(1)->Name("GramPolynomial");

BENCHMARK_MAIN();
