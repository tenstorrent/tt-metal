// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "metal/operations.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace {

struct GramMatmulShape {
    uint32_t M, K;
    std::string name;
};

const int num_warmup = 3;
const int num_iters = 10;

// Shapes from Llama 1.1B, 7B, 30B hidden/MLP dimensions
const std::vector<GramMatmulShape> shapes = {
    {2048, 2048, "2048x2048"},
    {2048, 5632, "2048x5632"},
    {4096, 4096, "4096x4096"},
    {4096, 11008, "4096x11008"},
    {8192, 8192, "8192x8192"},
};

ttnn::Tensor make_random_tensor(uint32_t M, uint32_t K, ttnn::distributed::MeshDevice* device, uint32_t seed) {
    std::vector<float> data(M * K);
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);
    auto shape = ttnn::Shape({1, 1, M, K});
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

void BM_GramMatmul(benchmark::State& state) {
    auto device = ttnn::device::open_mesh_device(0);
    device->enable_program_cache();

    auto compute_kernel_config = ttml::core::ComputeKernelConfig::matmul();
    auto device_grid = device->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(device_grid.x, device_grid.y);

    struct Result {
        std::string name;
        double gram_us, minimal_us, ttnn_us;
        double gram_tf, minimal_tf, ttnn_tf;
    };
    std::vector<Result> results;

    for (const auto& s : shapes) {
        const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(s.name));
        auto input = make_random_tensor(s.M, s.K, device.get(), seed);
        auto input_t = ttnn::transpose(input, -2, -1);

        double flops = 2.0 * s.M * s.M * s.K;
        auto tflops = [&](double us) { return flops / (us * 1e-6) / 1e12; };

        double gram_us = bench_op(
            [&]() {
                auto out = ttml::metal::gram_matmul(input);
                out.deallocate();
            },
            device.get());

        double minimal_us = bench_op(
            [&]() {
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
            },
            device.get());

        double ttnn_us = bench_op(
            [&]() {
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
            },
            device.get());

        results.push_back({s.name, gram_us, minimal_us, ttnn_us, tflops(gram_us), tflops(minimal_us), tflops(ttnn_us)});

        input.deallocate();
        input_t.deallocate();
    }

    // Print results table
    // Cell format: "%6.0f us %3.0f TF" = 16 chars
    std::cout << "\n";
    std::cout << "  ┌────────────┬──────────────────┬──────────────────┬──────────────────┬────────────┬─────────┐\n";
    std::cout << "  │   Shape    │  gram_matmul     │  minimal_matmul  │  ttnn::matmul    │ vs minimal │ vs ttnn │\n";
    std::cout << "  ├────────────┼──────────────────┼──────────────────┼──────────────────┼────────────┼─────────┤\n";
    for (const auto& r : results) {
        char line[256];
        std::snprintf(
            line,
            sizeof(line),
            "  │ %10s │ %6.0f us %3.0f TF │ %6.0f us %3.0f TF │ %6.0f us %3.0f TF │     %5.2fx │  %5.2fx │",
            r.name.c_str(),
            r.gram_us,
            r.gram_tf,
            r.minimal_us,
            r.minimal_tf,
            r.ttnn_us,
            r.ttnn_tf,
            r.minimal_us / r.gram_us,
            r.ttnn_us / r.gram_us);
        std::cout << line << "\n";
    }
    std::cout << "  └────────────┴──────────────────┴──────────────────┴──────────────────┴────────────┴─────────┘\n";
    std::cout << std::flush;

    state.SetIterationTime(results.back().gram_us * 1e-6);

    device->close();
}

}  // namespace

BENCHMARK(BM_GramMatmul)->Unit(benchmark::kMillisecond)->UseManualTime()->Iterations(1)->Name("GramMatmul");

BENCHMARK_MAIN();
