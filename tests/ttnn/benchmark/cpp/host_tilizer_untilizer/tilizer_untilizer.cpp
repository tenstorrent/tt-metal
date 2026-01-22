// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt_stl/indestructible.hpp>

// static std::vector<float> input_data;
namespace {
constexpr PhysicalSize shape = {10240, 1024};

const std::vector<float>& GetInputData() {
    static tt::stl::Indestructible<std::vector<float>> input_data([]() {
        std::vector<float> input_data;
        input_data.resize(shape[0] * shape[1]);
        std::mt19937 gen(42);  // fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        for (auto& val : input_data) {
            val = dist(gen);
        }
        return input_data;
    }());

    return input_data.get();
}

// Benchmark for LIN_ROW_MAJOR -> TILED_SWIZZLED
void BM_ConvertLayout_RowMajorToTiledSwizzled(benchmark::State& state) {
    const auto& input_data = GetInputData();
    for ([[maybe_unused]] auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::make_const_span(input_data),
            shape,
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_SWIZZLED);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for LIN_ROW_MAJOR -> TILED_NFACES
void BM_ConvertLayout_RowMajorToTiledNfaces(benchmark::State& state) {
    const auto& input_data = GetInputData();
    for ([[maybe_unused]] auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::make_const_span(input_data),
            shape,
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for TILED_SWIZZLED -> LIN_ROW_MAJOR
void BM_ConvertLayout_TiledSwizzledToRowMajor(benchmark::State& state) {
    // Pre-convert input_data to TILED_SWIZZLED for a fair benchmark
    const auto& input_data = GetInputData();
    static std::vector<float> tiled_data = convert_layout<float>(
        tt::stl::make_const_span(input_data), shape, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_SWIZZLED);

    for ([[maybe_unused]] auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::make_const_span(tiled_data),
            shape,
            TensorLayoutType::TILED_SWIZZLED,
            TensorLayoutType::LIN_ROW_MAJOR);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for TILED_SWIZZLED -> TILED_NFACES
void BM_ConvertLayout_TiledSwizzledToTiledNFaces(benchmark::State& state) {
    // Pre-convert input_data to TILED_SWIZZLED
    const auto& input_data = GetInputData();
    static std::vector<float> tiled_data = convert_layout<float>(
        tt::stl::make_const_span(input_data), shape, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_SWIZZLED);

    for ([[maybe_unused]] auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::make_const_span(tiled_data),
            shape,
            TensorLayoutType::TILED_SWIZZLED,
            TensorLayoutType::TILED_NFACES);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for TILED_NFACES -> LIN_ROW_MAJOR
void BM_ConvertLayout_TiledNFacesToRowMajor(benchmark::State& state) {
    // Pre-convert input_data to TILED_NFACES
    const auto& input_data = GetInputData();
    static std::vector<float> nfaces_data = convert_layout<float>(
        tt::stl::make_const_span(input_data), shape, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);

    for ([[maybe_unused]] auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::make_const_span(nfaces_data),
            shape,
            TensorLayoutType::TILED_NFACES,
            TensorLayoutType::LIN_ROW_MAJOR);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for TILED_NFACES -> TILED_SWIZZLED
void BM_ConvertLayout_TiledNFacesToTiledSwizzled(benchmark::State& state) {
    // Pre-convert input_data to TILED_NFACES
    const auto& input_data = GetInputData();
    static std::vector<float> nfaces_data = convert_layout<float>(
        tt::stl::make_const_span(input_data), shape, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);

    for ([[maybe_unused]] auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::make_const_span(nfaces_data),
            shape,
            TensorLayoutType::TILED_NFACES,
            TensorLayoutType::TILED_SWIZZLED);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_ConvertLayout_RowMajorToTiledSwizzled)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ConvertLayout_RowMajorToTiledNfaces)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ConvertLayout_TiledSwizzledToRowMajor)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ConvertLayout_TiledSwizzledToTiledNFaces)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ConvertLayout_TiledNFacesToRowMajor)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ConvertLayout_TiledNFacesToTiledSwizzled)->Unit(benchmark::kMillisecond);
}  // namespace

BENCHMARK_MAIN();
