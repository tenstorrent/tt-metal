#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/tilize_utils.hpp>

static std::vector<float> input_data;
static constexpr PhysicalSize shape = {10240, 1024};

static void GenerateRandomData() {
    input_data.resize(shape[0] * shape[1]);
    std::mt19937 gen(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (auto& val : input_data) {
        val = dist(gen);
    }
}

// Benchmark for LIN_ROW_MAJOR -> TILED_SWIZZLED
static void BM_ConvertLayout_RowMajorToTiledSwizzled(benchmark::State& state) {
    for (auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::MakeConstSpan(input_data),
            shape,
            tests::utils::TensorLayoutType::LIN_ROW_MAJOR,
            tests::utils::TensorLayoutType::TILED_SWIZZLED);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for LIN_ROW_MAJOR -> TILED_NFACES
static void BM_ConvertLayout_RowMajorToTiledNfaces(benchmark::State& state) {
    for (auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::MakeConstSpan(input_data),
            shape,
            tests::utils::TensorLayoutType::LIN_ROW_MAJOR,
            tests::utils::TensorLayoutType::TILED_NFACES);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for TILED_SWIZZLED -> LIN_ROW_MAJOR
static void BM_ConvertLayout_TiledSwizzledToRowMajor(benchmark::State& state) {
    // Pre-convert input_data to TILED_SWIZZLED for a fair benchmark
    static std::vector<float> tiled_data = convert_layout<float>(
        tt::stl::MakeConstSpan(input_data),
        shape,
        tests::utils::TensorLayoutType::LIN_ROW_MAJOR,
        tests::utils::TensorLayoutType::TILED_SWIZZLED);

    for (auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::MakeConstSpan(tiled_data),
            shape,
            tests::utils::TensorLayoutType::TILED_SWIZZLED,
            tests::utils::TensorLayoutType::LIN_ROW_MAJOR);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for TILED_SWIZZLED -> TILED_NFACES
static void BM_ConvertLayout_TiledSwizzledToTiledNFaces(benchmark::State& state) {
    // Pre-convert input_data to TILED_SWIZZLED
    static std::vector<float> tiled_data = convert_layout<float>(
        tt::stl::MakeConstSpan(input_data),
        shape,
        tests::utils::TensorLayoutType::LIN_ROW_MAJOR,
        tests::utils::TensorLayoutType::TILED_SWIZZLED);

    for (auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::MakeConstSpan(tiled_data),
            shape,
            tests::utils::TensorLayoutType::TILED_SWIZZLED,
            tests::utils::TensorLayoutType::TILED_NFACES);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for TILED_NFACES -> LIN_ROW_MAJOR
static void BM_ConvertLayout_TiledNFacesToRowMajor(benchmark::State& state) {
    // Pre-convert input_data to TILED_NFACES
    static std::vector<float> nfaces_data = convert_layout<float>(
        tt::stl::MakeConstSpan(input_data),
        shape,
        tests::utils::TensorLayoutType::LIN_ROW_MAJOR,
        tests::utils::TensorLayoutType::TILED_NFACES);

    for (auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::MakeConstSpan(nfaces_data),
            shape,
            tests::utils::TensorLayoutType::TILED_NFACES,
            tests::utils::TensorLayoutType::LIN_ROW_MAJOR);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

// Benchmark for TILED_NFACES -> TILED_SWIZZLED
static void BM_ConvertLayout_TiledNFacesToTiledSwizzled(benchmark::State& state) {
    // Pre-convert input_data to TILED_NFACES
    static std::vector<float> nfaces_data = convert_layout<float>(
        tt::stl::MakeConstSpan(input_data),
        shape,
        tests::utils::TensorLayoutType::LIN_ROW_MAJOR,
        tests::utils::TensorLayoutType::TILED_NFACES);

    for (auto _ : state) {
        auto out = convert_layout<float>(
            tt::stl::MakeConstSpan(nfaces_data),
            shape,
            tests::utils::TensorLayoutType::TILED_NFACES,
            tests::utils::TensorLayoutType::TILED_SWIZZLED);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_ConvertLayout_RowMajorToTiledSwizzled)->Unit(benchmark::kMillisecond);
;
BENCHMARK(BM_ConvertLayout_RowMajorToTiledNfaces)->Unit(benchmark::kMillisecond);
;
BENCHMARK(BM_ConvertLayout_TiledSwizzledToRowMajor)->Unit(benchmark::kMillisecond);
;
BENCHMARK(BM_ConvertLayout_TiledSwizzledToTiledNFaces)->Unit(benchmark::kMillisecond);
;
BENCHMARK(BM_ConvertLayout_TiledNFacesToRowMajor)->Unit(benchmark::kMillisecond);
;
BENCHMARK(BM_ConvertLayout_TiledNFacesToTiledSwizzled)->Unit(benchmark::kMillisecond);
;

int main(int argc, char** argv) {
    GenerateRandomData();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
