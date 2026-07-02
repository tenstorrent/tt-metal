// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pure-host microbenchmark for the optimized BFP host tilizer.
//
// Measures `pack_as_bfp8_tiles` (and BFP4_b for comparison) under whatever
// configuration is selected by these env vars at process startup:
//
//   TT_BFP_HOST_TILIZER_DISABLE_SIMD=1   force scalar fallback
//   TT_BFP_HOST_TILIZER_THREADS=N        cap worker thread count
//
// IMPORTANT: those env vars are cached at first use inside the production
// code, so we cannot toggle them between benchmarks within one process.
// To compare scalar vs SIMD, run this binary three times - one per config
// - with the env vars set before launch:
//
//   TT_BFP_HOST_TILIZER_DISABLE_SIMD=1 TT_BFP_HOST_TILIZER_THREADS=1 \
//     ./bench_bfp_pack | tee bench-scalar-1t.txt
//   TT_BFP_HOST_TILIZER_THREADS=1 \
//     ./bench_bfp_pack | tee bench-simd-1t.txt
//   ./bench_bfp_pack | tee bench-simd-nt.txt
//
// No device is required; this exercises only host code.

#include <benchmark/benchmark.h>

#include <cstdint>
#include <random>
#include <vector>

#include <tt_stl/indestructible.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>

namespace {

constexpr uint32_t FLOATS_PER_TILE = 1024;  // 32x32 tile

// ---------- Cached deterministic input data ------------------------------
// Build the largest input once, reuse for every benchmark size.
constexpr uint32_t MAX_TILES = 4096;

const std::vector<float>& GetFloatInput() {
    static tt::stl::Indestructible<std::vector<float>> data([]() {
        std::vector<float> v(MAX_TILES * FLOATS_PER_TILE);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        for (auto& f : v) {
            f = dist(gen);
        }
        return v;
    }());
    return data.get();
}

const std::vector<bfloat16>& GetBfloat16Input() {
    static tt::stl::Indestructible<std::vector<bfloat16>> data([]() {
        const auto& f = GetFloatInput();
        std::vector<bfloat16> v;
        v.reserve(f.size());
        for (float x : f) {
            v.emplace_back(x);
        }
        return v;
    }());
    return data.get();
}

template <typename T>
const std::vector<T>& GetInput();
template <>
const std::vector<float>& GetInput<float>() { return GetFloatInput(); }
template <>
const std::vector<bfloat16>& GetInput<bfloat16>() { return GetBfloat16Input(); }

// ---------- Throughput accounting ----------------------------------------
template <typename T>
void SetThroughput(benchmark::State& state, uint32_t num_tiles) {
    const std::int64_t bytes_per_iter = static_cast<std::int64_t>(num_tiles) * FLOATS_PER_TILE * sizeof(T);
    state.SetBytesProcessed(state.iterations() * bytes_per_iter);
    state.counters["tiles"] = benchmark::Counter(
        static_cast<double>(num_tiles) * state.iterations(), benchmark::Counter::kIsRate);
}

// ---------- Benchmark bodies ---------------------------------------------
// Bodies are intentionally minimal: NO PauseTiming/ResumeTiming, NO env
// var manipulation. The for-loop body is the timed region.

template <typename T>
void Bench_Bfp8b(benchmark::State& state) {
    const auto num_tiles = static_cast<uint32_t>(state.range(0));
    const auto& src = GetInput<T>();
    const tt::stl::Span<const T> input(src.data(), num_tiles * FLOATS_PER_TILE);

    for ([[maybe_unused]] auto _ : state) {
        auto packed = pack_as_bfp8_tiles(input, /*row_major_input=*/true, /*is_exp_a=*/false);
        benchmark::DoNotOptimize(packed.data());
        benchmark::ClobberMemory();
    }
    SetThroughput<T>(state, num_tiles);
}

template <typename T>
void Bench_Bfp4b(benchmark::State& state) {
    const auto num_tiles = static_cast<uint32_t>(state.range(0));
    const auto& src = GetInput<T>();
    const tt::stl::Span<const T> input(src.data(), num_tiles * FLOATS_PER_TILE);

    for ([[maybe_unused]] auto _ : state) {
        auto packed = pack_as_bfp4_tiles(input, /*row_major_input=*/true, /*is_exp_a=*/false);
        benchmark::DoNotOptimize(packed.data());
        benchmark::ClobberMemory();
    }
    SetThroughput<T>(state, num_tiles);
}

void BM_Bfp8b_Float(benchmark::State& s) { Bench_Bfp8b<float>(s); }
void BM_Bfp8b_Bf16(benchmark::State& s)  { Bench_Bfp8b<bfloat16>(s); }
void BM_Bfp4b_Float(benchmark::State& s) { Bench_Bfp4b<float>(s); }

}  // namespace

#define TILE_RANGES Arg(16)->Arg(256)->Arg(1024)->Arg(4096)

BENCHMARK(BM_Bfp8b_Float)->Unit(benchmark::kMillisecond)->TILE_RANGES;
BENCHMARK(BM_Bfp8b_Bf16) ->Unit(benchmark::kMillisecond)->TILE_RANGES;
BENCHMARK(BM_Bfp4b_Float)->Unit(benchmark::kMillisecond)->TILE_RANGES;

BENCHMARK_MAIN();
