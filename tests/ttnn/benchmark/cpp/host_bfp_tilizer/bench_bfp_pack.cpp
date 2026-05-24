// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pure-host microbenchmark for the optimized BFP host tilizer.
//
// Runs `pack_as_bfp8_tiles` (and BFP4_b for comparison) under three
// configurations toggled via the same env vars exposed in production:
//   * scalar, single-threaded   (TT_BFP_HOST_TILIZER_DISABLE_SIMD=1,
//                                TT_BFP_HOST_TILIZER_THREADS=1)
//   * SIMD,   single-threaded   (TT_BFP_HOST_TILIZER_THREADS=1)
//   * SIMD,   multi-threaded    (TT_BFP_HOST_TILIZER_THREADS unset -> hw default)
//
// No device is required; this exercises only host code.

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <tt_stl/indestructible.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>

namespace {

constexpr uint32_t FLOATS_PER_TILE = 1024;  // 32x32 tile

// ---------- Env-var scoping ----------------------------------------------
// Set/restore an env var for the lifetime of a benchmark invocation. We do
// this in `state.PauseTiming()` blocks so the setenv cost is not counted.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* prev = std::getenv(name);
        had_prev_ = (prev != nullptr);
        if (had_prev_) {
            prev_value_ = prev;
        }
        if (value != nullptr) {
            ::setenv(name, value, /*overwrite=*/1);
        } else {
            ::unsetenv(name);
        }
    }
    ~ScopedEnv() {
        if (had_prev_) {
            ::setenv(name_, prev_value_.c_str(), /*overwrite=*/1);
        } else {
            ::unsetenv(name_);
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

// ---------- Cached deterministic input data ------------------------------
// Build the largest input we'll need once, reuse for every benchmark size.
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

// ---------- Throughput accounting ----------------------------------------
// Report bytes processed per iteration (input bytes) so the framework prints
// a useful "bytes/s" column.
template <typename T>
void SetThroughput(benchmark::State& state, uint32_t num_tiles) {
    const std::int64_t bytes_per_iter = static_cast<std::int64_t>(num_tiles) * FLOATS_PER_TILE * sizeof(T);
    state.SetBytesProcessed(state.iterations() * bytes_per_iter);
    state.counters["tiles"] = benchmark::Counter(
        static_cast<double>(num_tiles) * state.iterations(), benchmark::Counter::kIsRate);
}

// ---------- Benchmark bodies ---------------------------------------------
// One body per (format, input dtype, configuration). Each body sets the env
// vars under PauseTiming so environment manipulation is excluded from the
// measured loop, then calls the public packer in the timed region.

template <typename T>
const std::vector<T>& GetInput();
template <>
const std::vector<float>& GetInput<float>() { return GetFloatInput(); }
template <>
const std::vector<bfloat16>& GetInput<bfloat16>() { return GetBfloat16Input(); }

template <typename T>
void Bench_Bfp8b(benchmark::State& state, const char* threads, const char* disable_simd) {
    const auto num_tiles = static_cast<uint32_t>(state.range(0));
    const auto& src = GetInput<T>();
    const auto input = tt::stl::make_const_span(src.data(), num_tiles * FLOATS_PER_TILE);

    state.PauseTiming();
    ScopedEnv t("TT_BFP_HOST_TILIZER_THREADS", threads);
    ScopedEnv s("TT_BFP_HOST_TILIZER_DISABLE_SIMD", disable_simd);
    state.ResumeTiming();

    for ([[maybe_unused]] auto _ : state) {
        auto packed = pack_as_bfp8_tiles(input, /*row_major_input=*/true, /*is_exp_a=*/false);
        benchmark::DoNotOptimize(packed.data());
        benchmark::ClobberMemory();
    }
    SetThroughput<T>(state, num_tiles);
}

template <typename T>
void Bench_Bfp4b(benchmark::State& state, const char* threads, const char* disable_simd) {
    const auto num_tiles = static_cast<uint32_t>(state.range(0));
    const auto& src = GetInput<T>();
    const auto input = tt::stl::make_const_span(src.data(), num_tiles * FLOATS_PER_TILE);

    state.PauseTiming();
    ScopedEnv t("TT_BFP_HOST_TILIZER_THREADS", threads);
    ScopedEnv s("TT_BFP_HOST_TILIZER_DISABLE_SIMD", disable_simd);
    state.ResumeTiming();

    for ([[maybe_unused]] auto _ : state) {
        auto packed = pack_as_bfp4_tiles(input, /*row_major_input=*/true, /*is_exp_a=*/false);
        benchmark::DoNotOptimize(packed.data());
        benchmark::ClobberMemory();
    }
    SetThroughput<T>(state, num_tiles);
}

// ---------- Configuration matrix -----------------------------------------
// Three configs:    label              threads    disable_simd
//   scalar-1T       "1"                "1"
//   simd-1T         "1"                nullptr
//   simd-NT         nullptr (= hw)     nullptr

void BM_Bfp8b_Float_Scalar1T(benchmark::State& s)  { Bench_Bfp8b<float>   (s, "1",     "1");      }
void BM_Bfp8b_Float_Simd1T  (benchmark::State& s)  { Bench_Bfp8b<float>   (s, "1",     nullptr);  }
void BM_Bfp8b_Float_SimdNT  (benchmark::State& s)  { Bench_Bfp8b<float>   (s, nullptr, nullptr);  }

void BM_Bfp8b_Bf16_Scalar1T (benchmark::State& s)  { Bench_Bfp8b<bfloat16>(s, "1",     "1");      }
void BM_Bfp8b_Bf16_Simd1T   (benchmark::State& s)  { Bench_Bfp8b<bfloat16>(s, "1",     nullptr);  }
void BM_Bfp8b_Bf16_SimdNT   (benchmark::State& s)  { Bench_Bfp8b<bfloat16>(s, nullptr, nullptr);  }

// BFP4_b has no SIMD path - we only measure threading.
void BM_Bfp4b_Float_Serial  (benchmark::State& s)  { Bench_Bfp4b<float>   (s, "1",     nullptr);  }
void BM_Bfp4b_Float_Parallel(benchmark::State& s)  { Bench_Bfp4b<float>   (s, nullptr, nullptr);  }

}  // namespace

// Tile counts: 16 (small), 256 (medium), 1024 (large), 4096 (very large).
// Each is registered once per configuration so the table groups nicely.
#define TILE_RANGES Arg(16)->Arg(256)->Arg(1024)->Arg(4096)

BENCHMARK(BM_Bfp8b_Float_Scalar1T)->Unit(benchmark::kMillisecond)->TILE_RANGES;
BENCHMARK(BM_Bfp8b_Float_Simd1T)  ->Unit(benchmark::kMillisecond)->TILE_RANGES;
BENCHMARK(BM_Bfp8b_Float_SimdNT)  ->Unit(benchmark::kMillisecond)->TILE_RANGES;

BENCHMARK(BM_Bfp8b_Bf16_Scalar1T) ->Unit(benchmark::kMillisecond)->TILE_RANGES;
BENCHMARK(BM_Bfp8b_Bf16_Simd1T)   ->Unit(benchmark::kMillisecond)->TILE_RANGES;
BENCHMARK(BM_Bfp8b_Bf16_SimdNT)   ->Unit(benchmark::kMillisecond)->TILE_RANGES;

BENCHMARK(BM_Bfp4b_Float_Serial)  ->Unit(benchmark::kMillisecond)->TILE_RANGES;
BENCHMARK(BM_Bfp4b_Float_Parallel)->Unit(benchmark::kMillisecond)->TILE_RANGES;

BENCHMARK_MAIN();
