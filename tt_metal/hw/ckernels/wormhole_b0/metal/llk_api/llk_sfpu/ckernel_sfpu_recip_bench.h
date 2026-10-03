#ifndef CKERNEL_SFPU_RECIP_BENCH_H
#define CKERNEL_SFPU_RECIP_BENCH_H

#include "ckernel_sfpu_recip.h"
#include <benchmark/benchmark.h>

namespace ckernel {

namespace sfpu {

static void BM_SfpuReciprocalFP32(benchmark::State& state) {
    for (auto _ : state) {
        float x = static_cast<float>(state.range(0));
        benchmark::DoNotOptimize(reciprocal(x));
    }
}

static void BM_SfpuReciprocalBF16(benchmark::State& state) {
    for (auto _ : state) {
        bfloat16 x = bfloat16(static_cast<float>(state.range(0)));
        benchmark::DoNotOptimize(reciprocal(x));
    }
}

BENCHMARK(BM_SfpuReciprocalFP32)->Range(1, 1000);
BENCHMARK(BM_SfpuReciprocalBF16)->Range(1, 1000);

} // namespace sfpu

} // namespace ckernel

#endif // CKERNEL_SFPU_RECIP_BENCH_H
