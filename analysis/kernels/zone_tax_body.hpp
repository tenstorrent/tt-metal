// SPDX-License-Identifier: Apache-2.0
// Zone-tax micro-benchmark body (T0.2). Included by zone_tax_dm.cpp (BRISC/NCRISC) and
// zone_tax_compute.cpp (TRISC0/1/2). Defines expected: ZT_MODE, ZT_N, ZT_LOOP.
//   ZT_MODE 0: baseline, loop body only
//   ZT_MODE 1: one DeviceZoneScopedN (raw start/end markers) per iteration
//   ZT_MODE 2: one DeviceZoneScopedSumN1 (native accumulate, needs --enable-sum-profiling) per iteration
//   ZT_MODE 3: one SDPA_ZACC custom accumulate scope per iteration (sdpa_zones.hpp), flushed at end
#pragma once
#include "tools/profiler/kernel_profiler.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zones.hpp"

inline __attribute__((always_inline)) void zt_body() {
#pragma GCC unroll 65534
    for (int j = 0; j < ZT_LOOP; j++) {
        asm volatile("nop");
    }
}

inline void zone_tax_main() {
    for (uint32_t i = 0; i < ZT_N; i++) {
#if ZT_MODE == 1
        DeviceZoneScopedN("ZT_RAW");
#elif ZT_MODE == 2
        DeviceZoneScopedSumN1("ZT_SUM");
#elif ZT_MODE == 3
        SDPA_ZACC(0);
#endif
        zt_body();
    }
#if ZT_MODE == 3
    SDPA_ZFLUSH(0, "ZT_ACC");
#endif
}
