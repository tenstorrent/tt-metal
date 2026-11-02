// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Streaming-profiler workload for the compute RISCs (TRISC0/1/2), tagged T0_, T1_ and T2_.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"

#if COMPILE_FOR_TRISC == 0
#define ZTAG "T0"
constexpr uint32_t kBenchSlot = 2;
#elif COMPILE_FOR_TRISC == 1
#define ZTAG "T1"
constexpr uint32_t kBenchSlot = 3;
#else
#define ZTAG "T2"
constexpr uint32_t kBenchSlot = 4;
#endif

#include "zones_workload.hpp"
