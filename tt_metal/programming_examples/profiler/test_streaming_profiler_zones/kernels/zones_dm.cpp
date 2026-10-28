// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Streaming-profiler workload for the data-movement RISCs (BRISC = RISCV_0, NCRISC = RISCV_1), tagged BR_ and NC_.
#include <cstdint>

#if defined(COMPILE_FOR_BRISC)
#define ZTAG "BR"
constexpr uint32_t kBenchSlot = 0;
#else
#define ZTAG "NC"
constexpr uint32_t kBenchSlot = 1;
#endif

#include "zones_workload.hpp"
