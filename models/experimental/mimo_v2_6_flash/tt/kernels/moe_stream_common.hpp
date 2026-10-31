// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared constants for the MiMo MoE decode streaming kernels (tt/moe_stream.py).
#pragma once

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"

// A profiler point (TT_METAL_DEVICE_PROFILER=1; nothing otherwise): the time this RISC reached point ``id`` of its
// kernel (tests/perf/moe_zone_profile.py reads them).
#define MOE_MARK(id) DeviceTimestampedData("MOE", id)

namespace mimo_moe {

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kFaceDim = 16;
constexpr uint32_t kBf16FaceBytes = kFaceDim * kFaceDim * 2;  // 512 B: face 1 of a bf16 32x32 tile starts here

}  // namespace mimo_moe
