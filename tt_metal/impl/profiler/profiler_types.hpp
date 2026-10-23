// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/device_types.hpp>
// Re-exports public profiler enums and adds internal-only types.
// Prefer <tt-metalium/profiler_types.hpp> for the public subset alone.
#include <tt-metalium/profiler_types.hpp>

namespace tt::tt_metal {

enum class ProfilerDataBufferSource { L1, DRAM, DRAM_AND_L1 };

struct DeviceProgramId {
    uint32_t base_program_id = 0;
    ChipId device_id = 0;
    bool is_host_fallback_op = false;
};

}  // namespace tt::tt_metal
