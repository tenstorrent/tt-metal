// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {

enum class ProfilerReadState : std::uint8_t { NORMAL, ONLY_DISPATCH_CORES, LAST_FD_READ };
enum class ProfilerSyncState : std::uint8_t { INIT, CLOSE_DEVICE };
enum class ProfilerDataBufferSource : std::uint8_t { L1, DRAM, DRAM_AND_L1 };

struct DeviceProgramId {
    uint32_t base_program_id = 0;
    ChipId device_id = 0;
    bool is_host_fallback_op = false;
};
}  // namespace tt::tt_metal
