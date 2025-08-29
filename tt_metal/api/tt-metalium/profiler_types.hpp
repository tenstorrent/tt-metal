// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

enum class ProfilerReadState { NORMAL, LAST_SD_L1_READ, LAST_FD_READ };
enum class ProfilerSyncState { INIT, CLOSE_DEVICE };
enum class ProfilerDataBufferSource { L1, DRAM, DRAM_AND_L1 };

struct DeviceProgramId {
    uint32_t base_program_id = 0;
    uint32_t device_id = 0;
    bool is_host_fallback_op = false;
};

}  // namespace tt::tt_metal
