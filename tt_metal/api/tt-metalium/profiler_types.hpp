// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

enum class ProfilerReadState { NORMAL, ONLY_DISPATCH_CORES, LAST_FD_READ };
enum class ProfilerSyncState { INIT, CLOSE_DEVICE };
enum class ProfilerDataBufferSource { L1, DRAM, DRAM_AND_L1 };

}  // namespace tt::tt_metal
