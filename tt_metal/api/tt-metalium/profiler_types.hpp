// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

enum class ProfilerDumpState { NORMAL, FORCE_UMD_READ, ONLY_DISPATCH_CORES, LAST_FD_DUMP };
enum class ProfilerSyncState { INIT, CLOSE_DEVICE };
enum class ProfilerDataBufferSource { L1, DRAM };

}  // namespace tt::tt_metal
