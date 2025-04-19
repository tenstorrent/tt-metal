// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

enum class ProfilerDumpState { NORMAL, CLOSE_DEVICE_SYNC, LAST_CLOSE_DEVICE, FORCE_UMD_READ, FORCE_PUSH_TO_TRACY };
enum class ProfilerSyncState { INIT, CLOSE_DEVICE };

}  // namespace tt::tt_metal
