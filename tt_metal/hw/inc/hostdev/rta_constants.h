// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Sentinel used in launch_msg rta_offset/crta_offset to mark no args present.
// Shared by host and firmware.
constexpr uint16_t RTA_CRTA_NO_ARGS_SENTINEL = 0xFFFF;

// Watcher debug pattern: upper 16 bits mark uninitialized RTA slots.
// Lower 16 bits contain random value to prevent accidental matches.
constexpr uint32_t WATCHER_RTA_UNSET_PATTERN = 0xBEEF0000;
