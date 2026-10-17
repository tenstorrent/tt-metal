// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "hostdev/remote_dfb_constants.h"  // REMOTE_DFB_OFFSET_NONE

// Launch-msg GlobalSemaphore address table (host and device).
//
// kernel_config_msg_t::global_sem_offset points from kernel_config_base to:
//   word[0]         num_slots
//   word[1 + slot]  L1 address of the GlobalSemaphore bound to that slot
// REMOTE_DFB_OFFSET_NONE means the launch has no GlobalSemaphore bindings.

inline constexpr uint32_t GLOBAL_SEM_REGION_HEADER_WORDS = 1;

constexpr uint32_t global_sem_region_words(uint32_t num_slots) { return GLOBAL_SEM_REGION_HEADER_WORDS + num_slots; }
