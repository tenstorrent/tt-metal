// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "risc_attribs.h"
#else
#define tt_l1_ptr
#define tt_reg_ptr
#define FORCE_INLINE inline
#endif

// TODO: move routing table here
namespace tt::tt_fabric {

constexpr uint32_t GATEKEEPER_INFO_SIZE_BYTES = 848;

}  // namespace tt::tt_fabric
