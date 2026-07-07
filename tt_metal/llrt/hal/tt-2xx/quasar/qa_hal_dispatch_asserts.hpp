// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides static asserts. Consumers really do want it even if they don't use any symbol from it.
// IWYU pragma: always_keep

#include "dev_mem_map.h"
#include "noc/noc_parameters.h"

static_assert(DISPATCH_MEM_MAP_END <= MEM_L1_SIZE, "Dispatch-engine L1 layout exceeds MEM_L1_SIZE");
static_assert(
    MEM_DISPATCH_DM0_KERNEL_BASE % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0,
    "Dispatch DM0 kernel base must be NOC-write aligned");
