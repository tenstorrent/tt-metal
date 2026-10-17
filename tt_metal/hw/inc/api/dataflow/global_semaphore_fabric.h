// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// GlobalSemaphore with the off-chip (fabric) overloads enabled. Include this instead of
// api/dataflow/global_semaphore.h in kernels that signal other chips; FABRIC_2D picks mesh vs line addressing.

#if defined(TT_METAL_GLOBAL_SEMAPHORE_H_) && !defined(GLOBAL_SEMAPHORE_WITH_FABRIC)
#error "Include api/dataflow/global_semaphore_fabric.h before (or instead of) api/dataflow/global_semaphore.h"
#endif

#if defined(FABRIC_2D)
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#else
#include "tt_metal/fabric/hw/inc/linear/api.h"
#endif
#include "tt_metal/fabric/hw/inc/noc_addr.h"

#define GLOBAL_SEMAPHORE_WITH_FABRIC
#include "api/dataflow/global_semaphore.h"
