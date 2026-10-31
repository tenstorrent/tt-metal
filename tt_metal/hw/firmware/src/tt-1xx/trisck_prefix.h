// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What trisck.cc includes ahead of the kernel source, in the same order: its own leading headers,
// chlkc_list.h's, the CB tables and the thread define that the per-thread source (chlkc_*.cpp)
// starts with. A kernel PCH precompiles this file (HalJitBuildQueryInterface::kernel_prefix_header)
// with chlkc_descriptors.h's declarations-only copy, so it must not include other generated
// per-kernel headers than these two.

#pragma once

#include "defines_generated.h"

#include "internal/firmware_common.h"
#include "api/compile_time_args.h"

#include "chlkc_list_prefix.h"
#include "chlkc_descriptors.h"

#if defined(UCK_CHLKC_UNPACK)
#define TRISC_UNPACK
#elif defined(UCK_CHLKC_MATH)
#define TRISC_MATH
#elif defined(UCK_CHLKC_PACK)
#define TRISC_PACK
#endif
