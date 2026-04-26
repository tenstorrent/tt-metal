// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KCT stub for ckernel_sfpu_atan2.h — same __builtin_rvtt_sfpmad issue.

#pragma once

#ifndef KCT_SFPI_RVTT_BUILTINS_DEFINED
#define KCT_SFPI_RVTT_BUILTINS_DEFINED
#define __builtin_rvtt_sfpmad(...) (::sfpi::vFloat{})
#define __builtin_rvtt_sfpload(...) (::sfpi::vFloat{})
#define __builtin_rvtt_sfpstore(...) ((void)0)
#endif

#include_next "ckernel_sfpu_atan2.h"
