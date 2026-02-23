// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TRISC is a compute processor, not a data movement processor.
// The NOC non-blocking API is not available for TRISC.
#if !defined(COMPILE_FOR_TRISC)

// NOC Non-blocking API version selector
// Define NOC_API_V2 to use the V2 implementation, otherwise V1 is used by default

#define NOC_API_V2
#if defined(NOC_API_V2)
#include "noc_nonblocking_api_v2.h"
#else
#include "noc_nonblocking_api_v1.h"
#endif

#endif  // !defined(COMPILE_FOR_TRISC)
