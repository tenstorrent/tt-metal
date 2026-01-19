// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOC Non-blocking API version selector
// Define NOC_API_V2 to use the V2 implementation, otherwise V1 is used by default

#define NOC_API_V2
#if defined(NOC_API_V2)
#include "noc_nonblocking_api_v2.h"
#else
#include "noc_nonblocking_api_v1.h"
#endif
