// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOC Nonblocking API Version Router
// This header provides backward compatibility and version selection for the NOC nonblocking API
//
// Usage:
//   - By default (no define), V0 is included for backward compatibility
//   - To use V1: #define NOC_NONBLOCKING_API_VERSION 1 before including this header
//
// Example:
//   #define NOC_NONBLOCKING_API_VERSION 1
//   #include "noc_nonblocking_api.h"

#ifndef NOC_NONBLOCKING_API_VERSION
// Default to V0 for backward compatibility
#define NOC_NONBLOCKING_API_VERSION 1
#endif

#if NOC_NONBLOCKING_API_VERSION == 0
#include "noc_nonblocking_api_v0.h"
#elif NOC_NONBLOCKING_API_VERSION == 1
#include "noc_nonblocking_api_v1.h"
#else
#error "Unsupported NOC_NONBLOCKING_API_VERSION. Supported versions: 0, 1"
#endif
