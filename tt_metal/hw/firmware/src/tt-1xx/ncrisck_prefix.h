// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What ncrisck.cc includes ahead of the kernel source. A kernel PCH precompiles this file
// (HalJitBuildQueryInterface::kernel_prefix_header) with chlkc_descriptors.h's declarations-only copy,
// so it must not include other generated per-kernel headers.

#pragma once

#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "internal/profiler/noc_debugging_profiler.hpp"  // RECORD_DFB_REGION_CLEAR
#include "internal/tensix_functions.h"
#include "c_tensix_core.h"
