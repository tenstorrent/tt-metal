// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared BRISC prelude; keep include order in sync with brisck.cc.
// Exclude firmware_common.h and dataflow_api.h: their __has_include checks and
// constexpr tables depend on per-kernel chlkc_descriptors.h. The generated body
// (kernel_includes.hpp) and subsequent includes also stay outside the PCH.
// DPRINT can still reach descriptors through kernel_profiler.hpp; without the
// per-kernel include roots, PCH creation then fails and normal builds fall back.

#pragma once

#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tools/profiler/noc_debugging_profiler.hpp"
#include "internal/debug/stack_usage.h"
