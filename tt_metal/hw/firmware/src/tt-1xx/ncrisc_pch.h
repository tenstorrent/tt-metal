// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Umbrella header for the optional NCRISC precompiled header (TT_METAL_JIT_PCH=1).
// These are the includes ncrisck.cc parses before it reaches kernel_includes.hpp
// and that are identical for every NCRISC target, in source order.
//
// Two headers from that prelude are deliberately omitted:
//   internal/firmware_common.h
//   api/dataflow/dataflow_api.h
// See brisc_pch.h for why: dataflow_api.h gates four data-format APIs behind
// __has_include("chlkc_descriptors.h"), a per-kernel generated header that is
// absent where the shared PCH is built.
//
// Everything from kernel_includes.hpp onwards is also left out: that is the
// generated per-kernel body and the conditional includes that follow it. Note
// that unlike brisck.cc, ncrisck.cc includes stack_usage.h after the generated
// body, so it does not belong here either.
//
// The PERF_DUMP block is reproduced verbatim: the PCH is built with the same
// defines as the compile that consumes it, so it resolves the same way.
//
// Keep this in sync with the prelude in ncrisck.cc.

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
#include "tools/profiler/kernel_profiler.hpp"
#include "tools/profiler/noc_debugging_profiler.hpp"
#include "internal/tensix_functions.h"
#include "c_tensix_core.h"
