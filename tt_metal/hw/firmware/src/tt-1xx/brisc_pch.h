// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Umbrella header for the optional BRISC precompiled header (TT_METAL_JIT_PCH=1).
// These are the includes brisck.cc parses before it reaches kernel_includes.hpp
// and that are identical for every BRISC target, in source order.
//
// Two headers from that prelude are deliberately omitted:
//   internal/firmware_common.h
//   api/dataflow/dataflow_api.h
// dataflow_api.h (reached both directly and via firmware_common.h) decides
// whether to define get_tile_size/get_tile_hw/get_tile_num_faces/get_dataformat
// with __has_include("chlkc_descriptors.h"). That descriptor header is generated
// per kernel and resolved via -I.. from the kernel's own build directory, so it
// does not exist where the shared PCH is built: the PCH would bake in "absent"
// and those four APIs would vanish for every kernel. Baking in one kernel's
// descriptors instead is worse, since the tables are constexpr and differ per
// kernel. Precompiling dataflow_api.h therefore needs it decoupled from the
// per-kernel descriptors first.
//
// Omitting those two does not keep every generated file out of reach: with
// TT_METAL_DPRINT_CORES set, kernel_profiler.hpp reaches dprint_tile.h, which
// includes chlkc_descriptors.h unconditionally. The PCH build strips the
// "-I."/"-I.." roots through which per-kernel files resolve, so in that
// configuration it fails outright -- logged by ensure_pch -- and every compile
// falls back to plain parsing rather than sharing one kernel's descriptors.
//
// Everything from kernel_includes.hpp onwards is also left out: that is the
// generated per-kernel body and the conditional includes that follow it.
//
// Keep this in sync with the prelude in brisck.cc.

#pragma once

#include <unistd.h>
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
