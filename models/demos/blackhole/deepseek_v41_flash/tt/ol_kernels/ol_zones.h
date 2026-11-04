// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// otherlayers: helpers of the JIT kernels.
//  * profiler zones, compiled in only with the define OL_ZONES (DSV41_OL_ZONES=1 on the host side): OLZ("name") is a
//  scoped device zone.
//  * ol_core_idx(gx): the index of this core in the row-major order of the logical grid (gx = grid width, passed as a
//  COMMON runtime argument): the kernels take their work index from it instead of from a
//    per-core runtime argument (every unique per-core argument set is a separate dispatcher write, about 0.1-0.25 us
//    each: 32 cores x 2 kernels cost ~10 us before the program can start).
#pragma once
#ifdef OL_ZONES
#include "api/debug/kernel_profiler.hpp"
#define OLZ(n) DeviceZoneScopedN(n)
#else
#define OLZ(n)
#endif

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
inline uint32_t ol_core_idx(uint32_t gx) { return get_absolute_logical_y() * gx + get_absolute_logical_x(); }
#endif
