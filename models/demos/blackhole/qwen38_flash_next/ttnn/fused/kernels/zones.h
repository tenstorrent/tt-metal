// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// FUSED_ZONE(name): one device profiler zone (DeviceZoneScopedN) of a fused kernel's phase -- the reader's setup and
// main loop, the compute's stages, the writer -- when the builder passes the define QWEN38_FUSED_ZONES (the study
// build: the environment QWEN38_FUSED_ZONES=1, `program.zone_defines`), and nothing at all otherwise (the served
// build: no include, no code, the same binary as before the zone was written).  Names start with `fz_` and name the
// kernel and the phase (`fz_mp_r_setup`: moe_post, reader, setup); the census (decode_step_device_profile
// --fused-table) joins the zones to their program by runtime id and splits the program's kernel time by them.
// DeviceZoneScopedN declares `hash` and `zone` in the enclosing scope, so every FUSED_ZONE sits in its own block.
#pragma once

#ifdef QWEN38_FUSED_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define FUSED_ZONE(name) DeviceZoneScopedN(name)
#else
#define FUSED_ZONE(name)
#endif
