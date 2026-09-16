#!/bin/bash
# Rewrite the TopK toggle header. Usage: set_topk_zone_config.sh ZONES   (0 off, 1 coarse, 2 with the fine zones)
Z=${1:-0}
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"
F=$TTM/ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/topk_zone_config.hpp
cat > $F <<EOF
// SPDX-License-Identifier: Apache-2.0
// Scratch-branch toggle header for the TopK zone decomposition (handoff/revamp campaign L7), the same
// shape as sdpa_zone_config.hpp: the campaign script rewrites the line between runs and every kernel is
// JIT-recompiled per process. Zones only compile in on a PROFILE_KERNEL build.
#pragma once
#define TOPK_ZONES $Z  // 0 off; 1 coarse (per row, per launch); 2 coarse plus the per-step and per-tile zones

#if TOPK_ZONES && defined(PROFILE_KERNEL)
#include "tools/profiler/kernel_profiler.hpp"
#define TOPK_ZONE(name) DeviceZoneScopedN(name)
#else
#define TOPK_ZONE(name)
#endif
#if TOPK_ZONES > 1 && defined(PROFILE_KERNEL)
// The per-step and per-tile zones fill the profiler buffer on a wide row (127 steps), which drops the
// zones that close after them, so they are a separate level.
#define TOPK_ZONE_FINE(name) DeviceZoneScopedN(name)
#else
#define TOPK_ZONE_FINE(name)
#endif
EOF
grep "#define TOPK_ZONES" $F
