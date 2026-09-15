#!/bin/bash
# Rewrite the scratch-branch toggle header. Usage: set_zone_config.sh ZONES READER_STUB MASK_OFF EXP_STUB BARRIER_THR
Z=${1:-0}; RS=${2:-0}; MO=${3:-0}; ES=${4:-0}; BT=${5:-0}
# Path resolution for both layouts (workspace or $TTM/analysis/campaigns): see campaign_paths.sh.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"   # TTM is the checkout whose toggle header is rewritten (TTM=$TTM_FRESH targets the fresh one)
F=$TTM/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp
cat > $F <<EOF
// SPDX-License-Identifier: Apache-2.0
// Scratch-branch toggle header for the SDPA zone decomposition campaign (handoff/revamp).
// Rewritten between runs by the campaign script; every kernel is JIT-recompiled per process.
#pragma once
#define SDPA_ZONES $Z            // 1: accumulate + per-q-chunk raw zones compiled in (PROFILE_KERNEL build only)
#define SDPA_ABL_READER_STUB $RS  // A4: reader reserves/pushes K and V CBs without NoC reads
#define SDPA_ABL_MASK_OFF $MO     // A2: causal lightweight mask bracket skipped on every k chunk
#define SDPA_ABL_EXP_STUB $ES     // A6: softmax exp_packthread_tile calls removed (STALLWAIT kept)
#define SDPA_ABL_BARRIER_THR $BT  // A4b: >0 overrides the reader barrier_threshold (reads in flight per barrier)
EOF
grep "#define" $F | awk '{print $2"="$3}' | tr '\n' ' '; echo
