#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Gemma4 defaults for the stress launcher from PR #58456.
# Five runs, each with 600 iterations of 20 x 8192-token chunks; no hardware resets.
set -eu
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$SCRIPTS/../../../.." && pwd)}"
export MODEL=GEMMA4
export TRACE_ID="${TRACE_ID:-traced}"
export ITERS_ID="${ITERS_ID:-iters600}"
export CHUNKS_ID="${CHUNKS_ID:-chunks20}"
export RESET_BETWEEN_RUNS="${RESET_BETWEEN_RUNS:-0}"
export LOG_ROOT="${LOG_ROOT:-$TT_METAL_HOME/generated/gemma4_stress}"
export HF_MODEL="${HF_MODEL:-google/gemma-4-31B-it}"
export HF_HOME="${HF_HOME:-/mnt/models/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it}"
export PREFILL_TRACE_DIR="${PREFILL_TRACE_DIR:-/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135}"
export SESSION="${SESSION:-gemma4_stress_${HOSTNAME}}"
if [ "$#" -eq 0 ]; then set -- 5; fi
exec "$TT_METAL_HOME/models/demos/deepseek_v3_d_p/scripts/launch.sh" "$@"
