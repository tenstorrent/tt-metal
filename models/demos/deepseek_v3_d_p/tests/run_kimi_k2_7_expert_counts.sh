#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi-K2.7 MoE expert-routing capture on a SINGLE galaxy (no mpirun, 1 iteration).
#
# Mirrors the CI "Blaze - Chunked Kimi perf (code_debug 55k, no trace)" job but:
#   * variant kimi_k2_7 instead of kimi_k2_6 (K2.7 checkpoint / cache / golden trace)
#   * num_iters=1 instead of 10 -- we need routing data, not timing statistics
#   * no mpirun -- one host owns all 32 chips
# and it records total_counts_per_expert + per-chip expert_histograms for every
# (layer, chunk, chip, expert).
#
# Usage:  ./run_kimi_k2_7_expert_counts.sh [-k <extra pytest selection>]
set -euo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}"
cd "${TT_METAL_HOME}"

# shellcheck disable=SC1091
[ -f python_env/bin/activate ] && source python_env/bin/activate

export MESH_DEVICE=TG
export LOGURU_LEVEL=INFO
export OMP_NUM_THREADS=$(nproc)

# HF checkpoint (config + safetensors index; needed by the weight_cache_path fixture)
export KIMI_K2_7_HF_MODEL=/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized
# Prebuilt .tensorbin cache -> <root>/kimi_k2_7_bh_32dev/8x4. Required; without it conftest falls
# back under model_path and would invent an HF download.
export TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill
# code_debug (56320) golden trace for K2.7; metadata.json token_ids feed the prefill.
export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320

export KIMI_EXPERT_DUMP="${KIMI_EXPERT_DUMP:-${TT_METAL_HOME}/generated/kimi_k2_7_expert_counts.jsonl}"
mkdir -p "$(dirname "${KIMI_EXPERT_DUMP}")"

SELECT="${1:-L61 and chunks_eleven and preload0 and iters1}"

python3 -m pytest \
  models/demos/deepseek_v3_d_p/tests/test_kimi_k2_7_expert_counts.py::test_kimi_k2_7_expert_token_counts \
  -k "${SELECT}" -xvs
