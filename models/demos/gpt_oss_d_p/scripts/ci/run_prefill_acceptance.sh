#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
set -euo pipefail

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must point to shared CI scratch}"
export PREFILL_MODEL=gpt_oss_d_p
export PREFILL_HF_MODEL="${PREFILL_HF_MODEL:-/mnt/models/blaze/openai/gpt-oss-120b}"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
mkdir -p "${PREFILL_SUMMARIES}"
cd "${TT_METAL_HOME}"

# Exercise the live address table on the allocated worker before model loading.
MPIRUN=$(command -v mpirun-ulfm || command -v mpirun)
"${MPIRUN}" --bind-to none --pernode --tag-output --allow-run-as-root \
  --hostfile "${TTRUN_DIR:-/etc/ttop}/hostfile" \
  -x PATH -x LD_LIBRARY_PATH -x TT_METAL_HOME -x PYTHONPATH -x PYTHONUNBUFFERED -x PREFILL_SUMMARIES \
  bash -lc '
    set -euo pipefail
    cd "${TT_METAL_HOME}"
    python3 -m pytest -v --tt-arch blackhole --timeout=1800 \
      --junitxml "${PREFILL_SUMMARIES}/gptoss120b_kv_table.xml" \
      models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py
  '

# The canonical launcher gates all-layer K/V PCC for both slots, two 1K chunks each, and clean shutdown.
exec bash "${TT_METAL_HOME}/models/demos/common/prefill/runners/ci/run_multirank_pcc.sh" gptoss120b sc1
