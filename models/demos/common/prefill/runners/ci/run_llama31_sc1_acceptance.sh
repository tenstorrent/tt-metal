#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
set -euo pipefail

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must point to shared CI scratch}"
export PREFILL_MODEL=llama_3p1_8b
export PREFILL_HF_MODEL="${PREFILL_HF_MODEL:-/mnt/models/meta-llama/Llama-3.1-8B-Instruct}"
export LLAMA31_8B_CHECKPOINT="${PREFILL_HF_MODEL}"
export OMP_NUM_THREADS=16
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
mkdir -p "${PREFILL_SUMMARIES}"
# Each CI invocation makes independent HF FP32 references from two different book passages.
# A fresh directory prevents a stale trace from hiding changes to tokens or reference generation.
GOLDEN_DIR=$(mktemp -d "${PREFILL_SUMMARIES}/llama31_golden.XXXXXX")
export GOLDEN_DIR
export PREFILL_PRODUCER_SLOT_TRACES="${GOLDEN_DIR}/slot0,${GOLDEN_DIR}/slot1"

# Run one container process on the allocated SC1 host. The direct E2E fixture detaches
# its runner/producer children from this MPI session, then verifies their clean shutdown.
MPIRUN=$(command -v mpirun-ulfm || command -v mpirun)
"${MPIRUN}" --bind-to none --pernode --tag-output \
  -x PATH -x LD_LIBRARY_PATH \
  -x TT_METAL_HOME -x PREFILL_HF_MODEL -x LLAMA31_8B_CHECKPOINT -x PREFILL_MODEL \
  -x OMP_NUM_THREADS -x PYTHONPATH -x PYTHONUNBUFFERED -x GOLDEN_DIR -x PREFILL_PRODUCER_SLOT_TRACES \
  bash -lc '
  set -euo pipefail
  cd "${TT_METAL_HOME}"
  python3 -m pytest -q --tt-arch blackhole \
    models/demos/common/prefill/tests/test_prefill_producer_gqa.py \
    models/demos/common/prefill/tests/test_prefill_producer_kv_decode.py \
    models/demos/llama_3p1_8b_d_p/tests/test_prefill_runtime.py \
    models/demos/deepseek_v3_d_p/tests/test_prefill_summary_utils.py
  python3 models/demos/llama_3p1_8b_d_p/scripts/generate_prefill_trace.py \
    --checkpoint "${PREFILL_HF_MODEL}" \
    --prompt-file models/demos/llama_3p1_8b_d_p/tests/model/fixtures/book/pride_and_prejudice.txt \
    --output-dir "${GOLDEN_DIR}" --seq-len 2048 --threads 16
  python3 -m pytest -v --tt-arch blackhole --timeout=600 \
    models/demos/llama_3p1_8b_d_p/tests/test_kv_cache_table.py
  PREFILL_RUNNER_LAUNCH=ci python3 -m pytest -v --tt-arch blackhole --timeout=1200 \
    "models/demos/common/prefill/tests/test_producer_runner_e2e.py::test_producer_runner_pcc[llama31_2k_two_slots]"
'

# The standard launcher tests discovery, rank binding, merged table publication and
# all-layer K/V readback through the same entry point used by the other prefill models.
export TABLE_WAIT_SECS="${TABLE_WAIT_SECS:-900}"
exec bash "${TT_METAL_HOME}/models/demos/common/prefill/runners/ci/run_multirank_pcc.sh" llama31 sc1
