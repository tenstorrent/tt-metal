#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Find the smallest DG_TRACE_REGION_SIZE that still captures the 48 up-front denoise traces.
#
# Why: every serving script and gate reserves 12 GiB (doc/decision_fidelity/gate/*.sh,
# doc/vllm_integration/README.md), a number sized from an estimate that our own later measurement
# refuted — doc/vllm_integration/traced_serving.md:86-89 measures the 48 resident traces at
# ~1.41-1.44 GiB/chip and explicitly calls the ~168 MB/trace estimate in
# multistep_trace_batching.md "confirmed WRONG". The reservation is carved out of DRAM whether or
# not it is used (that doc's sweep shows used 13.27 + free 8.60 = 21.87 = 31.87 - 10 GiB), so the
# difference is ~10 GiB/chip of DRAM that nothing can allocate. That is the budget any weight-side
# lever (e.g. a concat-experts MoE relayout) has to come out of.
#
# Method: walk DOWN a descending ladder and stop one step above the first failure, keeping margin.
# A trace-region overflow can poison the device (see doc/optimize_perf/README.md) so each step runs
# in its own process and the script stops at the first failure rather than probing past it.
#
# Usage:
#   bisect_trace_region.sh [SIZE_BYTES...]     # default ladder 12G 8G 6G 4G 3G 2G 1.5G 1G
#
# Report: for each size, capture success + free DRAM after model build.

# Gumbel source: `device`. `--upfront` needs a MATERIALIZED full-vocabulary Gumbel and `device` is
# now the only one: the `host` mode this script used was DELETED on 2026-07-28 after being measured
# NOT to be the TT language-drift cause (it drifts on exactly the same prompts as `device`, repairs
# 0, and costs 1.40x per request; the real cause was the canvas attending prefill pad keys, fixed in
# d0936d4da4f).

set -uo pipefail

TT_METAL_ROOT="${TT_METAL_ROOT:-/home/zni/tt-metal}"
MODEL_VENV="${MODEL_VENV:-/home/zni/venvs/tt-diffusion-gemma}"
DG_CKPT="${DG_CKPT:-/home/zni/dg_models/diffusiongemma-26B-A4B-it}"
MESH="${MESH:-P150x4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
REVEAL_PMAX="${REVEAL_PMAX:-4096}"
NUM_BLOCKS="${NUM_BLOCKS:-1}"
STEPS="${STEPS:-48}"
OUT_DIR="${OUT_DIR:-/tmp/dg_trace_region}"

SIZES=("$@")
if [ "${#SIZES[@]}" -eq 0 ]; then
    SIZES=(12884901888 8589934592 6442450944 4294967296 3221225472 2147483648 1610612736 1073741824)
fi

mkdir -p "$OUT_DIR"
PY="${MODEL_VENV}/bin/python"

echo "DG_TRACE_REGION_BISECT_BEGIN sizes=${#SIZES[@]} max_seq_len=${MAX_SEQ_LEN} reveal_pmax=${REVEAL_PMAX}"
printf '%14s %8s %s\n' "size_bytes" "GiB" "result"

last_ok=""
for size in "${SIZES[@]}"; do
    gib=$(awk -v s="$size" 'BEGIN{printf "%.2f", s/1073741824}')
    tag="trace_${size}"
    log="${OUT_DIR}/${tag}.log"
    env \
        TT_METAL_HOME="${TT_METAL_ROOT}" \
        PYTHONPATH="${TT_METAL_ROOT}" \
        MESH_DEVICE="${MESH}" \
        DG_TRACE_REGION_SIZE="${size}" \
        DG_DEGENERACY_POLICY=off \
        "${PY}" -u -m models.experimental.diffusion_gemma.demo.serving_smoke \
        --checkpoint "${DG_CKPT}" \
        --mesh "${MESH}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --num-blocks "${NUM_BLOCKS}" \
        --max-denoising-steps "${STEPS}" \
        --gumbel-mode device \
        --seed 0 \
        --entropy-stop-threshold -1 \
        --disable-eos-stop \
        --local-files-only \
        --upfront \
        --reveal-pmax "${REVEAL_PMAX}" \
        --metrics-json "${OUT_DIR}/${tag}.json" \
        >"${log}" 2>&1
    rc=$?
    if grep -q "DG_VLLM_SERVING_SMOKE_SUCCESS" "${log}"; then
        # _log_mesh_dram prints the free/used DRAM per phase; keep the last line for the report.
        dram=$(grep -oE "free[^,)]*" "${log}" | tail -1)
        printf '%14s %8s %s\n' "${size}" "${gib}" "OK   ${dram}"
        last_ok="${size}"
    else
        why=$(grep -m1 -iE "trace region|out of memory|TT_FATAL|TT_THROW|Error" "${log}" | head -c 160)
        printf '%14s %8s %s\n' "${size}" "${gib}" "FAIL rc=${rc} ${why}"
        echo
        echo "First failure at ${size} (${gib} GiB). Stopping — a trace-region overflow can leave the"
        echo "device needing a reset, and probing further buys nothing."
        break
    fi
done

echo
if [ -n "${last_ok}" ]; then
    gib=$(awk -v s="${last_ok}" 'BEGIN{printf "%.2f", s/1073741824}')
    echo "DG_TRACE_REGION_BISECT_RESULT smallest_verified=${last_ok} (${gib} GiB)"
    echo "Ship one ladder step ABOVE this, not this value: the capture set grows with reveal_pmax"
    echo "and with any future lever that adds a captured op."
else
    echo "DG_TRACE_REGION_BISECT_RESULT none_succeeded"
fi
