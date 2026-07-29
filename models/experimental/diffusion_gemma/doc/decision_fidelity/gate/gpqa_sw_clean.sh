#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# DG_DENOISE_SLIDING_WINDOW=1 REGRESSION arm: the 67 questions that were already CLEAN in the
# shipped-config baseline (/tmp/dg_gpqa198_stoparm, 131 questions, 64 collapses / 67 clean).
#
# Why this half is mandatory. The collapsed-question arm can only look like an improvement, so by
# itself it cannot distinguish a fix from a change that trades one failure mode for another. These
# questions run 1-17 blocks, so their committed prefixes cross 1023 as well and the retention mask
# changes their behaviour too. A flag that repairs 56 collapses while corrupting answers that used to
# be right is not a fix, and only this arm can show that.
#
# Scored by score_sw_regression.py on three questions of increasing strictness: does the guard now
# fire on a question that used to be clean; does the answer change and in which direction; and does
# agreement with the CUDA reference go up or down. The last is the one the flag's claim rests on --
# the argument for it is that it moves TT toward HF's geometry, so agreement must not fall.
#
# SINGLE ARM: DG_DENOISE_SLIDING_WINDOW is the only variable vs the baseline command. Gumbel mode,
# thinking, EOS stop, degeneracy policy, seed, context and block budget are unchanged.
#
# Ordered by baseline block count DESCENDING: the deepest committed prefixes are where the retention
# mask hides the most keys, so the highest-risk questions report first and a regression shows up
# early rather than after seven hours.

set -uo pipefail

R=/home/zni/tt-metal
PY=/home/zni/venvs/tt-diffusion-gemma/bin/python
CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
PROMPTS=/tmp/gpqa198
OUT=${OUT:-/tmp/dg_gpqa_sw_clean}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384}
NUM_BLOCKS=${NUM_BLOCKS:-60}

QUESTIONS=${QUESTIONS:-"62 103 34 56 68 10 121 4 55 78 93 123 67 53 77 114 2 5 6 25 58 66 84 98 107 41 50 54 70 73 83 112 113 118 0 1 3 9 32 38 108 124 20 57 87 100 117 8 11 16 19 31 40 44 116 119 130 43 49 52 60 65 72 21 26 39 74"}

mkdir -p "$OUT"
echo "DG_SW_CLEAN_BEGIN $(date -u +%FT%TZ) max_seq_len=${MAX_SEQ_LEN} blocks=${NUM_BLOCKS} n=$(echo $QUESTIONS | wc -w)"

for i in $QUESTIONS; do
    q=$(printf "q%03d" "$i")
    if [ -f "$OUT/m_${q}.json" ]; then
        echo "$q SKIP (already done)"
        continue
    fi
    started=$(date +%s)
    env TT_METAL_HOME=$R PYTHONPATH=$R MESH_DEVICE=P150x4 DG_TRACE_REGION_SIZE="${DG_TRACE_REGION_SIZE:-4294967296}" \
        DG_DENOISE_SLIDING_WINDOW=1 \
        "$PY" -u -m models.experimental.diffusion_gemma.demo.serving_smoke \
        --checkpoint "$CKPT" --mesh P150x4 \
        --max-seq-len "$MAX_SEQ_LEN" --num-blocks "$NUM_BLOCKS" \
        --gumbel-mode device --local-files-only \
        --upfront --reveal-pmax "$MAX_SEQ_LEN" --enable-thinking \
        --seed 0 --prompt "$(cat "$PROMPTS/${q}.txt")" \
        --metrics-json "$OUT/m_${q}.json" > "$OUT/${q}.log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    guard=$(grep -c "ending request at block" "$OUT/${q}.log" 2>/dev/null || echo 0)
    blocks=$(grep -c "DG_DEGENERACY " "$OUT/${q}.log" 2>/dev/null || echo 0)
    window=$(grep -c "DG_DENOISE_SLIDING_WINDOW=1" "$OUT/${q}.log" 2>/dev/null || echo 0)
    echo "$q exit=$rc guard=$guard blocks=$blocks window_active=$window ${elapsed}s  [$(date -u +%T)]"
done

echo "DG_SW_CLEAN_END $(date -u +%FT%TZ)"
