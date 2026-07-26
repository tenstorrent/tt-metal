#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# DG_DENOISE_SLIDING_SPAN=1 arm: the PERF half of #51080, on top of the retention mask.
#
# Once sliding layers are masked down to HF's retained window, reading the whole p_max prefix on 25
# of 30 layers is wasted SDPA key rows. The bounded read takes per-step key rows from
# 30*(p_max+C) to 25*(span+C) + 5*(p_max+C) and is bit-identical GIVEN the retention mask -- which is
# why denoise_sliding_span_enabled() refuses to engage without it, and why this arm could not be run
# before the window gate.
#
# SINGLE ARM: DG_DENOISE_SLIDING_SPAN is the only variable vs the window arm
# (/tmp/dg_gpqa_slidingwindow). Both runs have DG_DENOISE_SLIDING_WINDOW=1; everything else --
# Gumbel mode, thinking, EOS stop, degeneracy policy, seed, context, block budget -- is unchanged.
#
# Two things this measures, and the second is the one that can go wrong:
#   1. throughput: steady per-block latency and denoise steps/block vs the window arm.
#   2. bit-identity: the span is only bit-identical if mask and read stay in lockstep
#      (sliding_read_offset recomputes the same lo the reader uses). committed_sha256 in the metrics
#      JSON is the check -- it MUST match the window arm's for every question. A throughput win with
#      a changed sha is a visibility change, not the intended optimisation.
#
# Question set: the 10 window-arm questions with results plus 6 clean long-generation ones, so the
# comparison covers both regimes at a size that fits between the fidelity arms and the block-0 probe.

set -uo pipefail

R=/home/zni/tt-metal
PY=/home/zni/venvs/tt-diffusion-gemma/bin/python
CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
PROMPTS=/tmp/gpqa198
OUT=${OUT:-/tmp/dg_gpqa_slidingspan}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384}
NUM_BLOCKS=${NUM_BLOCKS:-60}

QUESTIONS=${QUESTIONS:-"17 29 104 125 14 15 22 23 30 33 62 103 34 56 68 10"}

mkdir -p "$OUT"
echo "DG_SPAN_ARM_BEGIN $(date -u +%FT%TZ) max_seq_len=${MAX_SEQ_LEN} blocks=${NUM_BLOCKS} n=$(echo $QUESTIONS | wc -w)"

for i in $QUESTIONS; do
    q=$(printf "q%03d" "$i")
    if [ -f "$OUT/m_${q}.json" ]; then
        echo "$q SKIP (already done)"
        continue
    fi
    started=$(date +%s)
    env TT_METAL_HOME=$R PYTHONPATH=$R MESH_DEVICE=P150x4 DG_TRACE_REGION_SIZE=12884901888 \
        DG_DENOISE_SLIDING_WINDOW=1 DG_DENOISE_SLIDING_SPAN=1 \
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
    span=$(grep -c "DG_DENOISE_SLIDING_SPAN=1" "$OUT/${q}.log" 2>/dev/null || echo 0)
    window=$(grep -c "DG_DENOISE_SLIDING_WINDOW=1" "$OUT/${q}.log" 2>/dev/null || echo 0)
    rows=$(grep -oE 'SDPA key rows/step [0-9]+ -> [0-9]+' "$OUT/${q}.log" 2>/dev/null | head -1)
    echo "$q exit=$rc guard=$guard window=$window span=$span ${elapsed}s  [${rows:-no key-row line}]"
done

echo "DG_SPAN_ARM_END $(date -u +%FT%TZ)"
