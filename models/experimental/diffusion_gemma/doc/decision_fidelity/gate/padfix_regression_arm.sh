#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# DG_DENOISE_HIDE_PREFILL_PADS regression arm, on questions that were already CLEAN.
#
# Hiding the pad keys fixed 7 of 7 block-0 collapses, but that arm could only look like an
# improvement. This asks the other question: does it HURT anywhere? It changes the mask for every
# prompt whose length is not a 32-multiple, which is most of them, so clean questions are affected
# too and a fix that repairs 7 while degrading the rest is not a fix.
#
# Measured at the mechanism rather than end-to-end, which makes it ~4x cheaper per question and
# sharper. The pads sit at fixed absolute prefix positions, so their damage is concentrated where the
# canvas is nearest to them -- block 0. Two blocks is therefore enough to see the effect, and the
# quantities to compare are exactly the ones the block-0 gate moved:
#
#   step-1 entropy   should not RISE
#   block-0 steps    should not INCREASE
#   halted           should not go true -> false
#
# A full-generation regression would cost 8 hours to answer a question that lives in the first block.
#
# SINGLE ARM: DG_DENOISE_HIDE_PREFILL_PADS is the only variable. DG_DENOISE_SLIDING_WINDOW is left at
# its (now default) ON in both arms, so this measures the pad flag alone on top of the shipped state.

set -uo pipefail

R=/home/zni/tt-metal
PY=/home/zni/venvs/tt-diffusion-gemma/bin/python
CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
PROMPTS=/tmp/gpqa198
OUT=${OUT:-/tmp/dg_padfix_regression}
NUM_BLOCKS=${NUM_BLOCKS:-2}
# The 30 clean questions with the LARGEST padding, i.e. where hiding the pads changes the mask most.
# Aligned prompts (pad 0) are unaffected by construction and would only dilute the sample.
QUESTIONS=${QUESTIONS:-"44 1 6 20 62 114 117 25 116 9 3 43 49 65 108 124 11 16 19 31 40 52 60 72 21 26 39 74 0 32"}

mkdir -p "$OUT"
echo "DG_PADFIX_REGRESSION_BEGIN $(date -u +%FT%TZ) n=$(echo $QUESTIONS | wc -w) blocks=${NUM_BLOCKS}"

for i in $QUESTIONS; do
    q=$(printf "q%03d" "$i")
    for arm in off on; do
        tag="${q}_${arm}"
        [ -f "$OUT/m_${tag}.json" ] && { echo "$tag SKIP"; continue; }
        env TT_METAL_HOME=$R PYTHONPATH=$R MESH_DEVICE=P150x4 DG_TRACE_REGION_SIZE=12884901888 \
            DG_DENOISE_HIDE_PREFILL_PADS=$([ "$arm" = on ] && echo 1 || echo 0) \
            "$PY" -u -m models.experimental.diffusion_gemma.demo.serving_smoke \
            --checkpoint "$CKPT" --mesh P150x4 \
            --max-seq-len 4096 --num-blocks "$NUM_BLOCKS" \
            --gumbel-mode device --local-files-only \
            --upfront --reveal-pmax 4096 --enable-thinking \
            --seed 0 --prompt "$(cat "$PROMPTS/${q}.txt")" \
            --metrics-json "$OUT/m_${tag}.json" > "$OUT/${tag}.log" 2>&1
        rc=$?
        guard=$(grep -c "ending request at block" "$OUT/${tag}.log" 2>/dev/null || echo 0)
        steps=$(grep -o '"steps_run": [0-9]*' "$OUT/${tag}.log" 2>/dev/null | head -1 | grep -oE '[0-9]+')
        halted=$(grep -o '"halted": [a-z]*' "$OUT/${tag}.log" 2>/dev/null | head -1 | awk '{print $2}')
        h1=$(grep -o '"halt_entropy_first": [0-9.]*' "$OUT/${tag}.log" 2>/dev/null | head -1 | awk '{print $2}')
        echo "$tag exit=$rc guard=$guard block0_steps=${steps:-?} halted=${halted:-?} step1_H=${h1:-?}"
    done
done

echo "DG_PADFIX_REGRESSION_END $(date -u +%FT%TZ)"
