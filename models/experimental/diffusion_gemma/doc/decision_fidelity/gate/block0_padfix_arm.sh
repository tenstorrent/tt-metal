#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# DG_DENOISE_HIDE_PREFILL_PADS=1 gate: the seven questions that collapse on BLOCK 0.
#
# Prefill right-pads the prompt to a tile multiple and writes K/V for the pad tokens, and the reveal
# predicate uses the PADDED length, so the canvas attends up to 31 garbage keys sitting immediately
# before it. Injecting that geometry into the HF reference (seeded canvas, otherwise identical) took
# it from 18 denoise steps to the 48-step CAP on q096 and from 12/10 to 35/35 on q106/q095; hiding
# the pads restored 20/12/11, i.e. baseline. This asks whether the same holds on device.
#
# SINGLE ARM: DG_DENOISE_HIDE_PREFILL_PADS is the only variable vs the shipped baseline command.
# DG_DENOISE_SLIDING_WINDOW stays OFF here: these seven have prompt_len 167-481, far below the 1023
# where the retention mask binds, so it would be bit-identical anyway -- and keeping it off means a
# result here is attributable to one flag.
#
# The prediction being tested, stated before the run: all seven collapse on block 0 today, and the
# reference says hiding the pads restores baseline convergence, so the guard should stop firing and
# each question should emit a real answer. Anything less than that (say 3 of 7) means the pads are a
# contributor rather than the whole block-0 story, which is still worth knowing precisely.
#
# 4 blocks, not 60: the failure is on block 0, so a short run answers the question cheaply, and the
# per-block halt telemetry shows whether block 0 now converges instead of running all 48 steps.

set -uo pipefail

R=/home/zni/tt-metal
PY=/home/zni/venvs/tt-diffusion-gemma/bin/python
CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
PROMPTS=/tmp/gpqa198
OUT=${OUT:-/tmp/dg_block0_padfix}
QUESTIONS=${QUESTIONS:-"106 96 122 95 7 90 64"}
NUM_BLOCKS=${NUM_BLOCKS:-4}

mkdir -p "$OUT"
echo "DG_BLOCK0_PADFIX_BEGIN $(date -u +%FT%TZ) questions=${QUESTIONS} blocks=${NUM_BLOCKS}"

for i in $QUESTIONS; do
    q=$(printf "q%03d" "$i")
    for arm in off on; do
        tag="${q}_${arm}"
        [ -f "$OUT/m_${tag}.json" ] && { echo "$tag SKIP"; continue; }
        started=$(date +%s)
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
        elapsed=$(( $(date +%s) - started ))
        guard=$(grep -c "ending request at block" "$OUT/${tag}.log" 2>/dev/null || echo 0)
        # Block 0's own trajectory is the mechanism check: does it halt now, and in how many steps?
        steps=$(grep -o '"steps_run": [0-9]*' "$OUT/${tag}.log" 2>/dev/null | head -1 | grep -oE '[0-9]+')
        halted=$(grep -o '"halted": [a-z]*' "$OUT/${tag}.log" 2>/dev/null | head -1 | awk '{print $2}')
        h1=$(grep -o '"halt_entropy_first": [0-9.]*' "$OUT/${tag}.log" 2>/dev/null | head -1 | awk '{print $2}')
        echo "$tag exit=$rc guard=$guard block0_steps=${steps:-?} halted=${halted:-?} step1_H=${h1:-?} ${elapsed}s"
    done
done

echo "DG_BLOCK0_PADFIX_END $(date -u +%FT%TZ)"
