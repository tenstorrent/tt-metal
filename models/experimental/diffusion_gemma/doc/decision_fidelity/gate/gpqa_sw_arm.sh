#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# DG_DENOISE_SLIDING_WINDOW=1 arm over the 64 questions that collapsed in the shipped-config
# baseline (/tmp/dg_gpqa198_stoparm, 131 questions, 64 collapses).
#
# Why this arm: HF's sliding layers retain only the last `sliding_window - 1` = 1023 committed
# tokens, and 25 of DiffusionGemma's 30 layers are sliding. TT's production denoise path is
# maskless all-attend, so past a 1023-token committed prefix those 25 layers attend to keys HF
# does not have -- and the dilution grows with every committed block (block 4: ~271 excess keys;
# block 13: ~2575 excess vs 1023 legitimate). 56 of the 64 collapses happen at or after the block
# where that prefix crosses 1023, clustered at blocks 12-14. The retention mask is already
# implemented (#51080, denoise_forward.denoise_sliding_window_enabled) and left default-OFF
# pending exactly this gate.
#
# SINGLE ARM: DG_DENOISE_SLIDING_WINDOW is the only variable vs the baseline command. Gumbel mode,
# thinking, EOS stop, degeneracy policy, seed, context and block budget are all unchanged.
#
# Ordered by baseline collapse block DESCENDING, so the questions where the window binds hardest
# report first and a no-op flag is visible within the first few results.
#
# Control: q007/q064/q090/q095/q096/q106/q122 collapsed at BLOCK 0, where prompt_len is 167-481
# and the window never binds. The mask there is bit-identical to today's, so these MUST still
# collapse. They are last in the order and act as the negative control.

set -uo pipefail

R=/home/zni/tt-metal
PY=/home/zni/venvs/tt-diffusion-gemma/bin/python
CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
PROMPTS=/tmp/gpqa198
OUT=${OUT:-/tmp/dg_gpqa_slidingwindow}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384}
NUM_BLOCKS=${NUM_BLOCKS:-60}

QUESTIONS="17 29 104 125 14 15 22 23 30 33 37 45 46 48 59 63 91 110 129 12 35 42 71 75 86 89 94 105 109 115 126 36 47 51 76 82 92 97 102 111 120 69 79 88 99 18 24 27 80 81 101 13 61 128 85 127 28 7 64 90 95 96 106 122"

mkdir -p "$OUT"
echo "DG_SW_ARM_BEGIN $(date -u +%FT%TZ) max_seq_len=${MAX_SEQ_LEN} blocks=${NUM_BLOCKS} n=$(echo $QUESTIONS | wc -w)"

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

echo "DG_SW_ARM_END $(date -u +%FT%TZ)"
