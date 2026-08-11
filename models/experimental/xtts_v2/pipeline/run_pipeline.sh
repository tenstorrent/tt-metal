#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# XTTS-v2 reference pipeline: the neural blocks run on Tenstorrent (TT) — conditioning
# encoder + Perceiver (Block 1), ResNet speaker encoder (Block 2), the full GPT decoder
# with parallel prefill + KV-cached decode (Block 3), and the HiFi-GAN vocoder (Block 4).
# The CPU-only front-end (tokenizer, mel/STFT, sampling glue) stays in coqui's TTS package.
#
# Produces a TT-in-loop wav AND a full-CPU baseline wav for comparison.
#
# Two Python environments are required (they have conflicting deps and must stay separate):
#   * the coqui venv  — has the `TTS` package (phases A and C); set XTTS_COQUI_PY
#   * tt-metal's venv — has `ttnn` (phase B); this script activates $TT_METAL_HOME/python_env
#
# Environment variables:
#   XTTS_COQUI_PY   (required) path to the coqui venv python, e.g. /path/xtts_cpu_venv/bin/python
#   XTTS_CKPT_DIR   (required) coqui XTTS-v2 checkpoint dir (config.json, model.pth, vocab.json)
#   TT_METAL_HOME   (optional) tt-metal root; defaults to this repo (derived from script path)
#
# Usage: run_pipeline.sh "<text>" <ref_audio.pt> [out_dir] [sr] [lang] [dtype]
#   <ref_audio.pt> is a torch-saved mono waveform tensor (or {"audio":{"array","sampling_rate"}}).
set -eo pipefail

TEXT="${1:?need text prompt}"
REF="${2:?need path to reference audio .pt}"
OUT_DIR="${3:-./xtts_pipeline_out}"
SR="${4:-22050}"
LANG="${5:-en}"
DTYPE="${6:-bf16}"

HERE="$(cd "$(dirname "$0")" && pwd)"
# repo root = pipeline/ -> xtts_v2 -> experimental -> models -> <root>
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$HERE/../../../.." && pwd)}"
COQUI_PY="${XTTS_COQUI_PY:?set XTTS_COQUI_PY to the coqui venv python}"
CKPT_DIR="${XTTS_CKPT_DIR:?set XTTS_CKPT_DIR to the XTTS-v2 checkpoint dir}"
WORK="$OUT_DIR/work"

mkdir -p "$WORK"
echo "=== Phase A: coqui (CPU) — conditioning, codes, capture emb, baseline ==="
XTTS_CKPT_DIR="$CKPT_DIR" "$COQUI_PY" "$HERE/phase_coqui_pre.py" \
  --text "$TEXT" --ref "$REF" --sr "$SR" --lang "$LANG" --work "$WORK"

echo "=== Phase B: TT — Blocks 1,2,3,4 (full GPT prefill + decode, HiFi-GAN) ==="
( cd "$TT_METAL_HOME" && source python_env/bin/activate && \
  TT_METAL_HOME="$TT_METAL_HOME" PYTHONPATH="$TT_METAL_HOME" \
  XTTS_CKPT="$CKPT_DIR/model.pth" \
  python "$HERE/phase_tt.py" --work "$WORK" )

echo "=== Phase C: coqui (CPU) — emit final wav (TT vocoder output, or coqui fallback) ==="
XTTS_CKPT_DIR="$CKPT_DIR" "$COQUI_PY" "$HERE/phase_coqui_voc.py" --work "$WORK" --out "$OUT_DIR/tt_in_loop.wav"

echo
echo "=== DONE ==="
echo "  baseline (full CPU): $WORK/baseline_cpu.wav"
echo "  TT-in-loop (GPT+vocoder on TT): $OUT_DIR/tt_in_loop.wav"
