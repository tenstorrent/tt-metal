#!/usr/bin/env bash
# Launch Qwen3.8-27B LoRA fine-tuning on GovReport (see README.md). Extra arguments go to finetune_govreport.py.
#
#   MODEL_PATH=/path/to/Qwen3.8-27B DATA_DIR=/path/to/govreport_tok OUT=/path/to/run_dir ./run_govreport.sh
#
# FORENSICS=1 also starts forensics.py (host memory, card telemetry, dmesg, tt-triage on a stall).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
: "${MODEL_PATH:?set MODEL_PATH to the Qwen3.8-27B checkpoint dir}"
: "${DATA_DIR:?set DATA_DIR to the prepare_govreport.py output dir}"
: "${OUT:?set OUT to the run output dir}"
PY=${PYTHON:-$REPO/python_env/bin/python}

export TT_METAL_HOME=$REPO TT_METAL_RUNTIME_ROOT=$REPO ARCH_NAME=blackhole PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO/build_Release/tt-train/sources/ttml:$REPO/tt-train/sources/ttml:$REPO
export TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-$REPO/tt-train/configs/mgd/bh_qb_1_4_ring_ring.textproto}

mkdir -p "$OUT"
: > "$OUT/train.log"
if [[ "${FORENSICS:-0}" == 1 ]]; then
    "$PY" "$HERE/forensics.py" "$OUT/train.log" "$OUT/forensics" &
fi
cd "$REPO/tt-train"
"$PY" "$HERE/finetune_govreport.py" --model-path "$MODEL_PATH" --data "$DATA_DIR" --out "$OUT" "$@" 2>&1 | tee -a "$OUT/train.log"
echo "=== training exited with status ${PIPESTATUS[0]} at $(date) ===" | tee -a "$OUT/train.log"
wait
