#!/usr/bin/env bash
# Dedicated Qwen3-ASR API server entrypoint: container boots straight into the
# OpenAI-compatible /v1/audio/transcriptions server on a Tenstorrent P150.
set -e
source /opt/venv/bin/activate
cd "${TT_METAL_HOME:-/work}"
# Wire ttnn to the mounted tt-metal source (editable; ~9s, uses the prebuilt _ttnn.so,
# no recompile). The baked dev-image ttnn doesn't match the mounted v0.71 source.
if ! python3 -c "import ttnn.device" >/dev/null 2>&1; then
  echo "[entrypoint] wiring ttnn to ${TT_METAL_HOME:-/work} ..."
  uv pip install --python /opt/venv/bin/python3 -e "${TT_METAL_HOME:-/work}" >/tmp/ttnn_wire.log 2>&1 \
    || { echo "[entrypoint] ttnn wire failed"; tail -8 /tmp/ttnn_wire.log; exit 1; }
fi

# First-run bootstrap: if the extracted Qwen3-ASR text-decoder checkpoint is missing,
# build it from the base Qwen/Qwen3-ASR-1.7B snapshot (HF cache) via extract_text_decoder.py.
# Needs the base model in the HF cache (already required for the audio tower + processor).
CKPT="${HF_MODEL:-/models/qwen3_asr_text_decoder}"
SNAP_BASE_C="${QWEN3ASR_SNAP_BASE:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots}"
if [ ! -f "$CKPT/model.safetensors" ] || [ ! -f "$CKPT/config.json" ]; then
  echo "[entrypoint] decoder checkpoint missing at $CKPT — extracting from base Qwen3-ASR-1.7B ..."
  if [ ! -d "$SNAP_BASE_C" ] || [ -z "$(ls -A "$SNAP_BASE_C" 2>/dev/null)" ]; then
    echo "[entrypoint] ERROR: base model not found at $SNAP_BASE_C"
    echo "[entrypoint]   mount the HF cache containing models--Qwen--Qwen3-ASR-1.7B (or set QWEN3ASR_SNAP_BASE)."
    exit 1
  fi
  mkdir -p "$CKPT"
  QWEN3ASR_SNAP_BASE="$SNAP_BASE_C" python3 \
    "${TT_METAL_HOME:-/work}/models/demos/audio/qwen3_asr/reference/extract_text_decoder.py" \
    --ckpt-out "$CKPT" --skip-golden \
    || { echo "[entrypoint] decoder extraction failed"; exit 1; }
  echo "[entrypoint] decoder checkpoint ready at $CKPT"
fi

exec python3 -m uvicorn models.demos.audio.qwen3_asr.server.qwen3_asr_server:app \
  --host 0.0.0.0 --port "${QWEN3ASR_PORT:-8002}"
