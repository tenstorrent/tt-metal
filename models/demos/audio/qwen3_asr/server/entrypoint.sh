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
exec python3 -m uvicorn models.demos.audio.qwen3_asr.server.qwen3_asr_server:app \
  --host 0.0.0.0 --port "${QWEN3ASR_PORT:-8002}"
