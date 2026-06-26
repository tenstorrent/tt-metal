#!/usr/bin/env bash
# Dedicated Qwen3-TTS API server entrypoint: container boots straight into the
# OpenAI-compatible /v1/audio/speech server on a Tenstorrent P300.
set -e
source /opt/venv/bin/activate
cd "${TT_METAL_HOME:-/work}"

# Wire ttnn to the mounted tt-metal source (editable; uses the prebuilt _ttnn.so, no
# recompile). Skip if the baked image already has a matching ttnn.
if ! python3 -c "import ttnn.device" >/dev/null 2>&1; then
  echo "[entrypoint] wiring ttnn to ${TT_METAL_HOME:-/work} ..."
  uv pip install --python /opt/venv/bin/python3 -e "${TT_METAL_HOME:-/work}" >/tmp/ttnn_wire.log 2>&1 \
    || { echo "[entrypoint] ttnn wire failed"; tail -8 /tmp/ttnn_wire.log; exit 1; }
fi

# Qwen3-TTS loads a single HF checkpoint (QWEN3TTS_MODEL) from the mounted HF cache —
# no decoder-extraction step needed (unlike the ASR server).
# Port: TTS_PORT (tt-home compose contract) takes priority, else QWEN3TTS_PORT, else 8003.
PORT="${TTS_PORT:-${QWEN3TTS_PORT:-8003}}"
exec python3 -m uvicorn models.demos.qwen3_tts.server.qwen3_tts_server:app \
  --host 0.0.0.0 --port "${PORT}"
