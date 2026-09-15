#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Launch the dedicated Qwen3-ASR API server container.
# Mounts: tt-metal (TT_METAL_HOME), extracted Qwen3 checkpoint, HF cache (audio weights).
#
# Every path/device is overridable so this runs on any host (the defaults derive from the
# checkout and the invoking user, not a baked-in home dir):
#   TT_DEVICE          Tenstorrent chip index to expose            (default 0)
#   TT_METAL_SRC       tt-metal checkout mounted at /work          (default: this checkout)
#   QWEN3ASR_CKPT      extracted text-decoder checkpoint dir       (default /tmp/qwen3_asr_text_decoder)
#   HF_CACHE_DIR       HF cache with the base Qwen3-ASR snapshot   (default ~/.cache/huggingface)
#   QWEN3ASR_WARMUP_WAV_HOST  optional warmup wav (mounted at /warmup.wav)
#   QWEN3ASR_PORT      host port to publish                        (default 8002)
set -e
TT_DEVICE="${TT_DEVICE:-0}"
TT_METAL_SRC="${TT_METAL_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
QWEN3ASR_CKPT="${QWEN3ASR_CKPT:-/tmp/qwen3_asr_text_decoder}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
QWEN3ASR_PORT="${QWEN3ASR_PORT:-8002}"

WARMUP_MOUNT=()
if [ -n "${QWEN3ASR_WARMUP_WAV_HOST:-}" ] && [ -f "${QWEN3ASR_WARMUP_WAV_HOST}" ]; then
  WARMUP_MOUNT=(-v "${QWEN3ASR_WARMUP_WAV_HOST}:/warmup.wav:ro")
fi

docker rm -f qwen3-asr-api 2>/dev/null || true
docker run -d --name qwen3-asr-api \
  --device "/dev/tenstorrent/${TT_DEVICE}" \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "${TT_METAL_SRC}:/work" \
  -v "${QWEN3ASR_CKPT}:/models/qwen3_asr_text_decoder" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${WARMUP_MOUNT[@]}" \
  -p "${QWEN3ASR_PORT}:8002" --cap-add ALL \
  qwen3-asr-server:latest
# Warmup wav is mounted when QWEN3ASR_WARMUP_WAV_HOST is set: now that every _infer runs at the fixed FIXED_INFER_SEC length,
# warmup compiles the single prefill shape all requests use, so the first request / stream
# segment is warm (no cold-JIT burst). It runs lazily on the first request (request context;
# the old startup-context warmup corrupted state). The earlier warmup flakiness was the
# variable-prefill-length bug, now fixed by the fixed-length pin.
echo "qwen3-asr-api started on :${QWEN3ASR_PORT}"
