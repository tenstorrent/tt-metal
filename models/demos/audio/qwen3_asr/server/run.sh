#!/usr/bin/env bash
# Launch the dedicated Qwen3-ASR API server container (chip 3 = fake P150).
# Mounts: tt-metal (TT_METAL_HOME), extracted Qwen3 checkpoint, HF cache (audio weights).
set -e
docker rm -f qwen3-asr-api 2>/dev/null || true
docker run -d --name qwen3-asr-api \
  --device /dev/tenstorrent/3 \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /home/ttuser/ttwork/tt-metal:/work \
  -v /home/ttuser/ttwork/qwen3_asr_text_decoder:/models/qwen3_asr_text_decoder \
  -v /home/ttuser/.cache/huggingface:/root/.cache/huggingface \
  -v /home/ttuser/ttwork/qwen3_asr_warmup.wav:/warmup.wav:ro \
  -p 8002:8002 --cap-add ALL \
  qwen3-asr-server:latest
# Warmup wav IS mounted: now that every _infer runs at the fixed FIXED_INFER_SEC length,
# warmup compiles the single prefill shape all requests use, so the first request / stream
# segment is warm (no cold-JIT burst). It runs lazily on the first request (request context;
# the old startup-context warmup corrupted state). The earlier warmup flakiness was the
# variable-prefill-length bug, now fixed by the fixed-length pin.
echo "qwen3-asr-api started on :8002"
