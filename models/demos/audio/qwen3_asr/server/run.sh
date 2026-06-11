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
  -p 8002:8002 --cap-add ALL \
  qwen3-asr-server:latest
# NOTE: no warmup wav mounted by default — in-process warmup proved flaky (it left the
# first heavily-padded short request empty). With 512-forced prefill all shapes share
# the 512/1024 kernels, so the first real request compiles them (cold ~1.5 RTF) and
# every later request — including streaming segments — is warm. To pre-warm after deploy,
# just send one throwaway request. (Mounting a real-speech wav at /warmup.wav re-enables
# the experimental lazy warmup, but it is NOT recommended until the transient corruption
# on the first post-warmup request is root-caused.)
echo "qwen3-asr-api started on :8002"
