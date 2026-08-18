#!/usr/bin/env bash
# Download Whisper large-v3 + Qwen3-TTS.
# Use a full `hf download`. A partial hub copy that only has *.safetensors
# can omit speech_tokenizer/config.json.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_HUB_CACHE="$CACHE"
mkdir -p "$CACHE"

echo "Downloading into $HF_HUB_CACHE"
hf download openai/whisper-large-v3
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base

QWEN_SNAP=$(find "$HF_HUB_CACHE/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
MISSING=0
echo "Verifying $QWEN_SNAP"
for f in \
    config.json \
    tokenizer_config.json \
    speech_tokenizer/config.json \
    speech_tokenizer/model.safetensors \
    model.safetensors
do
    if [ ! -e "$QWEN_SNAP/$f" ]; then
        echo "MISSING: $QWEN_SNAP/$f"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "ERROR: incomplete Qwen3-TTS cache. Re-run: hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    exit 1
fi
echo "OK"
du -sh "$HF_HUB_CACHE"/models--openai--whisper-large-v3 "$HF_HUB_CACHE"/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base
