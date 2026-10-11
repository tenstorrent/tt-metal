#!/usr/bin/env bash
# Persistent, reproducible environment for stage 6.
export TT_MODEL_BRINGUP_ROOT=/home/vkovacevic/muse-glimmer/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19
export TT_AUTODEBUG_ROOT=/home/vkovacevic/muse-glimmer/codex_home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
bringup_exports=$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py") || return
eval "$bringup_exports"
export HF_HOME=/mnt/models/huggingface
export TT_METAL_HOME=/home/vkovacevic/muse-glimmer/tt-metal
export PYTHONPATH="$TT_METAL_HOME:$PYTHONPATH"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
