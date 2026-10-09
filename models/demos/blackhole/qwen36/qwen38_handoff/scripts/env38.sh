#!/bin/bash
MAIN=/home/ttuser/atupe/tt-metal; NEW=$MAIN/.claude/worktrees/qwen38-optimizations
unset TT_VISIBLE_DEVICES TT_MESH_GRAPH_DESC_PATH
export TT_METAL_HOME=$MAIN
export ARCH_NAME=blackhole
export LOGURU_LEVEL=INFO
export HF_HUB_OFFLINE=1
export PYTHONPATH="$NEW:$MAIN/ttnn:$MAIN/tools"
export HF_MODEL=/home/ttuser/atupe/models/Qwen3.8-27B
export TT_CACHE_PATH=/home/ttuser/atupe/qwen38_tt_cache
export MESH_DEVICE=P150x4
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
if [ "$(basename $HF_MODEL)" != "Qwen3.8-27B" ]; then echo "env38: bad HF_MODEL $HF_MODEL" >&2; exit 5; fi
if [ "$TT_CACHE_PATH" != "/home/ttuser/atupe/qwen38_tt_cache" ]; then echo "env38: bad TT_CACHE_PATH $TT_CACHE_PATH" >&2; exit 5; fi
