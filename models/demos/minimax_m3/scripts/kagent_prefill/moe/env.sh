# source me: run MODIFIED tt-metal Python + JIT kernels from the kagent/m3-prefill-moe worktree against the canonical
# build (no rebuild). Device access only via tt-partition-run prefill. (Copy of dev/prefill/env.sh with KP changed.)
source /mnt/data/kernel-agent/bin/canonical-env.sh
export KP=/mnt/data/kernel-agent/dev/prefill-moe
export KP0=/mnt/data/kernel-agent/dev/prefill
export W=$KP/tt-metal
export TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W
export PYTHONPATH=$W:$W/ttnn:$W/tools
export TT_METAL_CACHE=$KP/jit-cache
export TMPDIR=$KP/tmp
export HF_MODEL=$KP0/hf/MiniMax-M3
export TT_CACHE_PATH=/mnt/data/kernel-agent/model-cache/prefill
export TT_MESH_GRAPH_DESC_PATH=$KP0/topology/prefill.textproto
export HF_HUB_OFFLINE=1 LOGURU_DIAGNOSE=NO OMP_NUM_THREADS=8
cd $W
