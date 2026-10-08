# source me: run MODIFIED tt-metal Python + JIT kernels from the kagent/m3-prefill worktree against the canonical
# build (no rebuild). Device access only via tt-partition-run prefill.
#   - python: PYTHONPATH puts the worktree first, so `import ttnn` / `import models` resolve to the worktree
#     (the editable-install finder in the canonical venv is appended to sys.meta_path, i.e. after PYTHONPATH).
#   - native libs: worktree symlinks build -> canonical build_RelWithDebInfo, ttnn/ttnn/_ttnn.so -> build/lib/_ttnn.so.
#   - JIT kernels / firmware: TT_METAL_HOME = TT_METAL_RUNTIME_ROOT = worktree (both the same tree, see jit_path_fix),
#     runtime/ (sfpi) and third_party submodules are symlinks to canonical. Kernel edits in the worktree are picked
#     up at the next run (JIT cache keyed on source hash). C++ host (program factory) changes still need a rebuild.
source /mnt/data/kernel-agent/bin/canonical-env.sh
export KP=/mnt/data/kernel-agent/dev/prefill
export W=$KP/tt-metal
export TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W
export PYTHONPATH=$W:$W/ttnn:$W/tools
export TT_METAL_CACHE=$KP/jit-cache
export TMPDIR=$KP/tmp
export HF_MODEL=$KP/hf/MiniMax-M3   # symlink to /mnt/blaze-data/minimax-m3/model (ModelArgs asserts a MiniMax-M3* dir name)
export TT_CACHE_PATH=/mnt/data/kernel-agent/model-cache/prefill
export TT_MESH_GRAPH_DESC_PATH=$KP/topology/prefill.textproto
export HF_HUB_OFFLINE=1 LOGURU_DIAGNOSE=NO OMP_NUM_THREADS=8
cd $W
