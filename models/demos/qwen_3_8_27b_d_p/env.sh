# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Source this before any qwen_3_8_27b_d_p run:  source models/demos/qwen_3_8_27b_d_p/env.sh
# Everything here is machine/worktree state, deliberately NOT baked into the package.

# JIT kernel builds fan out one g++ per RISC-V core per kernel, and LTO forks lto1/cc1plus under
# each. The default 512-process soft RLIMIT_NPROC runs out mid-build and surfaces as
# "posix_spawnp: Operation not permitted" / "Cannot fork" from the kernel compiler, which reads
# like a toolchain fault rather than a limit. Raised to a bounded 16384 rather than the hard limit
# (~2.3M): enough headroom for any JIT fan-out, while still leaving a guard in place.
if [ "$(ulimit -u)" -lt 16384 ] 2>/dev/null; then
    ulimit -u 16384 2>/dev/null || true
fi

export TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$TT_METAL_HOME/python_env/bin/activate"

# Weights: the shared HF hub cache already carries Qwen/Qwen3.8-27B (read-only, 52 GB).
export HF_HOME="${HF_HOME:-/mnt/models/huggingface}"
export QWEN35_HF_MODEL="${QWEN35_HF_MODEL:-/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
export HF_MODEL="${HF_MODEL:-$QWEN35_HF_MODEL}"

# Golden traces, per-module goldens and the tilized TTNN weight cache. NOT on /mnt/models: the
# share is 99% full, and the tilized cache is ~27 GB per mesh shape. Local disk has 2.9 TB.
export QWEN35_SCRATCH="${QWEN35_SCRATCH:-$HOME/qwen_3_8_27b_bringup}"
export QWEN35_GOLDEN_ROOT="${QWEN35_GOLDEN_ROOT:-$QWEN35_SCRATCH/golden}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-$QWEN35_SCRATCH/tensor_cache}"
mkdir -p "$QWEN35_GOLDEN_ROOT" "$TT_CACHE_PATH"

# Fabric: plain mesh descriptor maps on any galaxy, torus-wired or not (recipe section 4).
# QWEN35_TORUS=1 selects the torus descriptor where the pod offers it.
if [ "${QWEN35_TORUS:-0}" = "1" ]; then
    export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
else
    export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"
fi
