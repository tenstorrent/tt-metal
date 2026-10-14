#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run one Wan2.2 T2V pipeline test on bh-glx-120-b10u08 with the settings used in the handoff.
#
#   handoff/wan-bin6/run_wan.sh <480p|720p> <chunk|nochunk> [log_file]
#
# Always uses the 4x8 linear, is_fsdp=False config (4x8sp1tp0nl2_linear_is_fsdp0): the ring config from
# issue #59504 cannot run on this machine (no X-axis wraparound cabling, see README.md).
set -euo pipefail

res="${1:?usage: run_wan.sh <480p|720p> <chunk|nochunk> [log_file]}"
mode="${2:?usage: run_wan.sh <480p|720p> <chunk|nochunk> [log_file]}"
log="${3:-wan_${res}_${mode}_$(date +%H%M%S).log}"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source python_env/bin/activate

export TT_METAL_HOME="$repo"
export PYTHONPATH="$repo:$repo/handoff/wan-bin6"
export NO_PROMPT=1
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/tmp/TT_DIT_CACHE_wan_bin6}"
export VBENCH_CACHE_DIR=/mnt/models/vbench
export TORCH_HOME=/mnt/models/vbench/torch_hub
# Required: without an explicit descriptor, fabric auto-discovery fails on this board (README.md, finding 1).
export TT_MESH_GRAPH_DESC_PATH="$repo/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

plugin=()
[[ "$mode" == "chunk" ]] && plugin=(-p vae_chunk_patch)

test_id="models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py::test_pipeline_inference[blackhole-resolution_${res}-4x8sp1tp0nl2_linear_is_fsdp0-True]"

echo "log: $log"
timeout 2400 pytest "${plugin[@]}" --timeout 2100 "$test_id" -x > "$log" 2>&1
