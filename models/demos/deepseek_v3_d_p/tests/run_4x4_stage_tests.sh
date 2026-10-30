#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the 4x4-stage test rows on one BH Galaxy tray half: a single [4,4] mesh with SP on the axis-1
# ring and TP on the axis-0 line, the layout of each 4x4 prefill stage.
#
# The rows (ids containing "4x4") need the Galaxy narrowed to one tray half: TT_VISIBLE_DEVICES set to
# that half's 16 chips and TT_MESH_GRAPH_DESC_PATH naming the [LINE, RING] 4x4 descriptor, which is what
# this script sets up. Unnarrowed, the system mesh is 8x4 and those rows skip. The half -> device map is per-galaxy, so it comes from
# gen_pipeline_binding.py --layout 4x4 (generated once per host, galaxy must be idle).
#
# Usage, from TT_METAL_HOME:
#   models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh [--half top|bottom] [-- <extra pytest args>]
#   models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh -- path/to/test_x.py -k "4x4 and foo"
#
# With no test paths after --, runs the MoE-stack suite below (plus any options given).
set -euo pipefail

HALF=top
while [[ $# -gt 0 ]]; do
  case "$1" in
    --half) HALF="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$HALF" in
  top) RANK=0 ;;
  bottom) RANK=1 ;;
  *) echo "--half must be top or bottom" >&2; exit 2 ;;
esac

TOPO=models/demos/common/prefill/runners/topology_configuration
BINDING="$TOPO/pipeline_prefill_request_intragalaxy_2rank_4x4_sp1.$(hostname).yaml"
if [[ ! -f "$BINDING" ]]; then
  python3 models/demos/deepseek_v3_d_p/utils/gen_pipeline_binding.py --layout 4x4
fi

VISIBLE=$(sed -nE "s/.*\{rank: $RANK, .*TT_VISIBLE_DEVICES: \"([0-9,]+)\".*/\1/p" "$BINDING")
if [[ -z "$VISIBLE" ]]; then
  echo "no TT_VISIBLE_DEVICES for rank $RANK in $BINDING; regenerate it" >&2
  exit 1
fi

export TT_VISIBLE_DEVICES="$VISIBLE"
export TT_MESH_GRAPH_DESC_PATH="$TOPO/single_4x4_sp1_mgd.textproto"
echo "4x4 stage, $HALF half: TT_VISIBLE_DEVICES=$TT_VISIBLE_DEVICES"

T=models/demos/deepseek_v3_d_p/tests
# Test paths given after -- replace the default suite; bare options (-rA, -x, ...) add to it. pytest keeps
# only the last -k, so a user's -k is pulled out and put back last: as given with test paths, and merged
# with the 4x4 filter for the default suite.
has_paths=false
user_k=""
args=()
while (($#)); do
  case "$1" in
    -k) user_k=$2 && shift ;;
    -k*) user_k=${1#-k} ;;
    *.py | *.py::*) has_paths=true && args+=("$1") ;;
    *) args+=("$1") ;;
  esac
  shift
done
set -- ${args[@]+"${args[@]}"}
if $has_paths; then
  if [[ -n "$user_k" ]]; then set -- "$@" -k "$user_k"; fi
else
  set -- "$@" \
    "$T/op_unit_tests/test_ttnn_dispatch_combine.py::test_ttnn_dispatch_combine" \
    "$T/op_unit_tests/test_combine_fabric2d.py::test_combine_fabric2d" \
    "$T/pcc/test_rmsnorm.py" \
    "$T/pcc/test_shared_expert.py::test_shared_expert_pcc" \
    "$T/pcc/test_ffn.py::test_ffn_pcc" \
    "$T/pcc/test_moe_gate_prefill2d.py::test_forward_pass" \
    "$T/pcc/test_ttnn_moe.py::test_kimi_k3_moe" \
    -k "4x4${user_k:+ and ($user_k)}"
fi

exec pytest "$@"
