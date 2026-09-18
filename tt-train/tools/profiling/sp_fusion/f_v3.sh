#!/usr/bin/env bash
# Agent F, after the scratchpad wipe: remaining perf2 tables (HiFi4/fp32acc + bf16acc, bf16 + bfp8 payload, 2 links),
# the not-yet-rerun correctness runs of the current tree, and the tt-train SP suites. Usage: f_v3.sh <tag> [steps]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
TAG="${1:-v3}"; shift || true; STEPS="${*:-all}"; cd "$TT_METAL_HOME"
R=$SPFUSE/mgd/bh_galaxy_1_4_ring_ring.textproto; L=$SPFUSE/mgd/bh_galaxy_1_4_line_line.textproto
want() { [ "$STEPS" = "all" ] || [[ " $STEPS " == *" $1 "* ]]; }
run() { local name=$1; shift; want "$name" || return 0; $SPFUSE/devrun.sh "f_${TAG}_${name}" 400 2400 -- "$@" > /dev/null 2>&1; echo "$name rc=$? :: $(grep -E '[0-9]+ (passed|failed|error)' $SPFUSE/logs/f_${TAG}_${name}.log | tail -1 | cut -c1-110)"; }
PY=python_env/bin/python
AG=tests/ttnn/unit_tests/operations/ccl/test_all_gather_matmul_sp_async.py
RS=tests/ttnn/unit_tests/operations/ccl/test_matmul_reduce_scatter_sp_async.py
run ag_perf2_ring "TT_MESH_GRAPH_DESC_PATH=$R $PY -m pytest $AG -k 'ring and perf2 and links2' -q -p no:cacheprovider"
run rs_perf2_line "TT_MESH_GRAPH_DESC_PATH=$L $PY -m pytest $RS -k 'line and perf2 and links2' -q -p no:cacheprovider"
run ag_perf2_line "TT_MESH_GRAPH_DESC_PATH=$L $PY -m pytest $AG -k 'line and perf2 and links2' -q -p no:cacheprovider"
run rs_perf2_ring "TT_MESH_GRAPH_DESC_PATH=$R $PY -m pytest $RS -k 'ring and perf2 and links2' -q -p no:cacheprovider"
run ag_check_line "TT_MESH_GRAPH_DESC_PATH=$L $PY -m pytest $AG -k 'line and not perf and not decomp' -q -p no:cacheprovider"
run rs_check_ring "TT_MESH_GRAPH_DESC_PATH=$R $PY -m pytest $RS -k 'ring and not perf' -q -p no:cacheprovider"
run rs_check_line "TT_MESH_GRAPH_DESC_PATH=$L $PY -m pytest $RS -k 'line and not perf' -q -p no:cacheprovider"
run ag_check_ring "TT_MESH_GRAPH_DESC_PATH=$R $PY -m pytest $AG -k 'ring and not perf and not decomp' -q -p no:cacheprovider"
run sp_suite      "cd tt-train && ../$PY -m pytest tests/python/test_sequence_parallel.py -q -p no:cacheprovider"
run sp_1x4        "cd tt-train && ../$PY -m pytest tests/python/test_sp_linear_ops_1x4.py -q -p no:cacheprovider"
echo DONE
