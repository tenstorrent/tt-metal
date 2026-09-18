#!/usr/bin/env bash
# Final regression sweep through devrun. Usage: final_regress.sh [tag]  -> logs/final_<tag>_*.log + one-line verdicts.
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
TAG="${1:-$(date +%H%M)}"; cd "$TT_METAL_HOME"; R=$MGD_1x4_RING; L=$MGD_1x4_LINE
run() { local name=$1; shift; $SPFUSE/devrun.sh "final_${TAG}_${name}" 300 1500 -- "$@" > /dev/null 2>&1; echo "$name rc=$? :: $(grep -E '[0-9]+ (passed|failed|error)|RESULT' $SPFUSE/logs/final_${TAG}_${name}.log | tail -1 | cut -c1-110)"; }
run mm_smoke  "$PY -m pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k 'test_matmul_2d_mcast_block_float_ktile_padding_subblock_h or test_linear_with_non_tile_aligned_bias or test_matmul_2d_nd_sharded_in1' -q -p no:cacheprovider"
run sched     "$PY -m pytest tests/ttnn/unit_tests/operations/ccl/test_sp_matmul_schedule.py -q -p no:cacheprovider"
run rs_ring   "TT_MESH_GRAPH_DESC_PATH=$R $PY -m pytest tests/ttnn/unit_tests/operations/ccl/test_matmul_reduce_scatter_sp_async.py -k 'ring and not perf' -q -p no:cacheprovider"
run rs_line   "TT_MESH_GRAPH_DESC_PATH=$L $PY -m pytest tests/ttnn/unit_tests/operations/ccl/test_matmul_reduce_scatter_sp_async.py -k 'line and not perf' -q -p no:cacheprovider"
run ag_ring   "TT_MESH_GRAPH_DESC_PATH=$R $PY -m pytest tests/ttnn/unit_tests/operations/ccl/test_all_gather_matmul_sp_async.py -k 'ring and not perf' -q -p no:cacheprovider"
run ag_line   "TT_MESH_GRAPH_DESC_PATH=$L $PY -m pytest tests/ttnn/unit_tests/operations/ccl/test_all_gather_matmul_sp_async.py -k 'line and not perf' -q -p no:cacheprovider"
run ccl_base_ring "TT_MESH_GRAPH_DESC_PATH=$R $PY -m pytest -p conftest -s $SPFUSE/bench_sp_collectives.py -k torus_ring -p no:cacheprovider"
run ccl_base_line "TT_MESH_GRAPH_DESC_PATH=$L $PY -m pytest -p conftest -s $SPFUSE/bench_sp_collectives.py -k mesh_linear -p no:cacheprovider"
run sp_suite  "cd tt-train && ../$PY -m pytest tests/python/test_sequence_parallel.py -q -p no:cacheprovider"
run sp_1x4    "cd tt-train && ../$PY -m pytest tests/python/test_sp_linear_ops_1x4.py -q -p no:cacheprovider"
[ -f tt-train/tests/python/test_sp_overlap.py ] && run sp_overlap "cd tt-train && ../$PY -m pytest tests/python/test_sp_overlap.py -q -p no:cacheprovider"
echo "== done $(date +%T)"
