#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The model-shape comparison on a BH Galaxy (4x8): every path on the default grid, ours on the folded grids, and ours
# with the bias / addcmul as separate ops, one JSON-lines file each in OUT_DIR (default generated/mrs_cmp_galaxy).
#
#   tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/model_compare/run_galaxy.sh [OUT_DIR] [-m "not large"]
#
# Run from the tt-metal root with the python env active. Extra arguments go to pytest (e.g. -m "not large" to skip the
# largest shapes, -k h3_ff2 for one model). A failed case does not stop the run.
set -u
OUT=generated/mrs_cmp_galaxy
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    OUT=$1
    shift
fi
DIR=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/model_compare
T=$DIR/test_matmul_reduce_scatter_model_compare_bh.py
mkdir -p "$OUT"
export MRS_CMP_MESH=${MRS_CMP_MESH:-4x8}  # 1x4 runs the same set on a QuietBox 2
run() {  # name, env..., then pytest args
    local name=$1
    shift
    env "$@" MRS_CMP_OUT="$OUT/$name.jsonl" scripts/run_safe_pytest.sh --run-all "$T" "${PYTEST_ARGS[@]}" \
        > "$OUT/$name.log" 2>&1
    echo "$name: exit $? $(grep -E 'passed|failed' "$OUT/$name.log" | tail -1)"
}
PYTEST_ARGS=("$@")
git rev-parse HEAD > "$OUT/commit.txt"
run compare MRS_CMP_GRID=grid_2d
run ours_folded_n MRS_CMP_GRID=folded_n MRS_CMP_PATHS=ours
run ours_folded_m MRS_CMP_GRID=folded_m MRS_CMP_PATHS=ours
run ours_separate_epilogue MRS_CMP_SEPARATE_EPILOGUE=1 MRS_CMP_PATHS=ours
python3 "$DIR/summarize.py" "$OUT/compare.jsonl" > "$OUT/summary.txt" 2>&1
echo "results in $OUT (send the whole directory)"
