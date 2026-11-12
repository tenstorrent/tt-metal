#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Moving-origin sweep of minimal_matmul: run the test on one grid window at a time, to see
# whether an ND or hang follows a core column, a row, or one core. Prints PASS / HANG / ND per case, with the failing
# devices and cores in device logical coordinates. See "ND core localization" in tests/didt/README.md.
#   usage: tests/didt/nd_origin_sweep.sh [cases...]   cases: col excl rows cols (default: col excl rows)
#   env:   BAD_X suspect column (6)  GX/GY device grid (12/10)  ITERS (200)  INTERVAL (20)  CASE_TIMEOUT s (900)
#          OUT log dir (~/nd_validate/origin_sweep)  TT_SMI tt-smi binary used for the reset after a hang (tt-smi)
#          MESH test mesh id (galaxy; 1chips for one chip, select it with TT_VISIBLE_DEVICES; sets the p150 mesh
#          graph descriptor in TT_MESH_GRAPH_DESC_PATH if not set, as one ASIC of a p300 needs one)
#          RESET reset command after a hang ("$TT_SMI -glx_reset_auto"; "$TT_SMI -r" on a p150/p300 host)
#          TT_MM_IN0_SOURCE / TT_MM_IN1_SOURCE partner (default) or dram, see the README
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" && { [ -f python_env/bin/activate ] && source python_env/bin/activate; }
ITERS=${ITERS:-200}; INTERVAL=${INTERVAL:-20}; CASE_TIMEOUT=${CASE_TIMEOUT:-900}
BAD_X=${BAD_X:-6}; GX=${GX:-12}; GY=${GY:-10}; TT_SMI=${TT_SMI:-tt-smi}
O=${OUT:-~/nd_validate/origin_sweep}; mkdir -p $O
MESH=${MESH:-galaxy}; RESET=${RESET:-"$TT_SMI -glx_reset_auto"}
NODE="tests/didt/test_minimal_matmul.py::test_minimal_matmul[blackhole-bf16_HiFi2-$MESH]"
[ "$MESH" = 1chips ] && export TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-$ROOT/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}
export TT_MM_ALLOW_THIN_GRID=1
echo "mesh $MESH, grid ${GX}x$GY, TT_VISIBLE_DEVICES ${TT_VISIBLE_DEVICES:-all}"
echo "in0 source ${TT_MM_IN0_SOURCE:-partner}, in1 source ${TT_MM_IN1_SOURCE:-partner}"

run() {  # name gridXxY originXxY
  local name=$1 grid=$2 origin=$3 log=$O/$1.log
  echo "=== [$(date +%T)] $name grid=$grid origin=$origin"
  timeout $CASE_TIMEOUT python -m pytest "$NODE" --grid-size $grid --grid-origin $origin \
    --didt-workload-iterations $ITERS --determinism-check-interval $INTERVAL --timeout=0 -q -s > $log 2>&1
  local rc=$?
  if [ $rc -eq 124 ] || grep -q "TIMEOUT\|device timeout\|Timeout" $log; then
    echo "    HANG (rc=$rc), resetting"; $RESET > $O/$name.reset.log 2>&1
  elif grep -q "non-deterministic outputs on device [0-9]* is [1-9]" $log; then
    echo "    ND: $(grep -oE 'non-deterministic outputs on device [0-9]+ is [1-9][0-9]*' $log | tr '\n' ';')"
    grep -oE "ND summary device [0-9]+ \[[^]]*\]: [0-9]+/[0-9]+ checks; pattern [^;]+; failing checks per core: .*" $log | sed 's/^/    /'
  elif [ $rc -eq 0 ]; then echo "    PASS"
  else echo "    FAIL rc=$rc (see $log)"; fi
}

for c in ${@:-col excl rows}; do
  case $c in
    col)  run col_x$BAD_X 1x$GY ${BAD_X}x0 ;;                                   # the suspect column alone
    excl) run excl_left ${BAD_X}x$GY 0x0                                         # everything left of it
          run excl_right $((GX - BAD_X - 1))x$GY $((BAD_X + 1))x0 ;;              # everything right of it
    rows) for y in $(seq 0 $((GY - 1))); do run row_y$y ${GX}x1 0x$y; done ;;      # each row alone, all x
    cols) for x in $(seq 0 $((GX - 1))); do run col_x$x 1x$GY ${x}x0; done ;;      # each column alone
    *)    echo "unknown case $c" ;;
  esac
done
