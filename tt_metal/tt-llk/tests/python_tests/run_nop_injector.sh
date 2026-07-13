#!/usr/bin/env bash
#
# OpenMP NOP injector over a real pytest file (space-efficient).
# Per case: compile+baseline → ttnop batch 1..100 → re-run → keep fails only →
# wipe that case's artefacts → next case.
#
# Usage:   ./run_nop_injector.sh [pytest_file]
#          ./run_nop_injector.sh test_eltwise_unary_datacopy.py
#
# Env:     NOP_THREAD   thread to perturb (unpack|math|pack)   (default math)
#          NOP_COUNTS   count sweep, e.g. "1-100" or "1,2,4"   (default 1-100)
#          OMP_NOP_OUT  output root (default /tmp/tt-llk-build/nop_injector)
#          OMP_NOP_KEEP=1  keep all ELFs under work/ (no delete)
#          JOBS         xdist workers / Tensix cores           (default 8)
#          TTNOP        path to ttnop binary
#
# /tmp layout (single root for this flow):
#   /tmp/tt-llk-build/              LLK harness artefacts (ELFs)
#   /tmp/tt-llk-build/nop_injector/ this tool (work / fails / summary.log)
#   /tmp/tt-llk-build/pytest/       pytest basetemp (via --basetemp)
set -uo pipefail

PT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PT"

export CHIP_ARCH=blackhole
export OMP_NOP=1
export OMP_NOP_OUT="${OMP_NOP_OUT:-/tmp/tt-llk-build/nop_injector}"
export TTNOP="${TTNOP:-$PT/ttnop/ttnop}"
JOBS="${JOBS:-8}"
TEST_FILE="${1:-test_eltwise_unary_datacopy.py}"
PYTEST_BASETEMP="${PYTEST_BASETEMP:-/tmp/tt-llk-build/pytest}"

echo ">> CHIP_ARCH=${CHIP_ARCH} (forced)"
echo ">> TEST_FILE=${TEST_FILE}"
echo ">> NOP_THREAD=${NOP_THREAD:-math} NOP_COUNTS=${NOP_COUNTS:-1-100} JOBS=${JOBS}"
echo ">> OMP_NOP_OUT=${OMP_NOP_OUT} OMP_NOP_KEEP=${OMP_NOP_KEEP:-0}"
if [[ "${OMP_NOP_KEEP:-0}" == "1" ]]; then
  echo ">> mode: KEEP ELFs under ${OMP_NOP_OUT}/work/<item_key>/"
else
  echo ">> mode: wipe private work after each case (fails only retained)"
fi

echo ">> [1/2] building ttnop (OpenMP)..."
make -C "$PT/ttnop"

mkdir -p "$OMP_NOP_OUT" "$PYTEST_BASETEMP"

echo ">> [2/2] per-case baseline + OpenMP batch + sweep (${JOBS} xdist workers)..."
pytest -n "$JOBS" -p omp_nop_plugin -p no:randomly -v \
  --basetemp="$PYTEST_BASETEMP" \
  "$TEST_FILE"
rc=$?

echo
echo "Root:    /tmp/tt-llk-build/"
echo "Summary: ${OMP_NOP_OUT}/summary.log  (failures only)"
if [[ "${OMP_NOP_KEEP:-0}" == "1" ]]; then
  echo "Kept:    ${OMP_NOP_OUT}/work/<item_key>/{bk,batch/n<count>}/"
else
  echo "Fails:   ${OMP_NOP_OUT}/fails/<item_key>/n<count>/"
fi
exit $rc
