#!/usr/bin/env bash
#
# NOP injector: prepare (compile) → OpenMP ttnop batch → consume (device).
#
# Usage:  ./run_nop_injector.sh [pytest_file]
#         MAX_CASES=5 NOP_COUNTS=1,2,4 ./run_nop_injector.sh test_eltwise_unary_datacopy.py
#
# Env:    NOP_THREAD, NOP_COUNTS, MAX_CASES, JOBS (consume, default 8),
#         COMPILE_JOBS (prepare xdist, default nproc --all),
#         OMP_NUM_THREADS (ttnop OpenMP, default nproc --all), OPEN_MP_NOP_KEEP=1,
#         OPEN_MP_NOP_OUT / PYTEST_BASETEMP  (must stay outside /tmp/tt-llk-build/)
#
# OpenMP runs only inside ``ttnop batch``. Pytest plugins never call ttnop.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST_CPUS="$(nproc --all)"  # host CPU count (prepare xdist + OpenMP)

export CHIP_ARCH=blackhole
export OPEN_MP_NOP_OUT="${OPEN_MP_NOP_OUT:-/tmp/tt-llk-nop/injector}"  # work/fails/summary root
export TTNOP="${TTNOP:-$SCRIPT_DIR/ttnop/ttnop}" # path to ttnop
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$HOST_CPUS}"  # ttnop OpenMP parallelism (libgomp)
export NOP_COUNTS="${NOP_COUNTS:-$(seq -s, 1 100)}"
export NOP_THREAD="${NOP_THREAD:-math}" # which TRISC ELF to patch (unpack|math|pack)

JOBS="${JOBS:-8}" # consume xdist workers
COMPILE_JOBS="${COMPILE_JOBS:-$HOST_CPUS}"  # prepare xdist workers — host CPUs
TEST_FILE="${1:-test_eltwise_unary_datacopy.py}"
PYTEST_BASETEMP="${PYTEST_BASETEMP:-/tmp/tt-llk-nop/pytest}"
COUNTS_CSV="$NOP_COUNTS"

echo ">> CHIP_ARCH=${CHIP_ARCH}"
echo ">> TEST_FILE=${TEST_FILE}"
echo ">> NOP_THREAD=${NOP_THREAD} NOP_COUNTS=${NOP_COUNTS} JOBS=${JOBS}"
echo ">> COMPILE_JOBS=${COMPILE_JOBS} (prepare xdist) OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo ">> OPEN_MP_NOP_OUT=${OPEN_MP_NOP_OUT} OPEN_MP_NOP_KEEP=${OPEN_MP_NOP_KEEP:-0}"
if [[ "${OPEN_MP_NOP_KEEP:-0}" == "1" ]]; then
  echo ">> mode: KEEP ELFs under ${OPEN_MP_NOP_OUT}/work/<item_key>/"
else
  echo ">> mode: wipe work after consume (fails only retained)"
fi
# make the program
echo ">> [1/4] building ttnop"
make -C "$SCRIPT_DIR/ttnop" || exit $?
[[ -x "$TTNOP" ]] || { echo "!! ttnop not found: $TTNOP" >&2; exit 1; }
mkdir -p "$OPEN_MP_NOP_OUT" "$PYTEST_BASETEMP"

echo ">> [2/4] collecting nodeids from ${TEST_FILE}..."
# list of pytest Ids collected from test file
# Used by prepare.py know which test cases to snapshot and by batch_cases.py to know which cases to run ttnop-batch
NODEIDS_FILE="$(mktemp "${TMPDIR:-/tmp}/nop_nodeids.XXXXXX")"

# output of batch_cases, it has the cases where ttnop batch worked on and includes {pytestID, key, work}
# for consume.py and delete_dirs.py to know what to run and clean up.
CASE_LIST_FILE="$(mktemp "${TMPDIR:-/tmp}/nop_case_list.XXXXXX")"
trap 'rm -f "$NODEIDS_FILE" "$CASE_LIST_FILE"' EXIT # delete files at end of program

# collect pytest IDs and store in NODEIDS_FILE
pytest --collect-only -q -p no:randomly "$TEST_FILE" 2>/dev/null \
  | grep '::' | grep -v '^ERROR' | sed 's/[[:space:]]*$//' > "$NODEIDS_FILE" || true

# trim pytest IDs to MAX_CASES if necessary
if [[ -n "${MAX_CASES:-}" ]]; then
  head -n "$MAX_CASES" "$NODEIDS_FILE" > "${NODEIDS_FILE}.trim"
  mv "${NODEIDS_FILE}.trim" "$NODEIDS_FILE"
fi

mapfile -t ALL_NODEIDS < "$NODEIDS_FILE"
TOTAL=${#ALL_NODEIDS[@]}
[[ "$TOTAL" -gt 0 ]] || { echo "!! no nodeids from ${TEST_FILE}" >&2; exit 1; }
if [[ -n "${MAX_CASES:-}" ]]; then
  echo ">> MAX_CASES=${MAX_CASES} → ${TOTAL} case(s)"
else
  echo ">> collected ${TOTAL} case(s)"
fi

# exit status for the whole pipeline. Individual parts keep running
# after a failure (so cleanup still happens)
exit_status=0

echo
echo ">> [3a] prepare ${TOTAL} case(s) (compile-only, pytest -n ${COMPILE_JOBS})"

export OPEN_MP_NOP_PHASE=prepare
# Scratch dir for pytest workers during prepare.py
PREPARE_BASETEMP="${PYTEST_BASETEMP}/prepare"
rm -rf "$PREPARE_BASETEMP" && mkdir -p "$PREPARE_BASETEMP"

# For each pytest create a directory work/key/base_elfs and save the threads' ELF
if ! pytest -n "$COMPILE_JOBS" -p nop_injector.prepare -p no:randomly -q \
    --compile-producer --basetemp="$PREPARE_BASETEMP" \
    "${ALL_NODEIDS[@]}"
then
  echo "!! prepare failed" >&2
  exit_status=1
fi

# use ttnop --batch to insert NOPs in work/key/batch for each pytest
echo ">> [3b] host OpenMP ttnop batch (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
if ! python3 -m nop_injector.batch_cases \
    --nodeids "$NODEIDS_FILE" \
    --case-list "$CASE_LIST_FILE" \
    --counts "$COUNTS_CSV"
then
  exit_status=1
fi

export OPEN_MP_NOP_PHASE=consume
export OPEN_MP_NOP_CASE_LIST="$CASE_LIST_FILE"
# Scratch dir for pytest workers during consume.py
CONSUME_BASETEMP="${PYTEST_BASETEMP}/consume"
rm -rf "$CONSUME_BASETEMP" && mkdir -p "$CONSUME_BASETEMP"

# Device run: for each case, expand to one pytest item per NOP count and load batch/nN/ ELFs
echo ">> [3c] consume ${TOTAL} case(s) × counts (pytest -n ${JOBS})..."
if ! pytest -n "$JOBS" -p nop_injector.consume -p no:randomly -q \
    --compile-consumer --basetemp="$CONSUME_BASETEMP" \
    "${ALL_NODEIDS[@]}"
then
  echo "!! consume reported failures" >&2
  exit_status=1
fi

if [[ "${OPEN_MP_NOP_KEEP:-0}" == "1" ]]; then
  echo ">> kept work dirs"
else
  echo ">> wiping work dirs"
  python3 -m nop_injector.delete_dirs --case-list "$CASE_LIST_FILE"
fi

echo
echo "ELFs:    ${OPEN_MP_NOP_OUT}/"
echo "Summary: ${OPEN_MP_NOP_OUT}/summary.log  (failures only)"
if [[ "${OPEN_MP_NOP_KEEP:-0}" == "1" ]]; then
  echo "Kept:    ${OPEN_MP_NOP_OUT}/work/<item_key>/{base_elfs,batch/n<count>}/"
else
  echo "Fails:   ${OPEN_MP_NOP_OUT}/fails/<item_key>/n<count>/"
fi
exit "$exit_status"
