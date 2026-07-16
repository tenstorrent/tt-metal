#!/usr/bin/env bash
#
# Chunked NOP injector: prepare (xdist compile) → OpenMP ttnop batch → consume (xdist device).
#
# Usage:  ./run_nop_injector.sh [pytest_file]
#         CHUNK_SIZE=1000 ./run_nop_injector.sh test_eltwise_unary_datacopy.py
#
# Env:    NOP_THREAD, NOP_COUNTS, CHUNK_SIZE, MAX_CASES, JOBS (consume, default 8),
#         COMPILE_JOBS (prepare xdist, default nproc --all),
#         OMP_NUM_THREADS (ttnop, default nproc --all), OPENMP_NOP_KEEP=1,
#         OPENMP_NOP_OUT / PYTEST_BASETEMP  (must stay outside /tmp/tt-llk-build/)
#
set -uo pipefail

PT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PT"

HOST_CPUS="$(nproc --all)"

export CHIP_ARCH=blackhole
export OPENMP_NOP=1
export OPENMP_NOP_OUT="${OPENMP_NOP_OUT:-/tmp/tt-llk-nop/injector}"
export TTNOP="${TTNOP:-$PT/ttnop/ttnop}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$HOST_CPUS}"
# CSV for ttnop --counts (and plugin). Override e.g. NOP_COUNTS=1,2,4
export NOP_COUNTS="${NOP_COUNTS:-$(seq -s, 1 100)}"
export NOP_THREAD="${NOP_THREAD:-unpack}"

JOBS="${JOBS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
COMPILE_JOBS="${COMPILE_JOBS:-$HOST_CPUS}"
TEST_FILE="${1:-test_eltwise_unary_datacopy.py}"
PYTEST_BASETEMP="${PYTEST_BASETEMP:-/tmp/tt-llk-nop/pytest}"
COUNTS_CSV="$NOP_COUNTS"

echo ">> CHIP_ARCH=${CHIP_ARCH} (forced)"
echo ">> TEST_FILE=${TEST_FILE}"
echo ">> NOP_THREAD=${NOP_THREAD} NOP_COUNTS=${NOP_COUNTS} JOBS=${JOBS} CHUNK_SIZE=${CHUNK_SIZE}"
echo ">> COMPILE_JOBS=${COMPILE_JOBS} (prepare xdist) OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo ">> OPENMP_NOP_OUT=${OPENMP_NOP_OUT} OPENMP_NOP_KEEP=${OPENMP_NOP_KEEP:-0}"
if [[ "${OPENMP_NOP_KEEP:-0}" == "1" ]]; then
  echo ">> mode: KEEP ELFs under ${OPENMP_NOP_OUT}/work/<item_key>/"
else
  echo ">> mode: wipe chunk work after consume (fails only retained)"
fi

echo ">> [1/4] building ttnop (OpenMP)..."
make -C "$PT/ttnop" || exit $?
[[ -x "$TTNOP" ]] || { echo "!! ttnop not found: $TTNOP" >&2; exit 1; }
mkdir -p "$OPENMP_NOP_OUT" "$PYTEST_BASETEMP"

echo ">> [2/4] collecting nodeids from ${TEST_FILE}..."
NODEIDS_FILE="$(mktemp "${TMPDIR:-/tmp}/nop_nodeids.XXXXXX")"
MANIFEST_FILE="$(mktemp "${TMPDIR:-/tmp}/nop_chunk_manifest.XXXXXX")"
CHUNK_LIST="$(mktemp "${TMPDIR:-/tmp}/nop_chunk_list.XXXXXX")"
trap 'rm -f "$NODEIDS_FILE" "$MANIFEST_FILE" "$CHUNK_LIST"' EXIT

pytest --collect-only -q -p no:randomly "$TEST_FILE" 2>/dev/null \
  | grep '::' | grep -v '^ERROR' | sed 's/[[:space:]]*$//' > "$NODEIDS_FILE" || true

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

rc=0
chunk_i=0
for ((offset = 0; offset < TOTAL; offset += CHUNK_SIZE)); do
  chunk_i=$((chunk_i + 1))
  end=$((offset + CHUNK_SIZE))
  ((end > TOTAL)) && end=$TOTAL
  chunk_n=$((end - offset))

  echo
  echo ">> ========== CHUNK ${chunk_i}  cases $((offset + 1))–${end} / ${TOTAL} (${chunk_n} cases) =========="

  : > "$CHUNK_LIST"
  for ((j = offset; j < end; j++)); do
    echo "${ALL_NODEIDS[j]}" >> "$CHUNK_LIST"
  done
  mapfile -t CHUNK_NODEIDS < "$CHUNK_LIST"

  echo ">> [3a] prepare ${chunk_n} case(s) (compile-only, pytest -n ${COMPILE_JOBS})..."
  export OPENMP_NOP_PHASE=prepare
  unset OPENMP_NOP_CHUNK_MANIFEST || true
  PREPARE_BASETEMP="${PYTEST_BASETEMP}/prepare_chunk_${chunk_i}"
  rm -rf "$PREPARE_BASETEMP" && mkdir -p "$PREPARE_BASETEMP"
  if ! pytest -n "$COMPILE_JOBS" -p openmp_nop_plugin -p no:randomly -v \
      --compile-producer --basetemp="$PREPARE_BASETEMP" \
      "${CHUNK_NODEIDS[@]}"
  then
    echo "!! prepare failed for chunk ${chunk_i}" >&2
    rc=1
  fi

  echo ">> [3b] host OpenMP ttnop batch for chunk ${chunk_i} (OMP_NUM_THREADS=${OMP_NUM_THREADS})..."
  OPENMP_NOP_CHUNK_LIST="$CHUNK_LIST" \
  OPENMP_NOP_CHUNK_MANIFEST_OUT="$MANIFEST_FILE" \
  COUNTS_CSV="$COUNTS_CSV" \
  python3 - <<'PY'
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

out_root = Path(os.environ["OPENMP_NOP_OUT"])
ttnop, thread = os.environ["TTNOP"], os.environ.get("NOP_THREAD", "math")
counts_csv = os.environ["COUNTS_CSV"]
manifest_path = Path(os.environ["OPENMP_NOP_CHUNK_MANIFEST_OUT"])
entries, failed = [], 0

for nodeid in Path(os.environ["OPENMP_NOP_CHUNK_LIST"]).read_text().splitlines():
    nodeid = nodeid.strip()
    if not nodeid:
        continue
    key = hashlib.sha1(nodeid.encode()).hexdigest()[:16]
    work = out_root / "work" / key
    if not (work / "meta.json").is_file():
        print(f"!! skip batch (no prepare meta): {nodeid}", file=sys.stderr)
        failed += 1
        continue
    batch = work / "batch"
    shutil.rmtree(batch, ignore_errors=True)
    batch.mkdir(parents=True)
    print(f">> batch key={key}", flush=True)
    r = subprocess.run([
        ttnop, "batch",
        "--base-dir", str(work / "bk"),
        "--out-root", str(batch),
        "--thread", thread,
        "--counts", counts_csv,
    ])
    if r.returncode != 0:
        print(f"!! ttnop batch failed: {nodeid}", file=sys.stderr)
        failed += 1
        continue
    entries.append({"nodeid": nodeid, "key": key, "work": str(work)})

manifest_path.write_text(json.dumps(entries, indent=2) + "\n")
print(f">> chunk manifest: {len(entries)} ready, {failed} failed/skipped", flush=True)
sys.exit(0 if entries else 1)
PY
  [[ $? -eq 0 ]] || rc=1

  READY="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$MANIFEST_FILE" 2>/dev/null || echo 0)"
  if [[ "$READY" -eq 0 ]]; then
    echo "!! chunk ${chunk_i}: nothing to consume" >&2
    continue
  fi

  echo ">> [3c] consume chunk ${chunk_i}: ${READY} case(s) × counts (pytest -n ${JOBS})..."
  export OPENMP_NOP_PHASE=consume
  export OPENMP_NOP_CHUNK_MANIFEST="$MANIFEST_FILE"
  CONSUME_BASETEMP="${PYTEST_BASETEMP}/consume_chunk_${chunk_i}"
  rm -rf "$CONSUME_BASETEMP" && mkdir -p "$CONSUME_BASETEMP"
  mapfile -t READY_NODEIDS < <(
    python3 -c 'import json,sys; [print(e["nodeid"]) for e in json.load(open(sys.argv[1]))]' "$MANIFEST_FILE"
  )
  if ! pytest -n "$JOBS" -p openmp_nop_plugin -p no:randomly -v \
      --compile-consumer --basetemp="$CONSUME_BASETEMP" \
      "${READY_NODEIDS[@]}"
  then
    echo "!! consume reported failures for chunk ${chunk_i}" >&2
    rc=1
  fi

  if [[ "${OPENMP_NOP_KEEP:-0}" == "1" ]]; then
    echo ">> kept work dirs for chunk ${chunk_i}"
  else
    echo ">> wiping work dirs for chunk ${chunk_i}"
    OPENMP_NOP_CHUNK_MANIFEST="$MANIFEST_FILE" python3 - <<'PY'
import json, os, shutil
from pathlib import Path
for e in json.load(open(os.environ["OPENMP_NOP_CHUNK_MANIFEST"])):
    p = Path(e["work"])
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
        print(f"  wiped {p}", flush=True)
PY
  fi
done

echo
echo "ELFs:    ${OPENMP_NOP_OUT}/"
echo "Compile: /tmp/tt-llk-build/  (wiped each prepare session)"
echo "Summary: ${OPENMP_NOP_OUT}/summary.log  (failures only)"
if [[ "${OPENMP_NOP_KEEP:-0}" == "1" ]]; then
  echo "Kept:    ${OPENMP_NOP_OUT}/work/<item_key>/{bk,batch/n<count>}/"
else
  echo "Fails:   ${OPENMP_NOP_OUT}/fails/<item_key>/n<count>/"
fi
echo "Chunks:  size=${CHUNK_SIZE}  (override with CHUNK_SIZE=N)"
exit "$rc"
