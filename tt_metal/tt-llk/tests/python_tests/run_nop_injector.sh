#!/usr/bin/env bash
#
# Chunked NOP injector with host-only OpenMP.
#
# For each chunk of CHUNK_SIZE cases (default 25):
#   1. prepare  — one pytest --compile-producer for the chunk (no golden)
#   2. OpenMP   — host `ttnop batch` per case (OMP_NUM_THREADS=nproc)
#   3. consume  — ONE pytest -n JOBS over the whole chunk (25×100 items)
#   4. wipe chunk work dirs (keep fails/ unless OPENMP_NOP_KEEP=1)
#
# Usage:   ./run_nop_injector.sh [pytest_file]
#          CHUNK_SIZE=50 ./run_nop_injector.sh test_eltwise_unary_datacopy.py
#
# Env:     NOP_THREAD   unpack|math|pack                         (default math)
#          NOP_COUNTS   e.g. "1-100" or "1,2,4"                  (default 1-100)
#          CHUNK_SIZE   cases per prepare/batch/consume cycle    (default 25)
#          MAX_CASES    only first N collected nodeids           (default: all)
#          OPENMP_NOP_OUT   output root (default /tmp/tt-llk-build/nop_injector)
#          OPENMP_NOP_KEEP=1  keep work/ ELFs (no wipe after chunk)
#          JOBS         xdist workers / Tensix cores             (default 8)
#          OMP_NUM_THREADS  OpenMP threads for ttnop             (default nproc)
#          TTNOP        path to ttnop binary
#
set -uo pipefail

PT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PT"

export CHIP_ARCH=blackhole
export OPENMP_NOP=1
export OPENMP_NOP_OUT="${OPENMP_NOP_OUT:-/tmp/tt-llk-build/nop_injector}"
export TTNOP="${TTNOP:-$PT/ttnop/ttnop}"
JOBS="${JOBS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-25}"
TEST_FILE="${1:-test_eltwise_unary_datacopy.py}"
PYTEST_BASETEMP="${PYTEST_BASETEMP:-/tmp/tt-llk-build/pytest}"
NOP_COUNTS="${NOP_COUNTS:-1-100}"
NOP_THREAD="${NOP_THREAD:-math}"
export NOP_COUNTS NOP_THREAD

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  OMP_NUM_THREADS="$(nproc 2>/dev/null || echo 1)"
  export OMP_NUM_THREADS
fi

echo ">> CHIP_ARCH=${CHIP_ARCH} (forced)"
echo ">> TEST_FILE=${TEST_FILE}"
echo ">> NOP_THREAD=${NOP_THREAD} NOP_COUNTS=${NOP_COUNTS} JOBS=${JOBS} CHUNK_SIZE=${CHUNK_SIZE}"
echo ">> OMP_NUM_THREADS=${OMP_NUM_THREADS} OPENMP_NOP_OUT=${OPENMP_NOP_OUT} OPENMP_NOP_KEEP=${OPENMP_NOP_KEEP:-0}"
if [[ "${OPENMP_NOP_KEEP:-0}" == "1" ]]; then
  echo ">> mode: KEEP ELFs under ${OPENMP_NOP_OUT}/work/<item_key>/"
else
  echo ">> mode: wipe chunk work after consume (fails only retained)"
fi

echo ">> [1/4] building ttnop (OpenMP)..."
make -C "$PT/ttnop" || exit $?
if [[ ! -x "$TTNOP" ]]; then
  echo "!! ttnop binary not found/executable: $TTNOP" >&2
  exit 1
fi

mkdir -p "$OPENMP_NOP_OUT" "$PYTEST_BASETEMP"

COUNTS_CSV="$(
  python3 - <<'PY'
import os
spec = os.environ.get("NOP_COUNTS", "1-100").strip()
out = []
for part in spec.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        lo, hi = part.split("-", 1)
        out.extend(range(int(lo), int(hi) + 1))
    else:
        out.append(int(part))
print(",".join(str(c) for c in out))
PY
)"
if [[ -z "$COUNTS_CSV" ]]; then
  echo "!! NOP_COUNTS produced an empty list" >&2
  exit 1
fi

echo ">> [2/4] collecting nodeids from ${TEST_FILE}..."
NODEIDS_FILE="$(mktemp "${TMPDIR:-/tmp}/nop_nodeids.XXXXXX")"
MANIFEST_FILE="$(mktemp "${TMPDIR:-/tmp}/nop_chunk_manifest.XXXXXX")"
CHUNK_LIST="$(mktemp "${TMPDIR:-/tmp}/nop_chunk_list.XXXXXX")"
cleanup_tmp() { rm -f "$NODEIDS_FILE" "$MANIFEST_FILE" "$CHUNK_LIST"; }
trap cleanup_tmp EXIT

pytest --collect-only -q -p no:randomly "$TEST_FILE" 2>/dev/null \
  | grep '::' \
  | grep -v '^ERROR' \
  | sed 's/[[:space:]]*$//' \
  > "$NODEIDS_FILE" || true

NUM_CASES="$(wc -l < "$NODEIDS_FILE" | tr -d ' ')"
if [[ "$NUM_CASES" -eq 0 ]]; then
  echo "!! no nodeids collected from ${TEST_FILE}" >&2
  exit 1
fi
if [[ -n "${MAX_CASES:-}" ]]; then
  head -n "$MAX_CASES" "$NODEIDS_FILE" > "${NODEIDS_FILE}.trim"
  mv "${NODEIDS_FILE}.trim" "$NODEIDS_FILE"
  NUM_CASES="$(wc -l < "$NODEIDS_FILE" | tr -d ' ')"
  echo ">> MAX_CASES=${MAX_CASES} → ${NUM_CASES} case(s)"
else
  echo ">> collected ${NUM_CASES} case(s)"
fi

overall_rc=0
chunk_i=0

# Read nodeids into an array for chunking.
mapfile -t ALL_NODEIDS < "$NODEIDS_FILE"
TOTAL=${#ALL_NODEIDS[@]}

for ((offset = 0; offset < TOTAL; offset += CHUNK_SIZE)); do
  chunk_i=$((chunk_i + 1))
  end=$((offset + CHUNK_SIZE))
  if ((end > TOTAL)); then end=$TOTAL; fi
  chunk_n=$((end - offset))

  echo
  echo ">> ========== CHUNK ${chunk_i}  cases $((offset + 1))–${end} / ${TOTAL} (${chunk_n} cases) =========="

  # --- build chunk nodeid list ---
  : > "$CHUNK_LIST"
  for ((j = offset; j < end; j++)); do
    echo "${ALL_NODEIDS[j]}" >> "$CHUNK_LIST"
  done

  # --- 3a prepare: one pytest for the whole chunk ---
  echo ">> [3a] prepare ${chunk_n} case(s) (compile-only, one pytest)..."
  export OPENMP_NOP_PHASE=prepare
  unset OPENMP_NOP_WORK OPENMP_NOP_BASE_NODEID OPENMP_NOP_CHUNK_MANIFEST || true
  PREPARE_BASETEMP="${PYTEST_BASETEMP}/prepare_chunk_${chunk_i}"
  rm -rf "$PREPARE_BASETEMP"
  mkdir -p "$PREPARE_BASETEMP"

  # Nodeids contain spaces (e.g. input_dimensions:[64, 64]) — must not word-split.
  mapfile -t CHUNK_NODEIDS < "$CHUNK_LIST"
  if ! pytest -n0 -p openmp_nop_plugin -p no:randomly -v \
      --compile-producer \
      --basetemp="$PREPARE_BASETEMP" \
      "${CHUNK_NODEIDS[@]}"
  then
    echo "!! prepare failed for chunk ${chunk_i}" >&2
    overall_rc=1
    # continue to next chunk; skip batch/consume for failures without meta
  fi

  # --- 3b OpenMP batch each case that has meta ---
  echo ">> [3b] host OpenMP ttnop batch for chunk ${chunk_i} (OMP_NUM_THREADS=${OMP_NUM_THREADS})..."
  OPENMP_NOP_CHUNK_LIST="$CHUNK_LIST" \
  OPENMP_NOP_CHUNK_MANIFEST_OUT="$MANIFEST_FILE" \
  COUNTS_CSV="$COUNTS_CSV" \
  python3 - <<'PY'
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

out_root = Path(os.environ["OPENMP_NOP_OUT"])
ttnop = os.environ["TTNOP"]
thread = os.environ.get("NOP_THREAD", "math")
counts_csv = os.environ["COUNTS_CSV"]
chunk_list = Path(os.environ["OPENMP_NOP_CHUNK_LIST"])
manifest_path = Path(os.environ["OPENMP_NOP_CHUNK_MANIFEST_OUT"])
entries = []
failed = 0
for nodeid in chunk_list.read_text().splitlines():
    nodeid = nodeid.strip()
    if not nodeid:
        continue
    key = hashlib.sha1(nodeid.encode()).hexdigest()[:16]
    work = out_root / "work" / key
    meta = work / "meta.json"
    if not meta.is_file():
        print(f"!! skip batch (no prepare meta): {nodeid}", file=sys.stderr)
        failed += 1
        continue
    batch = work / "batch"
    if batch.exists():
        shutil.rmtree(batch, ignore_errors=True)
    batch.mkdir(parents=True, exist_ok=True)
    cmd = [
        ttnop, "batch",
        "--base-dir", str(work / "bk"),
        "--out-root", str(batch),
        "--thread", thread,
        "--counts", counts_csv,
    ]
    print(f">> batch key={key}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!! ttnop batch failed: {nodeid}", file=sys.stderr)
        with open(out_root / "summary.log", "a") as f:
            f.write(f"{nodeid}\tBATCH-ERR\tttnop batch failed\n")
        failed += 1
        continue
    entries.append({"nodeid": nodeid, "key": key, "work": str(work)})
manifest_path.write_text(json.dumps(entries, indent=2) + "\n")
print(f">> chunk manifest: {len(entries)} ready, {failed} failed/skipped", flush=True)
sys.exit(1 if failed and not entries else 0)
PY
  batch_rc=$?
  if [[ "$batch_rc" -ne 0 ]]; then
    overall_rc=1
  fi

  READY="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$MANIFEST_FILE" 2>/dev/null || echo 0)"
  if [[ "$READY" -eq 0 ]]; then
    echo "!! chunk ${chunk_i}: nothing to consume" >&2
    continue
  fi

  # --- 3c one pytest for the whole chunk ---
  echo ">> [3c] consume chunk ${chunk_i}: ${READY} case(s) × counts (pytest -n ${JOBS})..."
  export OPENMP_NOP_PHASE=consume
  export OPENMP_NOP_CHUNK_MANIFEST="$MANIFEST_FILE"
  unset OPENMP_NOP_WORK OPENMP_NOP_BASE_NODEID || true
  CONSUME_BASETEMP="${PYTEST_BASETEMP}/consume_chunk_${chunk_i}"
  rm -rf "$CONSUME_BASETEMP"
  mkdir -p "$CONSUME_BASETEMP"

  mapfile -t READY_NODEIDS < <(python3 -c 'import json,sys; [print(e["nodeid"]) for e in json.load(open(sys.argv[1]))]' "$MANIFEST_FILE")
  if ! pytest -n "$JOBS" -p openmp_nop_plugin -p no:randomly -v \
      --compile-consumer \
      --basetemp="$CONSUME_BASETEMP" \
      "${READY_NODEIDS[@]}"
  then
    echo "!! consume reported failures for chunk ${chunk_i}" >&2
    overall_rc=1
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
echo "Root:    /tmp/tt-llk-build/"
echo "Summary: ${OPENMP_NOP_OUT}/summary.log  (failures only)"
if [[ "${OPENMP_NOP_KEEP:-0}" == "1" ]]; then
  echo "Kept:    ${OPENMP_NOP_OUT}/work/<item_key>/{bk,batch/n<count>}/"
else
  echo "Fails:   ${OPENMP_NOP_OUT}/fails/<item_key>/n<count>/"
fi
echo "Chunks:  size=${CHUNK_SIZE}  (override with CHUNK_SIZE=N)"
exit "$overall_rc"
