#!/usr/bin/env bash
# laneMQ galaxy driver: shard binarypow's full 2^32 joint bf16^2 sweep across NPAR chips
# of ONE galaxy node, both certified legs per chip, then combine to one verdict. ONE op,
# run-to-completion-and-quit (exit frees the node). Resume-safe (cached band SHAs).
#
# Env: FARM_ROOT (has tests/ + build/tt-llk-build), VENV, NPAR (default 32),
#      BAND_BITS (per-slice band size, default 23), OUT.
set -uo pipefail
FARM_ROOT="${FARM_ROOT:?}" ; VENV="${VENV:?}" ; OUT="${OUT:?}"
NPAR="${NPAR:-32}"
BAND_BITS="${BAND_BITS:-23}"
PYDIR="$FARM_ROOT/tests/python_tests"
TOOLS="$FARM_ROOT/tests/corpus/tools"
BUILD="$FARM_ROOT/build"
LLK_HOME="$FARM_ROOT/tests"
SEM='test_sfpu_binary.py::test_fresh_cpp_binary_pow[formats:Float16_b->Float16_b-mathop:SfpuElwpow-dest_acc:No-fresh_cpp_impl:1]'
HAND='test_sfpu_binary.py::test_fresh_cpp_binary_pow[formats:Float16_b->Float16_b-mathop:SfpuElwpow-dest_acc:No-fresh_cpp_impl:3]'
TWO32=4294967296
mkdir -p "$OUT"
echo "HOST=$(hostname) NPAR=$NPAR BAND_BITS=$BAND_BITS $(date -u +%H:%M:%SZ)" | tee "$OUT/DRIVER.log"

# ---- object-identity gate (once): sem != hand .text, both non-empty ----
OBJ_SEM=$(find "$BUILD/tt-llk-build/sources" -path '*f7bbba208acd05cf64bc2d3c84915c479dc8478fdbb9212c4cf5837c22128de3/elf/math.elf')
OBJ_HAND=$(find "$BUILD/tt-llk-build/sources" -path '*4fdf2260eb2f4fd8ca80509ea5a57ef9e191ec7b2c74f05c10d952925e42154a/elf/math.elf')
sha_sem=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_SEM")
sha_hand=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_HAND")
echo "IDGATE sem_text=$sha_sem hand_text=$sha_hand" | tee -a "$OUT/DRIVER.log"
if [ -z "$sha_sem" ] || [ -z "$sha_hand" ] || [ "$sha_sem" = "$sha_hand" ]; then
  echo "OP=binarypow VERDICT=REFUSED-IDENTITY(sem==hand or empty)" | tee "$OUT/binarypow-VERDICT.txt"
  exit 1
fi

SLICE=$(( TWO32 / NPAR ))
echo "SLICE=$SLICE joints/chip" | tee -a "$OUT/DRIVER.log"
# Stagger chip launches: 32 simultaneous cold torch/ttexalens imports off the NFS venv
# stampede the fileserver and get SIGINT-killed mid-import. A few seconds apart spreads
# the one-time import so every chip's harness comes up. Streaming itself is device-bound
# and unaffected. STAGGER seconds between launches (default 3).
STAGGER="${STAGGER:-3}"
pids=()
for k in $(seq 0 $((NPAR-1))); do
  RT="/tmp/lanemq-rt-$k"
  [ -d "$RT/tt-llk-build/sources" ] || { mkdir -p "$RT"; cp -a "$BUILD/tt-llk-build" "$RT/"; }
  start=$(( k * SLICE ))
  sdir="$OUT/slice-$k"
  ( LANEMK_WAIT_TIMEOUT="${LANEMK_WAIT_TIMEOUT:-600}" \
    "$VENV" "$TOOLS/binary_stream_sweep.py" \
      --op "binarypow-s$k" --sem-node "$SEM" --hand-node "$HAND" \
      --farm "$PYDIR" --venv "$VENV" --llk-home "$LLK_HOME" --runner-temp "$RT" \
      --band-bits "$BAND_BITS" --chip "$k" --start-bit "$start" --total "$SLICE" \
      --out "$sdir" > "$OUT/slice-$k.log" 2>&1 ) &
  pids+=("$!")
  sleep "$STAGGER"
done
echo "launched $NPAR chip-slices $(date -u +%H:%M:%SZ)" | tee -a "$OUT/DRIVER.log"
for p in "${pids[@]}"; do wait "$p"; done
echo "all slices done $(date -u +%H:%M:%SZ)" | tee -a "$OUT/DRIVER.log"

# ---- combine ----
"$VENV" - "$OUT" "$NPAR" "$TWO32" <<'PY' | tee "$OUT/binarypow-VERDICT.txt"
import sys, glob, re, pathlib
out, npar, two32 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
covered = 0; all_eq = True; witness = []; missing = []
for k in range(npar):
    v = pathlib.Path(out)/f"slice-{k}"/f"binarypow-s{k}-VERDICT.txt"
    if not v.exists():
        missing.append(k); all_eq = False; continue
    t = v.read_text()
    m = re.search(r"covered=(\d+)", t); covered += int(m.group(1)) if m else 0
    if "VERDICT=BIT-EXACT-ALL-INPUTS" not in t:
        all_eq = False
        wb = re.search(r"witness_bands=(\[.*\])", t)
        witness.append((k, wb.group(1) if wb else "?"))
full = covered == two32 and not missing
verdict = "BIT-EXACT-ALL-INPUTS" if (all_eq and full) else ("DIVERGENT" if not all_eq and not missing else "INCOMPLETE")
print(f"OP=binarypow VERDICT={verdict} slices={npar} covered={covered} "
      f"(full 2^32={covered==two32}) missing={missing} witness={witness}")
PY
