#!/usr/bin/env bash
# galaxy_shard.sh — shard ONE op's full input space across the 32 chips of a
# galaxy node, both certified legs per chip, then combine to one verdict.
# Run-to-completion-and-quit (exit frees the node).  Resume-safe (cached band
# SHAs), so a re-run picks up where a killed one stopped.
#
# This is the only 32-chip leg.  The per-op runners (run_op.sh, one job per
# op) pass --chip 0 and fan out across OPS, one chip per node — fine for
# breadth, 31 chips idle per op.  Use this when you want one op finished.
#
#   REQUIRED  FARM_ROOT  has tests/ + build/tt-llk-build
#             VENV       python that can import the harness
#             OUT        evidence dir (NFS; slices land under it)
#
#   OP        op key, default binarypow
#   SWEEP     binary | fp32   which streamer (default binary)
#             binary = two-operand joint space (binary_stream_sweep.py)
#             fp32   = one-operand space      (fp32_stream_sweep.py)
#   SPACE     input-space size, default 2^32 (both sweeps today)
#   NPAR      chips, default 32
#   BAND_BITS per-slice band size, default 23
#   STAGGER   seconds between chip launches, default 3
#
#   Op identity — supply EITHER the four values directly, OR a table to read
#   them from.  Defaults reproduce the certified binarypow run exactly.
#     SEM / HAND                  pytest node ids for the two legs
#     SEM_VARIANT / HAND_VARIANT  build-dir variant hashes, for the gate
#     OPS_TSV    op<TAB>sem_node<TAB>hand_node   (as run_op.sh uses)
#     IDMAP      op<TAB>sem_variant<TAB>sem_sha<TAB>hand_variant<TAB>hand_sha
#                also forwarded to each slice, so the per-slice gate still runs
#
# Examples:
#   OUT=$EV FARM_ROOT=$F VENV=$V bash galaxy_shard.sh            # binarypow
#   OP=exp SWEEP=fp32 OPS_TSV=$T IDMAP=$M OUT=$EV ... bash galaxy_shard.sh
set -uo pipefail
FARM_ROOT="${FARM_ROOT:?}" ; VENV="${VENV:?}" ; OUT="${OUT:?}"
OP="${OP:-binarypow}"
SWEEP="${SWEEP:-binary}"
NPAR="${NPAR:-32}"
BAND_BITS="${BAND_BITS:-23}"
STAGGER="${STAGGER:-3}"
SPACE="${SPACE:-4294967296}"
PYDIR="$FARM_ROOT/tests/python_tests"
TOOLS="$FARM_ROOT/tests/corpus/tools"
BUILD="$FARM_ROOT/build"
LLK_HOME="$FARM_ROOT/tests"

case "$SWEEP" in
  binary) STREAMER="$TOOLS/binary_stream_sweep.py" ;;
  fp32)   STREAMER="$TOOLS/fp32_stream_sweep.py" ;;
  *) echo "FATAL: SWEEP must be 'binary' or 'fp32', got '$SWEEP'" >&2; exit 2 ;;
esac
[ -f "$STREAMER" ] || { echo "FATAL: no streamer at $STREAMER" >&2; exit 2; }

# ---- op identity ----------------------------------------------------------
# Certified binarypow defaults: the pinned SfpuElwpow pair and the two build
# variants their .text hashes came from.  Any other op must supply its own,
# directly or through OPS_TSV/IDMAP.
DEF_SEM='test_sfpu_binary.py::test_fresh_cpp_binary_pow[formats:Float16_b->Float16_b-mathop:SfpuElwpow-dest_acc:No-fresh_cpp_impl:1]'
DEF_HAND='test_sfpu_binary.py::test_fresh_cpp_binary_pow[formats:Float16_b->Float16_b-mathop:SfpuElwpow-dest_acc:No-fresh_cpp_impl:3]'
DEF_SEM_VARIANT=f7bbba208acd05cf64bc2d3c84915c479dc8478fdbb9212c4cf5837c22128de3
DEF_HAND_VARIANT=4fdf2260eb2f4fd8ca80509ea5a57ef9e191ec7b2c74f05c10d952925e42154a

SEM="${SEM:-}" ; HAND="${HAND:-}"
SEM_VARIANT="${SEM_VARIANT:-}" ; HAND_VARIANT="${HAND_VARIANT:-}"

# Precedence: explicit env > table row > built-in binarypow default.  A table
# that has no row for this op does NOT erase a default -- pass an op the table
# does not know and you get the named refusal below, not a silent binarypow.
if [ "$OP" = binarypow ]; then
  SEM="${SEM:-$DEF_SEM}" ; HAND="${HAND:-$DEF_HAND}"
  SEM_VARIANT="${SEM_VARIANT:-$DEF_SEM_VARIANT}"
  HAND_VARIANT="${HAND_VARIANT:-$DEF_HAND_VARIANT}"
fi
if [ -n "${OPS_TSV:-}" ]; then
  _s=$(awk -F'\t' -v o="$OP" '$1==o{print $2}' "$OPS_TSV")
  _h=$(awk -F'\t' -v o="$OP" '$1==o{print $3}' "$OPS_TSV")
  [ -n "$_s" ] && SEM=$_s
  [ -n "$_h" ] && HAND=$_h
fi
if [ -n "${IDMAP:-}" ]; then
  # same row layout the streamers read: op, sem_variant, sem_sha, hand_variant, hand_sha
  _sv=$(awk -F'\t' -v o="$OP" '$1==o{print $2}' "$IDMAP")
  _hv=$(awk -F'\t' -v o="$OP" '$1==o{print $4}' "$IDMAP")
  [ -n "$_sv" ] && SEM_VARIANT=$_sv
  [ -n "$_hv" ] && HAND_VARIANT=$_hv
fi
[ -n "$SEM" ] && [ -n "$HAND" ] || {
  echo "FATAL: no sem/hand nodes for '$OP' — set SEM and HAND, or pass OPS_TSV" >&2; exit 2; }
[ -n "$SEM_VARIANT" ] && [ -n "$HAND_VARIANT" ] || {
  echo "FATAL: no build variants for '$OP' — set SEM_VARIANT and HAND_VARIANT, or pass IDMAP" >&2; exit 2; }

mkdir -p "$OUT"
echo "HOST=$(hostname) OP=$OP SWEEP=$SWEEP NPAR=$NPAR BAND_BITS=$BAND_BITS SPACE=$SPACE $(date -u +%H:%M:%SZ)" \
  | tee "$OUT/DRIVER.log"

# ---- object-identity gate, ONCE --------------------------------------------
# sem != hand .text, both non-empty.  Done here rather than only per-slice so a
# mis-built pair costs one gate instead of 32 slice failures.  Each slice still
# re-gates through --idmap when one is supplied.
OBJ_SEM=$(find "$BUILD/tt-llk-build/sources" -path "*${SEM_VARIANT}/elf/math.elf" | head -1)
OBJ_HAND=$(find "$BUILD/tt-llk-build/sources" -path "*${HAND_VARIANT}/elf/math.elf" | head -1)
sha_sem=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_SEM" 2>/dev/null)
sha_hand=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_HAND" 2>/dev/null)
echo "IDGATE sem_text=$sha_sem hand_text=$sha_hand" | tee -a "$OUT/DRIVER.log"
if [ -z "$sha_sem" ] || [ -z "$sha_hand" ] || [ "$sha_sem" = "$sha_hand" ]; then
  echo "OP=$OP VERDICT=REFUSED-IDENTITY(sem==hand or empty)" | tee "$OUT/$OP-VERDICT.txt"
  exit 1
fi

SLICE=$(( SPACE / NPAR ))
[ $(( SLICE * NPAR )) -eq "$SPACE" ] \
  || { echo "FATAL: SPACE=$SPACE is not divisible by NPAR=$NPAR — slices would not cover it" >&2; exit 2; }
echo "SLICE=$SLICE inputs/chip" | tee -a "$OUT/DRIVER.log"

idmap_args=()
[ -n "${IDMAP:-}" ] && idmap_args=(--idmap "$IDMAP")
# The ULP/golden leg rides this pass at no extra device cost; GOLDEN=0 opts out.
golden_args=()
[ "${GOLDEN:-1}" = 1 ] && golden_args=(--golden "$OP")

# Stagger chip launches: 32 simultaneous cold torch/ttexalens imports off the
# NFS venv stampede the fileserver and get SIGINT-killed mid-import.  A few
# seconds apart spreads the one-time import so every chip's harness comes up.
# Streaming itself is device-bound and unaffected.
pids=()
for k in $(seq 0 $((NPAR-1))); do
  RT="/tmp/galaxy-shard-rt-$k"
  [ -d "$RT/tt-llk-build/sources" ] || { mkdir -p "$RT"; cp -a "$BUILD/tt-llk-build" "$RT/"; }
  start=$(( k * SLICE ))
  sdir="$OUT/slice-$k"
  ( LANEMK_WAIT_TIMEOUT="${LANEMK_WAIT_TIMEOUT:-600}" \
    "$VENV" "$STREAMER" \
      --op "$OP-s$k" --sem-node "$SEM" --hand-node "$HAND" \
      --farm "$PYDIR" --venv "$VENV" --llk-home "$LLK_HOME" --runner-temp "$RT" \
      --band-bits "$BAND_BITS" --chip "$k" --start-bit "$start" --total "$SLICE" \
      --out "$sdir" \
      ${idmap_args[@]+"${idmap_args[@]}"} \
      ${golden_args[@]+"${golden_args[@]}"} > "$OUT/slice-$k.log" 2>&1 ) &
  pids+=("$!")
  sleep "$STAGGER"
done
echo "launched $NPAR chip-slices $(date -u +%H:%M:%SZ)" | tee -a "$OUT/DRIVER.log"
for p in "${pids[@]}"; do wait "$p"; done
echo "all slices done $(date -u +%H:%M:%SZ)" | tee -a "$OUT/DRIVER.log"

# ---- combine ----
"$VENV" - "$OUT" "$NPAR" "$SPACE" "$OP" <<'PY' | tee "$OUT/$OP-VERDICT.txt"
import sys, re, pathlib
out, npar, space, op = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
covered = 0; all_eq = True; witness = []; missing = []
for k in range(npar):
    v = pathlib.Path(out)/f"slice-{k}"/f"{op}-s{k}-VERDICT.txt"
    if not v.exists():
        missing.append(k); all_eq = False; continue
    t = v.read_text()
    m = re.search(r"covered=(\d+)", t); covered += int(m.group(1)) if m else 0
    if "VERDICT=BIT-EXACT-ALL-INPUTS" not in t:
        all_eq = False
        wb = re.search(r"witness_bands=(\[.*\])", t)
        witness.append((k, wb.group(1) if wb else "?"))
full = covered == space and not missing
verdict = "BIT-EXACT-ALL-INPUTS" if (all_eq and full) else ("DIVERGENT" if not all_eq and not missing else "INCOMPLETE")
print(f"OP={op} VERDICT={verdict} slices={npar} covered={covered} "
      f"(full {space}={covered==space}) missing={missing} witness={witness}")
PY
