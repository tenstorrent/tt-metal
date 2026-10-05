#!/usr/bin/env bash
# galaxy_shard.sh — shard ONE op's full input space across the 32 chips of a
# galaxy node, both identity-gated legs per chip, then combine to one verdict.
# Run-to-completion-and-quit (exit frees the node).  Resume-safe only for band
# SHAs carrying exact campaign provenance, so a re-run cannot adopt old caches.
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
#   SPACE     how much of the space THIS RUN sweeps, default 2^32
#   FULL_SPACE the op's WHOLE input space, default 2^32 (both sweeps today).
#             BIT-EXACT-ALL-INPUTS needs covered==FULL_SPACE; a reduced SPACE is
#             labelled BIT-EXACT-PARTIAL-<covered>-OF-<FULL_SPACE> and can never
#             be promoted to an exhaustive claim by accident.
#   NPAR      chips, default 32
#   BAND_BITS per-slice band size, default 23
#   STAGGER   seconds between chip launches, default 3
#
#   Tri-arm mode (compiler validation + semantic uplift):
#     TRI_PROFILES  planner tri-profiles.tsv
#     TRI_IDMAP     op<TAB>A_variant<TAB>A_sha<TAB>B_variant<TAB>B_sha<TAB>C_variant<TAB>C_sha
#   Legacy two-arm mode remains available for a plain sem-vs-hand comparison:
#     SEM / HAND                  pytest node ids for the two legs
#     SEM_VARIANT / HAND_VARIANT  build-dir variant hashes, for the gate
#     OPS_TSV    op<TAB>sem_node<TAB>hand_node   (as run_op.sh uses)
#     IDMAP      op<TAB>sem_variant<TAB>sem_sha<TAB>hand_variant<TAB>hand_sha
#                also forwarded to each slice, so the per-slice gate still runs
#     FLAGS_TSV  op<TAB>exact compiler flag string. Required when the staged
#                variants were built with per-op tuning selections.
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
FULL_SPACE="${FULL_SPACE:-4294967296}"
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

# ---- knob validation ------------------------------------------------------
# Every geometry knob reaches `$(( ))`, where bash silently treats a
# non-numeric word as 0 (and `set -u` turns it into an unbound-variable abort
# mid-script instead of a named refusal).  Validate them as plain decimals up
# front so a typo is a refusal, never a zero-sized "proof".
_posint() { case "${2:-}" in ''|*[!0-9]*) echo "FATAL: $1 must be a decimal integer, got '${2:-}'" >&2; exit 2 ;; esac; }
_posint NPAR       "$NPAR"
_posint SPACE      "$SPACE"
_posint FULL_SPACE "$FULL_SPACE"
_posint BAND_BITS  "$BAND_BITS"
_posint STAGGER    "$STAGGER"
case "${GOLDEN:-1}" in 0|1) ;; *) echo "FATAL: GOLDEN must be 0 or 1, got '${GOLDEN:-}'" >&2; exit 2 ;; esac
[ "$NPAR"  -gt 0 ] || { echo "FATAL: NPAR must be positive" >&2; exit 2; }
[ "$SPACE" -gt 0 ] || { echo "FATAL: SPACE must be positive — a zero-sized space would 'cover' itself" >&2; exit 2; }
[ "$FULL_SPACE" -gt 0 ] || { echo "FATAL: FULL_SPACE must be positive" >&2; exit 2; }
# Sweeping MORE than the op's whole space is a mis-declared geometry, not a
# stronger proof; refuse rather than let covered overshoot FULL_SPACE.
[ "$SPACE" -le "$FULL_SPACE" ] \
  || { echo "FATAL: SPACE=$SPACE exceeds FULL_SPACE=$FULL_SPACE" >&2; exit 2; }
[ "$BAND_BITS" -gt 0 ] && [ "$BAND_BITS" -le 32 ] \
  || { echo "FATAL: BAND_BITS must be in 1..32, got '$BAND_BITS'" >&2; exit 2; }

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
TRI_MODE=0
if [ -n "${TRI_PROFILES:-}" ] || [ -n "${TRI_IDMAP:-}" ]; then
  [ -f "${TRI_PROFILES:-}" ] && [ -f "${TRI_IDMAP:-}" ] || {
    echo "FATAL: tri-arm mode requires both TRI_PROFILES and TRI_IDMAP" >&2; exit 2; }
  TRI_MODE=1
  _profile_rows=$(awk -F'\t' -v o="$OP" '$1==o{n++} END{print n+0}' "$TRI_PROFILES")
  _id_rows=$(awk -F'\t' -v o="$OP" '$1==o{n++} END{print n+0}' "$TRI_IDMAP")
  [ "$_profile_rows" -eq 1 ] && [ "$_id_rows" -eq 1 ] || {
    echo "FATAL: tri tables need exactly one '$OP' row (profiles=$_profile_rows identity=$_id_rows)" >&2
    exit 2
  }
  _profile_nf=$(awk -F'\t' -v o="$OP" '$1==o{print NF; exit}' "$TRI_PROFILES")
  _id_nf=$(awk -F'\t' -v o="$OP" '$1==o{print NF; exit}' "$TRI_IDMAP")
  [ "$_profile_nf" -eq 10 ] && [ "$_id_nf" -eq 7 ] || {
    echo "FATAL: malformed tri row for '$OP' (profiles fields=$_profile_nf identity fields=$_id_nf)" >&2
    exit 2
  }
  A_NODE=$(awk -F'\t' -v o="$OP" '$1==o{print $5; exit}' "$TRI_PROFILES")
  A_FLAGS=$(awk -F'\t' -v o="$OP" '$1==o{print $6; exit}' "$TRI_PROFILES")
  B_NODE=$(awk -F'\t' -v o="$OP" '$1==o{print $7; exit}' "$TRI_PROFILES")
  B_FLAGS=$(awk -F'\t' -v o="$OP" '$1==o{print $8; exit}' "$TRI_PROFILES")
  C_NODE=$(awk -F'\t' -v o="$OP" '$1==o{print $9; exit}' "$TRI_PROFILES")
  C_FLAGS=$(awk -F'\t' -v o="$OP" '$1==o{print $10; exit}' "$TRI_PROFILES")
  [ -n "$A_NODE" ] && [ -n "$B_NODE" ] && [ -n "$C_NODE" ] || {
    echo "FATAL: tri profile has an empty node for '$OP'" >&2; exit 2; }
  [ "$A_NODE" = "$B_NODE" ] || {
    echo "FATAL: A/B must be the identical semantic pytest node for '$OP'" >&2; exit 2; }
  [ "$B_FLAGS" = "$C_FLAGS" ] || {
    echo "FATAL: B/C baseline flags differ for '$OP'" >&2; exit 2; }
  A_VARIANT=$(awk -F'\t' -v o="$OP" '$1==o{print $2; exit}' "$TRI_IDMAP")
  A_SHA=$(awk -F'\t' -v o="$OP" '$1==o{print $3; exit}' "$TRI_IDMAP")
  B_VARIANT=$(awk -F'\t' -v o="$OP" '$1==o{print $4; exit}' "$TRI_IDMAP")
  B_SHA=$(awk -F'\t' -v o="$OP" '$1==o{print $5; exit}' "$TRI_IDMAP")
  C_VARIANT=$(awk -F'\t' -v o="$OP" '$1==o{print $6; exit}' "$TRI_IDMAP")
  C_SHA=$(awk -F'\t' -v o="$OP" '$1==o{print $7; exit}' "$TRI_IDMAP")
fi

# Precedence: explicit env > table row > built-in binarypow default.  A table
# that has no row for this op does NOT erase a default -- pass an op the table
# does not know and you get the named refusal below, not a silent binarypow.
if [ "$TRI_MODE" -eq 0 ] && [ "$OP" = binarypow ]; then
  SEM="${SEM:-$DEF_SEM}" ; HAND="${HAND:-$DEF_HAND}"
  SEM_VARIANT="${SEM_VARIANT:-$DEF_SEM_VARIANT}"
  HAND_VARIANT="${HAND_VARIANT:-$DEF_HAND_VARIANT}"
fi
if [ "$TRI_MODE" -eq 0 ] && [ -n "${OPS_TSV:-}" ]; then
  _s=$(awk -F'\t' -v o="$OP" '$1==o{print $2}' "$OPS_TSV")
  _h=$(awk -F'\t' -v o="$OP" '$1==o{print $3}' "$OPS_TSV")
  [ -n "$_s" ] && SEM=$_s
  [ -n "$_h" ] && HAND=$_h
fi
if [ "$TRI_MODE" -eq 0 ] && [ -n "${IDMAP:-}" ]; then
  # same row layout the streamers read: op, sem_variant, sem_sha, hand_variant, hand_sha
  _sv=$(awk -F'\t' -v o="$OP" '$1==o{print $2}' "$IDMAP")
  _ss=$(awk -F'\t' -v o="$OP" '$1==o{print $3}' "$IDMAP")
  _hv=$(awk -F'\t' -v o="$OP" '$1==o{print $4}' "$IDMAP")
  _hs=$(awk -F'\t' -v o="$OP" '$1==o{print $5}' "$IDMAP")
  [ -n "$_sv" ] && SEM_VARIANT=$_sv
  [ -n "$_hv" ] && HAND_VARIANT=$_hv
fi
if [ "$TRI_MODE" -eq 0 ]; then
  [ -n "$SEM" ] && [ -n "$HAND" ] || {
    echo "FATAL: no sem/hand nodes for '$OP' — set SEM and HAND, or pass OPS_TSV" >&2; exit 2; }
  [ -n "$SEM_VARIANT" ] && [ -n "$HAND_VARIANT" ] || {
    echo "FATAL: no build variants for '$OP' — set SEM_VARIANT and HAND_VARIANT, or pass IDMAP" >&2; exit 2; }
fi

# `--compile-consumer` recomputes the build key before loading a staged ELF.
# The producer's exact flags therefore select the binary; they are not merely
# provenance. Resolve them once and let the streamer's environment carry them
# to pytest on every slice.
if [ "$TRI_MODE" -eq 0 ] && [ -n "${FLAGS_TSV:-}" ]; then
  [ -f "$FLAGS_TSV" ] || { echo "FATAL: FLAGS_TSV does not exist: $FLAGS_TSV" >&2; exit 2; }
  _flag_rows=$(awk -F'\t' -v o="$OP" '$1==o{n++} END{print n+0}' "$FLAGS_TSV")
  [ "$_flag_rows" -eq 1 ] || {
    echo "FATAL: FLAGS_TSV needs exactly one row for '$OP', found $_flag_rows" >&2; exit 2; }
  TT_LLK_EXTRA_COMPILER_OPTIONS=$(awk -F'\t' -v o="$OP" \
    '$1==o{sub(/^[^\t]*\t/, ""); print; exit}' "$FLAGS_TSV")
  export TT_LLK_EXTRA_COMPILER_OPTIONS
fi

mkdir -p "$OUT" || { echo "FATAL: cannot create OUT=$OUT" >&2; exit 2; }
echo "HOST=$(hostname) OP=$OP SWEEP=$SWEEP NPAR=$NPAR BAND_BITS=$BAND_BITS SPACE=$SPACE $(date -u +%H:%M:%SZ)" \
  | tee "$OUT/DRIVER.log"

# ---- object-identity gate, ONCE --------------------------------------------
# sem != hand .text, both non-empty.  Done here rather than only per-slice so a
# mis-built pair costs one gate instead of 32 slice failures.  Each slice still
# re-gates through --idmap when one is supplied.
# The per-slice gate hashes ONE source subtree (binary_stream_sweep.py
# --idmap-source, default sfpu_binary_test.cpp; fp32_stream_sweep.py hardcodes
# eltwise_unary_sfpu_test.cpp).  An unscoped `find | head -1` can pick a
# same-variant ELF from a DIFFERENT subtree, so the driver would certify one
# file and all 32 slices would then refuse a different one.  Look in the
# sweep's own subtree first, fall back to the whole tree, and sort so the
# choice is reproducible rather than filesystem-order.
case "$SWEEP" in
  binary) IDMAP_SOURCE="${IDMAP_SOURCE:-sfpu_binary_test.cpp}" ;;
  fp32)   IDMAP_SOURCE="${IDMAP_SOURCE:-eltwise_unary_sfpu_test.cpp}" ;;
esac
_find_elf() {  # $1 = variant hash
  find "$BUILD/tt-llk-build/sources/$IDMAP_SOURCE" -path "*${1}/elf/math.elf" 2>/dev/null \
    | LC_ALL=C sort | head -1
}
_find_elf_anywhere() {
  find "$BUILD/tt-llk-build/sources" -path "*${1}/elf/math.elf" 2>/dev/null \
    | LC_ALL=C sort | head -1
}
if [ "$TRI_MODE" -eq 1 ]; then
  OBJ_A=$(_find_elf "$A_VARIANT"); [ -n "$OBJ_A" ] || OBJ_A=$(_find_elf_anywhere "$A_VARIANT")
  OBJ_B=$(_find_elf "$B_VARIANT"); [ -n "$OBJ_B" ] || OBJ_B=$(_find_elf_anywhere "$B_VARIANT")
  OBJ_C=$(_find_elf "$C_VARIANT"); [ -n "$OBJ_C" ] || OBJ_C=$(_find_elf_anywhere "$C_VARIANT")
  sha_a=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_A" 2>/dev/null)
  sha_b=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_B" 2>/dev/null)
  sha_c=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_C" 2>/dev/null)
  echo "IDGATE a_text=$sha_a b_text=$sha_b c_text=$sha_c" | tee -a "$OUT/DRIVER.log"
  if [ -z "$sha_a" ] || [ -z "$sha_b" ] || [ -z "$sha_c" ] \
     || [ "$sha_a" != "$A_SHA" ] || [ "$sha_b" != "$B_SHA" ] || [ "$sha_c" != "$C_SHA" ]; then
    echo "OP=$OP VERDICT=REFUSED-IDENTITY(tri-text-mismatch-or-empty)" | tee "$OUT/$OP-VERDICT.txt"
    exit 1
  fi
else
  OBJ_SEM=$(_find_elf "$SEM_VARIANT");   [ -n "$OBJ_SEM" ]  || OBJ_SEM=$(_find_elf_anywhere "$SEM_VARIANT")
  OBJ_HAND=$(_find_elf "$HAND_VARIANT"); [ -n "$OBJ_HAND" ] || OBJ_HAND=$(_find_elf_anywhere "$HAND_VARIANT")
  sha_sem=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_SEM" 2>/dev/null)
  sha_hand=$("$VENV" "$TOOLS/elf_text_sha.py" "$OBJ_HAND" 2>/dev/null)
  echo "IDGATE sem_text=$sha_sem hand_text=$sha_hand" | tee -a "$OUT/DRIVER.log"
  if [ -z "$sha_sem" ] || [ -z "$sha_hand" ] || [ "$sha_sem" = "$sha_hand" ]; then
    echo "OP=$OP VERDICT=REFUSED-IDENTITY(sem==hand or empty)" | tee "$OUT/$OP-VERDICT.txt"
    exit 1
  fi
  if [ -n "${IDMAP:-}" ] && { [ "$sha_sem" != "${_ss:-}" ] || [ "$sha_hand" != "${_hs:-}" ]; }; then
    echo "OP=$OP VERDICT=REFUSED-IDENTITY(text-mismatch)" | tee "$OUT/$OP-VERDICT.txt"
    exit 1
  fi
fi

# Every resumable slice is bound to this exact object identity map.  Direct
# variant arguments get a generated one; legacy bare-SHA caches are refused by
# the streamer rather than silently adopted.
# Op-scoped: two ops sharing one OUT would otherwise clobber each other's
# single-row map and each other's slices would refuse on the wrong row.
ACTIVE_IDMAP="${TRI_IDMAP:-${IDMAP:-$OUT/IDENTITY-MAP-$OP.tsv}}"
if [ "$TRI_MODE" -eq 0 ] && [ -z "${IDMAP:-}" ]; then
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$OP" "$SEM_VARIANT" "$sha_sem" "$HAND_VARIANT" "$sha_hand" > "$ACTIVE_IDMAP"
fi

SLICE=$(( SPACE / NPAR ))
[ "$(( SLICE * NPAR ))" -eq "$(( SPACE ))" ] \
  || { echo "FATAL: SPACE=$SPACE is not divisible by NPAR=$NPAR — slices would not cover it" >&2; exit 2; }
# SLICE==0 (SPACE<NPAR) makes every slice a no-op whose streamer trivially
# reports covered==total==0, and the combiner's expected_per_slice is then 0
# too, so 32 empty slices used to combine to BIT-EXACT-ALL-INPUTS having
# checked nothing.  Refuse instead.
[ "$SLICE" -ge 1 ] \
  || { echo "FATAL: SLICE=0 (SPACE=$SPACE < NPAR=$NPAR) — every slice would be empty" >&2; exit 2; }
echo "SLICE=$SLICE inputs/chip" | tee -a "$OUT/DRIVER.log"

idmap_args=(--idmap "$ACTIVE_IDMAP")
# The golden tolerance leg rides this pass at no extra device cost; max ULP is
# diagnostic, not certified. GOLDEN=0 opts out.
golden_args=()
[ "${GOLDEN:-1}" = 1 ] && golden_args=(--golden "$OP")
if [ "$TRI_MODE" -eq 1 ]; then
  arm_args=(--selected-sem-node "$A_NODE" "--selected-flags=$A_FLAGS"
            --baseline-sem-node "$B_NODE" "--baseline-flags=$B_FLAGS"
            --baseline-hand-node "$C_NODE")
else
  arm_args=(--sem-node "$SEM" --hand-node "$HAND")
fi

# Stagger chip launches: 32 simultaneous cold torch/ttexalens imports off the
# NFS venv stampede the fileserver and get SIGINT-killed mid-import.  A few
# seconds apart spreads the one-time import so every chip's harness comes up.
# Streaming itself is device-bound and unaffected.
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/galaxy-shard.XXXXXX") \
  || { echo "FATAL: cannot create node-local run directory" >&2; exit 2; }
trap 'rm -rf -- "$RUN_ROOT"' EXIT
pids=()
chips=()
for k in $(seq 0 $((NPAR-1))); do
  RT="$RUN_ROOT/rt-$k"
  mkdir -p "$RT"
  cp -a "$BUILD/tt-llk-build" "$RT/"
  start=$(( k * SLICE ))
  sdir="$OUT/slice-$k"
  # A slice verdict left by an earlier run must never stand in for THIS run's
  # slice: if the chip dies now, the stale file would make the combiner count
  # its range as covered.  Band caches (sdir/bands) are provenance-bound and
  # stay, so a re-run still resumes rather than re-streaming.
  rm -f "$sdir/$OP-VERDICT.txt" "$sdir/$OP-COMPILER-VERDICT.txt" \
        "$sdir/$OP-CORRECTNESS-VERDICT.txt"
  ( SFPU_WAIT_TIMEOUT="${SFPU_WAIT_TIMEOUT:-600}" \
    "$VENV" "$STREAMER" \
      --op "$OP" "${arm_args[@]}" \
      --farm "$PYDIR" --venv "$VENV" --llk-home "$LLK_HOME" --runner-temp "$RT" \
      --band-bits "$BAND_BITS" --chip "$k" --start-bit "$start" --total "$SLICE" \
      --out "$sdir" \
      ${idmap_args[@]+"${idmap_args[@]}"} \
      ${golden_args[@]+"${golden_args[@]}"} > "$OUT/slice-$k.log" 2>&1 ) &
  pids+=("$!")
  chips+=("$k")
  sleep "$STAGGER"
done
echo "launched $NPAR chip-slices $(date -u +%H:%M:%SZ)" | tee -a "$OUT/DRIVER.log"
shard_rc=0
failed_chips=""
# A slice's exit status carries TWO different things: "this process broke" and
# "my comparison did not come out equal" (the streamers `return 0 if all_equal
# and numeric_ok else 1`).  Treating every non-zero exit as a dead chip made
# DIVERGENT structurally unreachable: each diverging slice landed in
# failed_chips, so the combiner's `not all_equal and not invalid_list` branch
# could never fire and a real, fully-covered divergence was reported INCOMPLETE.
# Same for a GOLDEN=1 numeric-gate failure, which voided an established
# bit-exactness verdict.  Distinguish them by what the slice LEFT BEHIND: a
# decided verdict of its own for THIS run means it did its work.  Stale verdict
# files were removed before launch, so any file here was written by this run, and
# the combiner still re-checks its op/start/total/covered before believing it.
_slice_decided() {  # $1 = chip index
  local v="$OUT/slice-$1/$OP-VERDICT.txt"
  [ -s "$v" ] && grep -qE \
    'VERDICT=(BIT-EXACT-ALL-INPUTS|BIT-EXACT-PARTIAL-[0-9]+-OF-[^[:space:]]+|DIVERGENT)' "$v"
}
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    if _slice_decided "${chips[$i]}"; then
      echo "slice ${chips[$i]}: non-zero exit with a decided verdict (divergence" \
           "and/or numeric gate), not a dead chip" | tee -a "$OUT/DRIVER.log"
    else
      shard_rc=1
      failed_chips="${failed_chips:+$failed_chips,}${chips[$i]}"
    fi
  fi
done
echo "all slices done $(date -u +%H:%M:%SZ) failed_chips=[${failed_chips}]" | tee -a "$OUT/DRIVER.log"

# ---- combine ----
# --failed-chips names the dead slices so the combiner marks their ranges
# invalid by identity, instead of leaning on one shard_rc boolean that a
# hand re-run of galaxy_combine.py would not reproduce.
combine_args=("$OUT" "$NPAR" "$SPACE" "$OP" "${GOLDEN:-1}" "$shard_rc"
              --full-space "$FULL_SPACE")
[ -n "$failed_chips" ] && combine_args+=(--failed-chips "$failed_chips")
"$VENV" "$TOOLS/galaxy_combine.py" "${combine_args[@]}" | tee "$OUT/$OP-VERDICT.txt"
combine_rc=${PIPESTATUS[0]}
compiler_rc=0
if [ "$TRI_MODE" -eq 1 ]; then
  compiler_args=("$OUT" "$NPAR" "$SPACE" "$OP" 0 "$shard_rc"
                 --full-space "$FULL_SPACE" --verdict-suffix=-COMPILER-VERDICT.txt
                 --require-full-space)
  [ -n "$failed_chips" ] && compiler_args+=(--failed-chips "$failed_chips")
  "$VENV" "$TOOLS/galaxy_combine.py" "${compiler_args[@]}" \
    | tee "$OUT/$OP-COMPILER-VERDICT.txt"
  compiler_rc=${PIPESTATUS[0]}
fi
if [ "${GOLDEN:-1}" = 1 ]; then
  # Equivalence remains in OP-VERDICT.txt.  Numeric semantic-uplift admission
  # is independent: fold all sidecars across the complete population before
  # comparing per-class maxima.  A divergent semantic/hand pair may pass this
  # gate; missing/incomplete oracle evidence may not.
  "$VENV" "$TOOLS/galaxy_numeric_admission.py" "$OUT" "$OP" \
    --out-prefix "$OUT/$OP-NUMERIC-ADMISSION" \
    > "$OUT/$OP-NUMERIC-ADMISSION.log"
  numeric_rc=$?
  numeric_status=$(awk -F'\t' 'NR==2 {print $7}' \
    "$OUT/$OP-NUMERIC-ADMISSION.tsv")
  compiler_status=NOT_RUN
  if [ "$TRI_MODE" -eq 1 ]; then
    compiler_status=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^VERDICT=/){sub(/^VERDICT=/,"",$i); print $i; exit}}' \
      "$OUT/$OP-COMPILER-VERDICT.txt")
  fi
  echo "$(cat "$OUT/$OP-VERDICT.txt") compiler_gate=$compiler_status numeric_admission=$numeric_status"
  final_rc=1
  [ "$shard_rc" -eq 0 ] && [ "$compiler_rc" -eq 0 ] && [ "$numeric_rc" -eq 0 ] \
    && final_rc=0
  if [ "$TRI_MODE" -eq 1 ]; then
    semantic_status=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^VERDICT=/){sub(/^VERDICT=/,"",$i); print $i; exit}}' \
      "$OUT/$OP-VERDICT.txt")
    covered=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^covered=/){sub(/^covered=/,"",$i); print $i; exit}}' \
      "$OUT/$OP-VERDICT.txt")
    deployment=FAIL; [ "$final_rc" -eq 0 ] && deployment=PASS
    printf 'OP=%s VERDICT=%s compiler_gate=%s semantic_equivalence=%s numeric_admission=%s covered=%s full_space=%s\n' \
      "$OP" "$deployment" "$compiler_status" "$semantic_status" "$numeric_status" \
      "$covered" "$FULL_SPACE" > "$OUT/$OP-DEPLOYMENT-VERDICT.txt"
    cat "$OUT/$OP-DEPLOYMENT-VERDICT.txt"
  fi
  exit "$final_rc"
else
  if [ "$TRI_MODE" -eq 1 ]; then
    [ "$shard_rc" -eq 0 ] && [ "$compiler_rc" -eq 0 ]; exit $?
  else
    [ "$shard_rc" -eq 0 ] && [ "$combine_rc" -eq 0 ]; exit $?
  fi
fi
