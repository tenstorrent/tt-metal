#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# laneJO driver: run one board row's sem+hand legs on the instrumented
# pinned simulator (TTSIM_TRACE_SFPU_STREAM), then prove/refute bit-exact
# equivalence with formal_equiv.py.
#
# Usage: formal_equiv_row.sh <row> <out_dir> [sem_node] [hand_node]
#   Nodes default to the sweep_2x2_ops.tsv sem/hand FUNCTIONAL selectors.
# Env:
#   JO_SIM       instrumented libttsim.so (soc_descriptor.yaml beside it)
#   JO_TESTS     tt-llk tests dir of the candidate worktree
#   JO_TIMEOUT   z3 per-query timeout seconds (default 3600)
#   JO_FLAGS     exact candidate compiler flags (recorded; empty means defaults)
set -euo pipefail

ROW="$1"; OUT="$2"
SEM_NODE="${3:-}"; HAND_NODE="${4:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS="${JO_TESTS:-$(cd "$HERE/../.." && pwd)}"
SIM="${JO_SIM:?set JO_SIM to the instrumented libttsim.so}"
[ -x "$TESTS/.venv/bin/python" ] || {
    echo "REFUSED: harness venv python is missing" >&2; exit 3; }
"$TESTS/.venv/bin/python" -c 'import elftools, z3' || {
    echo "REFUSED: harness venv needs pyelftools and z3" >&2; exit 3; }

# The symbolic executor is transcribed from this exact instrumented simulator.
# A different simulator is not a current-toolchain experiment; it is an
# unvalidated semantics change and must use a separately reviewed executor.
EXPECTED_SIM_SHA=ba23c3f169126425998b53b0202a10a81e35fba0692ed4eca5964f073ec31113
SIM_SHA="$(sha256sum "$SIM" | cut -d' ' -f1)"
[ "$SIM_SHA" = "$EXPECTED_SIM_SHA" ] || {
    echo "REFUSED: JO simulator sha $SIM_SHA != transcribed $EXPECTED_SIM_SHA" >&2; exit 3; }
[ -f "$(dirname "$SIM")/soc_descriptor.yaml" ] || {
    echo "REFUSED: soc_descriptor.yaml missing beside JO simulator" >&2; exit 3; }

mapfile -t CC1S < <(find "$TESTS/sfpi/compiler/libexec/gcc/riscv-tt-elf" \
    -name cc1plus -type f 2>/dev/null | grep -v '/\.pin-backup/')
[ "${#CC1S[@]}" -eq 1 ] || {
    echo "REFUSED: expected one active cc1plus, found ${#CC1S[@]}" >&2; exit 3; }
CC1_SHA="$(sha256sum "${CC1S[0]}" | cut -d' ' -f1)"
LLK_ROOT="$(cd "$TESTS/.." && pwd)"
SOURCE_HEAD="$(git -C "$TESTS" rev-parse HEAD)"
SOURCE_ROOT="$(git -C "$TESTS" rev-parse --show-toplevel)"
OUT_ABS="$(realpath -m "$OUT")"
case "$OUT_ABS/" in
    "$SOURCE_ROOT/"*)
        echo "REFUSED: evidence output must be outside the source worktree" >&2; exit 3 ;;
esac
git -C "$LLK_ROOT" diff --quiet HEAD -- . || {
    echo "REFUSED: tracked tt-llk source is dirty" >&2; exit 3; }
[ -z "$(git -C "$LLK_ROOT" ls-files --others --exclude-standard -- .)" ] || {
    echo "REFUSED: untracked tt-llk source is present" >&2; exit 3; }
FLAGS="${JO_FLAGS:-}"
FLAGS_SHA="$(printf '%s' "$FLAGS" | sha256sum | cut -d' ' -f1)"
mkdir -p "$OUT"

if [ -z "$SEM_NODE" ] || [ -z "$HAND_NODE" ]; then
    line="$(awk -F'\t' -v r="$ROW" '$1==r {print; exit}' "$HERE/../sweep_2x2_ops.tsv")"
    [ -n "$line" ] || { echo "ERROR: row $ROW not in sweep_2x2_ops.tsv" >&2; exit 2; }
    [ -n "$SEM_NODE" ] || SEM_NODE="$(printf '%s' "$line" | cut -f6)"
    [ -n "$HAND_NODE" ] || HAND_NODE="$(printf '%s' "$line" | cut -f8)"
fi
[ -n "$SEM_NODE" ] || { echo "ERROR: no sem node for $ROW" >&2; exit 2; }
if [ -z "$HAND_NODE" ]; then
    echo "REFUSED: row $ROW has no distinct hand leg (kind=semantic)" | tee "$OUT/$ROW-refused.txt"
    exit 3
fi

run_leg() { # leg node
    local leg="$1" node="$2"
    local rt="$OUT/rt-$ROW-$leg"
    rm -rf "$rt"; mkdir -p "$rt"
    rm -f "$OUT/trace-$ROW-$leg.log"
    ( cd "$TESTS" && \
      RUNNER_TEMP="$rt" CHIP_ARCH=blackhole TT_METAL_SIMULATOR="$SIM" \
      TT_LLK_EXTRA_COMPILER_OPTIONS="$FLAGS" \
      TTSIM_TRACE_SFPU_STREAM=1 TTSIM_TRACE_SFPU_FILE="$OUT/trace-$ROW-$leg.log" \
      LLK_HOME="$(dirname "$TESTS")" \
      .venv/bin/python -m pytest -q -s --run-simulator "python_tests/$node" \
      > "$OUT/pytest-$ROW-$leg.log" 2>&1 ) || {
        echo "ERROR: $leg leg pytest failed; tail:" >&2
        tail -5 "$OUT/pytest-$ROW-$leg.log" >&2
        return 1
    }
    grep -q "SFPUJO I" "$OUT/trace-$ROW-$leg.log" || {
        echo "ERROR: $leg leg produced no SFPU stream" >&2; return 1; }
}

echo "== $ROW sem leg: $SEM_NODE"
run_leg sem "$SEM_NODE"
echo "== $ROW hand leg: $HAND_NODE"
run_leg hand "$HAND_NODE"

text_identity() { # leg
    local leg="$1"
    local rt="$OUT/rt-$ROW-$leg"
    local -a elfs
    mapfile -t elfs < <(find "$rt/tt-llk-build/sources" -path '*/elf/math.elf' -type f)
    [ "${#elfs[@]}" -eq 1 ] || {
        echo "REFUSED: $leg expected one math.elf, found ${#elfs[@]}" >&2; return 1; }
    "$TESTS/.venv/bin/python" "$HERE/elf_text_sha.py" "${elfs[0]}"
}
SEM_TEXT_SHA="$(text_identity sem)"
HAND_TEXT_SHA="$(text_identity hand)"
[ -n "$SEM_TEXT_SHA" ] && [ -n "$HAND_TEXT_SHA" ] && \
    [ "$SEM_TEXT_SHA" != "$HAND_TEXT_SHA" ] || {
    echo "REFUSED: semantic and hand math.elf .text identities are empty/equal" >&2; exit 3; }
SEM_TRACE_SHA="$(sha256sum "$OUT/trace-$ROW-sem.log" | cut -d' ' -f1)"
HAND_TRACE_SHA="$(sha256sum "$OUT/trace-$ROW-hand.log" | cut -d' ' -f1)"

cat > "$OUT/CURRENT-CANDIDATE-PROVENANCE.tsv" <<EOF
field	value
evidence_class	CURRENT-CANDIDATE-NOT-PIN59
source_head	$SOURCE_HEAD
cc1plus_path	${CC1S[0]}
cc1plus_sha256	$CC1_SHA
jo_sim_path	$SIM
jo_sim_sha256	$SIM_SHA
flags_sha256	$FLAGS_SHA
flags	$FLAGS
sem_node	$SEM_NODE
hand_node	$HAND_NODE
sem_text_sha256	$SEM_TEXT_SHA
hand_text_sha256	$HAND_TEXT_SHA
sem_trace_sha256	$SEM_TRACE_SHA
hand_trace_sha256	$HAND_TRACE_SHA
EOF

"$TESTS/.venv/bin/python" "$HERE/formal_equiv.py" --row "$ROW" \
    --trace-sem "$OUT/trace-$ROW-sem.log" \
    --trace-hand "$OUT/trace-$ROW-hand.log" \
    --out "$OUT" --timeout "${JO_TIMEOUT:-3600}"
