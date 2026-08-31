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
#   JO_TESTS     tt-llk tests dir of the pin-48 worktree
#   JO_TIMEOUT   z3 per-query timeout seconds (default 3600)
set -euo pipefail

ROW="$1"; OUT="$2"
SEM_NODE="${3:-}"; HAND_NODE="${4:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS="${JO_TESTS:-$(cd "$HERE/../.." && pwd)}"
SIM="${JO_SIM:?set JO_SIM to the instrumented libttsim.so}"
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
    ( cd "$TESTS" && \
      RUNNER_TEMP="$rt" CHIP_ARCH=blackhole TT_METAL_SIMULATOR="$SIM" \
      TTSIM_TRACE_SFPU_STREAM=1 LLK_HOME="$(dirname "$TESTS")" \
      .venv/bin/python -m pytest -q -s --run-simulator "python_tests/$node" \
      > "$OUT/trace-$ROW-$leg.log" 2>&1 ) || {
        echo "ERROR: $leg leg pytest failed; tail:" >&2
        tail -5 "$OUT/trace-$ROW-$leg.log" >&2
        return 1
    }
    grep -q "SFPUJO I" "$OUT/trace-$ROW-$leg.log" || {
        echo "ERROR: $leg leg produced no SFPU stream" >&2; return 1; }
}

echo "== $ROW sem leg: $SEM_NODE"
run_leg sem "$SEM_NODE"
echo "== $ROW hand leg: $HAND_NODE"
run_leg hand "$HAND_NODE"

python3 "$HERE/formal_equiv.py" --row "$ROW" \
    --trace-sem "$OUT/trace-$ROW-sem.log" \
    --trace-hand "$OUT/trace-$ROW-hand.log" \
    --out "$OUT" --timeout "${JO_TIMEOUT:-3600}"
