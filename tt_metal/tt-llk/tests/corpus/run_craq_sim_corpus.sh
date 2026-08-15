#!/usr/bin/env bash
# Run the tiered F1/F2 LLK corpus through craq-sim's device-cycle extractor.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$HERE/../../../.." && pwd)}"
CRAQ_SIM_ROOT="${CRAQ_SIM_ROOT:-/localdev/nkapre/craq-sim}"
ARCH=bh
TIER=1
SAMPLE=1
LIST_ONLY=0
RUN_ROOT=""

usage() {
    cat <<'EOF'
Usage: run_craq_sim_corpus.sh [--arch bh|wh] [--tier 1..4] [--sample N] [--run-root DIR] [--list]

Selects rows from f1_candidates.tsv, then invokes craq-sim's llk-sim-perf.sh.
N=0 selects every parametrized nodeid in each selected functional module.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --tier) TIER="$2"; shift 2 ;;
        --sample) SAMPLE="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$ARCH" in
    bh) SIM_ARCH=blackhole; SHORT_ARCH=bh ;;
    wh) SIM_ARCH=wormhole; SHORT_ARCH=wh ;;
    *) echo "ERROR: --arch must be bh or wh" >&2; exit 2 ;;
esac
case "$TIER" in 1|2|3|4) ;; *) echo "ERROR: --tier must be 1..4" >&2; exit 2 ;; esac
case "$SAMPLE" in ''|*[!0-9]*) echo "ERROR: --sample must be a non-negative integer" >&2; exit 2 ;; esac

MANIFEST="$HERE/f1_candidates.tsv"
SIM_RUNNER="$CRAQ_SIM_ROOT/scripts/perf/llk-sim-perf.sh"
HARNESS="$TT_METAL_HOME/tt_metal/tt-llk/tests"
[ -f "$MANIFEST" ] || { echo "ERROR: missing manifest: $MANIFEST" >&2; exit 2; }
[ -x "$SIM_RUNNER" ] || { echo "ERROR: missing craq-sim runner: $SIM_RUNNER" >&2; exit 2; }
[ -d "$HARNESS/python_tests" ] || { echo "ERROR: missing tt-llk harness: $HARNESS" >&2; exit 2; }

if [ -z "$RUN_ROOT" ]; then
    RUN_ROOT="$TT_METAL_HOME/tt_metal/tt-llk/tests/corpus/runs/$(date -u +%Y%m%dT%H%M%SZ)-$ARCH-tier$TIER"
fi
mkdir -p "$RUN_ROOT"
MODULES="$RUN_ROOT/modules.txt"
awk -F '\t' -v arch="$ARCH" -v tier="$TIER" '
    /^#/ { next }
    NF >= 7 && ($2 + 0) <= (tier + 0) && ("," $4 "," ~ "," arch ",") { print $7 }
' "$MANIFEST" | tr ',' '\n' | sort -u > "$MODULES"
[ -s "$MODULES" ] || { echo "ERROR: manifest selected no modules" >&2; exit 2; }

{
    printf 'tt_metal_head\t'; git -C "$TT_METAL_HOME" rev-parse HEAD
    printf 'craq_sim_head\t'; git -C "$CRAQ_SIM_ROOT" rev-parse HEAD
    printf 'arch\t%s\ntier\t%s\nsample\t%s\n' "$ARCH" "$TIER" "$SAMPLE"
} > "$RUN_ROOT/provenance.tsv"
cp "$MANIFEST" "$RUN_ROOT/f1_candidates.tsv"

if [ "$LIST_ONLY" = 1 ]; then
    cat "$MODULES"
    exit 0
fi

[ -d "$HARNESS/.venv" ] || { echo "ERROR: missing harness virtualenv: $HARNESS/.venv" >&2; exit 2; }

args=(--sample "$SAMPLE" --run-root "$RUN_ROOT/sim")
while IFS= read -r module; do args+=(--module "$module"); done < "$MODULES"
SIM_ARCH="$SIM_ARCH" SHORT_ARCH="$SHORT_ARCH" TT_METAL_HOME="$TT_METAL_HOME" HARNESS="$HARNESS" \
    "$SIM_RUNNER" "${args[@]}"
