#!/usr/bin/env bash
# Parameterised end-to-end prefill run. One runner for every e2e measurement in this
# report -- the per-experiment copies this replaces are how a `local A=$1` bug and a
# false-positive witness got into earlier results.
#
#   ./run_e2e.sh --chunk 2048 [--ctx 256k] [--out DIR] [--label NAME] [FLAG=VAL ...]
#
# Every flag you pass is applied as an env var AND recorded in the log header, so a log
# is self-describing. Witness checks are enforced for the flags that have one.
set -u
usage () { sed -n '2,12p' "$0"; exit 1; }

T=${TT_METAL_HOME:-/data/kmabee/tt-metal-2}
CHUNK=""; CTX=256k; OUT="."; LABEL=""
declare -a FLAGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --chunk) CHUNK=$2; shift 2;;
    --ctx)   CTX=$2;   shift 2;;
    --out)   OUT=$2;   shift 2;;
    --label) LABEL=$2; shift 2;;
    -h|--help) usage;;
    *=*)     FLAGS+=("$1"); shift;;
    *) echo "unknown arg: $1"; usage;;
  esac
done
[ -n "$CHUNK" ] || usage
[ -n "$LABEL" ] || LABEL="c${CHUNK}"

# GEMMA4_PREFILL_L1_ACT is DEAD on this base. The rebase onto mmanzoor/svuckovic
# inverted the knob: gemma4_d_p now places short-lived prefill activations in L1 by
# DEFAULT and the opt-out is GEMMA4_ACTIVATIONS_DRAM_ONLY=1. The old name survives only
# in models/demos/gemma4/ (a different model) and as a stale comment. Passing it is a
# silent no-op that produced two "different" configs with bit-identical numbers, so
# refuse it rather than measure it.
if printf '%s\n' "${FLAGS[@]:-}" | grep -q '^GEMMA4_PREFILL_L1_ACT='; then
  echo "ERROR: GEMMA4_PREFILL_L1_ACT is a no-op on this base (gemma4_d_p)." >&2
  echo "       L1 activations are ON by default; use GEMMA4_ACTIVATIONS_DRAM_ONLY=1 to turn them OFF." >&2
  exit 2
fi

mkdir -p "$OUT"
LOG="$OUT/${LABEL}.log"
DEMO=models/demos/gemma4_d_p/demo/text_demo_prefill.py
cd "$T" || exit 1

{
  echo "# label     $LABEL"
  echo "# chunk     $CHUNK"
  echo "# ctx       $CTX"
  echo "# git       $(git -C "$T" rev-parse --short HEAD)"
  echo "# dirty     $(git -C "$T" status --porcelain | grep -cv '^??')"
  echo "# host      $(hostname)"
  echo "# flags     ${FLAGS[*]:-none}"
  echo "# started   $(date -Is)"
} > "$LOG"

env "${FLAGS[@]}" PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-3600} \
  "$T/python_env/bin/python3" -m pytest \
  "$DEMO::test_prefill_long_context_traced[blackhole-readback_final-ctx_${CTX}-chunk${CHUNK}-text-8x4]" \
  -sv >> "$LOG" 2>&1
RC=$?

# ---- witness checks: a flag that did not reach the model invalidates the run ----
# Anchored to PROGRAM output. Never match anything this script itself echoed: an earlier
# version grepped "async=[01]" and matched its own header, certifying a run that never
# started.
fail=0
check () { # flag_value_expected  regex_on_program_output  human_name
  local want=$1 re=$2 name=$3
  local got; got=$(grep -oE "$re" "$LOG" | tail -1)
  if [ "$want" = "1" ] && [ -z "$got" ]; then
    echo "  WITNESS_FAIL: $name requested but never engaged"; fail=1
  elif [ "$want" = "0" ] && [ -n "$got" ]; then
    echo "  WITNESS_FAIL: $name engaged when NOT requested"; fail=1
  fi
}
has () { printf '%s\n' "${FLAGS[@]:-}" | grep -q "^$1=1$" && echo 1 || echo 0; }
check "$(has GEMMA4_NORM_SHARD)"  "prefill norm: block-sharded"                 GEMMA4_NORM_SHARD
check "$(has GEMMA4_MLP_MM_CFG)"  "MLP explicit matmul cfg ENGAGED"             GEMMA4_MLP_MM_CFG
check "$(has GEMMA4_ATTN_MM_PC)"  "ATTN explicit mm cfg ENGAGED"                GEMMA4_ATTN_MM_PC
check "$(has GEMMA4_ACTIVATIONS_DRAM_ONLY)" "prefill activations: DRAM-only" GEMMA4_ACTIVATIONS_DRAM_ONLY

N=$(grep -c 'traced_perf\] chunk' "$LOG")
echo "# rc=$RC chunks=$N witness_fail=$fail finished=$(date -Is)" >> "$LOG"
echo "$LABEL: rc=$RC chunks=$N $([ $fail = 0 ] && echo 'witness OK' || echo 'WITNESS FAIL')"
[ "$RC" = 0 ] && [ "$fail" = 0 ] && [ "$N" -gt 0 ]
