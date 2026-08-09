#!/bin/bash
# =============================================================================
# bounds_sweep.sh - Phase-diagram sweep of SFPU optimization APPLICABILITY BOUNDS.
#
# For each grid cell (variant, degree D, segments N, parity), generate a
# specialized kernel + benchmark, run it, and record:
#   COMPILED?  - a JIT register-ICE => the register-pressure frontier (a DATA POINT)
#   CORRECT?   - matches the exact reference
#   us         - device kernel time
# Output: bounds_results.csv  ->  phase_diagram.py renders the diagrams + bounds.
#
# DEVICE IS EXCLUSIVE: cells run strictly one at a time.
#
# Grid is configurable via env: VARIANTS, DEGREES, SEGMENTS, PARITIES.
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
EX="$REPO/tt_metal/programming_examples/generic_lut_activation_embedded"
BIN="$REPO/build_Release/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc"
SYSPY=/usr/bin/python3
source "$EX/profiler_helpers.sh"
OUT="$HERE/bounds_results.csv"

VARIANTS="${VARIANTS:-cascade dual blend blend_dual}"
DEGREES="${DEGREES:-2 4 6 8 12 16}"
SEGMENTS="${SEGMENTS:-4 16 64}"
PARITIES="${PARITIES:-0 1}"

printf "variant,degree,segments,parity,status,us,max_abs_err\n" > "$OUT"
n=0
for variant in $VARIANTS; do
 for par in $PARITIES; do
  for D in $DEGREES; do
   for N in $SEGMENTS; do
    n=$((n+1))
    "$SYSPY" "$HERE/gen_sweep.py" "$variant" "$D" "$N" "$par" >/dev/null 2>&1 || {
        printf "%s,%s,%s,%s,GEN_FAIL,,\n" "$variant" "$D" "$N" "$par" >> "$OUT"; continue; }
    # build host (always succeeds); JIT happens at run -> that's where ICE shows up
    ninja -C "$REPO/build_Release" programming_examples_generic_lut_activation_embedded_adhoc >/dev/null 2>&1
    pd="/tmp/bounds_${variant}_${D}_${N}_${par}"; rm -rf "$pd"; mkdir -p "$pd"
    dump="/tmp/bounds_dump_${variant}_${D}_${N}_${par}.csv"
    run_log="/tmp/bounds_run_${variant}_${D}_${N}_${par}.log"
    DUMP_OUTPUT_CSV="$dump" TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$pd" \
      "$BIN" --activation silu --precision fp32 --range-min -8 --range-max 8 --tiles 256 \
      > "$run_log" 2>&1
    rc=$?
    if grep -qiE "reload|maximum number of generated reload|internal compiler error" "$run_log"; then
        status="ICE"; us=""; err=""
    elif [[ $rc -ne 0 || ! -s "$dump" ]]; then
        status="RUN_FAIL"; us=""; err=""
    else
        us="$(extract_profiler_compute_time "$pd/.logs/profile_log_device.csv" "$EX" 2>/dev/null)"
        read -r _fma _insns err <<<"$("$SYSPY" "$HERE/lib/score.py" "p_${variant}" "$dump" "" 2>/dev/null)"
        # tolerance gate
        bad="$("$SYSPY" -c "import sys;print(1 if float(sys.argv[1])>1e-2 else 0)" "${err:-9.99}" 2>/dev/null || echo 1)"
        status="OK"; [[ "$bad" == "1" ]] && status="WRONG"
    fi
    printf "%s,%s,%s,%s,%s,%s,%s\n" "$variant" "$D" "$N" "$par" "$status" "${us:-}" "${err:-}" >> "$OUT"
    echo "  [$n] $variant D=$D N=$N par=$par -> $status ${us:+(${us}us)}"
   done
  done
 done
done
echo "BOUNDS_DONE -> $OUT"
