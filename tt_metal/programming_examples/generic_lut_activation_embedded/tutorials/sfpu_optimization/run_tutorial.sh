#!/bin/bash
# =============================================================================
# run_tutorial.sh — Build, profile, and score each rung of the SFPU
# optimization ladder. Reuses the adhoc host + Tracy profiler extractor.
#
# Usage: ./run_tutorial.sh [poly|rational|all]   (default: all)
#
# For each rung: regenerate bench -> swap rung kernel into the adhoc slot ->
# ninja the adhoc target -> run at a fixed shape under the device profiler
# (3x, take min) -> dump output -> score (correctness vs reference + static
# analysis) -> append to results.csv.
#
# Device is exclusive: rungs run strictly one at a time.
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
EX="$REPO/tt_metal/programming_examples/generic_lut_activation_embedded"
ADHOC_SLOT="$EX/kernels/compute/adhoc/adhoc.cpp"
BIN="$REPO/build_Release/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc"
SYSPY=/usr/bin/python3
SHAPE_TILES="${SHAPE_TILES:-256}"
OUT="$HERE/results.csv"
source "$EX/profiler_helpers.sh"

ACT="${1:-all}"
POLY="p0_naive p1_unrolled p2_dual p3_parity p4_adaptive p5_blend"
RAT="r0_naive r1_unrolled r2_interleaved r3_parity r4_deferred"
case "$ACT" in
  poly) RUNGS="$POLY" ;;
  rational) RUNGS="$RAT" ;;
  *) RUNGS="$POLY $RAT" ;;
esac

# Always regenerate the deterministic benchmark first.
"$SYSPY" "$HERE/gen_bench.py" >/dev/null

printf "rung,us,fma,sfpu_insns,max_abs_err,status\n" > "$OUT"

newest_trisc_obj() {
  ls -t "$HOME"/.cache/tt-metal-cache/*/kernels/adhoc/*/trisc1/*.o 2>/dev/null | head -1
}

for rung in $RUNGS; do
  src="$HERE/kernels/compute/${rung}.cpp"
  if [[ ! -f "$src" ]]; then
    printf "%s,,,,,MISSING_KERNEL\n" "$rung" >> "$OUT"; continue
  fi
  cp "$src" "$ADHOC_SLOT"
  if ! ninja -C "$REPO/build_Release" programming_examples_generic_lut_activation_embedded_adhoc >/dev/null 2>&1; then
    printf "%s,,,,,BUILD_FAIL\n" "$rung" >> "$OUT"; continue
  fi

  best="999999"; dump="/tmp/sfputut_${rung}.csv"
  for run in 1 2 3; do
    pd="/tmp/sfputut_prof_${rung}_${run}"; rm -rf "$pd"; mkdir -p "$pd"
    DUMPARG="/dev/null"; [[ "$run" -eq 1 ]] && DUMPARG="$dump"
    DUMP_OUTPUT_CSV="$DUMPARG" TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$pd" \
      "$BIN" --activation silu --precision fp32 \
      --range-min -8 --range-max 8 --tiles "$SHAPE_TILES" >/dev/null 2>&1
    t="$(extract_profiler_compute_time "$pd/.logs/profile_log_device.csv" "$EX" 2>/dev/null)"
    if [[ -n "$t" && "$t" != "0" ]] && (( $(echo "$t < $best" | bc -l 2>/dev/null || echo 0) )); then
      best="$t"
    fi
  done

  obj="$(newest_trisc_obj)"
  read -r fma insns err <<<"$("$SYSPY" "$HERE/lib/score.py" "$rung" "$dump" "$obj")"
  status="OK"
  [[ "$best" == "999999" ]] && status="NO_TIMING"
  # Use python for the threshold test: err is in scientific notation (e.g. 1.3e-03),
  # which `bc` cannot parse. Tolerance gate: max_abs_err < 1e-2.
  fail="$("$SYSPY" -c "import sys; print(1 if float(sys.argv[1])>1e-2 else 0)" "${err:-9.99}" 2>/dev/null || echo 1)"
  [[ "$fail" == "1" ]] && status="ACCURACY_FAIL"
  printf "%s,%s,%s,%s,%s,%s\n" "$rung" "$best" "$fma" "$insns" "$err" "$status" >> "$OUT"
  echo "  $rung: us=$best fma=$fma insns=$insns err=$err -> $status"
done

echo "results -> $OUT"
column -t -s, "$OUT"
