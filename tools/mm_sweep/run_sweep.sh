#!/bin/bash
# minimal_matmul "main optimized baseline vs branch" sweep driver.
#
# Runs, per shape, BOTH:
#   baseline = pure unicast (== main): explicit block sweep + TT_MM_NO_LARGE_LEVERS (best block wins)
#   branch   = the production auto path (no config)
# under the tracy profiler, on the CURRENTLY-CHECKED-OUT BRANCH BUILD (no main checkout needed).
#
# Usage:
#   bash tools/mm_sweep/run_sweep.sh <shapes_file> <results_dir>
# Example:
#   source /home/cglagovich/bh_env.sh && source python_env/bin/activate      # Blackhole
#   bash tools/mm_sweep/run_sweep.sh tools/mm_sweep/shapes_big.txt /tmp/mm_bh_big
#   python tools/mm_sweep/parse_sweep.py /tmp/mm_bh_big tools/mm_sweep/shapes_big.txt /tmp/mm_bh_big/results.md
#
# Env knobs: FC_REPS (default 12), SWEEP_TIMEOUT seconds per (shape,mode) (default 900),
#            FC_BPCM/FC_BPCN/FC_KBS forwarded to the baseline block sweep.
set -u
SHAPES=${1:?usage: run_sweep.sh <shapes_file> <results_dir>}
OUT=${2:?usage: run_sweep.sh <shapes_file> <results_dir>}
T=tests/ttnn/nightly/unit_tests/operations/experimental/test_mm_repro.py::test_mm_repro
FC_REPS=${FC_REPS:-12}
SWEEP_TIMEOUT=${SWEEP_TIMEOUT:-900}
mkdir -p "$OUT"
echo "shapes=$SHAPES out=$OUT reps=$FC_REPS timeout=${SWEEP_TIMEOUT}s"
while read -r M K N; do
  [ -z "${M:-}" ] && continue
  case "$M" in \#*) continue;; esac
  tag="${M}x${K}x${N}"
  for mode in baseline branch; do
    ( export FL_M=$M FL_K=$K FL_N=$N MM_MODE=$mode FC_REPS=$FC_REPS
      timeout "$SWEEP_TIMEOUT" python -m tracy -p -r -o "$OUT/${mode}_${tag}" -m "pytest -q $T" \
        > "$OUT/${mode}_${tag}.log" 2>&1 )
    pcc=$(grep -oE 'pcc=[0-9.]+' "$OUT/${mode}_${tag}.log" | tail -1)
    echo "$tag $mode done ${pcc:-NO_PCC}"
  done
done < "$SHAPES"
echo "ALL_DONE"
