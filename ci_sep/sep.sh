#!/usr/bin/env bash
# CI driver: SEP_ROUNDS rounds; in each round every case and form runs in its own process (orders rotated by round), all
# under one JIT cache; then the summary.
set +e +o pipefail
D=/work/ci_sep; O=/tmp/sep; mkdir -p $O; export TT_METAL_CACHE=/tmp/sep_cache PYTHONPATH=/work:${PYTHONPATH:-}
FORMS=(BA1 BOFF1 BA2 BOFF2); CASES=(acc perf)
for r in $(seq 1 ${SEP_ROUNDS:-3}); do
  for ci in 0 1; do case=${CASES[$(( (ci + r) % 2 ))]}
    for fi in 0 1 2 3; do f=${FORMS[$(( (fi + r) % 4 ))]}
      rm -rf $O/out_${r}_${case}_${f}; mkdir -p $O/praw_${r}_${case}_${f}
      RC_ALT=$f TT_METAL_PROFILER_DIR=$O/praw_${r}_${case}_${f} timeout -s INT -k 30 900 python3 -m tracy -r -p --no-web-server -o $O/out_${r}_${case}_${f} $D/sep.py $case $f > $O/log_${r}_${case}_${f}.txt 2>&1
      echo "== round $r $case $f rc=$? $(grep -c '^CASE' $O/log_${r}_${case}_${f}.txt) ok $(date -u +%T)"; grep -E "Traceback|TT_THROW|TT_FATAL" $O/log_${r}_${case}_${f}.txt | head -2 | cut -c1-300
    done
  done
done
python3 $D/sep_sum.py $O BA1,BA2,BOFF1,BOFF2
