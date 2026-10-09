#!/usr/bin/env bash
# CI driver: ALT_RUNS processes of alt.py under the tracy wrapper (one JIT cache), then the summary.
set +e +o pipefail
D=/work/ci_alt; O=/tmp/alt; mkdir -p $O; export TT_METAL_CACHE=/tmp/alt_cache PYTHONPATH=/work:${PYTHONPATH:-}
for r in $(seq 1 ${ALT_RUNS:-6}); do
  rm -rf $O/out_$r; mkdir -p $O/praw_$r
  TT_METAL_PROFILER_DIR=$O/praw_$r timeout -s INT -k 30 1500 python3 -m tracy -r -p --no-web-server -o $O/out_$r $D/alt.py $r ${ALT_CASES:-} > $O/log_$r.txt 2>&1
  echo "== run $r rc=$? $(grep -c '^CASE' $O/log_$r.txt) cases $(date -u +%T)"; grep -E "Traceback|TT_THROW|TT_FATAL|Error" $O/log_$r.txt | head -3 | cut -c1-300
done
python3 $D/alt_sum.py $O
