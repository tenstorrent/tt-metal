#!/usr/bin/env bash
# Traced denoise-only op-count profile. Env (device-profiler on, PROGRAM_SUPPORT_COUNT=3000,
# LTX_PROFILE_DENOISE_ONLY) comes from opt/env_prof_traced.yaml via the prewarm wrapper.
# The acid test below is the witness BOTTLENECK.md requires: rows carrying a non-empty METAL TRACE ID
# mean the profiler actually saw the traced denoise (the 1000-cap dropped all of them before).
set -uo pipefail
cd /home/smarton/tt-metal/.claude/worktrees/ltxperf-tip

CSV=generated/profiler/.logs/cpp_device_perf_report.csv
rm -f "$CSV" opt/prof_traced.log opt/traced_profile.csv

python_env/bin/python -m pytest \
  models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled \
  -k bh_4x8sp1tp0_ring -s -p no:cacheprovider --timeout=250 > opt/prof_traced.log 2>&1
echo "EXIT=$?"

if [ -s "$CSV" ]; then cp "$CSV" opt/traced_profile.csv; fi
echo "CSV_ROWS=$(wc -l < "$CSV" 2>/dev/null || echo 0)"
echo "CSV_SAVED_ROWS=$(wc -l < opt/traced_profile.csv 2>/dev/null || echo 0)"

# Acid test: per-op trace-ID witness + distinct replay sessions over the saved CSV.
awk -F',' '
  NR==1{for(i=1;i<=NF;i++){if($i ~ /METAL TRACE ID/)t=i; if($i ~ /REPLAY SESSION/)r=i}}
  NR>1{tot++; if(t&&$t!=""&&$t!=" ")tn++; if(r&&$r!=""&&$r!=" ")seen[$r]=1}
  END{ns=0; for(k in seen)ns++;
      printf "TRACE_ID_NONEMPTY_ROWS=%d TOTAL_DATA_ROWS=%d DISTINCT_REPLAY_SESSIONS=%d\n", tn+0, tot+0, ns+0}
' opt/traced_profile.csv 2>/dev/null

echo '---STEP_MS---';  grep -aE 'STEP_MS='  opt/prof_traced.log | tail -12
echo '---STAGES---';   grep -aiE 'Stage [12] denoise|denoise-only|1 passed|1 failed|deselected' opt/prof_traced.log | tail -6
echo '---ERRS---';     grep -aiE 'Timed out|ethernet core|TT_FATAL|TT_THROW|Segmentation|has_value|Aborted|Query mappings|error in [0-9]' opt/prof_traced.log \
                         | grep -avE '^[[:space:]]+File ' | tail -8
