#!/usr/bin/env bash
# MoE block power A/B (investigation only, not for main): CI's GLM-5.3 SC1 disaggregated 1M@5k runner twice in ONE
# job - MOE_IMPL=ttmoe then MOE_IMPL=block - with a 1 ms tt_umd telemetry sampler (AICLK / TDP / TDC / VCORE / temp /
# clock-limiter bitmask per chip, plus every chip's power limits and firmware versions at start). Everything lands
# under $PREFILL_SUMMARIES/pcc/pwr_ab (uploaded by CI); pwr_ab_summary.py prints the comparison into the job log.
set -uo pipefail
: "${TT_METAL_HOME:?}" "${PREFILL_SUMMARIES:?}"
cd "${TT_METAL_HOME}"
CI=models/demos/common/prefill/runners/ci
BASE="${PREFILL_SUMMARIES}"
OUT="${BASE}/pcc/pwr_ab"; mkdir -p "${OUT}/telemetry"
echo "pwr_ab $(git log --oneline -1 2>/dev/null) on $(hostname) $(date -u +%FT%TZ)" | tee "${OUT}/summary.txt"
PY=/opt/venv/bin/python3; "${PY}" -c "import tt_umd" 2>/dev/null || PY=python3
unset MOE_IMPL K3_MOE_IMPL
taskset -c $(( $(nproc) - 1 )) "${PY}" "${CI}/telem_sampler.py" "${OUT}/telemetry/telem.csv" 1 > "${OUT}/telemetry/telem.err" 2>&1 &
SP=$!
trap 'kill -TERM ${SP} 2>/dev/null; wait ${SP} 2>/dev/null' EXIT
sleep 5
kill -0 ${SP} 2>/dev/null && echo "sampler on (1 ms)" | tee -a "${OUT}/summary.txt" || { echo "sampler failed:"; cat "${OUT}/telemetry/telem.err"; }
for v in ttmoe block; do
  RS="${OUT}/run_multirank_pcc_${v}.sh"
  sed -e "s|^    RUNNER_ENV=\"export LOGURU_LEVEL=ERROR;\"$|    RUNNER_ENV=\"export LOGURU_LEVEL=ERROR; export MOE_IMPL=${v};\"|" \
    "${CI}/run_multirank_pcc.sh" > "${RS}"
  grep -q "MOE_IMPL=${v}" "${RS}" || { echo "MOE_IMPL injection did not apply"; exit 3; }
  echo "${v} start $(date +%s)" >> "${OUT}/phases.txt"
  t0=$(date +%s)
  PREFILL_SUMMARIES="${OUT}/${v}/prefill_summaries" bash "${RS}" glm53 sc1 > "${OUT}/runner_${v}.log" 2>&1
  rc=$?
  echo "${v} end $(date +%s) rc ${rc}" >> "${OUT}/phases.txt"
  echo "## runner_${v}: rc ${rc} | $(( $(date +%s) - t0 )) s" | tee -a "${OUT}/summary.txt"
  grep -a -E '^  (chunk_time|ttft|throughput) +@|PCC GATE|PERF GATE|TDP_LIMIT_MAX|Traceback' "${OUT}/runner_${v}.log" | sort -u | head -30 | tee -a "${OUT}/summary.txt"
  sleep 120
done
kill -TERM ${SP} 2>/dev/null; wait ${SP} 2>/dev/null; trap - EXIT
python3 "${CI}/pwr_ab_summary.py" "${OUT}" | tee -a "${OUT}/summary.txt"
gzip -f "${OUT}/telemetry/telem.csv"
ls -la "${OUT}" "${OUT}/telemetry"
