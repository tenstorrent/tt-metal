#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Collect Tracy device-op evidence for the MusicLLM warmed traced decode / warmed prefill.
#
#   source ~/mm3-bringup/common.sh && cd $MM3_WT
#   with_hw_lock models/autoports/minimaxai_minimax_music3/scripts/collect_llm_perf.sh decode
#   with_hw_lock models/autoports/minimaxai_minimax_music3/scripts/collect_llm_perf.sh prefill 104
#
# One Tracy session per measured window (tt-perf-report keys off the first start/end signpost
# pair in a CSV). Artifacts land under doc/llm/tracy/<mode>[_<size>]/:
#   ops.csv.gz                raw Tracy ops CSV (gzipped; the repo hook rejects >500 KB files)
#   perf_report.txt           tt-perf-report table for the signposted window (--no-summary --no-advice)
#   perf_report.csv           filtered per-op CSV (--csv)
#   perf_report.summary.txt   summary + advice
#   pytest.log                the tracy/pytest session log
set -euo pipefail
MODE="${1:?mode: prefill|decode}"
SIZE="${2:-}"
MODEL_DIR="models/autoports/minimaxai_minimax_music3"
case "${MODE}" in
  prefill) START=PERF_PREFILL; END=PERF_PREFILL_END; SELECTOR="test_prefill_perf[${SIZE:?prefill seq_len}]"; OUT="${MODEL_DIR}/doc/llm/tracy/prefill_${SIZE}" ;;
  decode)  START=PERF_DECODE;  END=PERF_DECODE_END;  SELECTOR="test_decode_perf"; OUT="${MODEL_DIR}/doc/llm/tracy/decode" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
mkdir -p "${OUT}"
PY="${MM3_PY:-python}"
BEFORE_CSV="$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1 || true)"
echo "+ ${PY} -m tracy -r -p -v -m pytest ${MODEL_DIR}/tests/test_llm_perf.py -k '${SELECTOR}'"
if ! timeout 3600 "${PY}" -m tracy -r -p -v -m pytest "${MODEL_DIR}/tests/test_llm_perf.py" -k "${SELECTOR}" \
      -p no:cacheprovider > "${OUT}/pytest.log" 2>&1; then
  echo "+ tracy post-processing failed; falling back to TT_METAL_DEVICE_PROFILER=1" | tee -a "${OUT}/pytest.log"
  TT_METAL_DEVICE_PROFILER=1 timeout 3600 "${PY}" -m pytest "${MODEL_DIR}/tests/test_llm_perf.py" -k "${SELECTOR}" \
      -p no:cacheprovider >> "${OUT}/pytest.log" 2>&1
  "${PY}" tools/tracy/process_ops_logs.py --date >> "${OUT}/pytest.log" 2>&1
fi
CSV="$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)"
if [ -z "${CSV}" ] || [ "${CSV}" = "${BEFORE_CSV}" ]; then
  echo "ERROR: no new ops CSV for ${SELECTOR}; see ${OUT}/pytest.log" >&2; exit 4
fi
grep -q "1 passed" "${OUT}/pytest.log" || { echo "ERROR: ${SELECTOR} did not pass; see ${OUT}/pytest.log" >&2; exit 5; }
gzip -c "${CSV}" > "${OUT}/ops.csv.gz"
echo "${CSV}" > "${OUT}/ops.csv.provenance"
tt-perf-report <(gzip -dc "${OUT}/ops.csv.gz") --start-signpost "${START}" --end-signpost "${END}" --no-summary --no-advice > "${OUT}/perf_report.txt"
tt-perf-report <(gzip -dc "${OUT}/ops.csv.gz") --start-signpost "${START}" --end-signpost "${END}" --csv "${OUT}/perf_report.csv" --no-advice > "${OUT}/perf_report.console.log"
tt-perf-report <(gzip -dc "${OUT}/ops.csv.gz") --start-signpost "${START}" --end-signpost "${END}" > "${OUT}/perf_report.summary.txt"
echo "+ wrote ${OUT}/perf_report.{txt,csv,summary.txt}"
