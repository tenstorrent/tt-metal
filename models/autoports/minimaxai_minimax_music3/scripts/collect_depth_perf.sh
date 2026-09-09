#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Tracy device-op evidence for one warmed 7-step depth-decoder frame (eager or traced).
#
#   source ~/mm3-bringup/common.sh && cd $MM3_WT
#   with_hw_lock models/autoports/minimaxai_minimax_music3/scripts/collect_depth_perf.sh eager
#   with_hw_lock models/autoports/minimaxai_minimax_music3/scripts/collect_depth_perf.sh traced
#
# Artifacts under doc/depth_decoder/tracy/<mode>/: ops.csv.gz, perf_report.{txt,csv,summary.txt}, pytest.log.
# Same tooling notes as scripts/collect_llm_perf.sh (tracy tools folder, tolerant post-processing).
set -euo pipefail
MODE="${1:?mode: eager|traced}"
MODEL_DIR="models/autoports/minimaxai_minimax_music3"
PY="${MM3_PY:-python}"
PERF_REPORT="$(dirname "${PY}")/tt-perf-report"
[ -x "${PERF_REPORT}" ] || PERF_REPORT="tt-perf-report"
TRACY_TOOLS="${MM3_TRACY_TOOLS:-${MM3_METAL_MAIN:-$HOME/tt-metal}/build_Release/tools/profiler/bin}"
case "${MODE}" in
  eager)  START=PERF_DEPTH_EAGER;  END=PERF_DEPTH_EAGER_END;  SELECTOR="test_eager_frame_perf" ;;
  traced) START=PERF_DEPTH_TRACED; END=PERF_DEPTH_TRACED_END; SELECTOR="test_traced_frame_perf" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
OUT="${MODEL_DIR}/doc/depth_decoder/tracy/${MODE}"
mkdir -p "${OUT}"
BEFORE_CSV="$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1 || true)"
if ! timeout 1800 "${PY}" -m tracy -r -p -v --tracy-tools-folder "${TRACY_TOOLS}" -m pytest \
      "${MODEL_DIR}/tests/test_depth_decoder_perf.py" -k "${SELECTOR}" -m slow -p no:cacheprovider > "${OUT}/pytest.log" 2>&1; then
  echo "+ tracy post-processing failed; re-processing the same logs tolerantly" | tee -a "${OUT}/pytest.log"
  "${PY}" "${MODEL_DIR}/scripts/tracy_postprocess_tolerant.py" >> "${OUT}/pytest.log" 2>&1
fi
CSV="$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)"
if [ -z "${CSV}" ] || [ "${CSV}" = "${BEFORE_CSV}" ]; then
  echo "ERROR: no new ops CSV for ${SELECTOR}; see ${OUT}/pytest.log" >&2; exit 4
fi
grep -q "1 passed" "${OUT}/pytest.log" || { echo "ERROR: ${SELECTOR} did not pass; see ${OUT}/pytest.log" >&2; exit 5; }
gzip -c "${CSV}" > "${OUT}/ops.csv.gz"
echo "${CSV}" > "${OUT}/ops.csv.provenance"
"${PERF_REPORT}" <(gzip -dc "${OUT}/ops.csv.gz") --start-signpost "${START}" --end-signpost "${END}" --no-summary --no-advice > "${OUT}/perf_report.txt"
"${PERF_REPORT}" <(gzip -dc "${OUT}/ops.csv.gz") --start-signpost "${START}" --end-signpost "${END}" --csv "${OUT}/perf_report.csv" --no-advice > "${OUT}/perf_report.console.log"
"${PERF_REPORT}" <(gzip -dc "${OUT}/ops.csv.gz") --start-signpost "${START}" --end-signpost "${END}" > "${OUT}/perf_report.summary.txt"
echo "+ wrote ${OUT}/perf_report.{txt,csv,summary.txt}"
