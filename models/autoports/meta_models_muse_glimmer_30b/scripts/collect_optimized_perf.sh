#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Collect warmed prefill / traced warmed decode performance artifacts for the
# Muse-Glimmer-30B optimized decoder layer.
#
#   scripts/collect_optimized_perf.sh prefill sliding_rope 8192
#   scripts/collect_optimized_perf.sh decode  sliding_rope 1
#
# One Tracy session per (mode, layer kind, size): tt-perf-report keys off the *first*
# start/end signpost in a CSV, so mixing several measured windows into one session would
# report only the first one.
#
# Artifacts (under doc/optimized_decoder/tracy/<kind>/):
#   <mode>_<size>_ops.csv.gz               raw Tracy ops CSV, gzipped (provenance)
#   <mode>_<size>_perf_report.txt          human-readable table  (--no-summary)
#   <mode>_<size>_perf_report.csv          filtered per-op CSV   (--csv)
#   <mode>_<size>_perf_report.console.log  --csv run stdout (boilerplate/provenance)
#   <mode>_<size>_perf_report.summary.txt  summary + advice table
#   <mode>_<size>_pytest.log               the pytest/tracy session log
set -euo pipefail

MODE="${1:?mode: prefill|decode}"
KIND="${2:?layer kind: sliding_rope|full_nope}"
SIZE="${3:?prefill seq_len or decode batch}"

MODEL_DIR="models/autoports/meta_models_muse_glimmer_30b"
ART="${MODEL_DIR}/doc/optimized_decoder"
OUT="${ART}/tracy/${KIND}"
mkdir -p "${OUT}"

case "${MODE}" in
  prefill) START=PERF_PREFILL; END=PERF_PREFILL_END; TESTNAME=test_prefill_perf ;;
  decode)  START=PERF_DECODE;  END=PERF_DECODE_END;  TESTNAME=test_decode_perf ;;
  *) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac

SELECTOR="${TESTNAME}[${SIZE}-${KIND}]"
PYTEST_LOG="${OUT}/${MODE}_${SIZE}_pytest.log"

# Remember the newest ops CSV *before* the run so a run that produces none cannot silently
# publish a previous run's data as this configuration's evidence (which is exactly what
# happened once when a device fault killed the pytest run mid-way).
BEFORE_CSV="$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1 || true)"

echo "+ python -m tracy -r -p -v -m pytest ${MODEL_DIR}/tests/test_optimized_decoder_perf.py -k '${SELECTOR}'"
if ! python -m tracy -r -p -v -m pytest \
      "${MODEL_DIR}/tests/test_optimized_decoder_perf.py" \
      -k "${SELECTOR}" > "${PYTEST_LOG}" 2>&1; then
  # Tracy's host-side post-processing can fail to match one op to the device CSV
  # ("Device data missing: Op N not present in cpp_device_perf_report.csv"), which is a
  # profiler-tooling failure, not a model failure. Fall back to the device-profiler flow
  # documented in tech_reports/LLMs/llms.md: same measured window, same signposts, ops CSV
  # produced by process_ops_logs.py instead of by the tracy wrapper.
  echo "+ tracy post-processing failed; falling back to TT_METAL_DEVICE_PROFILER=1" | tee -a "${PYTEST_LOG}"
  TT_METAL_DEVICE_PROFILER=1 python -m pytest \
    "${MODEL_DIR}/tests/test_optimized_decoder_perf.py" \
    -k "${SELECTOR}" >> "${PYTEST_LOG}" 2>&1
  python tools/tracy/process_ops_logs.py --date >> "${PYTEST_LOG}" 2>&1
fi

# `ls ... | head -1` under `set -o pipefail` fails with SIGPIPE once enough report
# directories accumulate for ls to fill the pipe buffer, which silently aborted three
# collections. Tolerate the pipe failure and check emptiness explicitly instead.
CSV="$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1 || true)"
if [ -z "${CSV}" ] || [ "${CSV}" = "${BEFORE_CSV}" ]; then
  echo "ERROR: no new ops CSV was produced for ${SELECTOR} (newest is still '${BEFORE_CSV}')." >&2
  echo "       Check ${PYTEST_LOG}; do not publish a stale CSV as this run's evidence." >&2
  exit 4
fi
if ! grep -q "1 passed" "${PYTEST_LOG}"; then
  echo "ERROR: ${SELECTOR} did not pass; see ${PYTEST_LOG}." >&2
  exit 5
fi
echo "+ newest ops CSV: ${CSV}"
# Gzipped: the raw ops CSV can be several MB and the repo's pre-commit hook rejects files
# over 500 KB. scripts/render_evidence.py reads either form.
rm -f "${OUT}/${MODE}_${SIZE}_ops.csv" "${OUT}/${MODE}_${SIZE}_ops.csv.gz"
gzip -c "${CSV}" > "${OUT}/${MODE}_${SIZE}_ops.csv.gz"
echo "${CSV}" > "${OUT}/${MODE}_${SIZE}_ops.csv.provenance"

tt-perf-report <(gzip -dc "${OUT}/${MODE}_${SIZE}_ops.csv.gz") \
  --start-signpost "${START}" --end-signpost "${END}" \
  --no-summary --no-advice \
  > "${OUT}/${MODE}_${SIZE}_perf_report.txt"

tt-perf-report <(gzip -dc "${OUT}/${MODE}_${SIZE}_ops.csv.gz") \
  --start-signpost "${START}" --end-signpost "${END}" \
  --csv "${OUT}/${MODE}_${SIZE}_perf_report.csv" --no-advice \
  > "${OUT}/${MODE}_${SIZE}_perf_report.console.log"

tt-perf-report <(gzip -dc "${OUT}/${MODE}_${SIZE}_ops.csv.gz") \
  --start-signpost "${START}" --end-signpost "${END}" \
  > "${OUT}/${MODE}_${SIZE}_perf_report.summary.txt"

echo "+ wrote ${OUT}/${MODE}_${SIZE}_perf_report.{txt,csv,console.log,summary.txt}"
