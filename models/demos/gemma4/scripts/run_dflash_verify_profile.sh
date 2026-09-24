#!/bin/bash
# Tracy/profiler capture of DFlash's "part after drafter": the target model's
# packed-verify forward (ttnn_packed_verify_forward, tt/model.py) -- the step
# that checks the drafter's candidate block against the real 31B target.
#
# Sibling of run_dflash_drafter_profile.sh (drafter only) and
# run_dflash_prefill_profile.sh (prefill only) -- together the three cover
# DFlash's three model-invocation phases.
#
# Reuses DFlashFusedDecoder's own real persistent buffers (built by a real
# prefill + drafter + capture() setup) rather than synthetic ones -- see
# test_dflash_verify_tracy.py's docstring. Candidate token ids are a
# shape-only placeholder: this measures verify's REAL device op cost at
# DFlash's REAL packed shape, but generated logits are not meaningful text.
#
# Two modes:
#
#   A) Device-only (default) — safe vs Tracy's 32K source-location cap.
#      CSV has timing only (NO "DEVICE ID" / "OP CODE") → NOT usable by
#      tt-perf-report. Good for raw device dumps.
#
#   B) Host Tracy (--tt-perf-report) — short single-instance capture under
#      `python -m tracy`, then merge host+device logs into a full
#      ops_perf_results CSV that tt-perf-report accepts.
#
# Usage:
#   # tt-perf-report-compatible (recommended for analysis)
#   ./models/demos/gemma4/scripts/run_dflash_verify_profile.sh --tt-perf-report
#
#   # device-only dump (default)
#   ./models/demos/gemma4/scripts/run_dflash_verify_profile.sh
#
#   # N measured repeats
#   ./models/demos/gemma4/scripts/run_dflash_verify_profile.sh --repeat 16
#
# Requires MODEL_WEIGHTS_DIR-discoverable target snapshot (for the drafter's
# tied lm_head) and GEMMA4_DFLASH_SHARD_ARGMAX=1 -- see
# test_dflash_verify_tracy.py's docstring.
#
# Env knobs (see test_dflash_verify_tracy.py for details):
#   GEMMA4_DFLASH_VERIFY_TRACY_NUM_LAYERS   [8]
#   GEMMA4_DFLASH_VERIFY_TRACY_ITERS        [8]
#   GEMMA4_DFLASH_VERIFY_TRACY_WARMUP       [1]
set -uo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$_SCRIPT_DIR/../../../.." && pwd)}"

die() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$TT_METAL_HOME/python_env/bin/activate" ] || die "no venv at $TT_METAL_HOME/python_env — run ./create_venv.sh first"
command -v tt-smi >/dev/null || die "tt-smi not on PATH — needed for the device health check"

FOR_TT_PERF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tt-perf-report)
      FOR_TT_PERF=1
      shift
      ;;
    --repeat)
      [ $# -ge 2 ] || die "--repeat needs N"
      export GEMMA4_DFLASH_VERIFY_TRACY_ITERS="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,38p' "$0"
      exit 0
      ;;
    *)
      die "unknown arg: $1 (use --tt-perf-report | --repeat N)"
      ;;
  esac
done

export HF_MODEL="${HF_MODEL:-google/gemma-4-31b-it}"
export GEMMA4_DFLASH_SHARD_ARGMAX=1

cd "$TT_METAL_HOME"
# shellcheck disable=SC1091
source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME"

export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_MID_RUN_DUMP="${TT_METAL_PROFILER_MID_RUN_DUMP:-1}"
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="${TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT:-100000}"

echo "device health check:"
tt-smi -s 2>&1 | grep -c '"FAULTS": "0x0"'
echo "ITERS=${GEMMA4_DFLASH_VERIFY_TRACY_ITERS:-8} NUM_LAYERS=${GEMMA4_DFLASH_VERIFY_TRACY_NUM_LAYERS:-8} FOR_TT_PERF=${FOR_TT_PERF}"

rm -rf generated/profiler/.logs generated/profiler/reports

TEST="models/demos/gemma4/tests/dflash/test_dflash_verify_tracy.py"
PYTEST_ARGS=( "$TEST" -k 1x8 -s --timeout 1800 )

if [ "$FOR_TT_PERF" = "1" ]; then
  echo "running host Tracy (needed for tt-perf-report): python -m tracy -p -r -m pytest ..."
  python -m tracy -p -r -m pytest "${PYTEST_ARGS[@]}"
  STATUS=$?
  PROCESS_ARGS=( --date )
else
  echo "running device-only (CSV will NOT work with tt-perf-report)"
  echo "  use --tt-perf-report if you need tt-perf-report"
  pytest "${PYTEST_ARGS[@]}"
  STATUS=$?
  PROCESS_ARGS=( --date --device-only )
fi

echo "processing device logs -> ops_perf_results_*.csv"
python tools/tracy/process_ops_logs.py "${PROCESS_ARGS[@]}"

CSV="$(find generated/profiler/reports -iname 'ops_perf_results_*.csv' 2>/dev/null | sort | tail -1)"
if [ -n "$CSV" ]; then
    echo ""
    echo "CSV: $TT_METAL_HOME/$CSV"
    if head -1 "$CSV" | tr ',' '\n' | grep -qx 'DEVICE ID'; then
      echo "OK: CSV has DEVICE ID / OP CODE — usable by tt-perf-report:"
      echo "  tt-perf-report \"$CSV\" --start-signpost verify_start --end-signpost verify_stop --no-host-ops"
    else
      echo "NOTE: this CSV is device-only (no DEVICE ID) — tt-perf-report will reject it."
      echo "  Re-run with: $0 --tt-perf-report"
    fi
else
    echo "WARNING: no ops_perf_results_*.csv found under generated/profiler/reports" >&2
fi

exit "$STATUS"
