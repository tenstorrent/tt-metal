#!/bin/bash
# Dev-only validation runner for the block-sparse ring SDPA (successor of run_sol_validation.sh).
# Build first with your usual flow. Each step is timeout-wrapped and logs to log_val_<name>.txt in the repo root,
# so a hang in one step cannot block the rest.
#
#   ./dev/run.sh            # = all: the correctness steps (smoke, check, prod, dense)
#   ./dev/run.sh smoke      # keep-all mask at r512k512 must match dense
#   ./dev/run.sh check      # every non-production case of test_sparse_ring_sdpa.py
#   ./dev/run.sh prod       # the production-size cases (5 s / 10 s)
#   ./dev/run.sh dense      # short dense ring-joint regression
#   ./dev/run.sh perf       # 5 s timing vs the cleanup baseline (test_sparse_ring_sdpa_perf.py)
#   ./dev/run.sh perf_all   # the whole benchmark sweep
#   ./dev/run.sh profile    # one 5 s sparse call under the device profiler + perf counters, summarized to
#                           # log_val_profile_summary.txt. Needs a build with SPARSE_STEP_PROFILE (DEBUG commit)
#                           # for the per-zone tables; PROFILE_K overrides the -k filter.
set -u
cd "$(dirname "$0")/.."
SD=tests/nightly/blackhole/sdpa
SPARSE=$SD/test_sparse_ring_sdpa.py
PERF=$SD/test_sparse_ring_sdpa_perf.py
DENSE=$SD/test_ring_joint_sdpa.py
ONLY=${1:-all}

run() {
    local name=$1 secs=$2; shift 2
    echo "=== $name  (log_val_$name.txt)"
    timeout -k 30 "$secs" "$@" > "log_val_$name.txt" 2>&1
    local rc=$?
    echo "    exit $rc  (124 = timeout)"
    return $rc
}
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

# The device profiler writes $TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv, else
# $TT_METAL_HOME/generated/profiler/.logs/ (on the cluster TT_METAL_HOME can be a node-local copy, not this
# directory). Summarize it in place to log_val_<name>.txt; the CSV never needs copying.
summarize_profile() {  # name, [per-core table file]
    local name=$1 csv="" d
    for d in "${TT_METAL_PROFILER_DIR:-}" "${TT_METAL_HOME:+$TT_METAL_HOME/generated/profiler}" generated/profiler; do
        if [ -n "$d" ] && [ -f "$d/.logs/profile_log_device.csv" ]; then
            csv="$d/.logs/profile_log_device.csv"
            break
        fi
    done
    if [ -z "$csv" ]; then
        { echo "no profile_log_device.csv in TT_METAL_PROFILER_DIR='${TT_METAL_PROFILER_DIR:-}'," \
               "TT_METAL_HOME='${TT_METAL_HOME:-}'/generated/profiler or ./generated/profiler; found:"
          find "${TT_METAL_HOME:-.}" generated -name 'profile_log_device.csv' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' \
              2>/dev/null | sort | tail -10; } > "log_val_$name.txt"
        echo "=== $name: no device CSV found (see log_val_$name.txt)"
        return 1
    fi
    echo "csv: $csv ($(du -h "$csv" | cut -f1), $(date -r "$csv"))" > "log_val_$name.txt"
    run "${name}_tmp" 1200 python3 dev/profile_summary.py "$csv" ${2:+"$2"}
    cat "log_val_${name}_tmp.txt" >> "log_val_$name.txt" && rm -f "log_val_${name}_tmp.txt"
}

want smoke && run smoke 900 pytest "$SPARSE" -s -k "keep_all_matches_dense and r512k512"
want check && run check 3600 pytest "$SPARSE" -s -k "not production"
want prod && run prod 2400 pytest "$SPARSE" -s -k production
# Dense: test_ring_joint_attention_logical_tensor_accuracy is 3 cases (causal / balanced / noncausal) at 256
# rows per device, 8 heads, with the CPU reference check: it exercises the shared ring/all-gather path and the
# logical_n tail masking the sparse op relies on, in about a minute. The model sweeps (sdpa_accuracy,
# determinism) and the *_perf* tests are much larger and stay out.
want dense && run dense 1200 pytest "$DENSE" -s -k "logical_tensor_accuracy"

# Not part of "all": timing and profiling.
# perf: the 5 s comparison with the cleanup-branch baseline (dense 7.2, tau 1/2/3 4.15/3.40/3.26, gather 3.09 ms):
# dense + gather, and the sparse op at 256-key pieces with pooled K/V at the three densities.
[ "$ONLY" = perf ] && run perf 2400 pytest "$PERF" -s -k "5s and (dense_and_gather or (k256 and sparse))"
# perf_all: every shape, piece size, density and mode (long; 15 s builds ~1.5 GB host tensors per input).
[ "$ONLY" = perf_all ] && run perf_all 10800 pytest "$PERF" -s
if [ "$ONLY" = profile ]; then
    # One untraced sparse call (the correctness test's 5 s case: 256-key pieces, density 0.15), so the per-core
    # profiler buffer holds the whole op; the traced benchmark's replays would overrun it. PROFILE_K picks another.
    run profile 1800 env TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=39 \
        pytest "$SPARSE" -s -k "${PROFILE_K:-test_production_5s and 256 and 0.15}"
    summarize_profile profile_summary
fi
exit 0
