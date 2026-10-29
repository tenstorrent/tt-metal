#!/bin/bash
# Dev-only validation runner for the block-sparse ring SDPA (successor of run_sol_validation.sh).
# Build first with your usual flow. Each step is timeout-wrapped and logs to log_val_<name>.txt in the repo root,
# so a hang in one step cannot block the rest.
#
#   ./dev/run.sh            # = all: the correctness steps (smoke, check, prod, dense)
#   ./dev/run.sh a b c      # several steps in one call; presets: quick, night (see below)
#   ./dev/run.sh smoke      # keep-all mask at r512k512 must match dense
#   ./dev/run.sh check      # every non-production case of test_sparse_ring_sdpa.py
#   ./dev/run.sh prod       # the production-size cases (5 s / 10 s)
#   ./dev/run.sh dense      # short dense ring-joint regression
#   ./dev/run.sh perf       # 5 s timing vs the cleanup baseline (test_sparse_ring_sdpa_perf.py)
#   ./dev/run.sh perf_all   # the whole benchmark sweep
#   ./dev/run.sh profile    # = profile5
#   ./dev/run.sh profile10  # the same from dev/profile_case.py at 10 s (also profile5, profile15), summarized to
#                           # log_val_profile10_summary.txt
#   ./dev/run.sh sum10      # whole-op per-core step totals at 10 s from sum zones (also sum5, sum15)
#   ./dev/run.sh perf_pairs # the 5 s perf subset with D4 pair steps forced off, then at the factory default
#   ./dev/run.sh perf_rescale  # the 5 s perf subset with TT_SPARSE_FAST_RESCALE=0, then 1 (fewer rescale drains)
#   ./dev/run.sh check_rescale # the check step with TT_SPARSE_FAST_RESCALE=1
#
# Knobs (dev branch only, read by the DEBUG factory code; set once per process, they are not in the op's program
# hash): TT_SPARSE_PROF_PASS / _ITER / _SKIP / _STEPS pick the profile window (defaults 0 / 1 / 0 / 100; ITER=-1 =
# every ring iteration): e.g. TT_SPARSE_PROF_ITER=3 TT_SPARSE_PROF_SKIP=100 ./dev/run.sh profile15.
# TT_SPARSE_PAIR_STEPS=0 forces D4 pair steps off.
set -u
cd "$(dirname "$0")/.."
SD=tests/nightly/blackhole/sdpa
SPARSE=$SD/test_sparse_ring_sdpa.py
PERF=$SD/test_sparse_ring_sdpa_perf.py
DENSE=$SD/test_ring_joint_sdpa.py
# Several steps in one call run one after another (one job pays the rsync / reset once); presets:
#   quick = smoke perf profile5                      (a kernel experiment whose correctness is known)
#   night = smoke check prod dense perf_pairs perf_rescale profile5 profile5_nopairs sum10 sum15
case "${1:-all}" in
    quick) set -- smoke perf profile5 ;;
    night) set -- smoke check prod dense perf_pairs perf_rescale profile5 profile5_nopairs sum10 sum15 ;;
esac
if [ $# -gt 1 ]; then
    worst=0
    for step in "$@"; do
        "$0" "$step"
        rc=$?
        [ "$rc" -gt "$worst" ] && worst=$rc
    done
    echo "=== steps: $*  worst exit code: $worst"
    exit "$worst"
fi
ONLY=${1:-all}

run() {
    local name=$1 secs=$2; shift 2
    echo "=== $name  (log_val_$name.txt)"
    timeout -k 30 "$secs" "$@" > "log_val_$name.txt" 2>&1
    local rc=$?
    note $rc
    echo "    exit $rc  (124 = timeout)"
    return $rc
}
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }
STEPS=" all smoke check check_rescale prod dense perf perf_all profile profile5_nopairs profile5 profile10 profile15 sum5 sum10 sum15 perf_pairs perf_rescale "
case "$STEPS" in *" $ONLY "*) ;; *) echo "unknown step '$ONLY'; one of:$STEPS" >&2; exit 2 ;; esac
WORST=0  # worst exit code of any step, returned at the end
note() { local rc=$1; [ "$rc" -gt "$WORST" ] && WORST=$rc; return 0; }
# pytest -k matches substrings of the whole node id: "5s" alone also matches "15s", and the module name
# test_sparse_ring_sdpa_perf.py contains "sparse", hence the "not 15s" / "not kept" forms below.
PERF5_K="5s and not 15s and (dense_and_gather or (k256 and not kept))"

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
        echo "=== $name: no device CSV found (see log_val_$name.txt)"; note 1
        return 1
    fi
    echo "csv: $csv ($(du -h "$csv" | cut -f1), $(date -r "$csv"))" > "log_val_$name.txt"
    run "${name}_tmp" 1200 python3 dev/profile_summary.py "$csv" ${2:+"$2"}
    cat "log_val_${name}_tmp.txt" >> "log_val_$name.txt" && rm -f "log_val_${name}_tmp.txt"
}

want smoke && run smoke 900 pytest "$SPARSE" -s -k "test_keep_all_matches_dense and r512k512"
want check && run check 3600 pytest "$SPARSE" -s -k "not production"
[ "$ONLY" = check_rescale ] && run check_rescale 3600 env TT_SPARSE_FAST_RESCALE=1 pytest "$SPARSE" -s -k "not production"
want prod && run prod 2400 pytest "$SPARSE" -s -k production
# Dense: test_ring_joint_attention_logical_tensor_accuracy is 3 cases (causal / balanced / noncausal) at 256
# rows per device, 8 heads, with the CPU reference check: it exercises the shared ring/all-gather path and the
# logical_n tail masking the sparse op relies on, in about a minute. The model sweeps (sdpa_accuracy,
# determinism) and the *_perf* tests are much larger and stay out.
want dense && run dense 1200 pytest "$DENSE" -s -k "logical_tensor_accuracy"

# Not part of "all": timing and profiling.
# perf: the 5 s comparison with the cleanup-branch baseline (dense 7.2, tau 1/2/3 4.15/3.40/3.26, gather 3.09 ms):
# dense + gather, and the sparse op at 256-key pieces with pooled K/V at the three densities.
[ "$ONLY" = perf ] && run perf 2400 pytest "$PERF" -s -k "$PERF5_K"
# perf_all: every shape, piece size, density and mode (long; 15 s builds ~1.5 GB host tensors per input).
[ "$ONLY" = perf_all ] && run perf_all 10800 pytest "$PERF" -s
[ "$ONLY" = profile ] && ONLY=profile5  # one sparse call only: the summary treats the longest run as the op
# profile5 / profile10 / profile15: one untraced sparse call at 5 / 10 / 15 s from dev/profile_case.py (inputs made on
# device, setup in seconds), 256-key pieces, density 0.2; PROFILE_K overrides the -k filter (see profile_case.py ids).
if [ "$ONLY" = profile5_nopairs ]; then  # the 5 s profile with D4 pair steps forced off (the pre-D4 baseline)
    run profile5_nopairs 1800 env TT_SPARSE_PAIR_STEPS=0 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=39 \
        pytest dev/profile_case.py -s -k "${PROFILE_K:-s05 and k256 and d020 and sparse_only}"
    summarize_profile profile5_nopairs_summary
fi
case "$ONLY" in
    profile5 | profile10 | profile15)
        n=${ONLY#profile}
        shape=s$(printf '%02d' "$n")
        run "$ONLY" 1800 env TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=39 \
            pytest dev/profile_case.py -s -k "${PROFILE_K:-$shape and k256 and d020 and sparse_only}"
        summarize_profile "profile${n}_summary"
        ;;
    # sum5 / sum10 / sum15: the same call under TT_METAL_PROFILER_SUM=1 with the zone window empty, so the buffer
    # keeps room for the per-RISC totals (RING_STEP_SUM); the summary prints min / median / max per device.
    sum5 | sum10 | sum15)
        n=${ONLY#sum}
        shape=s$(printf '%02d' "$n")
        run "$ONLY" 1800 env TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SUM=1 TT_SPARSE_PROF_STEPS=0 \
            pytest dev/profile_case.py -s -k "${PROFILE_K:-$shape and k256 and d020 and sparse_only}"
        summarize_profile "sum${n}_summary"
        ;;
    # perf_pairs: A/B of D4 pair steps at 5 s, separate processes (the op program cache is per process).
    # perf_rescale: A/B of the fast_rescale step (fewer PACK_DONE drains) at 5 s, separate processes.
    perf_rescale)
        for p in 0 1; do
            run "perf_rescale$p" 2400 env TT_SPARSE_FAST_RESCALE=$p pytest "$PERF" -s -k "$PERF5_K"
        done
        ;;
    perf_pairs)
        for p in 0 1; do
            run "perf_pairs$p" 2400 env TT_SPARSE_PAIR_STEPS=$p \
                pytest "$PERF" -s -k "$PERF5_K"
        done
        ;;
esac
[ "$WORST" -ne 0 ] && echo "=== worst exit code: $WORST"
exit "$WORST"
