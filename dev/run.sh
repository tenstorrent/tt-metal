#!/bin/bash
# Dev-only validation runner for the block-sparse ring SDPA (successor of run_sol_validation.sh).
# Build first with your usual flow. Each step is timeout-wrapped and logs to log_val_<name>.txt in the repo root,
# so a hang in one step cannot block the rest.
#
#   ./dev/run.sh            # = all: the correctness steps (smoke, check, prod, dense)
#   ./dev/run.sh a b c      # several steps in one call; presets: quick, night (see below)
#   ./dev/run.sh bench      # THE timing to quote: MiniMax-H3 attention module, traced, dense vs sol at tau 1 and
#                           # the gather alone, 5 / 10 / 15 s (bench5: 5 s only); log_val_bench.txt

#   DEV_LOG_COPY=<dir> ./dev/run.sh night   # also copy each log to <dir> as soon as its step ends
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
    night) set -- bench smoke check prod dense perf_pairs perf_rescale profile5 profile5_nopairs sum10 sum15 ;;
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
    keep "log_val_$name.txt"
    echo "    exit $rc  (124 = timeout)"
    return $rc
}
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }
STEPS=" all bench bench5 hang_ok hang_2p hang_2p19 smoke check check_rescale prod dense perf perf_all profile profile5_nopairs profile5 profile10 profile15 sum5 sum10 sum15 perf_pairs perf_rescale "
case "$STEPS" in *" $ONLY "*) ;; *) echo "unknown step '$ONLY'; one of:$STEPS" >&2; exit 2 ;; esac
# DEV_LOG_COPY=<dir>: copy every log there as soon as its step ends (survives a job that dies later).
keep() { [ -n "${DEV_LOG_COPY:-}" ] && mkdir -p "$DEV_LOG_COPY" && cp -f "$@" "$DEV_LOG_COPY"/; return 0; }
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
        echo "=== $name: no device CSV found (see log_val_$name.txt)"; note 1; keep "log_val_$name.txt"
        return 1
    fi
    echo "csv: $csv ($(du -h "$csv" | cut -f1), $(date -r "$csv"))" > "log_val_$name.txt"
    run "${name}_tmp" 1200 python3 dev/profile_summary.py "$csv" ${2:+"$2"}
    cat "log_val_${name}_tmp.txt" >> "log_val_$name.txt" && rm -f "log_val_${name}_tmp.txt"
    keep "log_val_$name.txt"
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
# bench: the MiniMax-H3 attention module, traced (capture once, 6 timed replays of the attention op), dense vs sol at
# tau 1 (real routing pre-pass, prefix recompute off) and the ring all-gather alone, at 5 / 10 / 15 s, in one process
# so all numbers come from the same machine state. These are the numbers to quote. bench5: 5 s only.
BENCH=models/tt_dit/tests/models/minimax_h3/test_sol_bench_minimax_h3.py
case "$ONLY" in
    bench | bench5)
        k="4x8"
        [ "$ONLY" = bench5 ] && k="4x8 and 768p_5s"
        run "$ONLY" 7200 env SOL_BENCH_TRACE=1 SOL_BENCH_REPS=6 SOL_TAU=1.0 SOL_BENCH_PREFIX=off \
            SOL_BENCH_OUT="$PWD/bench_results" pytest "$BENCH" -s -k "$k"
        keep "bench_results/results.jsonl" 2>/dev/null
        ;;
esac
# Hang repro (2 K/V pages with group delivery hung at 10 s, 19 Q blocks per core in one pass): one sparse call at
# 10 s, pair steps off. hang_ok: 3 pages, 10 blocks per pass (2 passes; ran before). hang_2p: 2 pages, 10 blocks per
# pass (separates the page count from the slot count). hang_2p19: 2 pages, every block in one pass (the hanging
# case). The two 2-page cases run under the watcher with a 15-minute timeout; a hang leaves the device stuck, so run
# them last or in separate jobs (hang_ok first). The watcher log is copied to log_val_<step>_watcher.txt.
case "$ONLY" in
    hang_ok | hang_2p | hang_2p19)
        pages=3; q=q10; watch=""
        [ "$ONLY" != hang_ok ] && { pages=2; watch="TT_METAL_WATCHER=1"; }
        [ "$ONLY" = hang_2p19 ] && q=qauto
        run "$ONLY" 900 env TT_SPARSE_PAIR_STEPS=0 TT_SPARSE_KV_PAGES=$pages $watch \
            pytest dev/profile_case.py -s -k "s10 and k256 and d020 and sparse_only and $q"
        for d in "${TT_METAL_HOME:-.}/generated/watcher" generated/watcher; do
            if [ -n "$watch" ] && [ -f "$d/watcher.log" ]; then
                cp -f "$d/watcher.log" "log_val_${ONLY}_watcher.txt"; keep "log_val_${ONLY}_watcher.txt"; break
            fi
        done
        ;;
esac
if [ "$ONLY" = profile5_nopairs ]; then  # the 5 s profile with D4 pair steps forced off (the pre-D4 baseline)
    run profile5_nopairs 1800 env TT_SPARSE_PAIR_STEPS=0 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=39 \
        pytest dev/profile_case.py -s -k "${PROFILE_K:-s05 and k256 and d020 and sparse_only and qauto}"
    summarize_profile profile5_nopairs_summary
fi
case "$ONLY" in
    profile5 | profile10 | profile15)
        n=${ONLY#profile}
        shape=s$(printf '%02d' "$n")
        run "$ONLY" 1800 env TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=39 \
            pytest dev/profile_case.py -s -k "${PROFILE_K:-$shape and k256 and d020 and sparse_only and qauto}"
        summarize_profile "profile${n}_summary"
        ;;
    # sum5 / sum10 / sum15: the same call under TT_METAL_PROFILER_SUM=1 with the zone window empty, so the buffer
    # keeps room for the per-RISC totals (RING_STEP_SUM); the summary prints min / median / max per device.
    sum5 | sum10 | sum15)
        n=${ONLY#sum}
        shape=s$(printf '%02d' "$n")
        run "$ONLY" 1800 env TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SUM=1 TT_SPARSE_PROF_STEPS=0 \
            pytest dev/profile_case.py -s -k "${PROFILE_K:-$shape and k256 and d020 and sparse_only and qauto}"
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
