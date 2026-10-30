#!/bin/bash
# Review probe (not for main): base vs head A/B of one LLK functional module and one perf module on this runner's card.
# usage: run.sh <arch> <reps> <func_module> <perf_module>; checkouts in $GITHUB_WORKSPACE/{base,head}, results in $GITHUB_WORKSPACE/out
set -u
ARCH=$1 REPS=$2 FUNC=$3 PERF=$4
WS=$GITHUB_WORKSPACE; P=$WS/probe/review-probe; OUT=$WS/out; mkdir -p "$OUT"
export CHIP_ARCH=$ARCH TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=0
T() { date -u +%T; }
FAIL=0
rc_of() { local rc; rc=$(tail -n 1 "$1" | sed -n "s/^rc=//p"); [ "${rc:-1}" = 0 ] || FAIL=1; }
step() { echo "$(T) $*" | tee -a "$OUT/timeline.txt"; }
step "start arch=$ARCH reps=$REPS runner=$(hostname)"
for side in base head; do
    step "setup $side"
    (cd "$WS/$side/tt_metal/tt-llk/tests" && ./setup_testing_env.sh) > "$OUT/setup_$side.log" 2>&1 || { tail -30 "$OUT/setup_$side.log"; exit 2; }
done
py() { cd "$WS/$1/tt_metal/tt-llk/tests/python_tests" && RUNNER_TEMP=$WS/rt-$1 PYTHONPATH=$P pytest -p no:cacheprovider "${@:2}"; }
for side in base head; do
    step "func $side"
    PROBE_HASH_OUT=$OUT/func_$side.hash.tsv py $side -p probe_hash -q -rA --timeout=300 --junitxml="$OUT/func_$side.xml" "$FUNC" > "$OUT/func_$side.log" 2>&1
    echo "rc=$?" >> "$OUT/func_$side.log"; tail -n 2 "$OUT/func_$side.log"; rc_of "$OUT/func_$side.log"
done
for side in base head; do
    step "perf build $side"
    py $side --compile-producer -n 10 -q --timeout=120 -m "perf and not accuracy" "$PERF" > "$OUT/perf_build_$side.log" 2>&1
    echo "rc=$?" >> "$OUT/perf_build_$side.log"; tail -n 2 "$OUT/perf_build_$side.log"; rc_of "$OUT/perf_build_$side.log"
done
for r in $(seq 1 "$REPS"); do
    for side in base head; do
        step "perf run $side r$r"
        rm -rf "$WS/$side/tt_metal/tt-llk/tests/python_tests/perf_data" "$WS/$side/tt_metal/tt-llk/perf_data"
        py $side --compile-consumer -q --timeout=300 -m "perf and not accuracy" --junitxml="$OUT/perf_${side}_r$r.xml" "$PERF" > "$OUT/perf_${side}_r$r.log" 2>&1
        echo "rc=$?" >> "$OUT/perf_${side}_r$r.log"; tail -n 2 "$OUT/perf_${side}_r$r.log"; rc_of "$OUT/perf_${side}_r$r.log"
        for d in "$WS/$side/tt_metal/tt-llk/tests/python_tests/perf_data" "$WS/$side/tt_metal/tt-llk/perf_data"; do
            [ -d "$d" ] && tar czf "$OUT/perf_${side}_r$r.$(basename "$(dirname "$d")").tgz" -C "$(dirname "$d")" perf_data
        done
    done
done
step "done fail=$FAIL"
exit $FAIL
