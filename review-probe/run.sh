#!/bin/bash
# Review probe (not for main): A/B of LLK test modules on this runner's card, every side on the same card.
# env: ARCH, REPS, SIDES ("base head" or "base head main"), MODULES (JSON list of {"func": file, "perf": file}, "" when
# absent), <SIDE>_SHA; checkouts in $GITHUB_WORKSPACE/<side>, results in $GITHUB_WORKSPACE/out.
# Per module and side: the functional module once, then once more only its failing ids (at most 50, at the first run's
# 300 s per test, so a slow test does not count as reproduced by a shorter timeout), or the whole module when pytest
# crashed (no junit xml); then the perf module is built and its reps run interleaved
# over the sides. out/summary.json (summary.py) lists every step's pytest summary line, rc and failing ids.
# The job fails when any step does.
set -u
WS=$GITHUB_WORKSPACE; P=$WS/probe/review-probe; OUT=$WS/out; rm -rf "$OUT"; mkdir -p "$OUT"
export CHIP_ARCH=$ARCH TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=0
T() { date -u +%T; }
FAIL=0
step() { echo "$(T) $*" | tee -a "$OUT/timeline.txt"; }
step "start arch=$ARCH reps=$REPS sides=$SIDES runner=$(hostname)"
command -v python3 > /dev/null || { step "no python3 in the image"; exit 2; }
python3 "$P/summary.py" modules "$MODULES" > "$OUT/modules.txt" || { step "bad modules input"; exit 2; }
mapfile -t MODS < "$OUT/modules.txt"

# pyt <side> <log> <pytest args...>: pytest in that side's python_tests with its own RUNNER_TEMP; "rc=<rc>" ends the log
pyt() {
    local side=$1 log=$2 rc; shift 2
    (cd "$WS/$side/tt_metal/tt-llk/tests/python_tests" && RUNNER_TEMP=$WS/rt-$side PYTHONPATH=$P \
        pytest -p no:cacheprovider "$@") > "$log" 2>&1
    rc=$?
    echo "rc=$rc" >> "$log"; tail -n 2 "$log"
    [ "$rc" = 0 ] || FAIL=1
    return "$rc"
}

READY=()
for side in $SIDES; do
    step "setup $side"
    (cd "$WS/$side/tt_metal/tt-llk/tests" && ./setup_testing_env.sh) > "$OUT/setup_$side.log" 2>&1
    rc=$?
    echo "rc=$rc" >> "$OUT/setup_$side.log"
    if [ "$rc" = 0 ]; then READY+=("$side"); else tail -30 "$OUT/setup_$side.log"; FAIL=1; fi
done

i=0
for pair in "${MODS[@]}"; do
    i=$((i + 1))
    IFS='|' read -r FUNC PERF <<< "$pair"
    if [ -n "$FUNC" ]; then
        for side in "${READY[@]}"; do
            b=$OUT/m${i}_func_$side
            step "m$i func $side $FUNC"
            PROBE_HASH_OUT=$b.hash.tsv pyt "$side" "$b.log" -p probe_hash -q -rA --timeout=300 --junitxml="$b.xml" "$FUNC"
            python3 "$P/summary.py" rerun "$b.log" "$b.xml" "$FUNC" > "$b.rerun.txt"
            mapfile -t ids < "$b.rerun.txt"
            if [ "${#ids[@]}" -gt 0 ]; then
                step "m$i func rerun $side ${#ids[@]} ids, first ${ids[0]}"
                PROBE_HASH_OUT=$b.rerun.hash.tsv pyt "$side" "$b.rerun.log" -p probe_hash -q -rA --timeout=300 \
                    --junitxml="$b.rerun.xml" "${ids[@]}"
            fi
        done
    fi
    [ -n "$PERF" ] || continue
    BUILT=()
    for side in "${READY[@]}"; do
        step "m$i perf build $side $PERF"
        pyt "$side" "$OUT/m${i}_perf_build_$side.log" --compile-producer -n 10 -q --timeout=120 -m "perf and not accuracy" "$PERF" \
            && BUILT+=("$side")  # never run a consumer whose producer failed
    done
    for r in $(seq 1 "$REPS"); do
        for side in "${BUILT[@]}"; do
            b=$OUT/m${i}_perf_${side}_r$r
            step "m$i perf run $side r$r"
            rm -rf "$WS/$side/tt_metal/tt-llk/tests/python_tests/perf_data" "$WS/$side/tt_metal/tt-llk/perf_data"
            pyt "$side" "$b.log" --compile-consumer -q --timeout=300 -m "perf and not accuracy" --junitxml="$b.xml" "$PERF"
            for d in "$WS/$side/tt_metal/tt-llk/tests/python_tests/perf_data" "$WS/$side/tt_metal/tt-llk/perf_data"; do
                [ -d "$d" ] && tar czf "$b.$(basename "$(dirname "$d")").tgz" -C "$(dirname "$d")" perf_data
            done
        done
    done
done

python3 "$P/summary.py" summary "$OUT" "$FAIL" > "$OUT/summary.json.tmp" && mv "$OUT/summary.json.tmp" "$OUT/summary.json" \
    || { step "summary.json failed"; FAIL=1; }
step "done fail=$FAIL"
exit $FAIL
