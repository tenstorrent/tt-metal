#!/bin/bash
# Review probe (not for main): A/B of LLK test modules on this runner's card, every side on the same card.
# env: ARCH, REPS, SIDES ("base head" or "base head main"), MODULES (JSON list of {"func": file, "perf": file}, "" when
# absent), <SIDE>_SHA; checkouts in $GITHUB_WORKSPACE/<side>, results in $GITHUB_WORKSPACE/out.
# Per module and side: the functional module once, then once more only its failing ids (at most 50, at the first run's
# 300 s per test, so a slow test does not count as reproduced by a shorter timeout), or the whole module when pytest
# crashed (no junit xml); then the perf module is built and its reps run interleaved
# over the sides. out/summary.json (summary.py) lists every step's pytest summary line, rc and failing ids.
# The job fails when any step does.
# Two optional extra sides, built from head's checkout after its setup, functional modules only (no perf, so a dispatch
# without them is unchanged and the extra cost is one more functional pass): "merged" (env MERGED=true, needs main:
# head's diff against base applied onto main's tree, 3-way; a conflict leaves the side out) and "patch" (env PATCH_B64,
# gzip+base64 of a unified diff against head, a mutant: applied to a copy of head after `git apply --check`; a patch
# that does not apply leaves the side out (an empty diff of head against base: the side is main itself, "applied (no head
# changes)"; a setup that fails on an extra side fails the job); no rerun of its failing ids, a failing test is the expected result and does
# not fail the job). summary.json gets "merged"/"patch": "applied" or "not applied: <why>" when asked for.
set -u
WS=$GITHUB_WORKSPACE; P=$WS/probe/review-probe; OUT=$WS/out; rm -rf "$OUT"; mkdir -p "$OUT"
export CHIP_ARCH=$ARCH TT_LLK_DISABLE_ASSERTS=1 PERF_KEEP_RUNS=0
T() { date -u +%T; }
FAIL=0
NOTES=$OUT/notes.txt
note() { printf '%s\t%s\n' "$1" "$(printf '%s' "$2" | tr '\n\t' '  ' | head -c 400)" >> "$NOTES"; }
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
    [ "$rc" = 0 ] || [ "$side" = patch ] || FAIL=1  # the mutant's tests are meant to fail
    return "$rc"
}

READY=()
setup_side() {
    local side=$1 rc
    step "setup $side"
    (cd "$WS/$side/tt_metal/tt-llk/tests" && ./setup_testing_env.sh) > "$OUT/setup_$side.log" 2>&1
    rc=$?
    echo "rc=$rc" >> "$OUT/setup_$side.log"
    if [ "$rc" = 0 ]; then READY+=("$side"); else tail -30 "$OUT/setup_$side.log"; FAIL=1; fi
    return "$rc"
}
for side in $SIDES; do setup_side "$side"; done

# extra sides: copies of head's checkout (its sfpi comes along, so their setup is a version check)
head_ready=0; for side in "${READY[@]}"; do [ "$side" = head ] && head_ready=1; done
if [ "${MERGED:-}" = true ]; then
    if [ -z "${MAIN_SHA:-}" ] || [ ! -d "$WS/main" ]; then
        note merged "not applied: no main sha given"
    elif [ "$head_ready" != 1 ]; then
        note merged "not applied: head setup failed"
    else
        step "merged: head onto main"
        cp -a "$WS/head" "$WS/merged"
        why=$(cd "$WS/merged" && H=$(git rev-parse HEAD) \
            && git fetch -q --depth=1 "$WS/base" HEAD 2>&1 && B=$(git rev-parse FETCH_HEAD) \
            && git fetch -q --depth=1 "$WS/main" HEAD 2>&1 && M=$(git rev-parse FETCH_HEAD) \
            && git diff --binary "$B" "$H" > "$WS/pr.diff" \
            && git checkout -q -f "$M" 2>&1 \
            && { [ ! -s "$WS/pr.diff" ] || git apply --3way --index "$WS/pr.diff" 2>&1; } \
            && echo "$H+$M" > "$WS/merged.sha") || why="${why:-git failed}"
        if [ -s "$WS/merged.sha" ] && [ -z "$(cd "$WS/merged" && git diff --name-only --diff-filter=U)" ]; then
            export MERGED_SHA=$(cat "$WS/merged.sha")
            if setup_side merged; then SIDES="$SIDES merged"; if [ -s "$WS/pr.diff" ]; then note merged applied; else note merged "applied (no head changes)"; fi; else note merged "not applied: setup failed"; fi
        else
            # git apply lists every file it applied cleanly before the one that failed: keep the lines that say why
            err=$(printf '%s\n' "$why" | grep -Ei "error|conflict|^U |does not|fatal|already exists" | head -5)
            note merged "not applied: ${err:-$(printf '%s' "$why" | tail -c 380)}"
            rm -rf "$WS/merged"
        fi
    fi
fi
if [ -n "${PATCH_B64:-}" ]; then
    if [ "$head_ready" != 1 ]; then
        note patch "not applied: head setup failed"
    elif ! why=$(python3 "$P/summary.py" patch "$PATCH_B64" "$OUT/patch.diff" 2>&1); then
        note patch "not applied: $why"
    else
        step "patch: $why"
        cp -a "$WS/head" "$WS/patch"
        if why=$(cd "$WS/patch" && git apply --check "$OUT/patch.diff" 2>&1 && git apply "$OUT/patch.diff" 2>&1); then
            note patch applied
            note patch_sha256 "$(sha256sum "$OUT/patch.diff" | cut -d' ' -f1)"
            if setup_side patch; then export PATCH_SHA=$HEAD_SHA; SIDES="$SIDES patch"; else note patch "not applied: setup failed"; fi
        else
            note patch "not applied: $why"
            rm -rf "$WS/patch"
        fi
    fi
fi
PERF_SIDES=(); for side in "${READY[@]}"; do [ "$side" = merged ] || [ "$side" = patch ] || PERF_SIDES+=("$side"); done

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
            if [ "${#ids[@]}" -gt 0 ] && [ "$side" != patch ]; then
                step "m$i func rerun $side ${#ids[@]} ids, first ${ids[0]}"
                PROBE_HASH_OUT=$b.rerun.hash.tsv pyt "$side" "$b.rerun.log" -p probe_hash -q -rA --timeout=300 \
                    --junitxml="$b.rerun.xml" "${ids[@]}"
            fi
        done
    fi
    [ -n "$PERF" ] || continue
    BUILT=()
    for side in "${PERF_SIDES[@]}"; do
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
