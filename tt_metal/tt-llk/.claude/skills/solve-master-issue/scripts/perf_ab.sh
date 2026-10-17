#!/bin/bash
# usage: perf_ab.sh <fix-worktree> <base-sha> <arch> <perf-module> <out-dir> [--goal no_regress|improve]
#                   [-k <pytest expr>] [--op <mathop substring>] [--primary-metric <m>] [--speed-of-light]
# Same-board perf A/B of one tt-llk perf module: the fix tree ("current") against <base-sha> ("baseline",
# a detached worktree created next to the fix worktree and reused on re-runs), judged by perf_eval.py.
#   arch         the harness's CHIP_ARCH name (wormhole | blackhole | quasar | …) — must be the board in this box
#   perf-module  tests/python_tests/<perf-module>.py, e.g. perf_eltwise_unary_sfpu
#   out-dir      where the evidence lands: current.post.csv, baseline.post.csv, verdict.json, *.log
# Each tree runs the CI two-phase flow (compile-producer then compile-consumer, both under one device-lock hold),
# with PERF_RUN_TAG pinned so the combined CSV is at perf_data/runs/<tag>/<module>/<module>.post.csv.
# Current runs first, then baseline, back to back on the same card, same toolchain, same selector.
# Exit: 0 = pass (neutral/improved per goal) · 1 = regression · 2 = not comparable (no baseline rows,
# schema mismatch, empty selection — read verdict.json reason_code) · 3 = a perf run failed (env/plan,
# see the logs; never a verdict on the fix). The PR body quotes the summary line and links both CSVs.
set -uo pipefail
fix=$1; base=$2; arch=$3; mod=$4; out=$5; shift 5
goal=no_regress; kexpr=; op=; metric=; sol=()
while [ $# -gt 0 ]; do case $1 in
    --goal) goal=$2; shift 2;; -k) kexpr=$2; shift 2;; --op) op=$2; shift 2;;
    --primary-metric) metric=$2; shift 2;; --speed-of-light) sol=(--speed-of-light); shift;;
    *) echo "unknown arg $1" >&2; exit 3;; esac; done
[[ $arch =~ ^[a-z][a-z0-9_]*$ ]] || { echo "arch must be an architecture name the harness takes (CHIP_ARCH: wormhole, blackhole, quasar, …)" >&2; exit 3; }
here=$(dirname "$(readlink -f "$0")")
mkdir -p "$out"
fix=$(readlink -f "$fix")
basewt="${fix}-perfbase"
if [ ! -d "$basewt" ]; then
    git -C "$fix" worktree add -q --detach "$basewt" "$base" || { echo "cannot create baseline worktree" >&2; exit 3; }
    for l in sfpi .venv; do
        src=$(readlink -f "$fix/tt_metal/tt-llk/tests/$l" 2>/dev/null || true)
        [ -n "$src" ] && [ -e "$src" ] && ln -sfn "$src" "$basewt/tt_metal/tt-llk/tests/$l"
    done
else
    git -C "$basewt" checkout -q --detach "$base" || { echo "baseline worktree is not at $base" >&2; exit 3; }
fi
export TT_LLK_DISABLE_ASSERTS=${TT_LLK_DISABLE_ASSERTS:-1}
run_one() {  # run_one <tree> <label>  -> prints the post.csv path
    local tree=$1 label=$2 tag llk csv log
    llk=$tree/tt_metal/tt-llk
    tag="ab-$label-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    log=$out/$label.log
    [ -f "$llk/tests/python_tests/$mod.py" ] || { echo "no $mod.py in $llk/tests/python_tests" >&2; return 3; }
    local k=(); [ -n "$kexpr" ] && k=(-k "$kexpr")
    # Both phases under ONE device-lock hold: the producer compiles into the shared /tmp/tt-llk-build, which any
    # other runner session wipes at its own compile start, so an unlocked producer races every other test run.
    (
        cd "$llk/tests" || exit 3
        export CHIP_ARCH=$arch PERF_RUN_TAG=$tag
        echo "== $label: $tree @ $(git -C "$tree" rev-parse --short=11 HEAD) tag=$tag" >> "$log"
        exec 9>"/tmp/tt-device-$(hostname).lock"; flock 9 || exit 3
        # A hung test from an earlier session leaves the card wedged (xdist workers die at startup): reset first.
        tt-smi -r 0 > /dev/null 2>&1 || true
        .venv/bin/python -m pytest -q --override-ini=log_cli=false "${sol[@]}" --compile-producer -n 10 \
            -m "perf and not accuracy" --timeout=120 "${k[@]}" "./python_tests/$mod.py" >> "$log" 2>&1 || exit 3
        .venv/bin/python -m pytest -q --override-ini=log_cli=false "${sol[@]}" --compile-consumer --dist loadgroup -n 15 \
            -m "perf and not accuracy" --timeout=120 "${k[@]}" "./python_tests/$mod.py" >> "$log" 2>&1 || exit 3
    ) || return 3
    csv="$llk/perf_data/runs/$tag/$mod/$mod.post.csv"
    [ -s "$csv" ] || { echo "no combined CSV at $csv (see $log)" >&2; return 3; }
    cp "$csv" "$out/$label.post.csv"
    echo "$out/$label.post.csv"
}
cur=$(run_one "$fix" current) || { echo "PERF_RUN_FAILED current (see $out/current.log)"; exit 3; }
bas=$(run_one "$basewt" baseline) || { echo "PERF_RUN_FAILED baseline (see $out/baseline.log) — not a verdict on the fix"; exit 3; }
args=(--current "$cur" --baseline "$bas" --goal "$goal" --test "$mod" --json-out "$out/verdict.json")
[ -n "$op" ] && args+=(--op "$op")
[ -n "$metric" ] && args+=(--primary-metric "$metric")
python3 -I "$here/perf_eval.py" "${args[@]}" 2> "$out/verdict.stderr.json"
rc=$?
echo "evidence: $out/{current,baseline}.post.csv verdict.json (fix $(git -C "$fix" rev-parse --short=11 HEAD) vs base $(git -C "$basewt" rev-parse --short=11 HEAD), $arch, $(hostname))"
exit $rc
