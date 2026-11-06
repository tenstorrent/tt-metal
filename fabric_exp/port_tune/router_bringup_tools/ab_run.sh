#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Router bring-up tools (temporary: drop before merge; see README.md).
# ab_run.sh: the same test_tt_fabric bandwidth yamls on two built trees (A and B, e.g. main and the new router), in two
# passes with the arm order reversed in the second (pass a: A then B per yaml; pass b: B then A), telemetry off. Each
# run's summary and per-sender CSVs and its log go to OUT; ab_report.py turns them into cycles/packet with pass noise.
#
#   ab_run.sh --tree-a DIR --tree-b DIR --out DIR [--name-a main] [--name-b new] [--passes 2] [--timeout 1800]
#             [--env-a "K=V ..."] [--env-b "K=V ..."] [--env "K=V ..."] YAML [YAML ...]
#
#   --tree-a/b   built tt-metal trees (each runs its own build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric, with
#                TT_METAL_HOME set to the tree); one kernel cache per arm under OUT/cache_<name>
#   --env*       extra environment for one arm or both (e.g. --env-b "TT_METAL_FABRIC_OPT_LEVEL=O3")
#   YAML         test_tt_fabric --test_config files (absolute or relative to the current directory)
# Every run: TT_FABRIC_NO_BW_TELEMETRY=1 (router bandwidth telemetry off), TT_METAL_CLEAR_L1=1,
# TT_METAL_DEVICE_PROFILER=0. Labels: <pass>_<name>_<yaml stem> (.log, .csv = summary, _tests.csv = per sender).
# A run that times out or fails stops the script (the device may need a reset before the next run).
set -uo pipefail
tree_a="" tree_b="" out="" name_a=main name_b=new passes=2 to=1800 env_a="" env_b="" env_all=""
while [ $# -gt 0 ]; do
    case $1 in
        --tree-a) tree_a=$2; shift 2 ;;
        --tree-b) tree_b=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --name-a) name_a=$2; shift 2 ;;
        --name-b) name_b=$2; shift 2 ;;
        --passes) passes=$2; shift 2 ;;
        --timeout) to=$2; shift 2 ;;
        --env-a) env_a=$2; shift 2 ;;
        --env-b) env_b=$2; shift 2 ;;
        --env) env_all=$2; shift 2 ;;
        -h|--help) sed -n '5,22p' "$0"; exit 0 ;;
        --*) echo "unknown option $1"; exit 1 ;;
        *) break ;;
    esac
done
[ -n "$tree_a" ] && [ -n "$tree_b" ] && [ -n "$out" ] && [ $# -gt 0 ] || { sed -n '5,22p' "$0"; exit 1; }
[ "$name_a" != "$name_b" ] || { echo "!!! --name-a and --name-b must differ"; exit 1; }
case $passes in 1|2) ;; *) echo "!!! --passes 1 or 2"; exit 1 ;; esac
yamls=()
for y in "$@"; do
    [ -f "$y" ] || { echo "!!! no yaml $y"; exit 1; }
    yamls+=("$(cd "$(dirname "$y")" && pwd)/$(basename "$y")")
done
for t in "$tree_a" "$tree_b"; do
    [ -x "$t/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric" ] ||
        { echo "!!! $t has no build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric (build it first)"; exit 1; }
done
tree_a=$(cd "$tree_a" && pwd); tree_b=$(cd "$tree_b" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
{ echo "tree_a=$tree_a ($name_a) $(git -C "$tree_a" rev-parse --short HEAD 2>/dev/null)"
  echo "tree_b=$tree_b ($name_b) $(git -C "$tree_b" rev-parse --short HEAD 2>/dev/null)"
  echo "env_a=$env_a"; echo "env_b=$env_b"; echo "env=$env_all"; echo "yamls=${yamls[*]}"
  echo "host=$(hostname) start=$(date -u +%FT%TZ)"; } > "$out/ab_run.txt"

one() {  # pass name tree extra-env yaml
    local p=$1 name=$2 tree=$3 extra=$4 y=$5 stem label rc
    stem=$(basename "$y" .yaml); label=${p}_${name}_${stem}
    [ -e "$out/$label.log" ] && { echo "!!! $label exists in $out: use another --out"; exit 1; }
    rm -f "$tree"/generated/fabric/bandwidth_summary_results_*.csv "$tree"/generated/fabric/bandwidth_results_*.csv
    echo "=== $label ($(date -u +%T))"
    # shellcheck disable=SC2086
    (cd "$tree" && env TT_METAL_HOME="$tree" TT_METAL_CACHE="$out/cache_$name" TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 $env_all $extra \
        timeout "$to" ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$y" \
        > "$out/$label.log" 2>&1)
    rc=$?
    for f in "$tree"/generated/fabric/bandwidth_summary_results_*.csv; do [ -f "$f" ] && cp "$f" "$out/$label.csv"; done
    for f in "$tree"/generated/fabric/bandwidth_results_*.csv; do [ -f "$f" ] && cp "$f" "$out/${label}_tests.csv"; done
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    echo "$label: exit $rc; ${res:-NO RESULT}"
    if [ $rc = 124 ]; then echo "!!! $label timed out after $to s (hang?): stopping"; exit 1; fi
    if [ $rc != 0 ] || ! echo "$res" | grep -q ", 0 failed"; then echo "!!! $label failed: stopping"; exit 1; fi
}
for y in "${yamls[@]}"; do
    one a "$name_a" "$tree_a" "$env_a" "$y"
    one a "$name_b" "$tree_b" "$env_b" "$y"
done
if [ "$passes" = 2 ]; then
    for y in "${yamls[@]}"; do
        one b "$name_b" "$tree_b" "$env_b" "$y"
        one b "$name_a" "$tree_a" "$env_a" "$y"
    done
fi
echo "done $(date -u +%T); report: ab_report.py $out"
