#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Wormhole LLK perf runner, shared by the 5 wh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# EXPERIMENT BRANCHES ONLY (mvlahovic/exp-*): knobs read from tests/exp_ci.env, all optional:
#   EXP_IDS=<argfile, relative to tests/python_tests>  run these pytest node ids instead of the pytest-split shard
#   EXP_IDS_SPLIT=1      split EXP_IDS round robin across the groups (default 0: every group runs all of EXP_IDS,
#                        which is the cross-runner check: five cards measure the same nodes)
#   EXP_LEGS="base= n4=-DLLK_EXP_NOP_PACK_INIT=4"   legs measured one after another on this runner (same-card A/B);
#                        a leg is <name>=<LLK_EXP_CFLAGS, commas for spaces>: its own producer (RUNNER_TEMP/leg_<name>)
#                        and consumer (PERF_RUN_TAG=<tag>-<name>); default one leg "base" with no flags
#   EXP_PRODUCER_TIMEOUT=<s>  per-test timeout of the producer (default 3600: layout work exceeds 60 s on CI CPUs)
# Each leg's results land in perf_data/runs/<tag>-<name>/ with build_manifest.json.gz (exp_manifest.py: loaded
# sections of every ELF and the layout choices, path independent, to compare the CI build with a local one).
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_wormhole.sh <group> <n_groups>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-true}"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"

case "$SPEED_OF_LIGHT" in
  true)
    SPEED_OF_LIGHT_ARGS=(--speed-of-light)
    ;;
  false)
    SPEED_OF_LIGHT_ARGS=()
    ;;
  *)
    echo "SPEED_OF_LIGHT must be 'true' or 'false', got '$SPEED_OF_LIGHT'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_IDS="" EXP_IDS_SPLIT=0 EXP_LEGS="base=" EXP_PRODUCER_TIMEOUT=3600
[ -f "$SCRIPT_DIR/exp_ci.env" ] && source "$SCRIPT_DIR/exp_ci.env"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

if [ -n "$EXP_IDS" ]; then
  mapfile -t ALL_IDS < <(grep -v '^\s*\(#\|$\)' "$EXP_IDS")
  SEL=()
  for i in "${!ALL_IDS[@]}"; do
    if [ "$EXP_IDS_SPLIT" != 1 ] || [ $((i % N_GROUPS)) -eq $((GROUP - 1)) ]; then SEL+=("${ALL_IDS[$i]}"); fi
  done
  echo "exp: ${#SEL[@]} of ${#ALL_IDS[@]} node ids from $EXP_IDS (split=$EXP_IDS_SPLIT)"
else
  SEL=(--splits "$N_GROUPS" --group "$GROUP" .)
fi

BASE_TAG="${PERF_RUN_TAG:?PERF_RUN_TAG is set by the workflow}"
BASE_RT="${RUNNER_TEMP:-/tmp}"
for LEG in $EXP_LEGS; do
  NAME="${LEG%%=*}"
  FLAGS="${LEG#*=}"
  FLAGS="${FLAGS//,/ }"
  echo "exp: leg $NAME LLK_EXP_CFLAGS='$FLAGS'"
  export RUNNER_TEMP="$BASE_RT/leg_$NAME" PERF_RUN_TAG="$BASE_TAG-$NAME" LLK_EXP_CFLAGS="$FLAGS"
  mkdir -p "$RUNNER_TEMP"
  T0=$(date +%s)
  pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 -m "perf and not accuracy" --timeout="$EXP_PRODUCER_TIMEOUT" \
    --junitxml="pytest-report-wormhole-${GROUP}-${NAME}-compile.xml" "${SEL[@]}"
  T1=$(date +%s)
  pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer --dist loadgroup -n 15 -x -m "perf and not accuracy" --timeout=60 \
    --junitxml="pytest-report-wormhole-${GROUP}-${NAME}-run.xml" "${SEL[@]}"
  T2=$(date +%s)
  OUT="../../perf_data/runs/$PERF_RUN_TAG"
  mkdir -p "$OUT"
  python3 "$SCRIPT_DIR/exp_manifest.py" "$RUNNER_TEMP" "$OUT/build_manifest.json.gz" || echo "exp: manifest failed"
  echo "leg=$NAME group=$GROUP producer_s=$((T1 - T0)) consumer_s=$((T2 - T1)) nproc=$(nproc) flags='$FLAGS'" | tee "$OUT/exp_leg.txt"
  rm -rf "$RUNNER_TEMP"
done
