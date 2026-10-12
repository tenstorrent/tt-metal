#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# EXPERIMENT BRANCHES ONLY (mvlahovic/exp-ncw-*): legs from tests/exp_ci.env, see the loop below.
# them (consumer) -- one invocation each over the whole perf suite.
#
# There is no durations file, so pytest-split gives every test the same weight.
# The default algorithm then cuts the suite into contiguous chunks by count, and
# a chunk that holds the compile-heavy SFPU and untilize modules takes ~7x longer
# than the rest. least_duration deals the tests out in turn, so every shard gets
# a share of each module. Both passes must use the same split.
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_blackhole.sh <group> <n_groups>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
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

ARCH_NAME=blackhole
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_IDS="" EXP_IDS_SPLIT=0 EXP_LEGS="base=" EXP_PRODUCER_TIMEOUT=3600 EXP_EXTRA="" EXP_ZONE_DUMP=0
[ -f "$SCRIPT_DIR/exp_ci.env" ] && source "$SCRIPT_DIR/exp_ci.env"
RT_DEFAULT="${LLK_PERF_RUN_TYPES:-}"
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
  printf '%s\n' "${SEL[@]}" > exp_sel_ids.txt
  SEL=(@exp_sel_ids.txt)
else
  SEL=(--splits "$N_GROUPS" --group "$GROUP" --splitting-algorithm least_duration .)
fi

BASE_TAG="${PERF_RUN_TAG:?PERF_RUN_TAG is set by the workflow}"
BASE_RT="${RUNNER_TEMP:-/tmp}"
for LEG in $EXP_LEGS; do
  NAME="${LEG%%=*}"
  FLAGS="${LEG#*=}"
  FLAGS="${FLAGS//,/ }"
  # tokens: RT:<a+b> this leg's LLK_PERF_RUN_TYPES, ENV:<NAME>=<value> an exported variable, ARG:<pytest arg> added to
  # both passes of this leg (e.g. ARG:--enable-perf-counters); the rest is LLK_EXP_CFLAGS
  LEG_RT=""; KEEP=""; LEG_ENV=""; LEG_ARGS=""
  for TOK in $FLAGS; do case "$TOK" in RT:*) LEG_RT="${TOK#RT:}" ;; ENV:*) export "${TOK#ENV:}"; LEG_ENV="$LEG_ENV ${TOK#ENV:}" ;; ARG:*) LEG_ARGS="$LEG_ARGS ${TOK#ARG:}" ;; *) KEEP="$KEEP $TOK" ;; esac; done
  FLAGS="${KEEP# }"
  if [ -n "$LEG_RT" ]; then export LLK_PERF_RUN_TYPES="${LEG_RT//+/,}"; elif [ -n "$RT_DEFAULT" ]; then export LLK_PERF_RUN_TYPES="$RT_DEFAULT"; else unset LLK_PERF_RUN_TYPES; fi
  echo "exp: leg $NAME LLK_EXP_CFLAGS='$FLAGS' args='$LEG_ARGS' LLK_PERF_RUN_TYPES='${LLK_PERF_RUN_TYPES:-}'"
  export RUNNER_TEMP="$BASE_RT/leg_$NAME" PERF_RUN_TAG="$BASE_TAG-$NAME" LLK_EXP_CFLAGS="$FLAGS"
  mkdir -p "$RUNNER_TEMP"
  OUT="../../perf_data/runs/$PERF_RUN_TAG"
  if [ "$EXP_ZONE_DUMP" = 1 ]; then export LLK_EXP_ZONE_DUMP="$RUNNER_TEMP/zones"; fi
  T0=$(date +%s)
  pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" $EXP_EXTRA $LEG_ARGS --compile-producer -n 10 -m "perf and not accuracy" --timeout="$EXP_PRODUCER_TIMEOUT" \
    --junitxml="pytest-report-${ARCH_NAME}-${GROUP}-${NAME}-compile.xml" "${SEL[@]}" || echo "exp: producer of leg $NAME failed"
  T1=$(date +%s)
  pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" $EXP_EXTRA $LEG_ARGS --compile-consumer --dist loadgroup -n 15 -m "perf and not accuracy" --timeout=120 \
    --junitxml="pytest-report-${ARCH_NAME}-${GROUP}-${NAME}-run.xml" "${SEL[@]}" || echo "exp: consumer of leg $NAME failed"
  T2=$(date +%s)
  mkdir -p "$OUT"
  python3 "$SCRIPT_DIR/exp_manifest.py" "$RUNNER_TEMP" "$OUT/build_manifest.json.gz" || echo "exp: manifest failed"
  if [ -d "$RUNNER_TEMP/zones" ]; then cat "$RUNNER_TEMP"/zones/*.jsonl | gzip > "$OUT/zones.jsonl.gz"; fi
  echo "leg=$NAME group=$GROUP producer_s=$((T1 - T0)) consumer_s=$((T2 - T1)) flags='$FLAGS' args='$LEG_ARGS' rts='${LLK_PERF_RUN_TYPES:-}'" | tee "$OUT/exp_leg.txt"
  rm -rf "$RUNNER_TEMP"
  for V in $LEG_ENV; do unset "${V%%=*}"; done
done
