#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_blackhole.sh <group> <n_groups>
# Optional: LLK_PERF_PHASE=all|compile|run, LLK_PERF_COMPILE_WORKERS, LLK_PERF_DEBUG_SYMBOLS,
# LLK_PERF_EXITFIRST, LLK_PERF_REPORT_TAG (JUnit name suffix). Defaults keep today's commands.
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

PHASE="${LLK_PERF_PHASE:-all}"
case "$PHASE" in
  all|compile|run) ;;
  *)
    echo "LLK_PERF_PHASE must be 'all', 'compile' or 'run', got '$PHASE'" >&2
    exit 2
    ;;
esac
COMPILE_WORKERS="${LLK_PERF_COMPILE_WORKERS:-10}"

case "${LLK_PERF_DEBUG_SYMBOLS:-true}" in
  true) DEBUG_SYMBOL_ARGS=() ;;
  false) DEBUG_SYMBOL_ARGS=(--no-debug-symbols) ;;
  *)
    echo "LLK_PERF_DEBUG_SYMBOLS must be 'true' or 'false', got '$LLK_PERF_DEBUG_SYMBOLS'" >&2
    exit 2
    ;;
esac

case "${LLK_PERF_EXITFIRST:-true}" in
  true) EXITFIRST_ARGS=(-x) ;;
  false) EXITFIRST_ARGS=() ;;
  *)
    echo "LLK_PERF_EXITFIRST must be 'true' or 'false', got '$LLK_PERF_EXITFIRST'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

REPORT="pytest-report-blackhole-${GROUP}${LLK_PERF_REPORT_TAG:-}"
COMPILE_XML="${REPORT}-compile.xml"
RUN_XML="${REPORT}-run.xml"

export PERF_KEEP_RUNS=200
TEMP_ROOT="${RUNNER_TEMP:-$(mktemp -d)}"
export PYTEST_ADDOPTS=""
# Experiment: bhperf (issues 60844, 46474). Samples every STRIDE-th test of this group plus all perf_pack_untilize tests, then
# measures them in legs: base (+ reruns), pure code shifts, NOPs at the top of the pack/unpack/math inits.
STRIDE="${EXP_STRIDE:-10}"
pytest -q --collect-only --compile-consumer "${SPEED_OF_LIGHT_ARGS[@]}" -p no:cacheprovider -m "perf and not accuracy" \
  --perf-splits "$N_GROUPS" --perf-group "$GROUP" . 2> /dev/null | grep '::' > grp_ids.txt || true
grep 'perf_pack_untilize.py' grp_ids.txt > u_ids.txt || true
grep -v 'perf_pack_untilize.py' grp_ids.txt | awk -v s="$STRIDE" 'NR % s == 0' > s_ids.txt
cat s_ids.txt u_ids.txt > leg_ids.txt
python3 - << 'PYEOF'
import re
out = []
for l in open("u_ids.txt"):
    m = re.search(r"benchmark:Standard-formats:Bfp8_b->\w+.*full_rt_dim:(\d+)-full_ct_dim:(\d+)", l)
    if m and int(m.group(1)) * int(m.group(2)) >= 32:
        out.append(l)
open("u46_ids.txt", "w").writelines(out)
PYEOF
tac u46_ids.txt > u46_ids_rev.txt
wc -l grp_ids.txt s_ids.txt u_ids.txt
files_of() { cut -d: -f1 "$1" | sort -u; }
run() { # run <tag> <idfile> <pytest args...>
  local tag=$1 ids=$2; shift 2
  mapfile -t FILES < <(files_of "$ids")
  env PERF_RUN_TAG="$tag" pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer --timeout=600 \
    -p no:cacheprovider -m "perf and not accuracy" --node-list="$(realpath "$ids")" "$@" "${FILES[@]}" || echo "run $tag rc=$?"
  echo "done $tag $(date -u +%T)"
}
build() { # build <leg> <cflags> <shift> <objcache>
  export RUNNER_TEMP="$TEMP_ROOT/leg-$1" LLK_EXP_CFLAGS="$2" LLK_CODE_SHIFT="$3"
  if [[ -n "$4" ]]; then export LLK_OBJ_CACHE="$TEMP_ROOT/objs-$4"; mkdir -p "$LLK_OBJ_CACHE"; else unset LLK_OBJ_CACHE; fi
  mapfile -t FILES < <(files_of leg_ids.txt)
  pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-producer -n "$COMPILE_WORKERS" -m "perf and not accuracy" \
    --timeout=600 --node-list="$(realpath leg_ids.txt)" "${FILES[@]}" || echo "build $1 rc=$?"
  echo "built $1 $(date -u +%T)"
}
X=(--dist loadgroup -n 15)
build base "" 0 base
run base_x1 leg_ids.txt "${X[@]}"
run base_x2 leg_ids.txt "${X[@]}"
[[ -s u46_ids.txt ]] && run base_ufwd u46_ids.txt -n 1
[[ -s u46_ids.txt ]] && run base_urev u46_ids_rev.txt -n 1
[[ -s u_ids.txt ]] && run base_ux3 u_ids.txt "${X[@]}" && run base_ux4 u_ids.txt "${X[@]}"
for sh in 4 36; do
  build sh$sh "" $sh base
  run sh${sh}_x1 leg_ids.txt "${X[@]}"
done
for t in PACK UNPACK MATH; do
  build nop$t "-DLLK_EXP_NOP_${t}_INIT=1" 0 ""
  run nop${t}_x1 leg_ids.txt "${X[@]}"
done
build nopALL "-DLLK_EXP_NOP_PACK_INIT=2 -DLLK_EXP_NOP_UNPACK_INIT=2 -DLLK_EXP_NOP_MATH_INIT=2" 0 ""
run nopALL_x1 leg_ids.txt "${X[@]}"
cp "$COMPILE_XML" "${REPORT}.xml" 2> /dev/null || touch "${REPORT}.xml"
exit 0

if [[ "$PHASE" != run ]]; then
  pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-producer -n "$COMPILE_WORKERS" -m "perf and not accuracy" --timeout=60 \
    --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
    --junitxml="$COMPILE_XML" .
fi
[[ "$PHASE" == compile ]] && exit 0

pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer --dist loadgroup -n 15 "${EXITFIRST_ARGS[@]}" -m "perf and not accuracy" --timeout=60 \
  --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
  --junitxml="$RUN_XML" .
if [[ -f "$COMPILE_XML" ]]; then
  junitparser merge "$COMPILE_XML" "$RUN_XML" "${REPORT}.xml"
else
  cp "$RUN_XML" "${REPORT}.xml"
fi
