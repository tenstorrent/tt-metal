#!/usr/bin/env bash
# Queue a 5-seed fl2va generation on the device broker; the only script here that touches the device.
#
#   gen_seeds.sh OUT_DIR [extra KEY=VALUE env ...]
#   e.g. gen_seeds.sh tt-project/baselines/dense BASE_SECONDS=10 BASE_HEIGHT=1088 BASE_WIDTH=1920
#        gen_seeds.sh tt-project/runs/vsa09 BASE_VSA_SPARSITY=0.9
#
# Seeds default to 0-4 (BASE_SEEDS overrides). Runs the code in REPO (default: the main checkout,
# the one with python_env and a build; point REPO at a built worktree to test its code). Prints the
# broker job id; follow it with
# `tt-device-mcp status -j ID` / `tt-device-mcp logs ID`. The artifacts land in OUT_DIR/<tag>/.
set -euo pipefail
[[ $# -ge 1 ]] || { sed -n '2,10p' "$0"; exit 2; }
out=$(realpath -m "$1"); shift
root=/home/smarton/fasth3/tt-metal
repo=${REPO:-$root}
[[ -x $repo/python_env/bin/python ]] || { echo "no python_env in $repo; build it or set REPO" >&2; exit 1; }
keyframes=$root/tt-project/baselines/keyframes

env_args=(
  MINIMAX_H3_MODEL_PATH="${MINIMAX_H3_MODEL_PATH:-/mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3}"
  TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$root/tt_dit_cache}"
  BASE_OUT="$out"
  BASE_SEEDS="${BASE_SEEDS:-0,1,2,3,4}"
  BASE_FIRST="${BASE_FIRST:-$keyframes/first.png}"
  BASE_LAST="${BASE_LAST:-$keyframes/last_pushin80.png}"
  "$@"
)
cmd="cd $repo && env ${env_args[*]} models/tt_dit/models/transformers/minimax_h3/scripts/run_h3_test.sh \
models/tt_dit/tests/models/minimax_h3/test_fasth3_baseline_minimax_h3.py -x -s --timeout ${GEN_TIMEOUT:-10700}"
echo "$cmd"
[[ -n "${DRY_RUN:-}" ]] && exit 0
tt-device-mcp run-bg -w "$repo" -t "${GEN_TIMEOUT:-10800}" "$cmd"
