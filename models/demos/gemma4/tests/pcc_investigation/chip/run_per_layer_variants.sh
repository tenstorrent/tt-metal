#!/bin/bash
# Per-layer PCC on the chips (QB2 1x4) for several versions of the Gemma 4 code, then back to the starting branch.
#   prefill: tt_per_layer_pcc.py on the unmodified port (b809f097c06) and the optimized tree (36b3922cc29)
#   decode:  tt_per_layer_pcc_decode.py on the fp32-switch branch (87505f06886), switches off (bf16) and on (fp32)
# The scripts are copied out first because checking out another commit removes them from the working tree.
# Results: $GEMMA4_PCC_DATA/tt_per_layer_pcc*_<label>.json, logs in $GEMMA4_PCC_DATA/logs/.
set -u
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DATA=${GEMMA4_PCC_DATA:-$HOME/benchmark-data/gemma4-pcc}
cd "$REPO"
BACK=$(git branch --show-current)
TMP=$(mktemp -d); cp "$(dirname "$0")"/tt_per_layer_pcc.py "$(dirname "$0")"/tt_per_layer_pcc_decode.py "$TMP"/
trap 'git checkout -q "$BACK"; rm -rf "$TMP"; echo "restored $(git branch --show-current) $(git log -1 --format=%h)"' EXIT
mkdir -p "$DATA/logs"
run() {  # <commit> <script> <label> [env assignments...]
  local c=$1 s=$2 label=$3; shift 3
  git checkout -q "$c" && echo "checked out $(git log -1 --format=%h) for $label"
  # each script points TT_CACHE_PATH at $DATA/tt_cache_perlayer*_<label>: start fresh, delete afterwards
  rm -rf "$DATA"/tt_cache_perlayer*_"$label"
  env "$@" GEMMA4_PCC_DATA="$DATA" GEMMA4_PCC_REPO="$REPO" timeout 3600 python_env/bin/python "$TMP/$s" "$label" > "$DATA/logs/tt-per-layer-$label.log" 2>&1
  echo "$label exit rc=$?"
  rm -rf "$DATA"/tt_cache_perlayer*_"$label"
}
run b809f097c06 tt_per_layer_pcc.py unmodified
run 36b3922cc29 tt_per_layer_pcc.py optimized
run 87505f06886 tt_per_layer_pcc_decode.py bf16
run 87505f06886 tt_per_layer_pcc_decode.py fp32 GEMMA4_FP32_ACTIVATIONS=1 GEMMA4_ROUTER_TOPK_ON_SCORES=1 GEMMA4_FP32_ATTENTION=1
