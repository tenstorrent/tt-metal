#!/bin/bash
# t288: one broker job running the Turbo e2e test (test_pipeline_turbo_minimax_h3.py) under the
# HyperFlow 8-step adapter, fl2va, 5 s, 4x8, with AdaLN precomputed (arm on) or on device (arm off).
# Args: <tag> <on|off> [cache]. Code: ~/fasth3/t286 at ttp/t288-adaln-ab (python-only on the t286 build).
# "cache" (or T288_CACHE_ONLY=1) builds the pipeline without warmup or generation: it writes the
# arm's transformer weight cache (and, for on, the precomputed AdaLN table) so the timed job is warm.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t288; W=${T288_W:-/home/smarton/fasth3/t286}
# Run the work in its own process group and kill the whole group on exit or signal, so no pytest
# child outlives the broker job.
if [ -z "$T288_INNER" ]; then
  T288_INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
TAG=${1:?tag}; ARM=${2:?on|off}; MODE=${3:-}
case $ARM in on) PRE=1;; off) PRE=0;; *) echo "[t288] arm must be on or off"; exit 2;; esac
case $MODE in cache) CACHE_ONLY=1;; "") CACHE_ONLY=${T288_CACHE_ONLY:-0};; *) echo "[t288] mode must be cache or empty"; exit 2;; esac
# T288_ONCE=1: this tag already passed in an earlier job, so a queued repeat exits before the device.
[ "${T288_ONCE:-0}" = 1 ] && [ -e $T/out_$TAG/PASS ] && { echo "[t288] $TAG already passed"; exit 0; }
# Stop before any large write if / is too full (blx03 cap 85%) or our outputs and caches have grown
# past T288_FOOT_MAX GB (the on arm adds a precomputed-AdaLN transformer cache).
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le ${T288_ROOT_MAX:-85} ] || { echo "[t288] / at $use%"; exit 5; }
gb=$(timeout 60 du -csxBG $T $F/cache/dit-h3hf $F/cache/tt-metal-cache-h3hf 2>/dev/null | tail -1 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le ${T288_FOOT_MAX:-200} ] || { echo "[t288] footprint ${gb}G"; exit 5; }
OUT=$T/out_$TAG; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
# Weights must be on local disk, checked readable before submit (never /mnt or other network fs).
export MINIMAX_H3_MODEL_PATH=$F/models/MiniMax-H3
[ -e $MINIMAX_H3_MODEL_PATH/READ_OK ] || { echo "[t288] no READ_OK in $MINIMAX_H3_MODEL_PATH"; exit 4; }
[ -e $F/models/hyperflow/READ_OK ] || { echo "[t288] no READ_OK in $F/models/hyperflow"; exit 4; }
export MINIMAX_H3_TURBO_LORA_PATH=$F/models/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors
export MINIMAX_H3_TURBO_KEYFRAME=$F/t209/kf_first.png MINIMAX_H3_TURBO_TASK=fl2va MINIMAX_H3_TURBO_POINT=hyperflow MINIMAX_H3_TURBO_NFE=8
for p in $MINIMAX_H3_MODEL_PATH $MINIMAX_H3_TURBO_LORA_PATH $MINIMAX_H3_TURBO_KEYFRAME; do
  r=$(readlink -f "$p"); fs=$(findmnt -n -o FSTYPE -T "$r" 2>/dev/null)
  case "$r" in /mnt/*) echo "[t288] $p resolves to $r under /mnt"; exit 4;; esac
  case "$fs" in nfs*|cifs|smb*|fuse*|sshfs|lustre|ceph*|9p|"") echo "[t288] $p is on fs '${fs:-unknown}', not local disk"; exit 4;; esac
done
[ -z "$(timeout 60 find $MINIMAX_H3_MODEL_PATH $F/models/hyperflow -type l | head -1)" ] || { echo "[t288] symlink in the weight dirs"; exit 4; }
export MINIMAX_H3_ADALN_PRECOMPUTE=$PRE
# Warm only what this request reaches (fl2va 5 s 1344x768 packs to 39773 rows, rung 41984; the
# pipeline adds the top rung) instead of 26 rungs and 28 keyframe layouts. Output is unchanged.
export MINIMAX_H3_TURBO_WARM_RUNGS=${T288_WARM_RUNGS:-41984} MINIMAX_H3_TURBO_WARM_USED_LAYOUTS=1
export MINIMAX_H3_TURBO_CACHE_ONLY=$CACHE_ONLY
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
cd $W || exit 3
# Without the warm-scope flags (ttp/t288-adaln-ab 22fcacb6b19) the warm walks 26 rungs and 28 layouts
# and passes 400 s at the 900 MHz clamp (jobs 853/855/857), so refuse older code before the device.
grep -q MINIMAX_H3_TURBO_WARM_RUNGS models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py || { echo "[t288] $W lacks the warm-scope flags"; exit 3; }
source ${T288_VENV:-/home/smarton/fasth3/tt-metal/python_env}/bin/activate || exit 3
# t286_skipvaewarm.py (in $F/t286): warm only this test's VAE canvases, with a warmup deadline.
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$F/t286
P=""; [ "${T286_SKIP_VAE_WARM:-0}" = 1 ] && P="-p t286_skipvaewarm"
MP4=$HOME/h3_turbo_artifacts/fl2va_turbo_1344x768_5s_8fwd.mp4; rm -f $MP4
CMD="python -u -m pytest -sv -p no:cacheprovider $P models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '5s and 4x8 and not 15s and not 4x32 and not WH'"
echo "[t288] host=$(hostname) commit=$(git -C $W rev-parse HEAD) tag=$TAG arm=$ARM cache_only=$CACHE_ONLY job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $OUT/run.log
echo "[t288] cmd: $CMD" | tee -a $OUT/run.log
env | grep -E '^(MINIMAX|TT_|T286_)' | sort >> $OUT/run.log
T0=$(date +%s)
eval "$CMD" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t288] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
[ "$CACHE_ONLY" = 1 ] || cp -p $MP4 $OUT/ 2>/dev/null
echo "T288_EXIT=$rc" | tee -a $OUT/run.log
[ $rc = 0 ] && touch $OUT/PASS
exit $rc
