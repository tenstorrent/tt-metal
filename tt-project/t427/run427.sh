#!/bin/bash
# t427: one broker job on blx01, one off/on A/B pair: MiniMax-H3 strip-stitch VAE decode (yuv420, 5 s 768p,
# 4x8 ring) with the temporal assembly after the last wave (off) vs streamed on the readback worker (on).
# Same latents per seed (T427_SEEDS, default 0-4), both arms in one process, so same box and clock.
# Code: ttp/t427-cut-the-fasth3-vae-decode-s-off-device-b (python-only on the t284 build) at $F/t427/b.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t427; W=$T/b; VENV=$F/t48/python_env
if [ -z "$T427_INNER" ]; then
  T427_INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
K="decode_stream_blend_ab and 5s"
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t427] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F 2>/dev/null | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t427] footprint ${gb}G"; exit 5; }
OUT=$T/out_ab; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
export MINIMAX_H3_MODEL_PATH=$F/models/MiniMax-H3
[ -e $MINIMAX_H3_MODEL_PATH/READ_OK ] || { echo "[t427] no READ_OK in $MINIMAX_H3_MODEL_PATH"; exit 4; }
r=$(readlink -f $MINIMAX_H3_MODEL_PATH); fs=$(findmnt -n -o FSTYPE -T "$r" 2>/dev/null)
case "$r" in /mnt/*) echo "[t427] weights under /mnt"; exit 4;; esac
case "$fs" in nfs*|cifs|smb*|fuse*|sshfs|lustre|ceph*|9p|wekafs|"") echo "[t427] weights on fs '${fs:-unknown}'"; exit 4;; esac
export MINIMAX_H3_VAE_AB_SEEDS=${T427_SEEDS:-0,1,2,3,4} MINIMAX_H3_VAE_AB_REPEATS=${T427_REPEATS:-2}
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
cd $W || exit 3
grep -q StreamedClipAssembly models/tt_dit/models/vae/minimax_h3/vae_minimax_h3.py || { echo "[t427] $W lacks StreamedClipAssembly"; exit 3; }
source $VENV/bin/activate || exit 3
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
LOG=$OUT/run.log
echo "[t427] host=$(hostname) commit=$(git -C $W rev-parse HEAD) seeds=$MINIMAX_H3_VAE_AB_SEEDS $(date -u '+%F %T') UTC" | tee $LOG
# 400 s warm deadline for the whole pytest; a 90 s output-stall watchdog catches a hang sooner.
timeout --foreground -k 10 400 python -u -m pytest -sv -p no:cacheprovider \
  models/tt_dit/tests/models/minimax_h3/test_performance_vae_minimax_h3.py -k "$K" >> $LOG 2>&1 &
PY=$!
while kill -0 $PY 2>/dev/null; do
  sleep 10
  idle=$(( $(date +%s) - $(stat -c %Y $LOG) ))
  if [ $idle -gt 90 ]; then echo "[t427] watchdog: no output for ${idle}s, stopping" | tee -a $LOG; kill -TERM $PY; sleep 12; kill -KILL $PY 2>/dev/null; break; fi
done
wait $PY; rc=$?
grep -E "STREAM_AB|DECODE_SPLIT|VAE decode profile|passed|failed|Error" $LOG | tail -60
echo "T427_EXIT=$rc" | tee -a $LOG
exit $rc
