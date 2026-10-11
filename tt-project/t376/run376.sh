#!/bin/bash
# t376 broker job on blx01 (4x8). Usage: run376.sh sweep <layer> <blockings> | run376.sh ab <new-blockings-dict>
# sweep: halo conv3d sweep of one LTX-2.5 conv VAE layer on a 2x4 submesh of the opened 4x8, only the listed
# blockings (plus the table one). ab: full 1080p/145f decode, real 2.5 conv VAE weights, old vs new blockings.
# Build: t376/b (drv376.sh, ttp/t48-ltx25-integrated 90ed8257bac + the t376 sweep/test files).
# The work runs in its own process group, killed whole on exit. timeout --foreground stays in that group
# (plain timeout calls setpgid and its pytest escaped the group kill, holding the device after the reap).
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
TP=
reap() {
  [ -n "$TP" ] || return 0
  local kids; kids=$(cat /proc/$TP/task/*/children 2>/dev/null)
  kill -TERM $TP $kids 2>/dev/null; sleep 3; kill -KILL $TP $kids 2>/dev/null; TP=
}
trap 'reap' EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
set -o pipefail
exec > >(tee -a /var/tmp/fasth3/t376/run.log) 2>&1
F=/var/tmp/fasth3; V=$F/t376; B=$V/b
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "[t376] df / $use% > 70%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-999}" -le 150 ] || { echo "[t376] $F ${gb}G > 150G"; exit 5; }
mkdir -p $F/tmp $V/res
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools TT_METAL_CACHE=$V/jit
export LTX_FUSE_YUV_OUTPUT=1 LTX_PIN_CORES=0
export VAE_CKPT=$F/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors AB_LATENT=$F/diffvae/latents/seed0.pt
for f in $B/ttnn/ttnn/_ttnn.so $VAE_CKPT $AB_LATENT; do [ -r $f ] || { echo "[t376] missing $f"; exit 3; }; done
cd $B || exit 6
echo "[t376] start $* $(date -u +%FT%TZ) host $(hostname)"
T0=$(date +%s)
case "$1" in
  sweep)
    export SWEEP_OUT_DIR=$V/res SWEEP_ONLY_BLOCKINGS="$3" SWEEP_MAX_SECONDS=330 SWEEP_COMBO_WATCHDOG_S=90
    timeout --foreground -k 10 560 python -m pytest -c $B/pytest.ini --rootdir=$B -sv -p no:cacheprovider --timeout=540 \
      models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py -k "halo and $2" & TP=$!;;
  ab)
    export T363_NEW_BLOCKINGS="$2" T363_REPS=3
    timeout --foreground -k 10 560 python -m pytest -c $B/pytest.ini --rootdir=$B -sv -p no:cacheprovider --timeout=540 \
      models/tt_dit/tests/models/ltx/test_vae_ltx_blk_ab_4x8.py & TP=$!;;
  *) echo "[t376] bad mode $1"; false;;
esac
if [ -n "$TP" ]; then wait $TP; rc=$?; TP=; else rc=1; fi
echo "[t376] rc=$rc wall $(( $(date +%s) - T0 )) s $(date -u +%FT%TZ)"
echo "T376_EXIT=$rc"
exit $rc
