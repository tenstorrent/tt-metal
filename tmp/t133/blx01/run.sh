#!/bin/bash
# One blx01 broker job for #135: one A/B pair of the traced conv VAE decode on the full 4x8 mesh,
# 1088x1920/145f, LTX_CONV3D_BLOCKING_MESH=4,8 (test_t133_4x8.py). Arms run in their own processes with their own
# JIT caches and no pre-compiled firmware. Usage: bash run.sh <job tag> <arm> <arm>   e.g. run.sh j1 A B
F=/var/tmp/fasth3; V=$F/t133; J=$1; shift
LOG=$V/run133_$J.log
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
source $F/t48/python_env/bin/activate
export HF_HUB_OFFLINE=1 LTX_FUSE_YUV_OUTPUT=1 LTX_CONV3D_BLOCKING_MESH=4,8 TT_METAL_DISABLE_PRECOMPILED_FW=1
export AB_LATENT=$V/lat.gen0.pt LTX_VAE_REF_DIR=$V/vae_ref
arm() {  # $1 A|B
  local a=$1 tag=$1$J W
  [ $a = A ] && W=$F/t48 || W=$V/b
  (
    export TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W PYTHONPATH=$W:$W/ttnn:$W/tools
    export TT_METAL_CACHE=$V/jit_$a AB_OUT_DIR=$V/out_$tag
    [ $a = A ] && export LTX_VAE_REF_RECORD=1
    cd $W
    echo "[t133] arm=$a tag=$tag tree=$(git rev-parse --short=11 HEAD) dirty=$(git status --short -uno | wc -l) boot=$(uptime -s) $(date -u +%T)"
    timeout 270 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=250 tmp/t133/test_t133_4x8.py 2>&1
    echo "T133_ARM_EXIT[$tag]=$?"
  ) | tee -a $LOG | grep -E '^\[t133\]|^AB|^VAE_REF|T133_ARM_EXIT|passed|failed'
}
echo "[t133] job=$J arms=$* start $(date -u "+%F %T") epoch=$(date +%s)" | tee $LOG
for a in "$@"; do arm $a; done
fail=$(grep -c 'T133_ARM_EXIT\[[AB][a-z0-9]*\]=[^0]' $LOG)
echo "T133_EXIT=$fail end_epoch=$(date +%s)" | tee -a $LOG
exit $fail
