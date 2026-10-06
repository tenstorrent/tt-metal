#!/bin/bash
# One blx03 broker job for #133: one A/B pair of the traced conv VAE decode on the full 4x8 mesh,
# 1088x1920/145f, real 2.5 latent, LTX_CONV3D_BLOCKING_MESH=4,8 (tmp/t133/test_t133_4x8.py).
# A = ~/fasth3/t133a (t48 c4409b1fa24), B = ~/fasth3/t133b (A + #56023). Each arm runs in its own process with
# its own JIT cache, so firmware and kernels compile from that tree's headers.
# Usage (on blx03, via tmp/blx03/submit.sh): bash run133.sh <job tag> <arm> <arm>, e.g. run133.sh j1 A B
BASE=/home/smarton/fasth3/tt-metal; V=/var/tmp/fasth3/t133; J=$1; shift
LOG=$V/run133_$J.log
mkdir -p $V/vae_ref
source $BASE/python_env/bin/activate
export HF_HUB_OFFLINE=1 LTX_FUSE_YUV_OUTPUT=1 LTX_CONV3D_BLOCKING_MESH=4,8
export AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt LTX_VAE_REF_DIR=$V/vae_ref
test -f "$AB_LATENT" || { echo "[t133] $AB_LATENT missing" | tee -a $LOG; exit 3; }

arm() {  # $1 A|B
  local a=$1 tag=$1$J W rev
  [ $a = A ] && { W=/home/smarton/fasth3/t133a; rev=c4409b1fa24; } || W=/home/smarton/fasth3/t133b
  (
    export TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W PYTHONPATH=$W:$W/ttnn:$W/tools
    export TT_METAL_CACHE=$V/jit_$a AB_OUT_DIR=$V/out_$tag
    # Only the first A run records the reference (halo-off decode); later runs check against it.
    [ $a = A ] && export LTX_VAE_REF_RECORD=1
    cd $W
    head=$(git rev-parse --short=11 HEAD)
    echo "[t133] arm=$a tag=$tag tree=$head dirty=$(git status --short -uno | wc -l) boot=$(uptime -s) $(date -u +%T)"
    if [ -n "$rev" ] && [ "$(git rev-parse HEAD)" != "$(git rev-parse "$rev^{commit}")" ]; then echo "[t133] A tree is not at $rev"; echo "T133_ARM_EXIT[$tag]=5"; exit; fi
    timeout 1000 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=960 \
      tmp/t133/test_t133_4x8.py 2>&1
    echo "T133_ARM_EXIT[$tag]=$?"
  ) | tee -a $LOG | grep -E '^\[t133\]|^AB|^VAE_REF|T133_ARM_EXIT|passed|failed'
}

echo "[t133] job=$J arms=$* start $(date -u '+%F %T')" | tee $LOG
for a in "$@"; do arm $a; done
fail=$(grep -c 'T133_ARM_EXIT\[[AB][a-z0-9]*\]=[^0]' $LOG)
echo "T133_EXIT=$fail" | tee -a $LOG
exit $fail
