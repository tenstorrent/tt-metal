#!/bin/bash
# blx03 broker job for #133: traced conv VAE decode A/B of the #56023 BH NoC MID-register skip on a 2x4 submesh
# (test_vae_ltx_trace_ab opens the full 4x8 mesh, then create_submesh(2,4)), 544x960/145f, real 2.5 latent,
# LTX_CONV3D_BLOCKING_MESH=4,8, 1 warmup + 3 eager + capture + 3 traced decodes per process.
# Arms: A = ~/fasth3/t133a (t48 tip), B = ~/fasth3/t133b (A + #56023), run A B A B (ROUNDS=2), each in its own
# process with its own JIT cache, so firmware and kernels compile from that tree's headers.
# If no stored VAE reference exists, a halo-off A run records one first (the harness's required ref check).
# Usage (on blx03, via tmp/blx03/submit.sh): bash /home/smarton/fasth3/t133b/tmp/t133/run133.sh
BASE=/home/smarton/fasth3/tt-metal; V=/var/tmp/fasth3/t133; LOG=$V/run133.log
ROUNDS=${ROUNDS:-2}
mkdir -p $V/vae_ref
source $BASE/python_env/bin/activate
export HF_HUB_OFFLINE=1 LTX_FUSE_YUV_OUTPUT=1 LTX_CONV3D_BLOCKING_MESH=4,8
export AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt LTX_VAE_REF_DIR=$V/vae_ref
# Reuse a shared reference decode if another task stored one.
cp -n /var/tmp/fasth3/vae_ref/ltx_conv_544x960x145_*.pt $V/vae_ref/ 2>/dev/null
test -f "$AB_LATENT" || { echo "[t133] $AB_LATENT missing" | tee -a $LOG; exit 3; }

arm() {  # $1 A|B, $2 tag; extra env via the caller
  local a=$1 tag=$2 W; [ $a = A ] && W=/home/smarton/fasth3/t133a || W=/home/smarton/fasth3/t133b
  (
    export TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W PYTHONPATH=$W:$W/ttnn:$W/tools
    export TT_METAL_CACHE=$V/jit_$a AB_OUT_DIR=$V/out_$tag
    cd $W
    echo "[t133] arm=$a tag=$tag tree=$(git rev-parse --short HEAD) halo_only=${LTX_VAE_HALO_ONLY:-1} $(date -u +%T)"
    timeout 900 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=840 \
      models/tt_dit/tests/models/ltx/test_vae_ltx_trace_ab.py 2>&1
    echo "T133_ARM_EXIT[$tag]=$?"
  ) | tee -a $LOG | grep -E '^\[t133\]|^AB|^VAE_REF|T133_ARM_EXIT|passed|failed'
}

echo "[t133] start $(date -u '+%F %T') rounds=$ROUNDS" | tee $LOG
if ! ls $V/vae_ref/ltx_conv_544x960x145_*.pt >/dev/null 2>&1; then
  LTX_VAE_HALO_ONLY=0 LTX_VAE_REF_RECORD=1 arm A ref
fi
for r in $(seq $ROUNDS); do arm A A$r; arm B B$r; done
fail=$(grep -c 'T133_ARM_EXIT\[[AB][0-9]*\]=[^0]' $LOG)
echo "T133_EXIT=$fail" | tee -a $LOG
exit $fail
