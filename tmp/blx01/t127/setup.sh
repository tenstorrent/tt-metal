#!/bin/bash
# CPU-only setup on blx01 for #127: reuse #135's lean t48 tree (bf7db12a149) and add only the test file.
set -eux
F=/var/tmp/fasth3; A=$F/t48; V=$F/t127
test "$(git -C $A rev-parse HEAD)" = "$(git -C $A rev-parse 'bf7db12a149^{commit}')"
mkdir -p $A/tmp/t127
cp $V/test_t119_4x8.py $A/tmp/t127/
source $A/python_env/bin/activate
cd $A && PYTHONPATH=$A:$A/ttnn:$A/tools python -c "
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import prefetch_shard_fits
for b in [(128,128,1,2,4),(64,128,3,2,4),(128,64,3,2,4),(128,128,3,2,2)]: print(b, prefetch_shard_fits(*b,(3,3,3),128))
"
echo SETUP_OK
