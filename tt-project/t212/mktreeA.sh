#!/bin/bash
# treeA: the t208 overlay (t48 5e4e0cd643a python) with the t212 test file swapped in for its DIFFVAE_DUMP_PIXELS
# and DIFFVAE_LATENT hooks only (verified diff). Hard links; the swapped file breaks its link, so t208 is never written.
set -eo pipefail
F=/var/tmp/fasth3; T=$F/t212; O=$T/treeA; S=$F/t208/tree
rm -rf $O; mkdir -p $O
cp -al $S/models $O/models
cp $S/conftest.py $S/pytest.ini $S/pyproject.toml $S/OVERLAY_COMMIT $O/
f=models/tt_dit/tests/models/vae/test_diffvae_ltx.py
git -C $F/t48 show 8b1167ef43b:$f > $O/$f.new && mv $O/$f.new $O/$f
grep -q DIFFVAE_DUMP_PIXELS $O/$f && cmp -s $S/$f $F/t48/$f
echo TREEA_OK
