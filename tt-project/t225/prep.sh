#!/bin/bash
# t225 prep on blx01 (no device): overlay ov = hardlinked t212/b/models with the 4 python files t48 036247a8eb6 changed
# since a40d78b8bae, each written as .new then mv'd (never in place: the hardlinks are shared with b).
set -eo pipefail
F=/var/tmp/fasth3; T=$F/t225; B=$F/t212/b
[ -d $T/ov/models ] || { mkdir -p $T/ov; cp -al $B/models $T/ov/models; }
for f in layers/neighborhood_attention.py layers/neighborhood_attention_plan.py models/vae/diffvae_ltx.py models/vae/diffvae_ltx_stage5.py; do
  cp $T/drv/ovsrc/$(basename $f) $T/ov/models/tt_dit/$f.new && mv $T/ov/models/tt_dit/$f.new $T/ov/models/tt_dit/$f
done
md5sum $T/ov/models/tt_dit/layers/neighborhood_attention*.py $T/ov/models/tt_dit/models/vae/diffvae_ltx*.py
git -C $B status --short | head -3
echo PREP_OK
