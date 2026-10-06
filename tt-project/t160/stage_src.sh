#!/bin/bash
# t160: stage the t48 tree (bf7db12a14) at /data/smarton/fasth3/t48 on exabox. No build here.
set -eo pipefail
F=/data/smarton/fasth3; W=$F/t48; REV=bf7db12a149bc1e8bf0559350ec2c874e86dc431
trap 'echo "STAGE_RC=$?" >> $F/stage_src.log' EXIT
[ -d $W/.git ] || git clone --filter=blob:none --no-checkout https://github.com/tenstorrent/tt-metal.git $W
cd $W
git bundle verify $F/t160.bundle
git fetch $F/t160.bundle ttp/t158-g15blx02-ltx25-4x8:refs/heads/fasth3-t48
git checkout --detach $REV
for n in tracy umd tt-cluster-descriptors; do git submodule update --init --depth 1 -- tt_metal/third_party/$n; done
git rev-parse HEAD
