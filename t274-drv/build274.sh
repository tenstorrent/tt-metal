#!/bin/bash
# t274: check out REV in $B (fetching the bundle into the t48 repo if needed) and build it.
set -eux
F=/var/tmp/fasth3; A=$F/t48; B=$F/t263/b; D=$F/t274/drv; REV=$1
git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q $D/t274.bundle "HEAD:refs/t274/head"
cd $B
git checkout -q --detach $REV
test "$(git rev-parse --short=11 HEAD)" = $REV
./build_metal.sh --build-type Release
test -f ttnn/ttnn/_ttnn.so
