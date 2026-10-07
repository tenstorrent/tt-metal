#!/bin/bash
# CPU-only setup on blx01 for #232: worktree of /var/tmp/fasth3/t48's repo at the t228 key-phase commit
# (b95861393fe, fetched from t232.bundle), its own submodules and a Release build. Everything under /var/tmp/fasth3/t232.
set -eux
F=/var/tmp/fasth3; A=$F/t48; T=$F/t232; B=$T/b; REV=b95861393fe
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q $T/drv/t232.bundle 'refs/heads/*:refs/t232/*'
[ -d $B ] || git -C $A worktree add --detach $B $REV
cd $B
test "$(git rev-parse --short=11 HEAD)" = $REV
git submodule update --init tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors tt_metal/third_party/umd
echo SETUP_OK B=$REV
./build_metal.sh --build-type Release
test -f ttnn/ttnn/_ttnn.so
echo "BUILD232_DONE rc=0 B=$REV"
