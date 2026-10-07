#!/bin/bash
# CPU-only setup on blx01 for #238: worktree of /var/tmp/fasth3/t48's repo at t48 head 76db90bbcfd (from t238.bundle),
# its own submodules and a Release build. Everything under /var/tmp/fasth3/t238. Marker: $T/build.rc
set -eux
F=/var/tmp/fasth3; A=$F/t48; T=$F/t238; B=$T/b; REV=76db90bbcfd
trap 'echo $? > $T/build.rc' EXIT
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q $T/drv/t238.bundle "$REV:refs/t238/base"
[ -d $B ] || git -C $A worktree add --detach $B $REV
cd $B
test "$(git rev-parse --short=11 HEAD)" = $REV
git submodule update --init tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors tt_metal/third_party/umd
echo SETUP_OK B=$REV
./build_metal.sh --build-type Release
test -f ttnn/ttnn/_ttnn.so
echo "BUILD238_DONE rc=0 B=$REV"
