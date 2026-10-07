#!/bin/bash
# CPU-only setup on blx01 for #209: worktree of /var/tmp/fasth3/t48's repo at ttp/t209-h3-e2e (fasth3-opt H3 code
# + harness knobs), its own submodules and a fresh Release build. Everything under /var/tmp/fasth3/t209.
set -eux
F=/var/tmp/fasth3; A=$F/t48; T=$F/t209; B=$T/b; REV=9fa26939b35
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q origin ttp/t209-h3-e2e
[ -d $B ] || git -C $A worktree add --detach $B $REV
cd $B
test "$(git rev-parse --short=11 HEAD)" = $REV
git submodule update --init tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors tt_metal/third_party/umd
git submodule status tt_metal/third_party/umd
echo SETUP_OK B=$REV
./build_metal.sh --build-type Release
test -f ttnn/ttnn/_ttnn.so
grep -q vsa_ring_sdpa ttnn/ttnn/_ttnn.so build_Release/lib/*.so
echo "BUILD209_DONE rc=0 B=$REV"
