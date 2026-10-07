#!/bin/bash
# CPU-only setup on blx01 for #155: B = worktree of /var/tmp/fasth3/t48 at bf7db12a149 (t149 guard) + the t152 C++ fix
# (fix.diff = conv3d_program_factory.cpp aa4ade43a28..c0c02788344), fresh Release build in B/build_Release.
# S = t152 test source (git archive c0c02788344: models conftest.py pytest.ini tests/scripts). All under /var/tmp/fasth3.
set -eux
F=/var/tmp/fasth3; A=$F/t48; T=$F/t155; B=$T/b; S=$T/src
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
P=ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp
test "$(git -C $A rev-parse HEAD)" = "$(git -C $A rev-parse 'bf7db12a149^{commit}')"
[ -d $B ] || git -C $A worktree add --detach $B bf7db12a149
cd $B
grep -q 'blocks get exactly num_patches pages' $P || {
  git apply $T/fix.diff
  git -c user.name='Steve Marton' -c user.email=smarton@tenstorrent.com commit -q -m 'conv3d: size vol2col_rm at num_patches pages for unaligned blocks (t152 be371d08a9f, C++ part)' $P
}
for s in tracy tt-cluster-descriptors umd; do
  rmdir tt_metal/third_party/$s 2>/dev/null || true
  [ -e tt_metal/third_party/$s/.git ] || [ -L tt_metal/third_party/$s ] || ln -s $A/tt_metal/third_party/$s tt_metal/third_party/$s
done
mkdir -p $S; [ -f $S/REV ] || { tar -xf $T/src.tar -C $S; echo c0c02788344 > $S/REV; }
echo SETUP_OK B=$(git rev-parse --short=11 HEAD)
./build_metal.sh --build-type Release
test -f ttnn/ttnn/_ttnn.so
echo "BUILD155_DONE rc=0 B=$(git rev-parse --short=11 HEAD)"
