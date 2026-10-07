#!/bin/bash
# CPU-only setup on blx01 for the #135 MID-skip A/B. A = /var/tmp/fasth3/t48 (bf7db12a149 = c4409b1fa24 + a host-only
# conv3d guard, lean build). B = a worktree of it at bf7db12a149 + #56023 (27a9c2f95c9). #56023 changes only device
# headers/kernels and tests, so B reuses A's host build via symlinks; both arms run with TT_METAL_DISABLE_PRECOMPILED_FW=1
# so firmware and dispatch kernels JIT-build from each tree's own headers. Everything under /var/tmp/fasth3/t133.
set -eux
F=/var/tmp/fasth3; A=$F/t48; T=$F/t133; B=$T/b
cd $A
test "$(git rev-parse HEAD)" = "$(git rev-parse 'bf7db12a149^{commit}')"
[ -d $B ] || git worktree add --detach $B bf7db12a149
cd $B
git log -1 --format=%s | grep -q 'NOC_TARG_ADDR_MID' ||
  git -c user.name='Steve Marton' -c user.email=smarton@tenstorrent.com am -q $T/56023.patch
for s in tracy tt-cluster-descriptors umd; do
  rmdir tt_metal/third_party/$s 2>/dev/null || true
  [ -e tt_metal/third_party/$s/.git ] || [ -L tt_metal/third_party/$s ] || ln -s $A/tt_metal/third_party/$s tt_metal/third_party/$s
done
[ -L build_Release ] || ln -s $A/build_Release build_Release
[ -L build ] || ln -s build_Release build
[ -L runtime ] || ln -s $A/runtime runtime
cp -r $A/tt_metal/llrt/hal/generated tt_metal/llrt/hal/
cp $A/ttnn/ttnn/_ttnn.so ttnn/ttnn/
mkdir -p $A/tmp/t133 $B/tmp/t133 $T/vae_ref
cp $T/test_t133_4x8.py $A/tmp/t133/; cp $T/test_t133_4x8.py $B/tmp/t133/
test -f $T/lat.gen0.pt
echo SETUP_OK B=$(git -C $B rev-parse --short=11 HEAD)
