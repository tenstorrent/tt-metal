#!/bin/bash
# t427 (CPU only, runs on blx01): tt-metal tree at ttp/t427-cut-the-fasth3-vae-decode-s-off-device-b, reusing the t284 Release
# build (the branch adds only Python on top of e24a2b93d79). Build outputs are hardlinked, not copied.
set -eo pipefail
F=/var/tmp/fasth3; A=$F/t48; S=$F/t284/b; B=$F/t427/b; REV=b0f45278d72
M=$F/t427/setup.marker
trap 'rc=$?; echo "rc=$rc step=${STEP:-?} $(date -u +%FT%TZ)" > $M' EXIT
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
set -x
STEP=fetch
git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q $F/t427/t427.bundle ttp/t427-cut-the-fasth3-vae-decode-s-off-device-b
STEP=worktree
[ -d $B ] || git -C $A worktree add --detach $B $REV
cd $B
test "$(git rev-parse --short=11 HEAD)" = $REV
STEP=submodules
for p in tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors tt_metal/third_party/umd; do
  git -c protocol.file.allow=always -c submodule.$p.url=$S/$p submodule update --init $p
done
STEP=artifacts
for d in build_Release generated runtime tt_metal/llrt/hal/generated tt_metal/pre-compiled; do
  [ -e $B/$d/.t427 ] && continue
  mkdir -p $B/$d && ionice -c3 nice -n 10 cp -al $S/$d/. $B/$d/ && touch $B/$d/.t427
done
ln -sfn build_Release build
cp -p $S/ttnn/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
STEP=verify
[ -z "$(git diff --name-only e24a2b93d79 HEAD | grep -vE '\.(py|md|sh|jpg|png)$')" ]
source $A/python_env/bin/activate
cd /tmp && TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn python -c "import ttnn, os; print('ttnn from', os.path.dirname(ttnn.__file__))" | grep -q "$B/ttnn/ttnn"
grep -q StreamedClipAssembly $B/models/tt_dit/models/vae/minimax_h3/vae_minimax_h3.py
echo SETUP427_DONE
STEP=done
