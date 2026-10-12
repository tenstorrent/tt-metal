#!/bin/bash
# t421 (CPU only, runs on blx01): tt-metal tree at the t421 branch (d4494e2c1a1), reusing the t284 Release
# build (e24a2b93d79; the branch adds only Python on top) as hard links. Worktree of /var/tmp/fasth3/t48's repo.
set -eo pipefail
F=/var/tmp/fasth3; A=$F/t48; S=$F/t284/b; B=$F/t421/b; REV=d4494e2c1a1
M=$F/t421/setup.marker
trap 'rc=$?; echo "rc=$rc step=${STEP:-?} $(date -u +%FT%TZ)" > $M' EXIT
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
set -x
STEP=fetch
git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q $F/t421/t421.bundle ttp/t421-fasth3-turbo-host-adaln-precompute-for-t
STEP=worktree
[ -d $B ] || git -C $A worktree add --detach $B $REV
cd $B
test "$(git rev-parse --short=11 HEAD)" = $REV
STEP=submodules
for p in tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors tt_metal/third_party/umd; do
  git -c protocol.file.allow=always -c submodule.$p.url=$S/$p submodule update --init $p
done
STEP=artifacts
# Build outputs from the t284 tree (ignored files; the RUNPATH of _ttnn.so is relative).
for d in build_Release generated runtime tt_metal/llrt/hal/generated tt_metal/pre-compiled; do
  ionice -c3 nice -n 10 rsync -a --bwlimit=100M --link-dest=$S/$d/ $S/$d/ $B/$d/
done
ln -sfn build_Release build
cp -pl $S/ttnn/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
STEP=verify
[ -z "$(git diff --name-only e24a2b93d79 HEAD | grep -vE '\.(py|md|jpg)$')" ]
source $A/python_env/bin/activate
cd /tmp && TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn python -c "import ttnn, os; print('ttnn from', os.path.dirname(ttnn.__file__))" | grep -q "$B/ttnn/ttnn"
grep -q FIXED_ADALN_STEPS $B/models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py
echo SETUP288_DONE
STEP=done
