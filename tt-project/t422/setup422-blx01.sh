#!/bin/bash
# t422 (CPU only, runs on blx01): tt-metal tree at ttp/t422-port-upstream-58676-tile-row-adaln-gathe and its own
# Release build. The port changes C++ (dit_fused_distributed_rmsnorm op, program factory, nanobind), so the t284
# build cannot be reused or hardlinked: a rebuild would write through the links into t284's files.
set -eo pipefail
F=/var/tmp/fasth3; A=$F/t48; S=$F/t284/b; T=$F/t422; B=$T/b; REV=${T422_REV:?commit}
M=$T/setup.marker
trap 'rc=$?; echo "rc=$rc step=${STEP:-?} $(date -u +%FT%TZ)" > $M' EXIT
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 85 ] || { echo "/ at $use%"; exit 5; }
set -x
STEP=fetch
git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q $T/t422.bundle ttp/t422-port-upstream-58676-tile-row-adaln-gathe
STEP=worktree
[ -d $B ] || git -C $A worktree add --detach $B $REV
cd $B
git checkout -q --detach $REV
test "$(git rev-parse HEAD)" = "$(git -C $A rev-parse $REV^{commit})"
STEP=submodules
for p in tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors tt_metal/third_party/umd; do
  git -c protocol.file.allow=always -c submodule.$p.url=$S/$p submodule update --init $p
done
STEP=build
ionice -c3 nice -n 10 ./build_metal.sh --build-type Release
STEP=verify
test -f ttnn/ttnn/_ttnn.so
grep -q affine_tile_row_map ttnn/ttnn/_ttnn.so build_Release/lib/*.so
grep -q dynamic_tile_row_map models/tt_dit/layers/normalization.py
source $A/python_env/bin/activate
cd /tmp && TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn python -c "import ttnn, os; print('ttnn from', os.path.dirname(ttnn.__file__))" | grep -q "$B/ttnn/ttnn"
echo SETUP422_DONE
STEP=done
