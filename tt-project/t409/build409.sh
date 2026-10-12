#!/bin/bash
# t409/t416 (CPU only, runs on blx01): tt-metal worktree of /var/tmp/fasth3/t48's repo at the t409 VSA code commit
# and a fresh Release build under /var/tmp/fasth3/t409. This is also the C++ compile check of the VSA ops on the t400
# base (t400 C++ == t284 e24a2b93d79). Arg: <rev> (11-char sha in t409.bundle). Idle I/O class, nice, 24 jobs.
set -eo pipefail
F=/var/tmp/fasth3; A=$F/t48; S=$F/t284/b; T=$F/t409; B=$T/b; REV=${1:?rev}
marker=$T/build409.marker; rm -f $marker
trap 'rc=$?; echo "rc=$rc step=${STEP:-?} rev=$REV $(date -u +%FT%TZ)" > "$marker"' EXIT
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache CMAKE_BUILD_PARALLEL_LEVEL=24
set -x
STEP=disk
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 85 ]
STEP=fetch
git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q $T/t409.bundle ttp/t409-port-vsa-sparse-attention-into-the-fasth
STEP=worktree
[ -d $B ] || git -C $A worktree add --detach $B $REV
cd $B
git checkout -q --detach $REV
test "$(git rev-parse --short=11 HEAD)" = $REV
STEP=submodules
for p in tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors tt_metal/third_party/umd; do
  git -c protocol.file.allow=always -c submodule.$p.url=$S/$p submodule update --init $p
done
STEP=build
ionice -c3 nice -n 10 ./build_metal.sh --build-type Release
STEP=verify
test -f ttnn/ttnn/_ttnn.so
source $A/python_env/bin/activate
cd /tmp && TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn python -c "
import ttnn, os
assert os.path.dirname(ttnn.__file__) == '$B/ttnn/ttnn', ttnn.__file__
for n in ('vsa_sdpa', 'vsa_ring_sdpa'):
    assert hasattr(ttnn.transformer, n), n
print('VSA ops bound')"
echo "BUILD409_DONE B=$REV"
STEP=done
