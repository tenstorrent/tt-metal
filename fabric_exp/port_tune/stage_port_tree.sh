#!/bin/bash
# Stage the receiver tuning sweep's port tree (RECEIVER_TUNING.md) for lb2 jobs: a lean copy of the worktree
# /home/aagarwal/tt-metal-port-tune (22cdb1e55c3 + port_tune.patch, built) at /data/aagarwal/lb2/trees/port_tune_22cd,
# which tune_runs.sh copies onto the node at the same path (the binaries' RUNPATH). What a run needs: the test binary
# and its four shared libraries, the runtime (sfpi, hw), the sources the JIT compiles (tt_metal, tt_stl, ttnn, the test
# kernels under tests/tt_metal), and ttnn's ttrun.py. Writes MD5SUMS (binaries), PORT_COMMIT and the patch beside them.
# Also refreshes fabric_exp/port_tune/port_tune.patch from the worktree (git diff HEAD).
set -euo pipefail
# PORT_TREE / PORT_TREE_SRC / PORT_BASE: another worktree, its staged copy and the commit it must be at (the allocator
# follow-up: /home/aagarwal/tt-metal-port-tune-alloc, .../port_tune_alloc_4b52, 4b524653490). The patch file is
# refreshed only for the default tree; any other tree must carry the same patch (checked: same diff).
W=${PORT_TREE:-/home/aagarwal/tt-metal-port-tune}
T=${PORT_TREE_SRC:-/data/aagarwal/lb2/trees/port_tune_22cd}
BASE=${PORT_BASE:-22cdb1e55c3}
HERE=$(cd "$(dirname "$0")" && pwd)
[ "$(git -C $W rev-parse --short=11 HEAD)" = $BASE ] || { echo "!!! $W is not at $BASE"; exit 1; }
# The jobs set TT_FABRIC_NO_BW_TELEMETRY=1; from ce2dee3cd6e on the port's test_tt_fabric reads it only with
# local_patches/test_tt_fabric_no_bw_telemetry.patch, so a tree without it would benchmark telemetry-on.
grep -q TT_FABRIC_NO_BW_TELEMETRY $W/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp ||
    { echo "!!! $W's test_tt_fabric does not read TT_FABRIC_NO_BW_TELEMETRY: apply local_patches/test_tt_fabric_no_bw_telemetry.patch"; exit 1; }
if [ "$W" = /home/aagarwal/tt-metal-port-tune ]; then
    git -C $W diff HEAD > $HERE/port_tune.patch
else  # the same patch, line for line (ignoring index lines and hunk offsets)
    norm() { grep -vE '^(index |@@ )'; }
    diff <(git -C $W diff HEAD | norm) <(norm < $HERE/port_tune.patch) > /dev/null || { echo "!!! $W does not carry port_tune.patch"; exit 1; }
fi
mkdir -p $T/build_Release
B=$W/build_Release
rsync -a --delete --relative \
    $B/./lib/libtracy.so $B/./lib/libtracy.so.0.14.1 \
    $B/./tt_metal/libtt_metal.so $B/./tt_stl/libtt_stl.so \
    $B/./tt_metal/third_party/umd/lib/ \
    $B/./test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
    $T/build_Release/
for d in runtime tt_metal tt_stl ttnn tests/tt_metal; do
    mkdir -p $T/$d
    rsync -a --delete --exclude=.git --exclude=__pycache__ $W/$d/ $T/$d/
done
ln -sfn build_Release $T/build
git -C $W rev-parse HEAD > $T/PORT_COMMIT
cp $HERE/port_tune.patch $T/port_tune.patch
(cd $T && md5sum build_Release/tt_metal/libtt_metal.so build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
    build_Release/tt_stl/libtt_stl.so > MD5SUMS)
(cd $W && md5sum -c --quiet $T/MD5SUMS 2>/dev/null) || { echo "!!! staged binaries differ from the worktree's"; exit 1; }
echo "staged $W ($(cat $T/PORT_COMMIT | cut -c1-11) + port_tune.patch, $(git -C $W diff HEAD | wc -l) patch lines) at $T ($(du -sh $T | cut -f1))"
