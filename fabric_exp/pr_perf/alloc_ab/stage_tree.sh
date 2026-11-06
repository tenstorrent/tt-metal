#!/bin/bash
# (alloc_ab copy of pr_perf/stage_pr_tree.sh: the tree must have NO local diff, since its own test_tt_fabric reads
# TT_FABRIC_NO_BW_TELEMETRY at these commits; checked in the source and the binary.) Stage a built tree for the PR perf refresh's lb2 jobs as ONE file (/data is short of inodes, 2026-11-05):
#   stage_pr_tree.sh TREE COMMIT NAME   ->  /data/aagarwal/lb2/trees/NAME.tar.zst (+ .sha256)
# TREE must be at COMMIT with exactly fabric_exp/port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch on top (its
# diff body; the jobs set TT_FABRIC_NO_BW_TELEMETRY=1, which the stock test_tt_fabric ignores), and its test binary must
# be newer than the patched source. Packed, as stage_port_tree.sh: the test binary and its shared libraries, the runtime
# (sfpi, hw), the sources the JIT compiles (tt_metal, tt_stl, ttnn, tests/tt_metal), the build -> build_Release link, and
# beside them COMMIT, PATCH (git diff HEAD) and MD5SUMS (binaries). The job unpacks it at TREE on the node (the binaries'
# RUNPATH) and checks MD5SUMS and COMMIT.
set -euo pipefail
W=$1 WANT=$2 NAME=$3
OUT=/data/aagarwal/lb2/trees/$NAME.tar.zst
P=/home/aagarwal/tt-metal/fabric_exp/port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch
[ "$(git -C $W rev-parse --short=11 HEAD)" = "$WANT" ] || { echo "!!! $W is not at $WANT"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no)" ] || { echo "!!! $W has local changes"; exit 1; }
grep -q TT_FABRIC_NO_BW_TELEMETRY $W/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp &&
    grep -aq TT_FABRIC_NO_BW_TELEMETRY $W/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric ||
    { echo "!!! $W's test_tt_fabric has no TT_FABRIC_NO_BW_TELEMETRY switch"; exit 1; }
B=$W/build_Release
[ $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric -nt $W/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp ] ||
    { echo "!!! $W's test_tt_fabric is older than its patched source: rebuild"; exit 1; }
[ ! -e $OUT ] || { echo "!!! $OUT exists (stages are never replaced)"; exit 1; }
M=$(mktemp -d)
git -C $W rev-parse HEAD > $M/COMMIT
git -C $W diff HEAD > $M/PATCH
(cd $W && md5sum build_Release/tt_metal/libtt_metal.so build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
    build_Release/tt_stl/libtt_stl.so > $M/MD5SUMS)
cd $W
tar -I 'zstd -T16 -3' -cf $OUT.part --exclude=.git --exclude=__pycache__ \
    build_Release/lib/libtracy.so build_Release/lib/libtracy.so.0.14.1 build_Release/tt_metal/libtt_metal.so \
    build_Release/tt_stl/libtt_stl.so build_Release/tt_metal/third_party/umd/lib \
    build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric runtime tt_metal tt_stl ttnn tests/tt_metal \
    -C $M COMMIT PATCH MD5SUMS
mv $OUT.part $OUT
(cd $(dirname $OUT) && sha256sum $(basename $OUT) > $(basename $OUT).sha256)
rm -rf $M
echo "staged $W ($WANT, no local diff) at $OUT ($(du -h $OUT | cut -f1))"
