#!/bin/bash
# Stage a tree for frs_glx_bfc.sbatch (RING_JAM_FIX.md §10): `bfc_stage_glx.sh bfc` (the branch worktree
# /home/aagarwal/tt-metal-mainline-bfc -> /data/aagarwal/regress/trees/bfc) or `bfc_stage_glx.sh base` (the 4db4f45790b
# build /home/aagarwal/tt-metal-mainline-bfcbase -> /data/aagarwal/regress/trees/bfcbase): rsync with the regress
# harness's STAGE_EXCLUDES, no python_env (the job's capture tool runs with the main /data tree's). Also copies the job's
# yamls and capture tool to /data/aagarwal/frs-glx/bfc. Refuses a worktree with uncommitted tracked changes. Checks that
# test_tt_fabric resolves nothing under /home with the staged lib directory.
set -euo pipefail
case ${1:-} in
    bfc) SRC=/home/aagarwal/tt-metal-mainline-bfc; DST=/data/aagarwal/regress/trees/bfc ;;
    base) SRC=/home/aagarwal/tt-metal-mainline-bfcbase; DST=/data/aagarwal/regress/trees/bfcbase ;;
    *) echo "usage: bfc_stage_glx.sh bfc|base"; exit 1 ;;
esac
DATA=/data/aagarwal/frs-glx/bfc
STUDY=/home/aagarwal/tt-metal/fabric_exp
[ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] || { echo "!!! $SRC has uncommitted changes"; exit 1; }
sha=$(git -C $SRC rev-parse HEAD)
[ $1 != base ] || [ $sha = 4db4f45790b8383aaf8288dfd9efd84ffea2f67d ] || { echo "!!! $SRC is at $sha, not 4db4f45790b"; exit 1; }
echo "stage $SRC @ $sha -> $DST"
mkdir -p $DST $DATA/yamls
rsync -a --delete --stats --exclude /.git --exclude /.cpmcache --exclude /generated --exclude /.pytest_cache \
    --exclude /.scratch --exclude /build_Release/CMakeFiles --exclude '/build_Release/_deps/*-build' \
    --exclude '/build_Release/_deps/*-subbuild' --exclude '/build_Release/**/*.o' --exclude '/build_Release/**/*.o.d' \
    --exclude /build_Release/Testing --exclude '/build_Release/.ninja_*' --exclude /build_Release/build.ninja \
    --exclude /build_Release/compile_commands.json --exclude '/build_Release/**/CMakeFiles' --exclude '/build_Release/**/*.a' \
    $SRC/ $DST/ | grep -E "files transferred|Total transferred"
printf '{"sha": "%s", "source": "%s", "staged": "%s"}\n' $sha $SRC "$(date -u +%FT%TZ)" > $DST/.bfc_tree.json
cp $STUDY/galaxy/frgap/glx_frg_*.yaml $STUDY/galaxy/glx_check_2d.yaml $DATA/yamls/
cp $STUDY/consolidation/bfc/hang_capture.py $DATA/
bad=$(env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=$DST/build/lib ldd $DST/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric |
    grep -E "/home/|not found" || true)
[ -z "$bad" ] || { echo "!!! test_tt_fabric resolves outside $DST:"; echo "$bad"; exit 1; }
echo "staged: $(cat $DST/.bfc_tree.json); yamls $(ls $DATA/yamls | wc -l) in $DATA/yamls"
