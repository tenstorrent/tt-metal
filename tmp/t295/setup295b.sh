#!/bin/bash
# t295 A/B overlays on blx01 (no device). treeA = t48 checkout models (hardlinks) + every file of f6547442b30's
# models/tt_dit and root config that differs; treeB = treeA with the 6 files f6547442b30 changed put back to
# 9e20d905481 (pre-revert t48 tip). Swapped files use --remove-destination, so $W's files are never written.
set -eo pipefail
F=/var/tmp/fasth3; W=$F/t48; T=$F/t295
rm -rf $T/treeA $T/treeB $T/files $T/filesB; mkdir -p $T/treeA $T/files $T/filesB
tar -xzf $T/py.tar.gz -C $T/files; tar -xzf $T/pyB.tar.gz -C $T/filesB
cp -al $W/models $T/treeA/models
cp $W/conftest.py $W/pytest.ini $W/pyproject.toml $T/treeA/
find $T/treeA -name __pycache__ -prune -exec rm -rf {} +
n=0; for f in $(cd $T/files && find . -type f); do
  cmp -s $T/files/$f $T/treeA/$f || { mkdir -p $(dirname $T/treeA/$f); cp --remove-destination $T/files/$f $T/treeA/$f; n=$((n+1)); }
done
echo "treeA: swapped $n files"
cp -al $T/treeA $T/treeB
for f in $(cd $T/filesB && find . -type f); do cp --remove-destination $T/filesB/$f $T/treeB/$f; done
echo f6547442b304d744711e80e2281f6bd368291673 > $T/treeA/OVERLAY_COMMIT
echo 9e20d905481 > $T/treeB/OVERLAY_COMMIT
for a in A B; do echo "tree$a: $(grep -n '^_DEFAULT_S2_SIGMAS' $T/tree$a/models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py)"; done
rm -rf $T/files $T/filesB
echo SETUP_OK
