#!/bin/bash
# t295 overlay (no device): hardlinked $W/models + root config, with every file from the t295 archive
# (models/tt_dit + root config at f6547442b30) that differs swapped in by --remove-destination,
# so the t48 tree's files are never written. Usage: bash setup295.sh   (archive at $T/py.tar.gz)
set -eo pipefail
F=/var/tmp/fasth3; W=$F/t48; T=$F/t295; O=$T/tree
rm -rf $O $T/files; mkdir -p $O $T/files
tar -xzf $T/py.tar.gz -C $T/files
cp -al $W/models $O/models
cp $W/conftest.py $W/pytest.ini $W/pyproject.toml $O/
n=0; for f in $(cd $T/files && find . -type f); do
  cmp -s $T/files/$f $O/$f || { cp --remove-destination $T/files/$f $O/$f; n=$((n+1)); }
done
echo "swapped $n files"
echo f6547442b304d744711e80e2281f6bd368291673 > $O/OVERLAY_COMMIT
grep -n "^_DEFAULT_S2_SIGMAS" $O/models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py
