#!/bin/bash
# Build /var/tmp/fasth3/t170/tree on blx01: hardlinked t48 (bf7db12a149) models/ + root test config, with the
# t164 python files (t48 abfd309e797 warmup cuts + t140 eval knobs) swapped in by --remove-destination, so
# the t48 originals are never written. Kernel sources, build and JIT cache stay the t48 tree's (job 621).
set -eo pipefail
F=/var/tmp/fasth3; W=$F/t48; T=$F/t170; O=$T/tree
rm -rf $O; mkdir -p $O
cp -al $W/models $O/models
cp $W/conftest.py $W/pytest.ini $W/pyproject.toml $O/
for f in $(cd $T/files && find . -type f); do cp --remove-destination $T/files/$f $O/$f; done
cd $O && for f in $(cd $T/files && find models -type f); do cmp -s $f $W/$f && echo "same $f" || echo "swapped $f"; done
