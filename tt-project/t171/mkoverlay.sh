#!/bin/bash
# Build /var/tmp/fasth3/t170/tree on blx01: hardlinked t48 (bf7db12a149) models/ + root test config, with the
# t48 f6b806516cc python files (new fused/exact-shard defaults) swapped in by --remove-destination, so
# the t48 originals are never written. Kernel sources, build and JIT cache stay the t48 tree's (job 621).
set -eo pipefail
F=/var/tmp/fasth3; W=$F/t48; T=$F/t171; O=$T/tree
rm -rf $O; mkdir -p $O
cp -al $W/models $O/models
cp $W/conftest.py $W/pytest.ini $W/pyproject.toml $O/
for f in $(cd $T/files && find . -type f); do cp --remove-destination $T/files/$f $O/$f; done
cd $O && for f in $(cd $T/files && find models -type f); do cmp -s $f $W/$f && echo "same $f" || echo "swapped $f"; done
echo f6b806516cc3c8322f0a8b7cebc672530d60e870 > $O/OVERLAY_COMMIT
