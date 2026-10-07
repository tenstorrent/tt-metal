#!/bin/bash
# Build /var/tmp/fasth3/t195/tree on blx01: hardlinked t171 tree (t48 f6b806516cc) models/ + root test config, with
# the python files changed f6b806516cc..b21f12b93a2 swapped in by --remove-destination, so the shared originals are
# never written. Kernel sources, build and JIT cache stay the t48 tree's.
set -eo pipefail
F=/var/tmp/fasth3; W=$F/t48; T=$F/t195; O=$T/tree
rm -rf $O; mkdir -p $O
cp -al $F/t171/tree/models $O/models
cp $W/conftest.py $W/pytest.ini $W/pyproject.toml $O/
for f in $(cd $T/files && find . -type f); do cp --remove-destination $T/files/$f $O/$f; done
cd $O && for f in $(cd $T/files && find models -type f); do cmp -s $f $W/$f && echo "same $f" || echo "swapped $f"; done
echo b21f12b93a2 > $O/OVERLAY_COMMIT
