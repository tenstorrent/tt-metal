#!/bin/bash
# Build /data/aagarwal/frs-glx/trees/$2 = trees/base (hard links) with variant $1's files swapped in (each file replaced,
# not overwritten, so the base tree's linked files stay as they are).
set -euo pipefail
V=$(realpath "$1"); NAME=$2; G=/data/aagarwal/frs-glx/trees  # the variant path is used after a cd
[ -e $G/$NAME ] && { echo "$G/$NAME exists"; exit 1; }
t0=$(date +%s)
cp -al $G/base $G/$NAME
echo "linked copy in $(( $(date +%s) - t0 )) s"
cd /home/aagarwal/tt-metal/fabric_exp/consolidation
python3 - "$V" "$G/$NAME" <<'PY'
import os, shutil, sys
sys.path.insert(0, ".")
from offline_build import resolve
variant, tree = os.path.abspath(sys.argv[1]), sys.argv[2]
for src, dst in resolve(variant):
    t = os.path.join(tree, dst)
    if os.path.lexists(t):
        os.remove(t)
    os.makedirs(os.path.dirname(t), exist_ok=True)
    shutil.copy2(src, t)
    print("  ", dst)
PY
# kernel compile caches of the base tree must not be reused
rm -rf $G/$NAME/generated
echo "done in $(( $(date +%s) - t0 )) s"
