#!/bin/bash
# Slot study (SLOT_STUDY.md): a lean Galaxy tree of the live study tree with the lead's router files, at
# /data/aagarwal/frs-glx/trees/$1 (default sls_lead). The /data base tree (2026-10-29) predates the lead's host named
# args (SENDER_CH_<i>_CONNECTION_SEMAPHORE_ADDR), so the host comes from the live tree's build_Release.
# What is copied (as the VC2 line agent's lean trees): tt_metal, runtime, tests/tt_metal, ttnn, tt_stl, the libraries and
# the test_tt_fabric / fabric_unit_tests binaries of build_Release, fabric_exp/{vc1,galaxy,slotstudy} and
# consolidation/hang_capture.py; python_env is a copy of the minimal system-python venv of trees/vc2l_p3 (ttrun.py's
# needs). A broker job may have a router variant swapped into the live tree: every file of the active swap's manifest is
# put back to its base copy (the backup) or removed, then the lead's files go in (offline_build.resolve, as
# mk_glx_tree.sh). The manifest is checked before and after the copy; verify against the idle tree with VERIFY=1 later.
set -euo pipefail
NAME=${1:-sls_lead}
S=/home/aagarwal/tt-metal
G=/data/aagarwal/frs-glx/trees
D=$G/$NAME
BK=/home/aagarwal/.tt-metal-p1-swap-backup
V=$S/fabric_exp/variants/v4rxv2i3
man_sum() { [ -f $BK/manifest.json ] && md5sum < $BK/manifest.json | cut -c1-12 || echo none; }

if [ "${VERIFY:-0}" = 1 ]; then  # compare with the live tree (no swap active): only the lead's files may differ
    [ "$(man_sum)" = none ] || { echo "a swap is active in the live tree: verify later"; exit 1; }
    cd $S
    { rsync -rcn --out-format='%n' tt_metal runtime ttnn tt_stl $D/ --exclude='__pycache__' --exclude='*.pyc'
      rsync -rcn --out-format='tests/%n' tests/tt_metal $D/tests/ --exclude='__pycache__' --exclude='*.pyc'
    } | grep -v '/$' > /tmp/sls_verify_$$.txt || true
    [ "$(man_sum)" = none ] || { echo "a swap became active during the verify: verify later"; rm -f /tmp/sls_verify_$$.txt; exit 1; }
    python3 - $V /tmp/sls_verify_$$.txt <<'PY'
import sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
from offline_build import resolve
lead = {d for _, d in resolve(sys.argv[1])}
diff = [l.strip() for l in open(sys.argv[2]) if l.strip()]
other = [d for d in diff if d not in lead]
print(f"differing files: {len(diff)}; the lead's: {len([d for d in diff if d in lead])}; others: {len(other)}")
for d in other[:20]:
    print("   ", d)
sys.exit(1 if other else 0)
PY
    rc=$?; rm -f /tmp/sls_verify_$$.txt; exit $rc
fi

[ -e $D ] && { echo "$D exists"; exit 1; }
m0=$(man_sum); t0=$(date +%s)
echo "swap manifest before: $m0"
mkdir -p $D/build_Release $D/tests $D/fabric_exp/consolidation
cd $S
rsync -a --exclude='__pycache__' --exclude='*.pyc' tt_metal runtime ttnn tt_stl $D/
rsync -a --exclude='__pycache__' tests/tt_metal $D/tests/
for p in tt_metal/libtt_metal.so tt_stl/libtt_stl.so lib tt_metal/third_party/umd/lib \
         test/tt_metal/tt_fabric/test_infra/test_tt_fabric test/tt_metal/tt_fabric/fabric_unit_tests; do
    mkdir -p $D/build_Release/$(dirname $p)
    rsync -a build_Release/$p $D/build_Release/$(dirname $p)/
done
ln -s build_Release $D/build
rsync -a fabric_exp/vc1 fabric_exp/galaxy fabric_exp/slotstudy $D/fabric_exp/
cp -a fabric_exp/consolidation/hang_capture.py $D/fabric_exp/consolidation/
cp -a $G/vc2l_p3/python_env $D/python_env
m1=$(man_sum)
echo "copied in $(( $(date +%s) - t0 )) s; swap manifest after: $m1"
[ "$m0" = "$m1" ] || { echo "!!! the live tree's swap changed during the copy: remove $D and retry"; exit 1; }
if [ "$m1" != none ]; then
    python3 - $BK $D <<'PY'
import json, os, shutil, sys
bk, d = sys.argv[1:3]
man = json.load(open(os.path.join(bk, "manifest.json")))
for p in man["saved"]:
    shutil.copy2(os.path.join(bk, p), os.path.join(d, p)); print("   base:", p)
for p in man["created"]:
    if os.path.lexists(os.path.join(d, p)):
        os.remove(os.path.join(d, p)); print("   removed:", p)
PY
fi
python3 - $V $D <<'PY'
import os, shutil, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
from offline_build import resolve
variant, tree = sys.argv[1:3]
for src, dst in resolve(variant):
    t = os.path.join(tree, dst)
    if os.path.lexists(t):
        os.remove(t)
    os.makedirs(os.path.dirname(t), exist_ok=True)
    shutil.copy2(src, t)
    print("   lead:", dst)
PY
rm -rf $D/generated
echo "$NAME: $(du -sh --apparent-size $D | cut -f1), $(find $D -type f | wc -l) files, done in $(( $(date +%s) - t0 )) s"
