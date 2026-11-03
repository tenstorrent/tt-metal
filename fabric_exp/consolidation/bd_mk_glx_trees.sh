#!/bin/bash
# REFERENCE ONLY (plan change 2026-11-02): a separate Galaxy agent runs the Galaxy points 2, 4 and 5 with its own
# job; this breakdown task submits no Galaxy job. Kept as a worked example of what the mapping needs (two trees,
# source snapshots, kept caches, compile logs): fabric_exp/report/BREAKDOWN_PLAN.md, "Galaxy samples: what to save".
# Cycle breakdown, phase 2 on the Galaxy (fabric_exp/report/BREAKDOWN_PLAN.md): two lean Galaxy trees under
# /data/aagarwal/frs-glx/trees/, one per router, because main's router (kernel v1) and the new router share headers
# that a variant replaces:
#   bd_main_TAG   the live study tree with no variant in it (main's router as shipped runs from it: kernel v1, flags 0)
#   bd_new_TAG    a hard-link copy of bd_main_TAG with NEW_VARIANT's files put in (each replaced file removed first, so
#                 the hard links to bd_main_TAG are never written through)
# Both get pc_sampler.py and hang_capture.py, and a python_env with tt-exalens (a copy of trees/$PYENV_FROM/python_env,
# default sls_lead's: a system-python venv with tt-exalens 0.4.1 and tt-umd 0.9.12, which hang captures ran from on the
# 110-a nodes, job 168774). What is copied, and the handling of a broker swap active in the live tree, follow
# slotstudy/mk_sls_tree.sh: every file of the active swap's manifest goes back to its base copy (the backup) or is
# removed, and the manifest is compared before and after the copy.
#   bd_mk_glx_trees.sh NEW_VARIANT TAG        (~20-30 min: an rsync of tt_metal, runtime, ttnn, tt_stl and the binaries)
# Writes only under /data/aagarwal/frs-glx/trees/bd_{main,new}_TAG. Phase 2 only (after the main session's go).
set -euo pipefail
NEW=${1:?usage: bd_mk_glx_trees.sh NEW_VARIANT TAG}; TAG=${2:?usage: bd_mk_glx_trees.sh NEW_VARIANT TAG}
S=/home/aagarwal/tt-metal
G=/data/aagarwal/frs-glx/trees
DM=$G/bd_main_$TAG
DN=$G/bd_new_$TAG
BK=/home/aagarwal/.tt-metal-p1-swap-backup
V=$S/fabric_exp/variants/$NEW
PYENV_FROM=${PYENV_FROM:-sls_lead}
man_sum() { [ -f $BK/manifest.json ] && md5sum < $BK/manifest.json | cut -c1-12 || echo none; }

[ -d $V ] || { echo "no variant $V"; exit 1; }
[ -e $DM ] || [ -e $DN ] && { echo "$DM or $DN exists"; exit 1; }
$G/$PYENV_FROM/python_env/bin/python -c "import ttexalens" || { echo "$PYENV_FROM's python_env has no tt-exalens"; exit 1; }
m0=$(man_sum); t0=$(date +%s)
echo "swap manifest before: $m0"
mkdir -p $DM/build_Release $DM/tests $DM/fabric_exp/consolidation
cd $S
rsync -a --exclude='__pycache__' --exclude='*.pyc' tt_metal runtime ttnn tt_stl $DM/
rsync -a --exclude='__pycache__' tests/tt_metal $DM/tests/
for p in tt_metal/libtt_metal.so tt_stl/libtt_stl.so lib tt_metal/third_party/umd/lib \
         test/tt_metal/tt_fabric/test_infra/test_tt_fabric test/tt_metal/tt_fabric/fabric_unit_tests; do
    mkdir -p $DM/build_Release/$(dirname $p)
    rsync -a build_Release/$p $DM/build_Release/$(dirname $p)/
done
ln -s build_Release $DM/build
rsync -a fabric_exp/galaxy $DM/fabric_exp/
mkdir -p $DM/fabric_exp/consolidation/bd_rows
cp -a fabric_exp/consolidation/pc_sampler.py fabric_exp/consolidation/hang_capture.py $DM/fabric_exp/consolidation/
cp -a fabric_exp/consolidation/bd_rows/glx_*.yaml $DM/fabric_exp/consolidation/bd_rows/
cp -a $G/$PYENV_FROM/python_env $DM/python_env
m1=$(man_sum)
echo "copied in $(( $(date +%s) - t0 )) s; swap manifest after: $m1"
[ "$m0" = "$m1" ] || { echo "!!! the live tree's swap changed during the copy: remove $DM and retry"; exit 1; }
if [ "$m1" != none ]; then  # put the base files back where a broker job's variant was in the live tree
    python3 - $BK $DM <<'PY'
import json, os, shutil, sys
bk, d = sys.argv[1:3]
man = json.load(open(os.path.join(bk, "manifest.json")))
for p in man["saved"]:
    os.remove(os.path.join(d, p)); shutil.copy2(os.path.join(bk, p), os.path.join(d, p)); print("   base:", p)
for p in man["created"]:
    if os.path.lexists(os.path.join(d, p)):
        os.remove(os.path.join(d, p)); print("   removed:", p)
PY
fi
# main's router sources must equal git HEAD's (+ the live tree's uncommitted worker-adapter fix, which is not swapped)
for f in tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp \
         tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h; do
    git -C $S show HEAD:$f | cmp -s - $DM/$f && echo "   = HEAD: $f" || echo "   !!! differs from HEAD: $f"
done
cp -al $DM $DN
python3 - $V $DN <<'PY'
import os, shutil, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
from offline_build import resolve
variant, tree = sys.argv[1:3]
for src, dst in resolve(variant):
    t = os.path.join(tree, dst)
    if os.path.lexists(t):
        os.remove(t)  # never write through a hard link into bd_main
    os.makedirs(os.path.dirname(t), exist_ok=True)
    shutil.copy2(src, t)
    print("   variant:", dst)
PY
echo "trees: $DM, $DN ($(( $(date +%s) - t0 )) s)"
