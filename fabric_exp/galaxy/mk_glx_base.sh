#!/bin/bash
# A fresh lean Galaxy base tree from the live study tree, at /data/aagarwal/frs-glx/trees/$1: main's router (kernel v1)
# runs from it, and mk_glx_tree.sh (BASE=$1) hard-links variant trees from it. Same copy as slotstudy/mk_sls_tree.sh and
# consolidation/bd_mk_glx_trees.sh: tt_metal, runtime, ttnn, tt_stl, tests/tt_metal, the libraries and the
# test_tt_fabric / fabric_unit_tests binaries of build_Release (the live host: per-VC args, worker-role arg, VC1 -O3),
# fabric_exp/galaxy, consolidation/{pc_sampler,hang_capture}.py, and python_env = a copy of trees/$PYENV_FROM/python_env
# (default sls_lead: system-python venv with tt-exalens 0.4.1 / tt-umd 0.9.12, the versions the P150 sampler uses;
# hang captures ran from it on the 110-a nodes, job 168774).
# A broker job may have a variant swapped into the live tree: the swap manifest is compared before and after the copy,
# and every file of an active swap goes back to its base copy (the backup) or is removed. The host library's mtime is
# compared too (a host rebuild during the copy stops it). main's router sources are checked against git HEAD.
#   mk_glx_base.sh NAME          (~5-25 min, NFS-bound). Writes only under /data/aagarwal/frs-glx/trees/NAME.
#   VERIFY=1 mk_glx_base.sh NAME (no swap may be active or happen meanwhile: checked by the manifest and its .done
#                                file): checksum-compare NAME with the live tree; FIX=1 also copies the differing files
#                                from the live tree (a swap that ran during the first copy can leave variant files).
set -euo pipefail
NAME=${1:?usage: mk_glx_base.sh NAME}
S=/home/aagarwal/tt-metal
G=/data/aagarwal/frs-glx/trees
D=$G/$NAME
BK=/home/aagarwal/.tt-metal-p1-swap-backup
PYENV_FROM=${PYENV_FROM:-sls_lead}
man_sum() { [ -f $BK/manifest.json ] && md5sum < $BK/manifest.json | cut -c1-12 || echo none; }
lib_stamp() { stat -c '%Y %s' $S/build_Release/tt_metal/libtt_metal.so; }

swap_stamp() { echo "$(man_sum) $(stat -c %Y $BK/manifest.json.done 2>/dev/null)"; }
if [ "${VERIFY:-0}" = 1 ]; then
    [ -d $D ] || { echo "no $D"; exit 1; }
    s0=$(swap_stamp); l0=$(lib_stamp)
    [ "${s0%% *}" = none ] || { echo "a swap is active in the live tree: verify later"; exit 1; }
    cd $S
    L=$(mktemp)
    { rsync -rcn --delete --out-format='%n' --exclude='__pycache__' --exclude='*.pyc' tt_metal runtime ttnn tt_stl $D/
      rsync -rcn --delete --out-format='tests/%n' --exclude='__pycache__' --exclude='*.pyc' tests/tt_metal $D/tests/ |
          sed 's#^deleting #deleting tests/#'
    } | grep -v '/$' > $L || true
    s1=$(swap_stamp); l1=$(lib_stamp)
    [ "$s0" = "$s1" ] && [ "$l0" = "$l1" ] || { echo "a swap or host rebuild ran during the verify: verify later"; rm -f $L; exit 1; }
    echo "files differing from the live tree: $(wc -l < $L)"
    sed 's/^/   /' $L | head -40
    if [ "${FIX:-0}" = 1 ] && [ -s $L ]; then
        while read -r f; do
            case $f in
                deleting\ *) f=${f#deleting }; rm -f "$D/$f"; echo "   removed (not in the live tree): $f" ;;
                *) [ -e "$S/$f" ] && { rm -f "$D/$f"; mkdir -p "$(dirname "$D/$f")"; cp -a "$S/$f" "$D/$f"; echo "   fixed: $f"; } ;;
            esac
        done < $L
        s2=$(swap_stamp)
        [ "$s0" = "$s2" ] || { echo "!!! a swap ran during the fix: verify again"; rm -f $L; exit 1; }
    fi
    rm -f $L
    exit 0
fi
[ -e $D ] && { echo "$D exists"; exit 1; }
$G/$PYENV_FROM/python_env/bin/python -c "import ttexalens, yaml" || { echo "$PYENV_FROM's python_env has no tt-exalens"; exit 1; }
m0=$(man_sum); l0=$(lib_stamp); t0=$(date +%s)
echo "swap manifest before: $m0; libtt_metal.so: $l0 ($(date -d @${l0%% *} -u +%FT%TZ))"
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
rsync -a --exclude='__pycache__' fabric_exp/galaxy $D/fabric_exp/
cp -a fabric_exp/consolidation/pc_sampler.py fabric_exp/consolidation/hang_capture.py $D/fabric_exp/consolidation/
cp -a $G/$PYENV_FROM/python_env $D/python_env
m1=$(man_sum); l1=$(lib_stamp)
echo "copied in $(( $(date +%s) - t0 )) s; swap manifest after: $m1; libtt_metal.so: $l1"
[ "$m0" = "$m1" ] || { echo "!!! the live tree's swap changed during the copy: remove $D and retry"; exit 1; }
[ "$l0" = "$l1" ] || { echo "!!! the host library changed during the copy: remove $D and retry"; exit 1; }
if [ "$m1" != none ]; then  # put the base files back where a broker job's variant was in the live tree
    python3 - $BK $D <<'PY'
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
# main's router sources must equal git HEAD's (the live tree's uncommitted host and worker-adapter changes are not
# router sources)
bad=0
for f in tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp \
         tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp \
         tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp \
         tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp \
         tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp \
         tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp \
         tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp \
         tt_metal/hw/inc/api/dataflow/dataflow_api.h \
         tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h; do
    if git -C $S show HEAD:$f | cmp -s - $D/$f; then echo "   = HEAD: $f"; else echo "   !!! differs from HEAD: $f"; bad=1; fi
done
[ -n "$(find $D/tt_metal/fabric/hw/inc/edm_fabric/router -type f 2>/dev/null)" ] && { echo "   !!! a variant's router/ files are in the base"; bad=1; }
[ $bad = 0 ] || { echo "!!! $D is not a clean base: remove it and retry"; exit 1; }
echo "$NAME: $(du -sh --apparent-size $D | cut -f1), $(find $D -type f | wc -l) files, done in $(( $(date +%s) - t0 )) s"
