#!/bin/bash
# Iteration 26 attribution: stage removal in the SDPA forward, Q backward and KV backward compute kernels at the
# TP=4 shard shape (chain mode). Each variant is patched in place (JIT recompiles), timed, then restored from git.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
ROOT=/home/ttuser/agobeljic/tt-metal
KFILES="tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute"
cd $ROOT/tt-train
export TT_METAL_HOME=$ROOT TT_METAL_RUNTIME_ROOT=$ROOT
unset TT_MESH_GRAPH_DESC_PATH
restore() { (cd $ROOT && git checkout -- $KFILES); }
run() {  # kernel variants
  restore
  ../python_env/bin/python $O/iter26_attrib_patch.py $1 $2; prc=$?
  if [ $prc -ne 0 ]; then echo "$1 $2: patch rc=$prc, skipped"; restore; return; fi
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  timeout 600 ../python_env/bin/python $O/iter26_sdpa_time.py --tag "$1/$2" --iters 20 > $O/iter26_attrib_${1}_${2//,/_}.log 2>&1; rc=$?
  grep -E "^$1/$2 case" $O/iter26_attrib_${1}_${2//,/_}.log || echo "$1 $2: rc=$rc (no result)"
  restore
}
echo "=== attribution start $(date -u +%FT%TZ)"
run fw base
for v in noqk nomax nosub noexp nopv nocorr nomask skeleton; do run fw $v; done
run kv base
for v in noqk noexp nodp nods nodv nodk skeleton; do run kv $v; done
run q base
for v in noqk noexp nodp nods nodq skeleton; do run q $v; done
restore
cd $ROOT && git status --short $KFILES
echo "=== attribution done $(date -u +%FT%TZ)"
