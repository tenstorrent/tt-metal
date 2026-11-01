#!/bin/bash
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
ROOT=/home/ttuser/agobeljic/tt-metal
KFILES="tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute"
cd $ROOT/tt-train
export TT_METAL_HOME=$ROOT TT_METAL_RUNTIME_ROOT=$ROOT
unset TT_MESH_GRAPH_DESC_PATH
restore() { (cd $ROOT && git checkout -- $KFILES); }
run() {
  restore
  ../python_env/bin/python $O/iter26_attrib_patch.py $1 $2; prc=$?
  if [ $prc -ne 0 ]; then echo "$1 $2: patch rc=$prc, skipped"; restore; return; fi
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  timeout 600 ../python_env/bin/python $O/iter26_sdpa_time.py --tag "$1/$2" --iters 20 > $O/iter26_attrib_${1}_${2//,/_}.log 2>&1; rc=$?
  grep -E "^$1/$2 case" $O/iter26_attrib_${1}_${2//,/_}.log || echo "$1 $2: rc=$rc (no result)"
  restore
}
echo "=== attribution 2 start $(date -u +%FT%TZ)"
for k in fw kv q; do run $k noacc; run $k nopack; run $k skeleton,nopack; done
restore
cd $ROOT && git status --short $KFILES
echo "=== attribution 2 done $(date -u +%FT%TZ)"
