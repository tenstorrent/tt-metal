#!/bin/bash
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
ROOT=/home/ttuser/agobeljic/tt-metal
SP=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
DF=$ROOT/tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/dataflow
KFILES="tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute"
cd $ROOT/tt-train
export TT_METAL_HOME=$ROOT TT_METAL_RUNTIME_ROOT=$ROOT
unset TT_MESH_GRAPH_DESC_PATH
restore() { (cd $ROOT && git checkout -- $KFILES); cp $SP/sdpa_fw_reader_kernel.cpp $SP/sdpa_fw_writer_kernel.cpp $DF/; }
run() {  # tag compute_variants dataflow_variants
  restore
  ../python_env/bin/python $O/iter26_attrib_patch.py fw $2; prc=$?
  if [ $prc -ne 0 ]; then echo "$1: patch rc=$prc, skipped"; restore; return; fi
  if [ "$3" != "-" ]; then ../python_env/bin/python $O/iter26_attrib_patch.py fwdf $3; prc=$?; if [ $prc -ne 0 ]; then echo "$1: df patch rc=$prc, skipped"; restore; return; fi; fi
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  timeout 120 ../python_env/bin/python $O/iter26_sdpa_time.py --tag "$1" --iters 20 > $O/iter26_attrib_$1.log 2>&1; rc=$?
  grep -E "^$1 case" $O/iter26_attrib_$1.log || echo "$1: rc=$rc (no result)"
  restore
}
echo "=== attribution 4 start $(date -u +%FT%TZ)"
run fw_skel_nostream skeleton,nopack nofwd,noread
run fw_skel_noreconfig skeleton,nopack,noregs,nocopy,noreconfig -
restore
cd $ROOT && git status --short $KFILES && git diff --stat | tail -1
echo "=== attribution 4 done $(date -u +%FT%TZ)"
