#!/bin/bash
# runs several kernel variants in one go (each its own container)
cd /mnt/tt-data/ssinghal/wt/hy4_pfpool/models/demos/blackhole/hy4_preview/tests
for v in "k3 FID=hifi2 PATS=256,mix" "k4 FID=lofi PATS=256,mix" "k5 FID=hifi2 KC=64 PATS=256" "k6 FID=hifi2 KC=256 PATS=256"; do
  set -- $v; tag=$1; shift
  SLOT=2 TMO=900 ./pfp_run.sh $tag pfp_kernel.py "$@"
done
