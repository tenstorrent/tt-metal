#!/bin/bash
# t41 device A/B on blx03 (one broker job): eager LTX-2.5 1080p/145f seed-0 gen, dense S2 self-attn
# with a Q/K/V dump, then the same gen with the temporal band (mask emulation). Run from ~/fasth3/t41.
# Usage: tmp/t41/job.sh <W> [<W> ...]   (one band run per W after the dense run)
set -o pipefail
T=/home/smarton/fasth3/t41
D=/var/tmp/fasth3/t41
OUT=/home/smarton/fasth3/out/t41
mkdir -p $D $OUT
common="LTX_TRACED=0 RUN_WARMUP=0 PYTEST_TIMEOUT=2400"
echo "T41_STEP dense start $(date +%T)"
W=$T OUT=$OUT bash $T/tmp/blx03/run25.sh dense $common \
  LTX_DUMP_QKV=38760:48:8:0:2 LTX_DUMP_QKV_DIR=$D/qkv LTX_DUMP_LATENTS=$D/lat_dense
echo "T41_STEP dense rc=$? $(date +%T)"
for w in "$@"; do
  W=$T OUT=$OUT bash $T/tmp/blx03/run25.sh band$w $common \
    LTX_SELF_BAND=38760:2040:$w LTX_DUMP_LATENTS=$D/lat_band$w
  echo "T41_STEP band$w rc=$? $(date +%T)"
done
echo T41_JOB_DONE
