#!/bin/bash
# Traced single 4x4 stage (12 layers, top half, FABRIC_2D) through the real runner at several chunk sizes.
# usage: chunk_sweep.sh <outdir> [chunk sizes...]   (default: 5120 2560)
# Each chunk size runs with PREFILL_KV_PERIOD_TOKENS equal to it; tuned configs stay keyed to the
# production 640 tokens/chip (plus the 1280 entries for 4x4 stages), so other per-chip sizes may fall
# back to default configs. Compare tokens per ms across sizes; cadence.py prints the throughput.
HERE=$(cd "$(dirname "$0")" && pwd)
OUTDIR=$1; shift; SIZES=${*:-5120 2560}
for CH in $SIZES; do
  N=$OUTDIR/stage_2d_$CH; rm -rf "$N"
  timeout 600 flock -w 300 /tmp/tt-device.lock tt-smi -r > /dev/null 2>&1
  PREFILL_KV_PERIOD_TOKENS=$CH NRANKS=1 NUM_LAYERS=12 WARMUP=2 BINDING_OVERRIDE=$HERE/binding_1rank_top_2d.yaml \
    timeout 3600 flock -w 600 /tmp/tt-device.lock "$HERE/launch_runner.sh" 2d $CH "$N" 1 > "$N.out" 2>&1
  echo "chunk $CH exit=$?"; grep -E "min_pcc" "$N.out" | cut -c1-120
  python3 "$HERE/cadence.py" "$N" $CH 2
done
