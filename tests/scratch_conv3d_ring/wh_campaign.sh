#!/bin/bash
# SCRATCH (conv3d trid-ring experiments, never merge): ring-on (host gate) vs ring forced off on a Wormhole
# CI runner via the custom test dispatch. Census of the gate decisions first, then the production-shape A/B
# for every ring-enabled entry, two rounds. Everything is printed to stdout between markers.
cd /work || exit 1
export PYTHONPATH=/work TT_METAL_HOME=/work TT_LOGGER_LEVEL=Error HF_HUB_OFFLINE=1
D=/work/generated/ring; mkdir -p $D
S=tests/scratch_conv3d_ring
PY=$(command -v python3)
$PY -c "import ttnn, sys; print('ttnn', ttnn.__file__, sys.version.split()[0]); d=ttnn.open_device(device_id=0); g=d.compute_with_storage_grid_size(); print('arch', d.arch(), 'grid', g.x, g.y); ttnn.close_device(d)" 2>&1 | tail -3
echo "=== $(date -u +%T) CENSUS ==="
TT_CONV3D_PROBE_ONLY=1 CENSUS_ANY_ARCH=1 timeout 1500 $PY $S/census.py $D/wh.jsonl wh 2>&1 | grep -v "^\s*$" | tail -4
echo "=== RING_CENSUS_BEGIN ==="; cat $D/wh.jsonl; echo "=== RING_CENSUS_END ==="
for r in 1 2; do
  for arm in gate d0; do
    envs=(TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_OPS_CSV=$D/r${r}_prod_${arm}.csv)
    [ $arm = d0 ] && envs+=(TT_CONV3D_FORCE_GATHER_TRIDS=0)
    echo "=== $(date -u +%T) round $r arm $arm ==="
    env "${envs[@]}" timeout 2400 $PY $S/prod_ab.py $D/wh.jsonl $D/r${r}_prod_${arm} 2 8 2>&1 | grep -E "^\[|fail|Error|done" | tail -200
  done
  echo "=== RING_RESULTS_BEGIN round$r ==="
  $PY $S/wh_summarize.py $D r1_prod_gate r1_prod_d0 r2_prod_gate r2_prod_d0
  echo "=== RING_RESULTS_END ==="
done
echo "=== $(date -u +%T) wh_campaign done ==="
