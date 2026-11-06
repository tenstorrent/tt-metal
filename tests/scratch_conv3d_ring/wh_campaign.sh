#!/bin/bash
# SCRATCH (conv3d trid-ring experiments, never merge): ring-on (host gate) vs ring forced off on a Wormhole
# CI runner via the custom test dispatch. Census of the gate decisions first, then the production-shape A/B
# for every ring-enabled entry, two rounds, timed by host wall-clock around synchronized call loops (the
# profiler-instrumented reader kernel overflows the Wormhole code region, so no device profiler here).
cd /work || exit 1
export PYTHONPATH=/work TT_METAL_HOME=/work TT_LOGGER_LEVEL=Error HF_HUB_OFFLINE=1
D=/work/generated/ring; mkdir -p $D
S=tests/scratch_conv3d_ring
PY=$(command -v python3)
$PY -c "import ttnn, sys; print('ttnn', ttnn.__file__, sys.version.split()[0]); d=ttnn.open_device(device_id=0); g=d.compute_with_storage_grid_size(); print('arch', d.arch(), 'grid', g.x, g.y); ttnn.close_device(d)" 2>&1 | tail -3
echo "=== $(date -u +%T) CENSUS ==="
TT_CONV3D_PROBE_ONLY=1 CENSUS_ANY_ARCH=1 timeout 1500 $PY $S/census.py $D/wh.jsonl wh 2>&1 | grep -v "^\s*$" | tail -4
echo "=== RING_CENSUS_BEGIN ==="; cat $D/wh.jsonl; echo "=== RING_CENSUS_END ==="
# Compile pass per arm (one call per entry, results discarded) so the kernel cache is warm before timing.
for arm in gate d0; do
  envs=(); [ $arm = d0 ] && envs+=(TT_CONV3D_FORCE_GATHER_TRIDS=0)
  echo "=== $(date -u +%T) compile pass arm $arm ==="
  env "${envs[@]}" timeout 2400 $PY $S/prod_ab.py $D/wh.jsonl $D/r0_prod_${arm} 1 1 2>&1 | grep -E "fail|Error|THROW|overflow|done" | tail -40
done
for r in 1 2; do
  for arm in gate d0; do
    envs=()
    [ $arm = d0 ] && envs+=(TT_CONV3D_FORCE_GATHER_TRIDS=0)
    echo "=== $(date -u +%T) round $r arm $arm ==="
    env "${envs[@]}" timeout 2400 $PY $S/prod_ab.py $D/wh.jsonl $D/r${r}_prod_${arm} 4 40 2>&1 | grep -E "^\[|fail|Error|THROW|info:|overflow|done" | tail -400
    echo "=== RING_RESULTS_BEGIN round$r arm $arm ==="
    $PY $S/wh_summarize.py $D r${r}_prod_${arm} 2>&1 || true
    echo "=== RING_RESULTS_END ==="
  done
done
echo "=== $(date -u +%T) wh_campaign done ==="
