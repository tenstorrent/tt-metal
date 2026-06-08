#!/bin/bash
# Sweep moe_compute combine configs to find one that doesn't hang on BH 4x8 (cluster_axis=0, epd=8).
# Each config is its own timeout-guarded subprocess (so a hang only kills that attempt); device
# health is checked between configs; stop on success or a device wedge.
cd /home/ubuntu/tt-metal/deepseek_codegen/graph_0
source /home/ubuntu/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/ubuntu/tt-metal PYTHONPATH=/home/ubuntu/tt-metal ARCH_NAME=blackhole
export TT_METAL_CCACHE_KERNEL_SUPPORT=1 PYTHONUNBUFFERED=1
mkdir -p logs

sanity() { timeout 120 python3 /tmp/sanity_dev.py >/tmp/sweep_sanity.log 2>&1; grep -q DEVICE_OPEN_OK /tmp/sweep_sanity.log; }

# config = "TOPOLOGY NUM_LINKS BH_RING(0=auto)"  (already-tried & hung: auto-Ring, Linear/nl=2)
configs=(
  "Ring 1 0"
  "Linear 1 0"
  "Ring 2 8"
  "Linear 1 8"
  "Ring 1 16"
)

echo "@@@ SWEEP START $(date +%H:%M:%S)"
for c in "${configs[@]}"; do
  set -- $c; TOPO=$1; NL=$2; BHR=$3; tag="t${TOPO}_l${NL}_r${BHR}"
  echo "@@@ CONFIG $tag  $(date +%H:%M:%S)"
  extra=""; [ "$BHR" != 0 ] && extra="MOE_BH_RING=$BHR"
  env MOE_USE_COMPUTE=1 MOE_DEBUG_SYNC=1 MOE_TOPOLOGY=$TOPO MOE_NUM_LINKS=$NL $extra \
      timeout 420 python3 moe_test.py > "logs/sweep_$tag.log" 2>&1
  ec=$?
  if grep -q "\[moe_dbg\] moe_compute OK" "logs/sweep_$tag.log"; then
    echo "@@@ RESULT $tag: *** COMBINE RAN (moe_compute OK) ***  ec=$ec"
    grep -E "moe_dbg|PCC=|PASS|FAIL" "logs/sweep_$tag.log" | grep -v leaked | tail -8
    echo "@@@ STOP: found a working combine config -> $tag"
    break
  fi
  echo "@@@ RESULT $tag: combine did NOT complete (no 'moe_compute OK'); ec=$ec (124=timeout/hang)"
  grep -E "\[moe_dbg\]|TT_FATAL|RuntimeError|overlaps|Traceback" "logs/sweep_$tag.log" | grep -v leaked | tail -4
  if ! sanity; then
    echo "@@@ DEVICE WEDGED after $tag (sanity open failed) — STOPPING sweep; needs host reboot."
    break
  fi
  echo "@@@ device healthy after $tag, continuing"
done
echo "@@@ SWEEP DONE $(date +%H:%M:%S)"
