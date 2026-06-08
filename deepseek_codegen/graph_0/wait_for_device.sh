#!/bin/bash
# Poll the wedged 4x8 BH galaxy for recovery (DEVICE_OPEN_OK), then re-invoke the agent.
# Context: the device is wedged ("topology_mapper: node 0 not mapped"); the ONLY known recovery is a
# HOST REBOOT (tt-smi -r is harmful here — see memory tt-device-timeout-recovery). A reboot kills this
# tmux/agent session, so this watcher can only catch a NON-reboot recovery (or a reboot+manual relaunch
# where someone re-attaches). Each probe is a clean open->fail->exit (~30s) under a hard timeout — safe.
# Usage: wait_for_device.sh [iterations] [interval_sec]   (default 16 x 1800s = ~8h)
source /home/ubuntu/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/ubuntu/tt-metal PYTHONPATH=/home/ubuntu/tt-metal ARCH_NAME=blackhole TT_METAL_CCACHE_KERNEL_SUPPORT=1
N=${1:-16}; INTERVAL=${2:-1800}
echo "@@@ DEVWATCH START $(date -u +%Y-%m-%dT%H:%M:%SZ)  iters=$N interval=${INTERVAL}s  boot=$(uptime -s)"
for i in $(seq 1 "$N"); do
  timeout 150 python3 /tmp/sanity_dev.py >/tmp/devwatch_sanity.log 2>&1
  if grep -q DEVICE_OPEN_OK /tmp/devwatch_sanity.log; then
    echo "@@@ DEVICE_BACK iter=$i $(date -u +%H:%M:%S)"
    exit 0
  fi
  reason=$(grep -oE "not mapped to any global node: 0|0xffffffff|Failed to allocate TLB|NOC address of a hugepage|Segmentation" /tmp/devwatch_sanity.log | head -1)
  echo "@@@ iter=$i/$N still-wedged $(date -u +%H:%M:%S) boot=$(uptime -s) reason='${reason:-unknown}'"
  [ "$i" -lt "$N" ] && sleep "$INTERVAL"
done
echo "@@@ DEVWATCH_TIMEOUT after $N iters $(date -u +%H:%M:%S) — device never recovered (likely awaiting host reboot)"
exit 2
