#!/bin/bash
# Poll for the dropped device. Chips dropped off PCIe ("No chips detected"; tt-smi -r fails "No such
# device") — needs a host reboot/power-cycle to re-enumerate. This watcher: each iter probes a (4,8)
# device open; if it fails but chips are VISIBLE again (tt-smi -ls), runs the user-authorized
# `tt-smi -r` from the venv and re-probes; exits 0 on DEVICE_OPEN_OK (re-invokes the agent to resume
# the main.py profile + optimization). Each probe is bounded by a timeout.
# Usage: device_watch_recover.sh [iterations] [interval_sec]   (default 48 x 1800s = 24h)
source /home/ubuntu/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/ubuntu/tt-metal PYTHONPATH=/home/ubuntu/tt-metal ARCH_NAME=blackhole TT_METAL_CCACHE_KERNEL_SUPPORT=1
N=${1:-48}; INTERVAL=${2:-1800}
echo "@@@ DEVWATCH-RECOVER START $(date -u +%Y-%m-%dT%H:%M:%SZ) iters=$N interval=${INTERVAL}s"
for i in $(seq 1 "$N"); do
  timeout 150 python3 /tmp/sanity_dev.py >/tmp/dwr.log 2>&1
  if grep -q DEVICE_OPEN_OK /tmp/dwr.log; then echo "@@@ DEVICE_BACK iter=$i $(date -u +%H:%M:%S)"; exit 0; fi
  if timeout 60 tt-smi -ls 2>/dev/null | grep -qiE "Blackhole|galaxy"; then
    echo "@@@ iter=$i chips VISIBLE but open failed -> tt-smi -r $(date -u +%H:%M:%S)"
    tt-smi -r >/tmp/dwr_reset.log 2>&1
    timeout 150 python3 /tmp/sanity_dev.py >/tmp/dwr.log 2>&1
    if grep -q DEVICE_OPEN_OK /tmp/dwr.log; then echo "@@@ DEVICE_BACK after reset iter=$i $(date -u +%H:%M:%S)"; exit 0; fi
  else
    echo "@@@ iter=$i/$N no chips detected (needs host reboot/power-cycle) $(date -u +%H:%M:%S)"
  fi
  [ "$i" -lt "$N" ] && sleep "$INTERVAL"
done
echo "@@@ DEVWATCH_TIMEOUT after $N iters — chips still gone $(date -u +%H:%M:%S)"; exit 2
