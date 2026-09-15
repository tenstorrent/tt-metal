#!/bin/bash
# T0.2 zone-tax micro-benchmark under tracy with sum profiling enabled.
# Usage: run_zone_tax.sh <tag> "<ZT_GRID>" [ZT_CORES]
set -e
TAG=$1; GRID=$2; CORES=${3:-1x1}
# Path resolution for both layouts (workspace or $TTM/analysis/campaigns): see campaign_paths.sh.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SD/campaign_paths.sh"
cd $TTM
export TT_METAL_HOME=$TTM ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1
source $PYENV
TP="python -m tracy --enable-sum-profiling -m pytest"
rm -f generated/profiler/.logs/profile_log_device.csv
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "### $TAG zone_tax grid=$GRID cores=$CORES start $T0 sha=$(git rev-parse --short HEAD)"
ZT_GRID="$GRID" ZT_CORES=$CORES $TP analysis/zone_tax.py::test_zone_tax -s > $DD/$TAG.log 2>&1 || true
if ! grep -q "1 passed" $DD/$TAG.log || [ ! -s generated/profiler/.logs/profile_log_device.csv ]; then echo "RUN FAILED, see $DD/$TAG.log"; grep -E "Error|FAILED|Traceback|error:" $DD/$TAG.log | head -8; exit 1; fi
{
  FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $DD/$TAG.log | sed "s/firmware bundle version: //"); echo "PROVENANCE: p100a 1350MHz fw_bundle=$FW; tt-metal $(git rev-parse HEAD) branch $(git branch --show-current); $T0; cmd: ZT_GRID=$GRID ZT_CORES=$CORES $TP analysis/zone_tax.py::test_zone_tax -s; PROFILE_KERNEL flags: TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SUM=1 (PROFILE_KERNEL=1|PROFILER_OPT_DO_SUM=9); zone_config: $(grep -h '#define SDPA_ZONES' ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp)"
  cat generated/profiler/.logs/profile_log_device.csv
} > $DD/$TAG.csv
tail -n +2 $DD/$TAG.csv > /tmp/zt_$$.csv
python analysis/zone_reduce.py /tmp/zt_$$.csv --out $DD/$TAG | tee -a $DD/$TAG.log | grep -v "^wrote"
rm -f /tmp/zt_$$.csv
grep -c "PROFILE_KERNEL" $DD/$TAG.log >/dev/null && grep -o "\-DPROFILE_KERNEL=[0-9]*" $DD/$TAG.log | sort | uniq -c
echo "### $TAG done $(date -u +%H:%M:%SZ)"
