#!/bin/bash
# T2.8 decode runner: tracy with ops report (-r) so DEVICE KERNEL DURATION per invocation is in the ops CSV; also the raw device log.
set -e
TAG=$1; MODE=${2:-plain}
# Path resolution for both layouts (workspace or $TTM/analysis/campaigns): see campaign_paths.sh.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"
cd $TTM; export TT_METAL_HOME=$TTM ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1; source $PYENV
export DEC_B=${DEC_B:-32} DEC_POS=${DEC_POS:-1024} DEC_GRID=${DEC_GRID:-8x8} DEC_KV_DTYPE=${DEC_KV_DTYPE:-bfp8_b} DEC_ITERS=${DEC_ITERS:-3} DEC_NH=${DEC_NH:-32} DEC_NKV=${DEC_NKV:-8} DEC_D=${DEC_D:-128} DEC_BLOCK=${DEC_BLOCK:-32}
if [ "$MODE" = mp ]; then TP="python -m tracy -r --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest"; else TP="python -m tracy -r -m pytest"; fi
rm -f generated/profiler/.logs/profile_log_device.csv; rm -rf generated/profiler/reports/*
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ); ENVS="b=$DEC_B pos=$DEC_POS grid=$DEC_GRID kv_dtype=$DEC_KV_DTYPE nh=$DEC_NH nkv=$DEC_NKV d=$DEC_D block=$DEC_BLOCK iters=$DEC_ITERS maxseq=${DEC_MAXSEQ:-auto}"
echo "### $TAG DECODE mode=$MODE start $T0 sha=$(git rev-parse --short HEAD) env: $ENVS"
$TP analysis/decode_sweep.py::test_decode_sweep -s > $DD/$TAG.log 2>&1 || true
if ! grep -q "1 passed" $DD/$TAG.log || [ ! -s generated/profiler/.logs/profile_log_device.csv ]; then echo "RUN FAILED, see $DD/$TAG.log"; grep -E "^E |Error|FAILED" $DD/$TAG.log | head -5; exit 1; fi
FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $DD/$TAG.log | sed "s/firmware bundle version: //")
{ echo "PROVENANCE: p100a fw_bundle=$FW; tt-metal $(git rev-parse HEAD) branch $(git branch --show-current) (kernels 72620d5, zones off); $T0; cmd: $TP analysis/decode_sweep.py::test_decode_sweep -s; env $ENVS mode=$MODE"; cat generated/profiler/.logs/profile_log_device.csv; } > $DD/$TAG.csv
sleep 2; R=$(ls -d generated/profiler/reports/*/ 2>/dev/null | sort | tail -1); [ -n "$R" ] && cp $R/ops_perf_results*.csv $DD/${TAG}_ops_perf_results.csv
tail -n +2 $DD/$TAG.csv > /tmp/zr_$$.csv; python analysis/zone_reduce.py /tmp/zr_$$.csv --out $DD/$TAG | tee -a $DD/$TAG.log | grep -v "^wrote"; rm -f /tmp/zr_$$.csv
echo "### $TAG done $(date -u +%H:%M:%SZ)"
