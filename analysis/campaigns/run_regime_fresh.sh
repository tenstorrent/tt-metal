#!/bin/bash
# Generic regime runner: run_regime.sh <tag> <mode plain|mp> <pytest target incl -k> ; env carries the harness knobs.
set -e
TAG=$1; MODE=$2; TARGET=$3
# Path resolution for both layouts (workspace or $TTM/analysis/campaigns): see campaign_paths.sh.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"   # TTM_FRESH is the fresh checkout, TTM the one whose python env is used
cd $TTM_FRESH; export TT_METAL_HOME=$TTM_FRESH ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1 PYTHONPATH=$TTM_FRESH:$TTM_FRESH/ttnn:$TTM_FRESH/tools; source $PYENV
if [ "$MODE" = mp ]; then TP="python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest"; else TP="python -m tracy -r -m pytest"; fi
rm -rf generated/profiler/reports/*
rm -f generated/profiler/.logs/profile_log_device.csv
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ENVSTR=$(env | grep -E "^(P2_|MLA_|MLAD_|CK_|SDPA_|R1_|SP_|JT_|DEC_)" | tr '\n' ' ')
echo "### $TAG mode=$MODE start $T0 sha=$(git rev-parse --short HEAD) target=$TARGET env: $ENVSTR"
eval $TP $TARGET -s > $DD/$TAG.log 2>&1 || true
if ! grep -q " passed" $DD/$TAG.log || [ ! -s generated/profiler/.logs/profile_log_device.csv ]; then echo "RUN FAILED, see $DD/$TAG.log"; grep -E "Error|FAILED|Traceback" $DD/$TAG.log | head -5; exit 1; fi
FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $DD/$TAG.log | sed "s/firmware bundle version: //")
{ echo "PROVENANCE: p100a 110 cores 1350MHz fw_bundle=$FW; tt-metal-fresh $(git rev-parse HEAD) (main tip 2dbd14bf632 merged + BRISC fw-size patch) branch $(git branch --show-current); $T0; cmd: $TP $TARGET -s; env $ENVSTR mode=$MODE; zone_config: $(grep -h '#define SDPA_' ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp | tr '\n' ' ')"; cat generated/profiler/.logs/profile_log_device.csv; } > $DD/$TAG.csv
R=$(ls -d generated/profiler/reports/*/ 2>/dev/null | sort | tail -1); sleep 2; if [ -n "$R" ] && ls $R/ops_perf_results*.csv >/dev/null 2>&1; then cp $R/ops_perf_results*.csv $DD/${TAG}_ops_perf_results.csv; fi
tail -n +2 $DD/$TAG.csv > /tmp/zr_$$.csv; python $TTM/analysis/zone_reduce.py /tmp/zr_$$.csv --out $DD/$TAG | tee -a $DD/$TAG.log | grep -v "^wrote"; rm -f /tmp/zr_$$.csv
echo "### $TAG done $(date -u +%H:%M:%SZ)"
