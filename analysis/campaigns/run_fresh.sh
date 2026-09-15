#!/bin/bash
# Same as run_zone.sh but on the fresh main-tip build (tt-metal-fresh, env per bh/fresh_build.md). Zones do not exist there.
set -e
TAG=$1; MODE=${2:-plain}
# Path resolution for both layouts (workspace or $TTM/analysis/campaigns): see campaign_paths.sh.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"   # TTM_FRESH is the fresh checkout, TTM the one whose python env is used
cd $TTM_FRESH; export TT_METAL_HOME=$TTM_FRESH ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1 PYTHONPATH=$TTM_FRESH:$TTM_FRESH/ttnn:$TTM_FRESH/tools
source $PYENV
export SDPA_NH=${SDPA_NH:-32} SDPA_NKV=${SDPA_NKV:-8} SDPA_HEAD_DIM=${SDPA_HEAD_DIM:-128} SDPA_DTYPE=${SDPA_DTYPE:-bfp8_b}
export SDPA_FIDELITY=${SDPA_FIDELITY:-HiFi2} SDPA_EXP_APPROX=${SDPA_EXP_APPROX:-1} SDPA_ITERS=${SDPA_ITERS:-3} SDPA_SEQ=${SDPA_SEQ:-4096}
export SDPA_QCHUNK=${SDPA_QCHUNK:-128}; export SDPA_KCHUNK=${SDPA_KCHUNK:-$SDPA_QCHUNK}; export SDPA_CAUSAL=${SDPA_CAUSAL:-1}
if [ "$MODE" = mp ]; then TP="python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest"; else TP="python -m tracy -m pytest"; fi
rm -f generated/profiler/.logs/profile_log_device.csv
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ENVS="S=$SDPA_SEQ nh=$SDPA_NH nkv=$SDPA_NKV d=$SDPA_HEAD_DIM q=$SDPA_QCHUNK k=$SDPA_KCHUNK causal=$SDPA_CAUSAL dtype=$SDPA_DTYPE fid=$SDPA_FIDELITY exp_approx=$SDPA_EXP_APPROX iters=$SDPA_ITERS kv_dtype=${SDPA_KV_DTYPE:-} grid=${SDPA_GRID:-full} fp32_acc=${SDPA_FP32_ACC:-0} packer_l1_acc=${SDPA_PACKER_L1_ACC:-0} math_approx=${SDPA_MATH_APPROX:-1}"
echo "### $TAG FRESH mode=$MODE start $T0 sha=$(git rev-parse --short HEAD) branch=$(git branch --show-current) env: $ENVS"
$TP analysis/zone_sweep.py::test_zone_sweep -s > $DD/$TAG.log 2>&1 || true
if ! grep -q "1 passed" $DD/$TAG.log || [ ! -s generated/profiler/.logs/profile_log_device.csv ]; then echo "RUN FAILED, see $DD/$TAG.log"; grep -E "Error|FAILED|Traceback" $DD/$TAG.log | head -5; exit 1; fi
FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $DD/$TAG.log | sed "s/firmware bundle version: //")
{ echo "PROVENANCE: p100a 110 cores 1350MHz fw_bundle=$FW; tt-metal-fresh $(git rev-parse HEAD) branch $(git branch --show-current) (main tip 2dbd14bf632 merged, local BRISC firmware-size patch); $T0; cmd: $TP analysis/zone_sweep.py::test_zone_sweep -s; env $ENVS mode=$MODE"; cat generated/profiler/.logs/profile_log_device.csv; } > $DD/$TAG.csv
tail -n +2 $DD/$TAG.csv > /tmp/zr_$$.csv; python $TTM/analysis/zone_reduce.py /tmp/zr_$$.csv --out $DD/$TAG | tee -a $DD/$TAG.log | grep -v "^wrote"; rm -f /tmp/zr_$$.csv
echo "### $TAG done $(date -u +%H:%M:%SZ)"
