#!/bin/bash
# Run one SDPA zone_sweep config under tracy device profiling, save CSV + log, reduce.
# Usage: run_zone.sh <tag> [counters] ; config comes from SDPA_* env vars (see analysis/zone_sweep.py).
#   counters = "mp" -> perf-counter multipass capture (5 replays), else plain device profiling.
set -e
TAG=$1; MODE=${2:-plain}
# Two layouts, no edit needed: this script in $WORK/handoff/revamp/data/bh_zones (DD is that directory), or
# copied into $TTM/analysis/campaigns of a checkout (TTM is that checkout, DD is $PWD, never the repo tree).
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SD/campaign_paths.sh"
cd $TTM
export TT_METAL_HOME=$TTM ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1
source $PYENV
export SDPA_NH=${SDPA_NH:-32} SDPA_NKV=${SDPA_NKV:-8} SDPA_HEAD_DIM=${SDPA_HEAD_DIM:-128} SDPA_DTYPE=${SDPA_DTYPE:-bfp8_b}
export SDPA_FIDELITY=${SDPA_FIDELITY:-HiFi2} SDPA_EXP_APPROX=${SDPA_EXP_APPROX:-1} SDPA_ITERS=${SDPA_ITERS:-3} SDPA_SEQ=${SDPA_SEQ:-4096}
export SDPA_QCHUNK=${SDPA_QCHUNK:-128}
export SDPA_KCHUNK=${SDPA_KCHUNK:-$SDPA_QCHUNK}
export SDPA_CAUSAL=${SDPA_CAUSAL:-1}
if [ "$MODE" = mp ]; then
  TP="python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest"
else
  TP="python -m tracy -m pytest"
fi
rm -f generated/profiler/.logs/profile_log_device.csv
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "### $TAG mode=$MODE start $T0 sha=$(git rev-parse --short HEAD) branch=$(git branch --show-current)"
echo "### env: S=$SDPA_SEQ nh=$SDPA_NH nkv=$SDPA_NKV d=$SDPA_HEAD_DIM q=$SDPA_QCHUNK k=$SDPA_KCHUNK causal=$SDPA_CAUSAL dtype=$SDPA_DTYPE fid=$SDPA_FIDELITY exp_approx=$SDPA_EXP_APPROX iters=$SDPA_ITERS kv_dtype=${SDPA_KV_DTYPE:-} grid=${SDPA_GRID:-full} fp32_acc=${SDPA_FP32_ACC:-0} packer_l1_acc=${SDPA_PACKER_L1_ACC:-0} math_approx=${SDPA_MATH_APPROX:-1}"
echo "### zone_config: $(grep -h '#define SDPA_' ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp 2>/dev/null | tr '\n' ' ')"
$TP analysis/zone_sweep.py::test_zone_sweep -s > $DD/$TAG.log 2>&1 || true
if ! grep -q "1 passed" $DD/$TAG.log || [ ! -s generated/profiler/.logs/profile_log_device.csv ]; then echo "RUN FAILED, see $DD/$TAG.log"; grep -E "Error|FAILED|Traceback" $DD/$TAG.log | head -5; exit 1; fi
{
  FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $DD/$TAG.log | sed "s/firmware bundle version: //"); echo "PROVENANCE: p100a 110 cores 1350MHz fw_bundle=$FW; tt-metal $(git rev-parse HEAD) branch $(git branch --show-current); $T0; cmd: $TP analysis/zone_sweep.py::test_zone_sweep -s; env S=$SDPA_SEQ nh=$SDPA_NH nkv=$SDPA_NKV d=$SDPA_HEAD_DIM q=$SDPA_QCHUNK k=$SDPA_KCHUNK causal=$SDPA_CAUSAL dtype=$SDPA_DTYPE fid=$SDPA_FIDELITY exp_approx=$SDPA_EXP_APPROX iters=$SDPA_ITERS kv_dtype=${SDPA_KV_DTYPE:-} grid=${SDPA_GRID:-full} fp32_acc=${SDPA_FP32_ACC:-0} packer_l1_acc=${SDPA_PACKER_L1_ACC:-0} math_approx=${SDPA_MATH_APPROX:-1} mode=$MODE; zone_config: $(grep -h '#define SDPA_' ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp 2>/dev/null | tr '\n' ' ')"
  cat generated/profiler/.logs/profile_log_device.csv
} > $DD/$TAG.csv
# reducer expects 2 header lines then data; strip the PROVENANCE line for it
tail -n +2 $DD/$TAG.csv > /tmp/zr_$$.csv
python analysis/zone_reduce.py /tmp/zr_$$.csv --out $DD/$TAG | tee -a $DD/$TAG.log | grep -v "^wrote"
rm -f /tmp/zr_$$.csv
echo "### $TAG done $(date -u +%H:%M:%SZ)"
