#!/bin/bash
# Capture the per-core pf_publish (W_gate arrival) profile for one K split.
#   rowprobe_run.sh <tag> <kr_split_csv>
set -u
tag=$1; split=$2
cd /localdev/mstaletovic/tt-metal
rm -f generated/profiler/.logs/profile_log_device.csv
TT_METAL_DEVICE_PROFILER=1 \
MOE_FUSED_SWIGLU_KR_SPLIT="$split" \
MOE_FUSED_SWIGLU_GU_CHUNKS=${GC:-2} \
MOE_FUSED_SWIGLU_DEFINES="MOE_PF_PROFILE=1" \
BENCH_M=256 BENCH_ITERS=1 BENCH_EXPERTS=8 BENCH_DISTINCT_W=1 BENCH_WSHARD=1 \
BENCH_TAG=rp_$tag \
timeout 1800 scripts/run_safe_pytest.sh --run-all routed_expert_work/test_bench.py >routed_expert_work/pf_zones/rp_$tag.log 2>&1
cp generated/profiler/.logs/profile_log_device.csv routed_expert_work/pf_zones/rowprobe_$tag.csv
echo "captured routed_expert_work/pf_zones/rowprobe_$tag.csv  (split=$split)"
