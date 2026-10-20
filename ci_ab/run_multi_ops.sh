#!/usr/bin/env bash
# Scratch, not for merge: multi-arm op-level A/B of topk_large_indices (arms H V1 V4 V2 V3 H2), ABBA.
cd /work; export HWLOCK_HELD=ci
export OPS_NMEAS=16 OPS_CASES="ds_idx_r640_n6144_k2048_odd ds_idx_r640_n4096_k2048 ds_idx_r640_n8192_k2048 ds_idx_r640_n12288_k2048 ds_idx_r64_n4096_k2048 ds_idx_r1_n4096_k2048 r640_n4096_k1024 r640_n8192_k1024 r640_n2048_k512 r640_n4096_k512 prefill_r640_n51200_k1536"
bash /work/ci_ab/run_multi.sh
