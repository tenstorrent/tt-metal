#!/bin/bash
cd /mnt/tt-data/ssinghal/wt/hy4_lattn/models/demos/blackhole/hy4_preview/tests
SCRIPT=run_attn_decode.py N=0 ./go.sh m_d4_L1_bf16 LAYER=1 USERS=4 KVFMT=bf16 POSITIONS=512,2560 TIMING=1 > /dev/null
SCRIPT=profile_decode.py N=0 ./go.sh m_prof_L1_bf16 LAYER=1 KVFMT=bf16 POS=2560 > /dev/null
SCRIPT=profile_decode.py N=0 ./go.sh m_prof_L1_bf16_v1 LAYER=1 KVFMT=bf16 POS=2560 V1=1 > /dev/null
SCRIPT=profile_decode.py N=0 ./go.sh m_prof_L1_fp8 LAYER=1 KVFMT=scaled_fp8 POS=2560 > /dev/null
SCRIPT=probe_row_stride.py N=0 ./go.sh m_stride > /dev/null
N=0 ./go.sh m_wbfp8_pf_L1 LAYER=1 MODE=prefill KVFMT=scaled_fp8 CHUNKS=6 DIAG=0 WDT=bfp8 TIMING=1 > /dev/null
SCRIPT=run_attn_decode.py N=0 ./go.sh m_wbfp8_d4_L1 LAYER=1 USERS=4 KVFMT=scaled_fp8 POSITIONS=512,2560 WDT=bfp8 TIMING=1 > /dev/null
echo done > /mnt/tt-data/ssinghal/hy4-logs/program/layer-attn/batch2.done
