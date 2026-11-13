#!/bin/bash
# batch of S1 matrix runs (sequential); logs program/layer-attn/run_<tag>.log
cd /mnt/tt-data/ssinghal/wt/hy4_lattn/models/demos/blackhole/hy4_preview/tests
for L in 1 0 2; do for F in bf16 scaled_fp8; do
  N=0 ./go.sh m_pf_L${L}_$F LAYER=$L MODE=prefill KVFMT=$F CHUNKS=6 DIAG=0 TIMING=1 > /dev/null
  SCRIPT=run_attn_decode.py N=0 ./go.sh m_d1_L${L}_$F LAYER=$L USERS=1 KVFMT=$F POSITIONS=512,2048,2560 TIMING=1 > /dev/null
  SCRIPT=run_attn_decode.py N=0 ./go.sh m_d4_L${L}_$F LAYER=$L USERS=4 KVFMT=$F POSITIONS=512,2560 TIMING=1 > /dev/null
done; done
for L in 1 0 2; do SCRIPT=run_attn_decode.py N=0 ./go.sh m_d1v1_L${L} LAYER=$L USERS=1 KVFMT=bf16 POSITIONS=512,2048,2560 TIMING=1 V1=1 > /dev/null; done
echo BATCH1_DONE > /mnt/tt-data/ssinghal/hy4-logs/program/layer-attn/batch1.done
