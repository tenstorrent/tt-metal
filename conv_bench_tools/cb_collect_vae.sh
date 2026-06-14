#!/bin/bash
WT=/localdev/wransom/tt-metal/.claude/worktrees/agent-a48fa14207415d0cb
cd "$WT"
# SDXL VAE REAL config: bf8 weights, bf16 out, LoFi, fp32_accum=False, packer_l1_acc OFF, bias, TILE out, auto abh.
# (VAE uses DRAM activation slicing on WH to fit; harness can't express it -> these test whether BH's larger L1
#  fits them WITHOUT slicing. Labels are out<-in.)
VAE="CB_WEIGHTS_DTYPE=bfloat8_b CB_OUT_DTYPE=bfloat16 CB_IN_DTYPE=bfloat16 CB_FP32_ACCUM=false CB_FIDELITY=LoFi CB_L1_ACC=false CB_OUT_LAYOUT=tile CB_BIAS=true CB_ABH=none CB_BATCH=1 CB_FILTER=3 CB_PAD=1,1,1,1"
run () { echo ">>> $1"; env MODEL=sdxl_vae LABEL="$1" $VAE $2 bash "$WT/conv_bench_tools/cb_bench.sh" 2 main helper_sbm helper_trm; }
run "vae_128<-128_1024x1024_HS" "CB_IN_CH=128 CB_OUT_CH=128 CB_H=1024 CB_W=1024 CB_SHARD=HS"
run "vae_128<-256_1024x1024_HS" "CB_IN_CH=256 CB_OUT_CH=128 CB_H=1024 CB_W=1024 CB_SHARD=HS"
run "vae_256<-256_1024x1024_BS" "CB_IN_CH=256 CB_OUT_CH=256 CB_H=1024 CB_W=1024 CB_SHARD=BS"
run "vae_256<-128_512x512_HS"   "CB_IN_CH=128 CB_OUT_CH=256 CB_H=512  CB_W=512  CB_SHARD=HS"
run "vae_128<-128_512x512_HS"   "CB_IN_CH=128 CB_OUT_CH=128 CB_H=512  CB_W=512  CB_SHARD=HS"
run "vae_256<-256_512x512_BS"   "CB_IN_CH=256 CB_OUT_CH=256 CB_H=512  CB_W=512  CB_SHARD=BS"
run "vae_512<-256_512x512_BS"   "CB_IN_CH=256 CB_OUT_CH=512 CB_H=512  CB_W=512  CB_SHARD=BS"
run "vae_256<-512_512x512_BS"   "CB_IN_CH=512 CB_OUT_CH=256 CB_H=512  CB_W=512  CB_SHARD=BS"
run "vae_512<-512_512x512_BS"   "CB_IN_CH=512 CB_OUT_CH=512 CB_H=512  CB_W=512  CB_SHARD=BS"
run "vae_256<-256_256x256_BS"   "CB_IN_CH=256 CB_OUT_CH=256 CB_H=256  CB_W=256  CB_SHARD=BS"
run "vae_512<-256_256x256_BS"   "CB_IN_CH=256 CB_OUT_CH=512 CB_H=256  CB_W=256  CB_SHARD=BS"
run "vae_512<-512_256x256_BS"   "CB_IN_CH=512 CB_OUT_CH=512 CB_H=256  CB_W=256  CB_SHARD=BS"
run "vae_512<-512_128x128_BS"   "CB_IN_CH=512 CB_OUT_CH=512 CB_H=128  CB_W=128  CB_SHARD=BS"
run "vae_512<-512_64x64_BS"     "CB_IN_CH=512 CB_OUT_CH=512 CB_H=64   CB_W=64   CB_SHARD=BS"
echo "=== SDXL VAE COLLECTION COMPLETE ==="
