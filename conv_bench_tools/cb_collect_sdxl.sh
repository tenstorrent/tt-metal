#!/bin/bash
cd /localdev/wransom/tt-metal
# SDXL UNet REAL config: bf8 weights, bf16 out/in, HiFi2, fp32_accum=False, l1_acc OFF, BS, bias, TILE out, auto abh
# NOTE: SDXL uses a custom 5x8 BS core_grid + act_block_w_div slicing (not expressible here) -> these are
# n150-native auto-grid baselines for the real shapes; big-channel convs may OOM (model slices to fit).
SD="CB_WEIGHTS_DTYPE=bfloat8_b CB_OUT_DTYPE=bfloat16 CB_IN_DTYPE=bfloat16 CB_FP32_ACCUM=false CB_FIDELITY=HiFi2 CB_L1_ACC=false CB_OUT_LAYOUT=tile CB_BIAS=true CB_SHARD=BS CB_ABH=none CB_FILTER=3 CB_PAD=1,1,1,1 CB_BATCH=1"
run () { echo ">>> $1"; env MODEL=sdxl_unet LABEL="$1" $SD $2 bash conv_bench_tools/cb_bench.sh 2 main helper_sbm helper_trm; }
run "sdxl_384<-1152_128x128_s1"  "CB_IN_CH=1152 CB_OUT_CH=384  CB_H=128 CB_W=128 CB_STRIDE=1"
run "sdxl_768<-1152_64x64_s1"    "CB_IN_CH=1152 CB_OUT_CH=768  CB_H=64  CB_W=64  CB_STRIDE=1"
run "sdxl_1536<-1536_16x16_s1"   "CB_IN_CH=1536 CB_OUT_CH=1536 CB_H=16  CB_W=16  CB_STRIDE=1"
run "sdxl_1536<-1536_32x32_s1"   "CB_IN_CH=1536 CB_OUT_CH=1536 CB_H=32  CB_W=32  CB_STRIDE=1"
run "sdxl_1536<-1536_64x64_s1"   "CB_IN_CH=1536 CB_OUT_CH=1536 CB_H=64  CB_W=64  CB_STRIDE=1"
run "sdxl_768<-1536_64x64_s1"    "CB_IN_CH=1536 CB_OUT_CH=768  CB_H=64  CB_W=64  CB_STRIDE=1"
run "sdxl_1536<-2304_32x32_s1"   "CB_IN_CH=2304 CB_OUT_CH=1536 CB_H=32  CB_W=32  CB_STRIDE=1"
run "sdxl_768<-2304_64x64_s1"    "CB_IN_CH=2304 CB_OUT_CH=768  CB_H=64  CB_W=64  CB_STRIDE=1"
run "sdxl_1536<-3072_16x16_s1"   "CB_IN_CH=3072 CB_OUT_CH=1536 CB_H=16  CB_W=16  CB_STRIDE=1"
run "sdxl_1536<-3072_32x32_s1"   "CB_IN_CH=3072 CB_OUT_CH=1536 CB_H=32  CB_W=32  CB_STRIDE=1"
run "sdxl_384<-384_128x128_s1"   "CB_IN_CH=384  CB_OUT_CH=384  CB_H=128 CB_W=128 CB_STRIDE=1"
run "sdxl_768<-384_64x64_s1"     "CB_IN_CH=384  CB_OUT_CH=768  CB_H=64  CB_W=64  CB_STRIDE=1"
run "sdxl_384<-768_128x128_s1"   "CB_IN_CH=768  CB_OUT_CH=384  CB_H=128 CB_W=128 CB_STRIDE=1"
run "sdxl_768<-768_128x128_s1"   "CB_IN_CH=768  CB_OUT_CH=768  CB_H=128 CB_W=128 CB_STRIDE=1"
run "sdxl_1536<-768_32x32_s1"    "CB_IN_CH=768  CB_OUT_CH=1536 CB_H=32  CB_W=32  CB_STRIDE=1"
run "sdxl_768<-768_64x64_s1"     "CB_IN_CH=768  CB_OUT_CH=768  CB_H=64  CB_W=64  CB_STRIDE=1"
run "sdxl_1536<-1536_32x32_s2"   "CB_IN_CH=1536 CB_OUT_CH=1536 CB_H=32  CB_W=32  CB_STRIDE=2"
run "sdxl_384<-384_128x128_s2"   "CB_IN_CH=384  CB_OUT_CH=384  CB_H=128 CB_W=128 CB_STRIDE=2"
run "sdxl_768<-768_64x64_s2"     "CB_IN_CH=768  CB_OUT_CH=768  CB_H=64  CB_W=64  CB_STRIDE=2"
echo "=== SDXL UNET COLLECTION COMPLETE ==="
