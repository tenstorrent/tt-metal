#!/bin/bash
# R1 re-measurement campaign (all walls the pages use). Blocks R1d(start) R1a R1d(mid) R1b R1c R1e R1d(end).
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
Z(){ $SD/set_zone_config.sh $1 0 0 0 0 >/dev/null; }
anchor(){ Z 0; SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 $SD/run_zone.sh $1 || true; }
zpair(){ # tag env... : zone_sweep point zoff then zon
  T=$1; shift; Z 0; env "$@" $SD/run_zone.sh ${T}_zoff || true; Z 1; env "$@" $SD/run_zone.sh ${T}_zon || true; Z 0; }
rpair(){ # tag target env... : regime harness point zoff then zon
  T=$1; TGT=$2; shift 2; Z 0; env "$@" $SD/run_regime.sh ${T}_zoff plain "$TGT" || true; Z 1; env "$@" $SD/run_regime.sh ${T}_zon plain "$TGT" || true; Z 0; }
roff(){ T=$1; TGT=$2; MODE=$3; shift 3; Z 0; env "$@" $SD/run_regime.sh $T $MODE "$TGT" || true; }
echo "### R1d start $(date -u +%FT%TZ)"; anchor r1d_anchor_start_causal_S4096_q128k128_zoff
echo "### R1a start $(date -u +%FT%TZ)"
for S in 2048 8192; do zpair r1a_causal_S${S}_q128k128 SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=$S; done
for S in 2048 8192; do zpair r1a_noncausal_S${S}_q128k128 SDPA_CAUSAL=0 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=$S; done
RG="analysis/r1_regimes.py::test_r1_regime"
rpair r1a_cross_1024_8192_nh16 "$RG" R1_MODE=cross R1_SQ=1024 R1_SK=8192 R1_NH=16 R1_D=128 R1_QCHUNK=128 R1_ITERS=3
rpair r1a_cross_4096_16384_nh16 "$RG" R1_MODE=cross R1_SQ=4096 R1_SK=16384 R1_NH=16 R1_D=128 R1_QCHUNK=128 R1_ITERS=3
rpair r1a_window_S8192_W4096_nh16 "$RG" R1_MODE=window R1_S=8192 R1_W=4096 R1_NH=16 R1_D=128 R1_QCHUNK=128 R1_ITERS=3
rpair r1a_window_S8192_W2048_nh16 "$RG" R1_MODE=window R1_S=8192 R1_W=2048 R1_NH=16 R1_D=128 R1_QCHUNK=128 R1_ITERS=3
rpair r1a_window_S4096_W1024_nh16 "$RG" R1_MODE=window R1_S=4096 R1_W=1024 R1_NH=16 R1_D=128 R1_QCHUNK=128 R1_ITERS=3
rpair r1a_mask_S4096_d0.25_nh16 "$RG" R1_MODE=mask R1_S=4096 R1_DENSITY=0.25 R1_NH=16 R1_D=128 R1_QCHUNK=128 R1_ITERS=3
rpair r1a_mask_S8192_d0.5_nh16 "$RG" R1_MODE=mask R1_S=8192 R1_DENSITY=0.5 R1_NH=16 R1_D=128 R1_QCHUNK=128 R1_ITERS=3
CK="analysis/chunked_sweep.py::test_chunked_prefill"
rpair r1a_chunked_start2048_S8192_Sq2048 "$CK" CK_STARTS=2048 CK_S=8192 CK_SQ=2048 CK_ITERS=3
rpair r1a_chunked_start8192_S16384_Sq2048 "$CK" CK_STARTS=8192 CK_S=16384 CK_SQ=2048 CK_ITERS=3
rpair r1a_chunked_start4096_S8192_Sq1024 "$CK" CK_STARTS=4096 CK_S=8192 CK_SQ=1024 CK_ITERS=3
ML="analysis/mla_perf_sweep.py::test_mla_sweep"
rpair r1a_mla_nh16_S1024 "$ML" MLA_SEQ=1024 MLA_NH=16 MLA_NKV=1 MLA_KVLORA=512 MLA_DROPE=64 MLA_ITERS=3
rpair r1a_mla_nh16_S4096 "$ML" MLA_SEQ=4096 MLA_NH=16 MLA_NKV=1 MLA_KVLORA=512 MLA_DROPE=64 MLA_ITERS=3
rpair r1a_mla_nh32_S2048 "$ML" MLA_SEQ=2048 MLA_NH=32 MLA_NKV=1 MLA_KVLORA=512 MLA_DROPE=64 MLA_ITERS=3
SP="analysis/sparse_sweep.py::test_sparse"
roff r1a_sparse_H32_S2048_T8192_topk2048_zoff "$SP" plain SP_H=32 SP_S=2048 SP_T=8192 SP_TOPKS=2048 SP_KC=128 SP_ITERS=3
roff r1a_sparse_H32_S2048_T8192_topk2048_zoff_mp "$SP" mp SP_H=32 SP_S=2048 SP_T=8192 SP_TOPKS=2048 SP_KC=128 SP_ITERS=3
roff r1a_sparse_H32_S2048_T16384_topk2048_zoff "$SP" plain SP_H=32 SP_S=2048 SP_T=16384 SP_TOPKS=2048 SP_KC=128 SP_ITERS=3
roff r1a_sparse_H32_S2048_T32768_topk2048_zoff "$SP" plain SP_H=32 SP_S=2048 SP_T=32768 SP_TOPKS=2048 SP_KC=128 SP_ITERS=3
roff r1a_sparse_H32_S2048_T8192_topk1024_zoff "$SP" plain SP_H=32 SP_S=2048 SP_T=8192 SP_TOPKS=1024 SP_KC=128 SP_ITERS=3
JT="analysis/joint_sweep.py::test_joint_sweep"
roff r1a_joint_N4096_L333_nh24_d128_q128k512_zoff "$JT" plain JT_SEQS=4096 JT_JOINT=333 JT_NH=24 JT_D=128 JT_QC=128 JT_KC=512 JT_ITERS=1
roff r1a_joint_N4096_L333_nh24_d128_q128k512_zoff_mp "$JT" mp JT_SEQS=4096 JT_JOINT=333 JT_NH=24 JT_D=128 JT_QC=128 JT_KC=512 JT_ITERS=1
roff r1a_joint_N4096_L333_nh24_d64_q128k512_zoff "$JT" plain JT_SEQS=4096 JT_JOINT=333 JT_NH=24 JT_D=64 JT_QC=128 JT_KC=512 JT_ITERS=1
echo "### R1a done $(date -u +%FT%TZ)"
echo "### R1d mid $(date -u +%FT%TZ)"; anchor r1d_anchor_mid_causal_S4096_q128k128_zoff
echo "### R1b start $(date -u +%FT%TZ)"
DE="analysis/r1_decode.py::test_r1_decode"
roff r1b_decode_nonpaged_b32_kvbfp8_pos1024_4096_g110_zoff "$DE" plain DEC_B=32 DEC_KV_DTYPE=bfp8_b DEC_POS=1024,4096 DEC_NH=32 DEC_NKV=8 DEC_D=128 DEC_ITERS=3
roff r1b_decode_nonpaged_b32_kvbfp8_pos1024_g110_zoff_mp "$DE" mp DEC_B=32 DEC_KV_DTYPE=bfp8_b DEC_POS=1024 DEC_NH=32 DEC_NKV=8 DEC_D=128 DEC_ITERS=3
roff r1b_decode_nonpaged_b8_kvbf16_pos1024_4096_g110_zoff "$DE" plain DEC_B=8 DEC_KV_DTYPE=bfloat16 DEC_POS=1024,4096 DEC_NH=32 DEC_NKV=8 DEC_D=128 DEC_ITERS=3
Z 0; DEC_B=32 DEC_POS=8192 DEC_MAXSEQ=16384 DEC_GRID=11x10 $SD/run_decode.sh r1b_decode_paged_b32_g110_pos8192 || true
MD="analysis/r1_mla_decode.py::test_r1_mla_decode"
roff r1b_mla_decode_paged_b4_nh128_pos1024_4096_8192_zoff "$MD" plain MLAD_POS=1024,4096,8192 MLAD_CACHE=16384 MLAD_B=4 MLAD_NH=128 MLAD_KVLORA=512 MLAD_DROPE=64 MLAD_QCORES=64 MLAD_ITERS=3
echo "### R1b done $(date -u +%FT%TZ)"
echo "### R1c start $(date -u +%FT%TZ)"
Z 0
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_HEAD_DIM=64 $SD/run_zone.sh r1c_causal_S4096_q128k128_hd64_zoff || true
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_HEAD_DIM=64 $SD/run_zone.sh r1c_causal_S4096_q128k128_hd64_zoff_mp mp || true
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_DTYPE=bfloat16 $SD/run_zone.sh r1c_causal_S4096_q128k128_allbf16_zoff || true
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_DTYPE=bfloat16 $SD/run_zone.sh r1c_causal_S4096_q128k128_allbf16_zoff_mp mp || true
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_FIDELITY=HiFi4 $SD/run_zone.sh r1c_causal_S4096_q128k128_hifi4_fp32off_zoff || true
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_FIDELITY=LoFi $SD/run_zone.sh r1c_causal_S4096_q128k128_lofi_zoff || true
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_NH=16 SDPA_NKV=16 $SD/run_zone.sh r1c_causal_S4096_q128k128_mha_nh16_zoff || true
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_NH=16 SDPA_NKV=8 SDPA_BATCH=2 $SD/run_zone.sh r1c_causal_S4096_q128k128_b2_nh16_zoff || true
SDPA_CAUSAL=1 SDPA_QCHUNK=64 SDPA_KCHUNK=64 SDPA_SEQ=2048 $SD/run_zone.sh r1c_causal_S2048_q64k64_zoff || true
SDPA_CAUSAL=1 SDPA_QCHUNK=64 SDPA_KCHUNK=64 SDPA_SEQ=512 $SD/run_zone.sh r1c_causal_S512_q64k64_zoff || true
SDPA_CAUSAL=0 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=4096 SDPA_HEAD_DIM=64 $SD/run_zone.sh r1c_noncausal_S4096_q128k128_hd64_zoff || true
echo "### R1c done $(date -u +%FT%TZ)"
echo "### R1e start $(date -u +%FT%TZ)"
SP_H=32 SP_S=2048 SP_T=8192 SP_TOPKS=2048 SP_KC=128 SP_ITERS=3 $SD/run_regime_fresh.sh r1e_fresh_sparse_H32_S2048_T8192_topk2048_zoff plain "$SP" || true
JT_SEQS=4096 JT_JOINT=333 JT_NH=24 JT_D=128 JT_QC=128 JT_KC=512 JT_ITERS=1 $SD/run_regime_fresh.sh r1e_fresh_joint_N4096_L333_nh24_d128_q128k512_zoff plain "$JT" || true
echo "### R1e done $(date -u +%FT%TZ)"
echo "### R1d end $(date -u +%FT%TZ)"; anchor r1d_anchor_end_causal_S4096_q128k128_zoff
Z 0; SDPA_CAUSAL=1 SDPA_FIDELITY=HiFi4 SDPA_EXP_APPROX=0 SDPA_MATH_APPROX=0 SDPA_FP32_ACC=1 SDPA_PACKER_L1_ACC=1 SDPA_DTYPE=bfp8_b SDPA_KV_DTYPE=bfp8_b SDPA_GRID=8x8 SDPA_SEQ=4096 SDPA_QCHUNK=256 SDPA_KCHUNK=256 $SD/run_zone.sh r1d_prod_end_causal_S4096_q256k256_g64_zoff || true
echo "### R1 campaign done $(date -u +%FT%TZ)"
