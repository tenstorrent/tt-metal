#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible runner for bench_e2e_p150.py: pins every QWEN36_*/QWEN_GDN_*/QWEN_* env flag
# explicitly (so results never depend on a stale shell), checks the device is free, and runs the
# benchmark under a 30-minute watchdog. See ../README_P150_PERF.md.
#
# Usage: run_bench_e2e_p150.sh [isl|demo] [osl] [runs]
#   isl  (default 4096) - or the literal "demo": passes --demo-prompt to bench_e2e_p150.py
#                          instead of --isl, and names the output file
#                          demo_osl<osl>_<timestamp>.json.
#   osl  (default 8)
#   runs (default 5)
#
# QWEN36_ONDEV_ARGMAX is a QWEN* flag the caller may set (it survives the reset below):
#   1 (default) = greedy argmax on device (4-byte token read); 0 = logits read + host argmax.
# The I-1 item flags QWEN36_I1_D3 / _D15 / _D4A / _D6 / _P5 / _ROPEWARM also survive the reset
#   (1 = item on; 0 = the pre-I-1 code path). Unset = the code default (tt/tp_common.py
#   I1_FLAG_DEFAULTS): 1 except D4A=0 (D3=1 since 2026-09-25, accepted after an HF fp32 check;
#   D4A changes greedy tokens and stays off until an accuracy acceptance).
# The I-2 prefill-matmul item flags QWEN36_I2_S1 / _S2 / _S3 / _PICKER also survive the reset
#   (tt/tp_common.py I2_FLAG_DEFAULTS; unset = code default: S1=2, PICKER=1 (both bit-identical to the
#   pre-I-2 path, 2026-09-25), S2=0, S3=0; 0 = the pre-I-2 path): S1 = fused SwiGLU
#   minimal_matmul blocking (1 = 7/8/10 1x5, 2 = 7/8/16 1x8 bit-identical); S2 = MLP down-proj 2D-mcast
#   linear; S3 = GDN qkv in-proj 2D-mcast linear (S2/S3: 1 = in0_block_w 16, 8 = in0_block_w 8);
#   PICKER = prefill picker subblock area cap 8 (fp32 dest off). Values: S1 0|1|2, PICKER 0|1, S2/S3 0|1|8.
# The M1 prefill-matmul item flags QWEN36_M1_S2 / _S3 / _S4 also survive the reset (tt/tp_common.py
#   M1_FLAG_DEFAULTS: code default 0 = the pre-M1 path; values 0|1; T == 2048 chunks only):
#   S2 = MLP down-proj, S3 = GDN q|k|v in-proj (attention_norm output + in-proj output in L1),
#   S4 = GDN z|a|0|b|0 in-proj (attention_norm output + in-proj output in L1); each = 2D-mcast
#   ttnn.matmul 13x10 bw8 pcM7 (H V2_bw8). Runner defaults (differ from the code default 0): S4=1
#   (2026-09-26: bit-exact, passed all M1 gates); S2=1, S3=1 together with QWEN36_M3_ZB=1 (M3 gate
#   2026-09-26: with the zero bias S2/S3 are bit-exact; S2/S3 with ZB=0 change numerics, HF KL worse).
#   A/B e.g. QWEN36_M1_S4=0 bash run_bench_e2e_p150.sh.
# The M2 traced-prefill item flags QWEN36_M2_NOWHERE / _REPACK_LATE / _LASTROW also survive the reset
#   (tt/tp_common.py M2_FLAG_DEFAULTS; unset = code default 0 = the current path; values 0|1; single
#   device traced chunked prefill only): NOWHERE = no vision-splice where in the traced chunk for a
#   text-only model (bit-exact); REPACK_LATE = the exact-multiple path defers the GDN conv-history repack
#   from before the LM head to the next decode entry (after the first-token read; bit-exact); LASTROW =
#   the traced chunk computes the last layer's gate/o_proj/MLP for row chunk-1 only (numerics change).
#   Runner defaults (M3 run 2026-09-26; differ from the code default 0): NOWHERE=1, LASTROW=1
#   (HF KL not worse), REPACK_LATE=0 (M3 REPACK_TRACE replaces it). A/B e.g. QWEN36_M2_LASTROW=0 bash run_bench_e2e_p150.sh.
# The M3 item flags QWEN36_M3_ZB / _REPACK_TRACE also survive the reset (tt/tp_common.py
#   M3_FLAG_DEFAULTS; unset = code default 0 = the current path; values 0|1; single device):
#   ZB = with QWEN36_M1_S2 / _S3 on, the M1 2D-mcast matmul runs as ttnn.linear + a zero bias [1, N]
#   allocated once at model load (FUSE_BIAS; bit-identical to minimal_matmul); REPACK_TRACE = the
#   exact-multiple traced prefill replays a captured trace of the GDN conv-history repack instead of
#   the eager ops (same point in the flow; bit-exact; QWEN36_M2_REPACK_LATE=1 takes precedence).
#   Runner defaults: REPACK_TRACE=1 (2026-09-26, passed the M3 bit-exact gates; differs from the code
#   default 0). ZB: the runner default is 0 since 2026-09-30 (drops the zero bias on the 2D-mcast
#   MLP-down and GDN in-proj matmuls; -0.5 ms TTFT; numerics change accepted under the coherence
#   gate; QWEN36_M3_ZB=1 restores the bit-exact-with-minimal_matmul path; it was 1 from 2026-09-26).
#   A/B e.g. QWEN36_M3_REPACK_TRACE=0 bash run_bench_e2e_p150.sh.
# The C2 item flag QWEN36_C2_SGRN also survives the reset (tt/tp_common.py C2_FLAG_DEFAULTS; unset = code
#   default 0 = the current path; values 0|1; single device, unmasked T == 2048 GDN chunks with QWEN36_M1_S4=1):
#   SGRN = the 5 GDN post-scan glue ops (z slice, typecast, per-head rms_norm, nlp_concat_heads, SILU gate
#   multiply) become one ttnn.experimental.kda.sigmoid_gated_rms_norm (gate_activation="silu") that reads z
#   in place from the L1 z|a|0|b|0 in-proj output (kept alive through ChunkGdnFused); numerics change.
#   Runner default 1 since MG3 (2026-09-26; differs from the code default 0); A/B e.g. QWEN36_C2_SGRN=0 bash run_bench_e2e_p150.sh.
# The R3 item flags QWEN36_R3_FLA_IN_L1 / _SGRN_GAB_L1 / _O_L1 also survive the reset (tt/tp_common.py
#   R3_FLAG_DEFAULTS; unset = code default 0 = the current path; values 0|1; unmasked T == 2048 GDN chunks;
#   need the R3 Ct==1 ChunkGdnFused CB shrink in the C++ op; placement only, bit-exact):
#   FLA_IN_L1 = tiled KDA conv q/k/v + beta/g chain in L1; SGRN_GAB_L1 = with QWEN36_C2_SGRN=1 gab stays L1
#   (C2 variant a); O_L1 = FLA o + final state in L1. Runner defaults 1 (all three) since MG3 (2026-09-26;
#   differ from the code default 0); A/B e.g. QWEN36_R3_FLA_IN_L1=0 QWEN36_R3_O_L1=0 bash run_bench_e2e_p150.sh.
# QWEN36_LAYER_RESID_L1 and QWEN36_LAYER_L1_MAX_T also survive the reset (E1, 2026-09-26; read in tt/layer.py
#   and tt/model.py; single device prefill only; placement only, bit-exact by the E1 gate):
#   QWEN36_LAYER_RESID_L1  0|1 (default 1 in this runner since C3, 2026-09-26; code default 0): 1 = the residual
#                          stream (post-embedding x and both residual adds of every layer) in L1 interleaved when
#                          T <= 2048 (the decoder norms then read L1 and, with no explicit output config, also
#                          write L1); bit-exact, -4.8 ms TTFT at ISL 4096 (E1). 0 = residual stream in DRAM.
#   QWEN36_LAYER_L1_MAX_T  non-negative integer (default 0 = code default = off): prefill norm outputs
#                          (attention_norm, ffn_norm) are copied to L1 when T <= this value (extra copy op).
#   A/B e.g. QWEN36_LAYER_RESID_L1=0 bash run_bench_e2e_p150.sh.
# QWEN36_GDN_DECODE_FUSED and QWEN36_GDN_CONV_REPACK also survive the reset (T8 fused GDN decode):
#   QWEN36_GDN_DECODE_FUSED  2 (default; user decision 2026-09-25) = fused gdn_decode_step op with FP32
#                            GDN state; 0 = the composite GDN decode (pre-T8 path).
#   QWEN36_GDN_CONV_REPACK   gather (runner default since P6_INT1C item 3) | batched | perlayer (code
#                            default) (conv-history repack, fused decode only; bit-exact). gather = one
#                            shared index table + one ttnn.embedding per layer (P3_SMALL/item_D.patch).
# QWEN36_GDN_CONV_KDA_TILED also survives the reset (INT-3 tiled KDA conv):
#   1 (default) = the KDA conv reads the TILE in-proj output and TILE conv state directly and returns
#   new_state (no untilize/zeros/slice/tilize glue); 0 = the ROW_MAJOR KDA path. Bit-identical results.
# QWEN36_I4_SDPA_Q64 and QWEN36_I4_SDPA_EXP_COMPAT also survive the reset (INT-4 SDPA, needs upstream
#   PR #57395 + the T3d causal K/V chains for the flexible chunked paged path in the op):
#   QWEN36_I4_SDPA_Q64        1 (default) = flexible chunked prefill SDPA q_chunk 64 / k_chunk 128,
#                             exp_approx_mode=True; 0 = q/k_chunk 128 (the pre-INT-4 config).
#   QWEN36_I4_SDPA_EXP_COMPAT 1 (default) = the other gated-attention SDPA configs pass exp_approx_mode=True,
#                             the kernel the pre-PR False ran (#57395 inverted the flag); 0 = literal False.
# The I-3 decode items QWEN36_I3_FA_PROGCFG / _MLP_PROGCFG / _MLP_FUSED_GU / _LMHEAD also survive the reset
#   (tt/tp_common.py I3_FLAG_DEFAULTS; unset = code default: FA_PROGCFG=1, MLP_PROGCFG=1 (both bit-identical
#   to the pre-I-3 path, 2026-09-25), MLP_FUSED_GU=0, LMHEAD=0; 0 = the pre-I-3 path):
#   FA_PROGCFG   0|1  full-attn decode projections via swept 1D progcfgs (bit-exact).
#   MLP_PROGCFG  0|1  MLP decode gate/up 1D 13x4 bw4 pcN4 progcfg (bit-exact).
#   MLP_FUSED_GU 0|1  MLP decode gate|up as one DRAM-sharded matmul + slices + silu*mul (numerics change).
#   LMHEAD       0|A|A2|A3|B|C  LM head (every value changes numerics): A = 4 x DRAM-sharded linear (LoFi),
#                A2 = 8 x DRAM-sharded linear (HiFi2, fp32 dest), A3 (M4) = 10 x DRAM-sharded linear (HiFi2,
#                fp32 dest, bw 2, in0 8x8; chunks built at load, unsplit weight freed), B = unsplit 1D linear
#                13x10 (LoFi), C = 4 x minimal_matmul 13x2 K16 (1 bf16 ulp on rare logits); 0 = the
#                QWEN36_LMHEAD_SPLIT=8 path. Runner default LMHEAD=A3 since MG3 (2026-09-26; code default 0);
#                A/B e.g. QWEN36_I3_LMHEAD=0 bash run_bench_e2e_p150.sh.
# The M4 item flags QWEN36_M4_R4A / _R4B also survive the reset (tt/tp_common.py M4_FLAG_DEFAULTS; unset =
#   code default 0 = the current path; values 0|1; single device, traced chunked prefill, only with
#   QWEN36_M2_LASTROW=1): R4A = the last layer's 3 one-row reads use tile-aligned [T-32:T] block slices
#   (bit-exact); R4B = the last layer's prefill SDPA is the paged decode SDPA for the chunk's last row
#   (numerics change). Runner defaults R4A=1, R4B=1 since MG3 (2026-09-26; differ from the code default 0);
#   A/B e.g. QWEN36_M4_R4B=0 bash run_bench_e2e_p150.sh.
# The M5 item flags QWEN36_M5_TAIL_TRACE / _ADDNORM also survive the reset (tt/tp_common.py M5_FLAG_DEFAULTS;
#   unset = code default 0 = the current path; values 0|1; single device, traced chunked prefill; both bit-exact):
#   TAIL_TRACE = for an exact-multiple prompt the final norm + LM head (+ argmax) after the last chunk replay run
#   as one captured trace into a persistent DRAM output; ADDNORM = in T == 2048 traced chunks each full-T residual
#   add + the RMSNorm that reads it run as one rms_norm(residual_output_tensor=h). Runner defaults 1 (both)
#   since MG3 (2026-09-26; differ from the code default 0); A/B e.g. QWEN36_M5_TAIL_TRACE=0 QWEN36_M5_ADDNORM=0 bash run_bench_e2e_p150.sh.
# The F item flag QWEN36_F_MLP_GU_BF8 also survives the reset (tt/tp_common.py F_FLAG_DEFAULTS; unset = code default 0
#   = the current path; values 0|1; single device): 1 = the MLP gate/up weights (decode w1/w3 + prefill packed
#   w_gate_up) are bfloat8_b instead of bfloat4_b (accuracy item; numerics change; new weight-cache files).
#   Runner default 1 since 2026-09-28 (differ from the code default 0; gated: PCC 0.97-0.99 vs bfp4 on isl4096/demo/
#   isl1000, KL-vs-HF lower than bfp4 at every measured length, needle 48/50 vs 49/50 at bfp4); A/B e.g.
#   QWEN36_F_MLP_GU_BF8=0 bash run_bench_e2e_p150.sh.
# The W flag QWEN36_W_BF4 also survives the reset (tt/tp_common.py w_bf4_enabled / mm_weight_dtype; unset = runner
#   default 0 = today's dtypes; values 0|1; single device): 1 = every matmul weight (GDN / FA projections, MLP
#   gate/up/down, LM head) is bfloat4_b instead of bfloat8_b, and the MLP gate/up are bfloat4_b whatever
#   QWEN36_F_MLP_GU_BF8 says (numerics change; new weight-cache files); A/B e.g. QWEN36_W_BF4=1 bash run_bench_e2e_p150.sh.
# The LM flag QWEN36_LM_BF4FAST also survives the reset (tt/tp_common.py lm_bf4fast_enabled; unset = runner default 1;
#   the code default in tp_common.py stays 0 = today's A3 LM head; values 0|1; single device, QWEN36_I3_LMHEAD=A3): 1 AND a
#   bfloat4_b LM head weight (QWEN36_W_BF4=1) = the A3 LM head chunks use LoFi (fp32 dest / packer L1 acc unchanged) and 3
#   workers per DRAM bank (shard 99 tiles/bank); bit-exact vs HiFi2 / 2 workers (T17A/T17B, one chunk: 103.1 -> 73.7 us).
#   With a bfloat8_b LM head (default QWEN36_W_BF4=0) the flag changes nothing, so runner default 1 alters no default run.
#   A/B e.g. QWEN36_W_BF4=1 QWEN36_LM_BF4FAST=0 bash run_bench_e2e_p150.sh.
# The N item flag QWEN36_N_GAMMA_L1 also survives the reset (tt/tp_common.py N_FLAG_DEFAULTS; unset =
#   code default 0 = the current path; values 0|1; single device): 1 = layer.py's attention_norm / ffn_norm
#   gamma (RMSNorm weight) lives in L1 interleaved instead of DRAM (placement only, bit-exact; standalone
#   P2_NORM measurement: -1.3 us/call of 92 calls). Runner default 1 since 2026-09-28 (differs from the
#   code default 0); A/B e.g. QWEN36_N_GAMMA_L1=0 bash run_bench_e2e_p150.sh.
# The R5 item flag QWEN36_R5_GLU also survives the reset (tt/tp_common.py R5_FLAG_DEFAULTS; unset = code default 0
#   = the current path; values 0|1|2; single device, T == 2048 prefill chunks with bfloat8_b gate/up; 2 = in0_block_w 16 + glu_last_block + glu_sfpu_on_pack variant, runner default since P10_INT1H; needle 47/50 vs 48, within noise): 1 = the fused-SwiGLU
#   gate/up matmul runs as the 2D-mcast ttnn.matmul with the fused SwiGLU epilogue (needs the C++ fuse_swiglu config
#   field) instead of minimal_matmul(fuse_swiglu=True); numerics change. Runner default 1 since 2026-09-28 (differs
#   from the code default 0; gated: PCC 0.99905-0.99937 top-1 equal, needle 48/50, G0 coherence OK); A/B e.g.
#   QWEN36_R5_GLU=0 bash run_bench_e2e_p150.sh.
# The MM item flag QWEN36_MM_BW16 also survives the reset (tt/tp_common.py MM_FLAG_DEFAULTS; unset = code default 0
#   = the current path; values 0|1; single device, T == 2048 prefill chunks): 1 = in0_block_w 16 instead of 8 for
#   MLP down (M1 S2), GDN z|a|0|b|0 in-proj (M1 S4), the o-proj family (FA/GDN o_proj) and the FA q|k|v fused proj
#   (P3_MMSWEEP shapes); excludes GDN q|k|v in-proj (M1 S3, bw16 overflows a kernel-config limit there). Numerics
#   change (PCC ~0.99995, not bit-exact). Runner default 1 since 2026-09-28 (differs from the code default 0);
#   A/B e.g. QWEN36_MM_BW16=0 bash run_bench_e2e_p150.sh.
# The SGRN item flag QWEN36_SGRN_VARIANT also survives the reset (tt/tp_common.py sgrn_kernel_variant();
#   only takes effect with QWEN36_C2_SGRN=1): kernel_variant passed to sigmoid_gated_rms_norm. 0 = legacy
#   7-pass kernel (bit-exact with the pre-P6_INT1C op); 1-3 = fused kernel (bit-exact with each other);
#   4 = fused kernel with an exp_21f sigmoid (within 1 bf16 ulp of 0 for >99.9% of values; not bit-exact);
#   5 = fused gated RMSNorm compute kernel (P9_SGRN2; numerics differ slightly from 4).
#   Runner default 5 since P9_INT1G (4 since P6_INT1C; code default unset -> the op's own default, 4); A/B e.g.
#   QWEN36_SGRN_VARIANT=0 bash run_bench_e2e_p150.sh.
# QWEN36_GDN_PCFG also survives the reset (PR #57440 port: program_config of the fused FLA prefill op,
#   parsed in tt/gdn/gated_deltanet.py; it replaces the removed QWEN_GDN_NP/_NV/_PLACEMENT C++ knobs):
#   nv1np5 (runner default, C1 2026-09-26; kept by MG3) = NV=1 NP=5 row-local (with QWEN36_GDN_WYINV=horner:
#   bit-exact vs the pre-#57440 nv1np6 path); nv1np6 = NV=1 NP=6 row-major, the pre-#57440 geometry;
#   auto = the op's cost model (code default when unset; NV=1 NP=5 row-local on P150 13x10);
#   nv2np4 = NV=2 NP=4 row-local; nv2np6 = NV=2 NP=6 row-major.
# QWEN36_GDN_WYINV also survives the reset (PR #57445 WY-inverse selector, parsed in tt/gdn/decode.py):
#   sfpu (runner default since MG3 2026-09-26) = pinned ttnn.ChunkGdnWyInverse.SFPU (numerics change vs horner);
#   horner (runner default C1 2026-09-26 .. MG3) = ttnn.ChunkGdnWyInverse.HORNER, the pre-#57445 numerics;
#   auto (code default when unset) = wy_inverse not passed (op default AUTO = SFPU on Blackhole at chunk 32). A/B e.g. QWEN36_GDN_PCFG=nv2np4 QWEN36_GDN_WYINV=sfpu bash run_bench_e2e_p150.sh.
# QWEN36_GDN_GATE_FUSE also survives the reset (P7_INT1D / P5_GATING, models/experimental/gated_attention_gated_deltanet:
#   fuses the GDN beta sigmoid+scale and the a+dt_bias+softplus into single BinaryNg ops, same math, bit-exact;
#   distinct from the pre-existing QWEN36_GDN_GATE_FUSED, which fuses the output-gate multiply): 1 (runner default
#   since P7_INT1D) = fused chain; 0 (code default) = separate ops.
# QWEN36_ROPE_L1 also survives the reset (P7_INT1D / P7_ROPE, tt/model.py: the persistent per-chunk RoPE cos/sin
#   buffers go in L1 interleaved instead of DRAM interleaved; bit-exact): 1 (runner default since P7_INT1D) = L1;
#   0 (code default) = DRAM.
# QWEN36_GDN_STATE_INPLACE also survives the reset (P9_INT1F / P7_STATECOPY, tt/gdn/decode.py: on the traced chunked
#   prefill the fused FLA op and the KDA conv op write the recurrent / conv state straight into the persistent buffers
#   instead of a new tensor plus a copy; bit-exact): 1 (runner default since P9_INT1F) = in place; 0 (code default) = copy.
# QWEN36_FLA_SCAN_FID also survives the reset (R10B hook, chunk_gdn_fused_program_factory.cpp: math-fidelity
#   override for the fused FLA scan/receiver compute kernel only): HiFi3 (runner default since P7_INT1D item B2;
#   +/-2% vs f64, inside the coherence gate, -1.7 ms/4k vs HiFi4) | HiFi4 | HiFi2 | LoFi; unset (code default) =
#   HiFi4 exactly (today's fixed behaviour). QWEN36_FLA_PREP_FID is the same hook for the prep/producer kernel;
#   the runner leaves it unset (no pin) -- the P4_FLARCV producer path is already bit-exact at HiFi4.
# QWEN36_FLA_SCAN_FID_BY_LEN also survives the reset (P11_FLALEN, tt/gdn/gated_deltanet.py + tt/model.py: the FLA
#   fidelity follows each request's prompt length -- scan HiFi2 up to 65536 tokens, scan HiFi3 above, prep HiFi4 in
#   both -- through the hashed ChunkGdnFusedProgramConfig.scan_math_fidelity / prep_math_fidelity fields;
#   prepare compiles both when max_prompt_len > 65536 and the chunk trace is re-captured when a request needs the
#   other one): 1 (runner default since P11_FLALEN) = by length; 0 (code default) = the fixed QWEN36_FLA_SCAN_FID
#   above. With 1 the runner does NOT export QWEN36_FLA_SCAN_FID (the env var overrides the field); an explicit
#   QWEN36_FLA_SCAN_FID=<fid> with 1 still pins every request's scan to <fid> (experiments; the runner prints a
#   NOTE). QWEN36_FLA_PREP_FID is never exported by the runner (an explicit one overrides the prep field the same way).
# Non-QWEN overrides (not touched by the reset): BENCH_GDN_FLAT_GB (default 1) -> QWEN_GDN_FLAT_GB;
#   BENCH_TRACE_GUARD (default 1) -> QWEN36_TRACE_GUARD + TT_METAL_TRACE_ALLOC_TRACKING=1.
#
# This branch has a single GDN-prefill configuration: the fused FLA prim with an explicit program
# config (QWEN_GDN_PATH=fused + QWEN36_GDN_PCFG), which the runner always exports. QWEN_GDN_PATH no
# longer reaches the C++ op (#57440 removed the env knobs); the model still reads it to default
# QWEN36_GDN_FLA_INPUTS_DRAM=1. It only works in a tree that has the #57440 program configs built
# in (see the FUSED_FLA_GREP check below); it refuses to run otherwise.
set -euo pipefail

ISL="${1:-4096}"
OSL="${2:-8}"
RUNS="${3:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/python_env/bin/activate" ]; then
  echo "ERROR: $REPO_ROOT/python_env/bin/activate not found -- run ./create_venv.sh first (see README_P150_PERF.md)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/python_env/bin/activate"

echo "== repo: $REPO_ROOT =="
echo "== branch: $(git rev-parse --abbrev-ref HEAD) commit: $(git rev-parse HEAD) dirty_files: $(git status --porcelain | wc -l) =="

# ---------------------------------------------------------------------------------------------
# 1. Unset EVERY QWEN* var already in the shell, so nothing is inherited from a previous session.
# (QWEN36_ONDEV_ARGMAX, the QWEN36_I1_* / QWEN36_I2_* / QWEN36_M1_* / QWEN36_M2_* / QWEN36_M3_* / QWEN36_C2_* / QWEN36_R3_* / QWEN36_M4_* / QWEN36_M5_* / QWEN36_I3_* / QWEN36_F_* / QWEN36_N_* / QWEN36_R5_* / QWEN36_MM_* / QWEN36_SGRN_* item flags, QWEN36_W_BF4, QWEN36_LM_BF4FAST, QWEN36_GDN_DECODE_FUSED,
#  QWEN36_GDN_CONV_REPACK, QWEN36_GDN_CONV_KDA_TILED, QWEN36_GDN_PCFG, QWEN36_GDN_WYINV, QWEN36_GDN_GATE_FUSE,
#  QWEN36_GDN_GATES_OP,
#  QWEN36_ROPE_L1, QWEN36_ACT_BF8_RESID, QWEN36_ACT_BF8_NORM, QWEN36_RESID_HS, QWEN36_GDN_STATE_INPLACE, QWEN36_FLA_SCAN_FID, QWEN36_FLA_SCAN_FID_BY_LEN, the QWEN36_I4_*
#  flags, QWEN36_LAYER_RESID_L1 and QWEN36_LAYER_L1_MAX_T are read first and re-exported in section 3.)
# ---------------------------------------------------------------------------------------------
ONDEV_ARGMAX="${QWEN36_ONDEV_ARGMAX:-1}"
case "$ONDEV_ARGMAX" in
  0|1) ;;
  *) echo "ERROR: QWEN36_ONDEV_ARGMAX must be 0 or 1 (got '$ONDEV_ARGMAX')" >&2; exit 1 ;;
esac
I1_ITEMS="D3 D15 D4A D6 P5 ROPEWARM"
declare -A I1_DEFAULTS=([D3]=1 [D15]=1 [D4A]=0 [D6]=1 [P5]=1 [ROPEWARM]=1)  # = tp_common.I1_FLAG_DEFAULTS
declare -A I1_VALS
for item in $I1_ITEMS; do
  var="QWEN36_I1_$item"
  val="${!var:-${I1_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  I1_VALS[$item]="$val"
done
I2_ITEMS="S1 S2 S3 PICKER"
declare -A I2_DEFAULTS=([S1]=2 [S2]=0 [S3]=0 [PICKER]=1)  # = tp_common.I2_FLAG_DEFAULTS
declare -A I2_VALS
for item in $I2_ITEMS; do
  var="QWEN36_I2_$item"
  val="${!var:-${I2_DEFAULTS[$item]}}"
  case "$item:$val" in
    S1:0|S1:1|S1:2|PICKER:0|PICKER:1|S2:0|S2:1|S2:8|S3:0|S3:1|S3:8) ;;
    *) echo "ERROR: $var must be 0 or 1 (S1 also 2; S2/S3 also 8) (got '$val')" >&2; exit 1 ;;
  esac
  I2_VALS[$item]="$val"
done
M1_ITEMS="S2 S3 S4"
declare -A M1_DEFAULTS=([S2]=1 [S3]=1 [S4]=1)  # runner defaults (tp_common.M1_FLAG_DEFAULTS are all 0)
declare -A M1_VALS
for item in $M1_ITEMS; do
  var="QWEN36_M1_$item"
  val="${!var:-${M1_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  M1_VALS[$item]="$val"
done
M2_ITEMS="NOWHERE REPACK_LATE LASTROW"
declare -A M2_DEFAULTS=([NOWHERE]=1 [REPACK_LATE]=0 [LASTROW]=1)  # runner defaults (tp_common.M2_FLAG_DEFAULTS are all 0)
declare -A M2_VALS
for item in $M2_ITEMS; do
  var="QWEN36_M2_$item"
  val="${!var:-${M2_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  M2_VALS[$item]="$val"
done
M3_ITEMS="ZB REPACK_TRACE"
declare -A M3_DEFAULTS=([ZB]=0 [REPACK_TRACE]=1)  # runner defaults (tp_common.M3_FLAG_DEFAULTS are all 0)
declare -A M3_VALS
for item in $M3_ITEMS; do
  var="QWEN36_M3_$item"
  val="${!var:-${M3_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  M3_VALS[$item]="$val"
done
C2_ITEMS="SGRN"
declare -A C2_DEFAULTS=([SGRN]=1)  # runner default 1 since MG3 2026-09-26 (tp_common.C2_FLAG_DEFAULTS: code default 0)
declare -A C2_VALS
for item in $C2_ITEMS; do
  var="QWEN36_C2_$item"
  val="${!var:-${C2_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  C2_VALS[$item]="$val"
done
R3_ITEMS="FLA_IN_L1 SGRN_GAB_L1 O_L1"
declare -A R3_DEFAULTS=([FLA_IN_L1]=1 [SGRN_GAB_L1]=1 [O_L1]=1)  # runner defaults 1 since MG3 2026-09-26 (tp_common.R3_FLAG_DEFAULTS are all 0)
declare -A R3_VALS
for item in $R3_ITEMS; do
  var="QWEN36_R3_$item"
  val="${!var:-${R3_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  R3_VALS[$item]="$val"
done
M4_ITEMS="R4A R4B"
declare -A M4_DEFAULTS=([R4A]=1 [R4B]=1)  # runner defaults 1 since MG3 2026-09-26 (tp_common.M4_FLAG_DEFAULTS are all 0)
declare -A M4_VALS
for item in $M4_ITEMS; do
  var="QWEN36_M4_$item"
  val="${!var:-${M4_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  M4_VALS[$item]="$val"
done
M5_ITEMS="TAIL_TRACE ADDNORM"
declare -A M5_DEFAULTS=([TAIL_TRACE]=1 [ADDNORM]=1)  # runner defaults 1 since MG3 2026-09-26 (tp_common.M5_FLAG_DEFAULTS are all 0)
declare -A M5_VALS
for item in $M5_ITEMS; do
  var="QWEN36_M5_$item"
  val="${!var:-${M5_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  M5_VALS[$item]="$val"
done
F_ITEMS="MLP_GU_BF8"
declare -A F_DEFAULTS=([MLP_GU_BF8]=1)  # runner default 1 since 2026-09-28 (tp_common.F_FLAG_DEFAULTS code default 0)
declare -A F_VALS
for item in $F_ITEMS; do
  var="QWEN36_F_$item"
  val="${!var:-${F_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  F_VALS[$item]="$val"
done
N_ITEMS="GAMMA_L1"
declare -A N_DEFAULTS=([GAMMA_L1]=1)  # runner default 1 since 2026-09-28 (tp_common.N_FLAG_DEFAULTS code default 0)
declare -A N_VALS
for item in $N_ITEMS; do
  var="QWEN36_N_$item"
  val="${!var:-${N_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  N_VALS[$item]="$val"
done
R5_ITEMS="GLU"
declare -A R5_DEFAULTS=([GLU]=2)  # runner default 2 since P10_INT1H (accepted: needle 47/50 vs 48, within noise; 1 = previous config) (tp_common.R5_FLAG_DEFAULTS code default 0)
declare -A R5_VALS
for item in $R5_ITEMS; do
  var="QWEN36_R5_$item"
  val="${!var:-${R5_DEFAULTS[$item]}}"
  case "$val" in
    0|1|2) ;;
    *) echo "ERROR: $var must be 0, 1 or 2 (got '$val')" >&2; exit 1 ;;
  esac
  R5_VALS[$item]="$val"
done
MM_ITEMS="BW16"
declare -A MM_DEFAULTS=([BW16]=1)  # runner default 1 since 2026-09-28 (tp_common.MM_FLAG_DEFAULTS code default 0)
declare -A MM_VALS
for item in $MM_ITEMS; do
  var="QWEN36_MM_$item"
  val="${!var:-${MM_DEFAULTS[$item]}}"
  case "$val" in
    0|1) ;;
    *) echo "ERROR: $var must be 0 or 1 (got '$val')" >&2; exit 1 ;;
  esac
  MM_VALS[$item]="$val"
done
SGRN_ITEMS="VARIANT"
declare -A SGRN_DEFAULTS=([VARIANT]=5)  # runner default 5 since P9_INT1G (fused gated RMSNorm variant 5); 4 = P6_INT1C fused kernel with exp_21f sigmoid (op default)
declare -A SGRN_VALS
for item in $SGRN_ITEMS; do
  var="QWEN36_SGRN_$item"
  val="${!var:-${SGRN_DEFAULTS[$item]}}"
  case "$val" in
    0|1|2|3|4|5) ;;
    *) echo "ERROR: $var must be 0, 1, 2, 3, 4 or 5 (got '$val')" >&2; exit 1 ;;
  esac
  SGRN_VALS[$item]="$val"
done
LAYER_RESID_L1="${QWEN36_LAYER_RESID_L1:-1}"  # C3 runner default 1 (code default 0: tt/layer.py, tt/model.py)
case "$LAYER_RESID_L1" in
  0|1) ;;
  *) echo "ERROR: QWEN36_LAYER_RESID_L1 must be 0 or 1 (got '$LAYER_RESID_L1')" >&2; exit 1 ;;
esac
LAYER_L1_MAX_T="${QWEN36_LAYER_L1_MAX_T:-0}"  # = code default (tt/layer.py)
case "$LAYER_L1_MAX_T" in
  ''|*[!0-9]*) echo "ERROR: QWEN36_LAYER_L1_MAX_T must be a non-negative integer (got '$LAYER_L1_MAX_T')" >&2; exit 1 ;;
  *) ;;
esac
GDN_DECODE_FUSED="${QWEN36_GDN_DECODE_FUSED:-2}"
case "$GDN_DECODE_FUSED" in
  0|2) ;;
  *) echo "ERROR: QWEN36_GDN_DECODE_FUSED must be 0 or 2 (got '$GDN_DECODE_FUSED')" >&2; exit 1 ;;
esac
GDN_CONV_REPACK="${QWEN36_GDN_CONV_REPACK:-gather}"  # runner default gather since P6_INT1C (item 3; code default: perlayer)
case "$GDN_CONV_REPACK" in
  perlayer|batched|gather) ;;
  *) echo "ERROR: QWEN36_GDN_CONV_REPACK must be perlayer, batched or gather (got '$GDN_CONV_REPACK')" >&2; exit 1 ;;
esac
GDN_CONV_KDA_TILED="${QWEN36_GDN_CONV_KDA_TILED:-1}"
case "$GDN_CONV_KDA_TILED" in
  0|1) ;;
  *) echo "ERROR: QWEN36_GDN_CONV_KDA_TILED must be 0 or 1 (got '$GDN_CONV_KDA_TILED')" >&2; exit 1 ;;
esac
GDN_PCFG="${QWEN36_GDN_PCFG:-nv1np5}"  # runner default (C1): NV1 NP5 row-local (code default: auto)
case "$GDN_PCFG" in
  auto|nv1np6|nv1np5|nv2np4|nv2np6) ;;
  *) echo "ERROR: QWEN36_GDN_PCFG must be auto, nv1np6, nv1np5, nv2np4 or nv2np6 (got '$GDN_PCFG')" >&2; exit 1 ;;
esac
GDN_WYINV="${QWEN36_GDN_WYINV:-sfpu}"  # runner default sfpu since MG3 2026-09-26 (C1: horner; code default: auto = not passed)
case "$GDN_WYINV" in
  auto|horner|sfpu) ;;
  *) echo "ERROR: QWEN36_GDN_WYINV must be auto, horner or sfpu (got '$GDN_WYINV')" >&2; exit 1 ;;
esac
GDN_GATE_FUSE="${QWEN36_GDN_GATE_FUSE:-1}"  # runner default 1 since P7_INT1D (P5_GATING; code default: 0)
case "$GDN_GATE_FUSE" in
  0|1) ;;
  *) echo "ERROR: QWEN36_GDN_GATE_FUSE must be 0 or 1 (got '$GDN_GATE_FUSE')" >&2; exit 1 ;;
esac
REPACK_AFTER_TTFT="${QWEN36_REPACK_AFTER_TTFT:-1}"  # P23_REPACK/INT2j: runner default 1 (code default 0); 1 = replay the M3 repack trace at the first decode step, not in TTFT (code default 0)
case "$REPACK_AFTER_TTFT" in
  0|1) ;;
  *) echo "ERROR: QWEN36_REPACK_AFTER_TTFT must be 0 or 1 (got '$REPACK_AFTER_TTFT')" >&2; exit 1 ;;
esac
GDN_GATES_OP="${QWEN36_GDN_GATES_OP:-1}"  # P10_GDNGATE: runner default 1 (code default: 0); 1 = one fused gdn_gates op for beta/g
case "$GDN_GATES_OP" in
  0|1) ;;
  *) echo "ERROR: QWEN36_GDN_GATES_OP must be 0 or 1 (got '$GDN_GATES_OP')" >&2; exit 1 ;;
esac
ROPE_L1="${QWEN36_ROPE_L1:-1}"  # runner default 1 since P7_INT1D (P7_ROPE; code default: 0)
case "$ROPE_L1" in
  0|1) ;;
  *) echo "ERROR: QWEN36_ROPE_L1 must be 0 or 1 (got '$ROPE_L1')" >&2; exit 1 ;;
esac
FA_GATE_FAST="${QWEN36_FA_GATE_FAST:-1}"  # P14_FAGATE2: FA prefill gate: SIGMOID fused into the gate matmul (B2) (runner default 1; code default: 0)
case "$FA_GATE_FAST" in
  0|1) ;;
  *) echo "ERROR: QWEN36_FA_GATE_FAST must be 0 or 1 (got '$FA_GATE_FAST')" >&2; exit 1 ;;
esac
SDPA_CONCAT_OUT="${QWEN36_SDPA_CONCAT_OUT:-1}"  # P15: chunked SDPA writes [B,1,T,H*D] directly, no concatenate_heads (runner default 1; code default: 0)
case "$SDPA_CONCAT_OUT" in
  0|1) ;;
  *) echo "ERROR: QWEN36_SDPA_CONCAT_OUT must be 0 or 1 (got '$SDPA_CONCAT_OUT')" >&2; exit 1 ;;
esac
ROPE_PARTIAL_INPLACE="${QWEN36_ROPE_PARTIAL_INPLACE:-1}"  # P10_ROPE: in-place partial RoPE op (runner default 1; code default: 0)
case "$ROPE_PARTIAL_INPLACE" in
  0|1) ;;
  *) echo "ERROR: QWEN36_ROPE_PARTIAL_INPLACE must be 0 or 1 (got '$ROPE_PARTIAL_INPLACE')" >&2; exit 1 ;;
esac
# P6_BF8ACT opt-in activation-dtype flags (default 0; read here, re-exported after the QWEN reset below).
#   QWEN36_ACT_BF8_RESID=1: bfloat8_b output for the prefill o-proj / MLP down matmuls (G3, F3, M2).
#   QWEN36_ACT_BF8_NORM=1:  bfloat8_b fused add+RMSNorm / layer-0 norm output n at T == 2048 (residual h stays bf16).
ACT_BF8_RESID="${QWEN36_ACT_BF8_RESID:-1}"
case "$ACT_BF8_RESID" in
  0|1) ;;
  *) echo "ERROR: QWEN36_ACT_BF8_RESID must be 0 or 1 (got '$ACT_BF8_RESID')" >&2; exit 1 ;;
esac
ACT_BF8_NORM="${QWEN36_ACT_BF8_NORM:-1}"
case "$ACT_BF8_NORM" in
  0|1) ;;
  *) echo "ERROR: QWEN36_ACT_BF8_NORM must be 0 or 1 (got '$ACT_BF8_NORM')" >&2; exit 1 ;;
esac
# P11_SHARDRES_B: QWEN36_RESID_HS=1 keeps the T == 2048 prefill residual stream h and the o-proj / down-proj outputs
# HEIGHT_SHARDED in L1 on the 64 fused add+RMSNorm cores (bit-exact; needs the P11_SHARDRES_A C++). Runner default 0 (code default: 0).
RESID_HS="${QWEN36_RESID_HS:-1}"
case "$RESID_HS" in
  0|1) ;;
  *) echo "ERROR: QWEN36_RESID_HS must be 0 or 1 (got '$RESID_HS')" >&2; exit 1 ;;
esac
STATE_INPLACE="${QWEN36_GDN_STATE_INPLACE:-1}"  # runner default 1 since P9_INT1F (P7_STATECOPY; code default: 0)
case "$STATE_INPLACE" in
  0|1) ;;
  *) echo "ERROR: QWEN36_GDN_STATE_INPLACE must be 0 or 1 (got '$STATE_INPLACE')" >&2; exit 1 ;;
esac
FLA_SCAN_FID_BY_LEN="${QWEN36_FLA_SCAN_FID_BY_LEN:-1}"  # runner default 1 since P11_FLALEN (code default: 0)
case "$FLA_SCAN_FID_BY_LEN" in
  0|1) ;;
  *) echo "ERROR: QWEN36_FLA_SCAN_FID_BY_LEN must be 0 or 1 (got '$FLA_SCAN_FID_BY_LEN')" >&2; exit 1 ;;
esac
# QWEN36_FLA_SCAN_FID: with BY_LEN=0 the runner default HiFi3 (since P7_INT1D item B2; code default: unset -> HiFi4).
# With BY_LEN=1 it stays unset unless the caller set it (then it overrides the by-length field for every request).
if [ "$FLA_SCAN_FID_BY_LEN" = "1" ]; then
  FLA_SCAN_FID="${QWEN36_FLA_SCAN_FID:-}"
  if [ -n "$FLA_SCAN_FID" ]; then
    echo "NOTE: QWEN36_FLA_SCAN_FID=$FLA_SCAN_FID is set explicitly: it overrides QWEN36_FLA_SCAN_FID_BY_LEN=1 (every request runs the FLA scan at $FLA_SCAN_FID)" >&2
  fi
else
  FLA_SCAN_FID="${QWEN36_FLA_SCAN_FID:-HiFi3}"
fi
case "$FLA_SCAN_FID" in
  ""|HiFi4|HiFi3|HiFi2|LoFi) ;;
  *) echo "ERROR: QWEN36_FLA_SCAN_FID must be HiFi4, HiFi3, HiFi2 or LoFi (got '$FLA_SCAN_FID')" >&2; exit 1 ;;
esac
I4_SDPA_Q64="${QWEN36_I4_SDPA_Q64:-1}"  # = ttnn_gated_attention.I4_SDPA_Q64_DEFAULT
case "$I4_SDPA_Q64" in
  0|1) ;;
  *) echo "ERROR: QWEN36_I4_SDPA_Q64 must be 0 or 1 (got '$I4_SDPA_Q64')" >&2; exit 1 ;;
esac
I4_SDPA_EXP_COMPAT="${QWEN36_I4_SDPA_EXP_COMPAT:-1}"  # = ttnn_gated_attention.I4_SDPA_EXP_COMPAT_DEFAULT
case "$I4_SDPA_EXP_COMPAT" in
  0|1) ;;
  *) echo "ERROR: QWEN36_I4_SDPA_EXP_COMPAT must be 0 or 1 (got '$I4_SDPA_EXP_COMPAT')" >&2; exit 1 ;;
esac
I3_ITEMS="FA_PROGCFG MLP_PROGCFG MLP_FUSED_GU LMHEAD"
declare -A I3_DEFAULTS=([FA_PROGCFG]=1 [MLP_PROGCFG]=1 [MLP_FUSED_GU]=0 [LMHEAD]=A3)  # = tp_common.I3_FLAG_DEFAULTS except LMHEAD (runner default A3 since MG3 2026-09-26; code default 0)
declare -A I3_VALS
for item in $I3_ITEMS; do
  var="QWEN36_I3_$item"
  val="${!var:-${I3_DEFAULTS[$item]}}"
  case "$item:$val" in
    FA_PROGCFG:0|FA_PROGCFG:1|MLP_PROGCFG:0|MLP_PROGCFG:1|MLP_FUSED_GU:0|MLP_FUSED_GU:1|LMHEAD:0|LMHEAD:A|LMHEAD:A2|LMHEAD:A3|LMHEAD:B|LMHEAD:C) ;;
    *) echo "ERROR: $var must be 0 or 1 (LMHEAD: 0, A, A2, A3, B or C) (got '$val')" >&2; exit 1 ;;
  esac
  I3_VALS[$item]="$val"
done
# P18_PRELUDE: QWEN36_PRELUDE_TRACE (runner default 1; code default 0) survives the reset below.
PRELUDE_TRACE="${QWEN36_PRELUDE_TRACE:-1}"
case "$PRELUDE_TRACE" in
  0|1) ;;
  *) echo "ERROR: QWEN36_PRELUDE_TRACE must be 0 or 1 (got '$PRELUDE_TRACE')" >&2; exit 1 ;;
esac
# QWEN36_W_BF4 (runner default 0 = code default; tt/tp_common.py w_bf4_enabled) survives the reset below.
W_BF4="${QWEN36_W_BF4:-0}"
case "$W_BF4" in
  0|1) ;;
  *) echo "ERROR: QWEN36_W_BF4 must be 0 or 1 (got '$W_BF4')" >&2; exit 1 ;;
esac
# QWEN36_LM_BF4FAST (runner default 1; code default in tt/tp_common.py lm_bf4fast_enabled stays 0; acts only with a
# bfloat4_b LM head weight, QWEN36_W_BF4=1) survives the reset below.
LM_BF4FAST="${QWEN36_LM_BF4FAST:-1}"
case "$LM_BF4FAST" in
  0|1) ;;
  *) echo "ERROR: QWEN36_LM_BF4FAST must be 0 or 1 (got '$LM_BF4FAST')" >&2; exit 1 ;;
esac
while IFS='=' read -r name _; do
  [ -n "$name" ] && unset "$name"
done < <(env | grep -E '^QWEN' || true)
export QWEN36_PRELUDE_TRACE="$PRELUDE_TRACE"
export QWEN36_W_BF4="$W_BF4"
export QWEN36_LM_BF4FAST="$LM_BF4FAST"

# ---------------------------------------------------------------------------------------------
# 2. Core env.
# ---------------------------------------------------------------------------------------------
export HF_MODEL="Qwen/Qwen3.5-2B"
export MESH_DEVICE="P150"
export HF_HUB_OFFLINE=1
export TT_METAL_HOME="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# ---------------------------------------------------------------------------------------------
# 3. Pin every QWEN36_*/QWEN35_*/QWEN9B_*/QWEN_* flag at its documented default (see the
# ALL_QWEN_FLAG_DEFAULTS table in bench_e2e_p150.py -- keep these two lists in sync; regenerate
# the flag NAMES with:
#   grep -h "os.environ.get(\"QWEN" -r models/demos/blackhole/qwen36 \
#       models/experimental/gated_attention_gated_deltanet | sed 's/.*environ.get(//' | cut -d, -f1 \
#       | sort -u
#
# A handful of flags are deliberately NOT exported here (left truly unset): they are read via a
# bare `if os.environ.get(X):` truthiness check, where exporting ANY non-empty string (even "0")
# would wrongly enable them, or their fallback is computed from other locals at call time (not a
# fixed literal), so a made-up literal here would silently override the correct computed default.
# Those are listed and explained at the bottom of this section.
# ---------------------------------------------------------------------------------------------
export QWEN35_GDN_DECODE_BF16=0
export QWEN35_GDN_STATE_BF16=0
export QWEN35_NO_REPEAT_NGRAM=0
export QWEN35_REP_PENALTY=1.0
export QWEN35_TEMP=0
export QWEN35_TOP_K=0
export QWEN35_TOP_P=1.0
export QWEN35_TP_DECODE_EAGER=0
export QWEN35_TP_PREFILL_EAGER=0

export QWEN36_ATTN_FUSED_QKV=1
export QWEN36_ATTN_GATE_FUSED=1
export QWEN36_ATTN_KV_BF8=0
export QWEN36_ATTN_L1_MAX_T=2048
export QWEN36_ATTN_QKNORM_HIFI2=1
export QWEN36_BATCHED_DECODE_MODE=shard
export QWEN36_BUCKET_TEST_CTX=8192
export QWEN36_BUCKET_TEST_WIDTHS=1,8
export QWEN36_CAPACITY_TEST_BMAX=8
export QWEN36_CAPACITY_TEST_EAGER_PROFILE=0
export QWEN36_CAPACITY_TEST_ITERS=20
export QWEN36_CAPACITY_TEST_TRIALS=5
export QWEN36_CAPACITY_TEST_WARMUP=3
export QWEN36_DEBUG_DECODE_TIMING=0
export QWEN36_DECODE_PROGCFG=1
export QWEN36_GDN_CONV_KDA=1
export QWEN36_GDN_CONV_KDA_FP32ACC=0
export QWEN36_GDN_CONV_KDA_TILED="$GDN_CONV_KDA_TILED"
export QWEN36_GDN_CONV_LEGACY=0
export QWEN36_GDN_CONV_SILU_SHARDED=0
export QWEN36_GDN_CONV_T3_MAX=0
export QWEN36_GDN_CONV_REPACK="$GDN_CONV_REPACK"
export QWEN36_GDN_CONV_TILED_SPLIT=0
export QWEN36_GDN_DECODE_FUSED="$GDN_DECODE_FUSED"
export QWEN36_GDN_FUSED_PREFILL=1
export QWEN36_GDN_GATE_CLIP=0
export QWEN36_GDN_GATE_FUSE="$GDN_GATE_FUSE"
export QWEN36_GDN_GATES_OP="$GDN_GATES_OP"
export QWEN36_REPACK_AFTER_TTFT="$REPACK_AFTER_TTFT"
export QWEN36_ACT_BF8_RESID="$ACT_BF8_RESID"
export QWEN36_ACT_BF8_NORM="$ACT_BF8_NORM"
export QWEN36_RESID_HS="$RESID_HS"
export QWEN36_GDN_GATE_FUSED=1
export QWEN36_GDN_GB_LAYOUT=0
export QWEN36_GDN_L1_MAX_T=0
export QWEN36_GDN_NATIVE_CONV1D=1
export QWEN36_GDN_PCFG="$GDN_PCFG"
export QWEN36_GDN_WYINV="$GDN_WYINV"
export QWEN36_GDN_POST_L1=1
export QWEN36_GDN_POST_L1_OUTPROJ=1
export QWEN36_GDN_POST_L1_SCAN=1
export QWEN36_GDN_SPLIT_PROJ=1
I1_SUMMARY=""
for item in $I1_ITEMS; do
  export "QWEN36_I1_$item=${I1_VALS[$item]}"
  I1_SUMMARY="$I1_SUMMARY QWEN36_I1_$item=${I1_VALS[$item]}"
done
I2_SUMMARY=""
for item in $I2_ITEMS; do
  export "QWEN36_I2_$item=${I2_VALS[$item]}"
  I2_SUMMARY="$I2_SUMMARY QWEN36_I2_$item=${I2_VALS[$item]}"
done
M1_SUMMARY=""
for item in $M1_ITEMS; do
  export "QWEN36_M1_$item=${M1_VALS[$item]}"
  M1_SUMMARY="$M1_SUMMARY QWEN36_M1_$item=${M1_VALS[$item]}"
done
M2_SUMMARY=""
for item in $M2_ITEMS; do
  export "QWEN36_M2_$item=${M2_VALS[$item]}"
  M2_SUMMARY="$M2_SUMMARY QWEN36_M2_$item=${M2_VALS[$item]}"
done
M3_SUMMARY=""
for item in $M3_ITEMS; do
  export "QWEN36_M3_$item=${M3_VALS[$item]}"
  M3_SUMMARY="$M3_SUMMARY QWEN36_M3_$item=${M3_VALS[$item]}"
done
C2_SUMMARY=""
for item in $C2_ITEMS; do
  export "QWEN36_C2_$item=${C2_VALS[$item]}"
  C2_SUMMARY="$C2_SUMMARY QWEN36_C2_$item=${C2_VALS[$item]}"
done
R3_SUMMARY=""
for item in $R3_ITEMS; do
  export "QWEN36_R3_$item=${R3_VALS[$item]}"
  R3_SUMMARY="$R3_SUMMARY QWEN36_R3_$item=${R3_VALS[$item]}"
done
M4_SUMMARY=""
for item in $M4_ITEMS; do
  export "QWEN36_M4_$item=${M4_VALS[$item]}"
  M4_SUMMARY="$M4_SUMMARY QWEN36_M4_$item=${M4_VALS[$item]}"
done
M5_SUMMARY=""
for item in $M5_ITEMS; do
  export "QWEN36_M5_$item=${M5_VALS[$item]}"
  M5_SUMMARY="$M5_SUMMARY QWEN36_M5_$item=${M5_VALS[$item]}"
done
F_SUMMARY=""
for item in $F_ITEMS; do
  export "QWEN36_F_$item=${F_VALS[$item]}"
  F_SUMMARY="$F_SUMMARY QWEN36_F_$item=${F_VALS[$item]}"
done
N_SUMMARY=""
for item in $N_ITEMS; do
  export "QWEN36_N_$item=${N_VALS[$item]}"
  N_SUMMARY="$N_SUMMARY QWEN36_N_$item=${N_VALS[$item]}"
done
R5_SUMMARY=""
for item in $R5_ITEMS; do
  export "QWEN36_R5_$item=${R5_VALS[$item]}"
  R5_SUMMARY="$R5_SUMMARY QWEN36_R5_$item=${R5_VALS[$item]}"
done
MM_SUMMARY=""
for item in $MM_ITEMS; do
  export "QWEN36_MM_$item=${MM_VALS[$item]}"
  MM_SUMMARY="$MM_SUMMARY QWEN36_MM_$item=${MM_VALS[$item]}"
done
SGRN_SUMMARY=""
for item in $SGRN_ITEMS; do
  export "QWEN36_SGRN_$item=${SGRN_VALS[$item]}"
  SGRN_SUMMARY="$SGRN_SUMMARY QWEN36_SGRN_$item=${SGRN_VALS[$item]}"
done
I3_SUMMARY=""
for item in $I3_ITEMS; do
  export "QWEN36_I3_$item=${I3_VALS[$item]}"
  I3_SUMMARY="$I3_SUMMARY QWEN36_I3_$item=${I3_VALS[$item]}"
done
export QWEN36_FLA_SCAN_FID_BY_LEN="$FLA_SCAN_FID_BY_LEN"
if [ -n "$FLA_SCAN_FID" ]; then
  export QWEN36_FLA_SCAN_FID="$FLA_SCAN_FID"
fi  # else left unset (BY_LEN=1 default): the env var would override the by-length field
export QWEN36_I4_SDPA_EXP_COMPAT="$I4_SDPA_EXP_COMPAT"
export QWEN36_I4_SDPA_Q64="$I4_SDPA_Q64"
export QWEN36_LAYER_L1_MAX_T="$LAYER_L1_MAX_T"
export QWEN36_LAYER_RESID_L1="$LAYER_RESID_L1"
export QWEN36_LMHEAD_MINIMAL=1
export QWEN36_LMHEAD_SPLIT=8
export QWEN36_MLP_FUSED_SWIGLU=1
export QWEN36_MLP_L1_OUT=1
export QWEN36_MLP_LEGACY_SHORT=0
export QWEN36_MLP_MINIMAL_MM=1
export QWEN36_ONDEV_ARGMAX="$ONDEV_ARGMAX"
export QWEN36_PREFILL_DEBUG=0
export QWEN36_PREFILL_MINIMAL_CFG=1
export QWEN36_PREFILL_MM_FP32_ACC=0
export QWEN36_PREFILL_MM_PACKER_L1_ACC=1
export QWEN36_PREFILL_OVERLAP=1
export QWEN36_PREFILL_PROGCFG=1
export QWEN36_PREFILL_PROGCFG_OVERRIDES=1
export QWEN36_PREFIX_WRITE_ITERS=100
export QWEN36_PREFIX_WRITE_WIDTH=1
export QWEN36_ROPE_DEVICE_TABLE=1
export QWEN36_ROPE_L1="$ROPE_L1"
export QWEN36_FA_GATE_FAST="$FA_GATE_FAST"
export QWEN36_SDPA_CONCAT_OUT="$SDPA_CONCAT_OUT"
export QWEN36_ROPE_PARTIAL_INPLACE="$ROPE_PARTIAL_INPLACE"
export QWEN36_GDN_STATE_INPLACE="$STATE_INPLACE"
export QWEN36_ROPE_LEGACY=0

export QWEN9B_MLP_DOWN_AUTO=0
export QWEN9B_MLP_UP_AUTO=0
export QWEN9B_SDPA_QK64=0

export QWEN_BATCHED_GROUPED=1
export QWEN_GDN_DIAG_ALPHA=0.25
# g/beta flat read in the FLA op (T7 gb_flat). Default ON in this runner (INT-1, 2026-09-25: bit-exact
# vs off, -3.7 ms TTFT); the code default (unset) stays off. Pinned like every other flag; the only
# way to turn it off is the explicit non-QWEN override BENCH_GDN_FLAT_GB=0 (section 1 unsets QWEN*).
export QWEN_GDN_FLAT_GB="${BENCH_GDN_FLAT_GB:-1}"
case "$QWEN_GDN_FLAT_GB" in
  0|1) ;;
  *) echo "ERROR: BENCH_GDN_FLAT_GB must be 0 or 1 (got '$QWEN_GDN_FLAT_GB')" >&2; exit 1 ;;
esac
echo "== QWEN_GDN_FLAT_GB=$QWEN_GDN_FLAT_GB (from BENCH_GDN_FLAT_GB) =="
# Trace guard (T7): default ON. Enables Metal's trace-allocation tracker, so ttnn.execute_trace fails
# on the host BEFORE replay if any buffer allocated while a trace was parked is still alive (it would
# be corrupted by the replay) -> the bench exits non-zero. bench_e2e_p150.py self-checks at start
# that the tracker is really on (QWEN36_TRACE_GUARD=1 => TT_METAL_TRACE_ALLOC_TRACKING=1 before the
# ttnn import, and Metal reports it enabled) and records the tracker env in the JSON header. (No log
# grep: with the tracker on, Metal never prints "Allocating device buffers is potentially unsafe".)
# Turn off only via the explicit non-QWEN override BENCH_TRACE_GUARD=0.
export QWEN36_TRACE_GUARD="${BENCH_TRACE_GUARD:-1}"
if [ "$QWEN36_TRACE_GUARD" = "1" ]; then
  export TT_METAL_TRACE_ALLOC_TRACKING=1
fi
echo "== QWEN36_TRACE_GUARD=$QWEN36_TRACE_GUARD TT_METAL_TRACE_ALLOC_TRACKING=${TT_METAL_TRACE_ALLOC_TRACKING:-} =="
export QWEN_GDN_FP32_STATE=0
export QWEN_GDN_INV_DOUBLING=0
export QWEN_SDPA_BF8=0

# Deliberately LEFT UNSET (see the comment above): bare-truthy or dynamic-default flags.
#   QWEN35_NO_THINK                 - bare truthy: any value seeds an empty <think> block.
#   QWEN35_REF_PROMPT               - bare truthy: any value switches to the 64k reference prompt.
#   QWEN36_CAPACITY_TEST_LAYER_INDEX / QWEN36_CAPACITY_TEST_N_LAYERS - bare truthy TEST-harness only.
#   QWEN36_GDN_CONV_CHUNKS          - bare truthy override; unset = auto chunk count.
#   QWEN36_GDN_CONV_KDA_CCS         - fallback is str(channel_chunk_size), a runtime local.
#   QWEN36_GDN_CONV_XIN_L1_MAX_T    - fallback is str(xin_l1_max_t), a runtime local.
#   QWEN36_GDN_FLA_INPUTS_DRAM      - fallback depends on QWEN_GDN_PATH ("1" if fused else "0");
#                                     leaving it unset lets the code compute the right value for
#                                     the fused FLA configuration this invocation is running.
#   QWEN36_MAX_TOKENS_ALL_USERS     - bare truthy; vLLM-serving only, unused by this bench.
#   QWEN36_SDPA_PREFILL_CHUNKS      - bare truthy; experimental gated-attention path only.
#   QWEN9B_GDN_DBG                  - bare truthy debug-print switch.
#   QWEN_GDN_PATH                   - set explicitly below (always "fused"; see section 4 below).
export QWEN_GDN_PATH=""

# PR #57440 port: the fused prim is selected by ttnn.ChunkGdnFusedProgramConfig (QWEN36_GDN_PCFG), not
# by env. Grep for the config type, not QWEN_GDN_PATH (a stale comment could still mention the old knob).
FUSED_FLA_GREP() {
  grep -q "ChunkGdnFusedProgramConfig" \
    "$REPO_ROOT/ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp" \
    2>/dev/null
}

# ---------------------------------------------------------------------------------------------
# 4. This branch has a single configuration: the fused FLA prim (QWEN_GDN_PATH=fused + QWEN36_GDN_PCFG).
# Refuse to run if the tree does not actually have the #57440 fused program config wired in.
# ---------------------------------------------------------------------------------------------
if ! FUSED_FLA_GREP; then
  echo "ERROR: the fused FLA configuration (QWEN36_GDN_PCFG) requires this tree's" >&2
  echo "       chunk_gated_delta_rule.cpp to mention ChunkGdnFusedProgramConfig -- the PR #57440" >&2
  echo "       fused FLA prim is not wired into this build. Build/checkout the tree that has" >&2
  echo "       the #57440 port and re-run there. Refusing to run." >&2
  exit 1
fi
export QWEN_GDN_PATH="fused"
echo "== QWEN_GDN_PATH=fused QWEN36_GDN_PCFG=$QWEN36_GDN_PCFG (fused FLA configuration) =="

# ---------------------------------------------------------------------------------------------
# 5. Device must be free: exactly one process on the device at a time.
# ---------------------------------------------------------------------------------------------
if [ -e /dev/tenstorrent/0 ]; then
  HOLDERS="$(lsof /dev/tenstorrent/0 2>/dev/null || true)"
  if [ -n "$HOLDERS" ]; then
    echo "ERROR: /dev/tenstorrent/0 is already held by another process:" >&2
    echo "$HOLDERS" >&2
    echo "Refusing to start -- only one process may hold the device at a time." >&2
    exit 1
  fi
else
  echo "WARNING: /dev/tenstorrent/0 not found -- skipping the busy-device check." >&2
fi

# ---------------------------------------------------------------------------------------------
# 6. tt-smi summary (best-effort; bench_e2e_p150.py also captures this into the JSON header).
# ---------------------------------------------------------------------------------------------
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -s 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    dev0 = (data.get('device_info') or [{}])[0]
    board = (dev0.get('board_info') or {}).get('board_type', 'N/A')
    fw = (dev0.get('firmwares') or {}).get('arc_fw', 'N/A')
    aiclk = (dev0.get('telemetry') or {}).get('aiclk', 'N/A')
    print(f'tt-smi: board_type={board} arc_fw={fw} aiclk={aiclk}')
except Exception as e:
    print(f'tt-smi: could not parse -s output ({e})')
" || echo "tt-smi: -s query failed"
else
  echo "tt-smi: not found on PATH -- skipping"
fi

# ---------------------------------------------------------------------------------------------
# 7. Run.
# ---------------------------------------------------------------------------------------------
RESULTS_DIR="$REPO_ROOT/models/demos/blackhole/qwen36/demo/bench_results"
mkdir -p "$RESULTS_DIR"
if [ ! -f "$RESULTS_DIR/.gitignore" ]; then
  {
    echo "*"
    echo "!.gitignore"
    echo "!README.md"
  } > "$RESULTS_DIR/.gitignore"
fi

if [ "$ISL" = "demo" ]; then
  OUT="$RESULTS_DIR/demo_osl${OSL}_$(date +%Y%m%d_%H%M%S).json"
  PROMPT_ARGS=(--demo-prompt)
else
  OUT="$RESULTS_DIR/isl${ISL}_osl${OSL}_$(date +%Y%m%d_%H%M%S).json"
  PROMPT_ARGS=(--isl "$ISL")
fi
echo "== running: isl=$ISL osl=$OSL runs=$RUNS chunk=2048 QWEN36_ONDEV_ARGMAX=$QWEN36_ONDEV_ARGMAX$I1_SUMMARY$I2_SUMMARY$M1_SUMMARY$M2_SUMMARY$M3_SUMMARY$C2_SUMMARY$R3_SUMMARY$M4_SUMMARY$M5_SUMMARY$I3_SUMMARY$F_SUMMARY QWEN36_W_BF4=$QWEN36_W_BF4 QWEN36_LM_BF4FAST=$QWEN36_LM_BF4FAST$N_SUMMARY$R5_SUMMARY$MM_SUMMARY$SGRN_SUMMARY QWEN36_LAYER_RESID_L1=$QWEN36_LAYER_RESID_L1 QWEN36_LAYER_L1_MAX_T=$QWEN36_LAYER_L1_MAX_T QWEN36_GDN_DECODE_FUSED=$QWEN36_GDN_DECODE_FUSED QWEN36_GDN_CONV_REPACK=$QWEN36_GDN_CONV_REPACK QWEN36_GDN_CONV_KDA_TILED=$QWEN36_GDN_CONV_KDA_TILED QWEN36_GDN_PCFG=$QWEN36_GDN_PCFG QWEN36_GDN_WYINV=$QWEN36_GDN_WYINV QWEN36_REPACK_AFTER_TTFT=$QWEN36_REPACK_AFTER_TTFT QWEN36_GDN_GATE_FUSE=$QWEN36_GDN_GATE_FUSE QWEN36_GDN_GATES_OP=$QWEN36_GDN_GATES_OP QWEN36_ROPE_L1=$QWEN36_ROPE_L1 QWEN36_FA_GATE_FAST=$QWEN36_FA_GATE_FAST QWEN36_SDPA_CONCAT_OUT=$QWEN36_SDPA_CONCAT_OUT QWEN36_ROPE_PARTIAL_INPLACE=$QWEN36_ROPE_PARTIAL_INPLACE QWEN36_ACT_BF8_RESID=$QWEN36_ACT_BF8_RESID QWEN36_ACT_BF8_NORM=$QWEN36_ACT_BF8_NORM QWEN36_RESID_HS=$QWEN36_RESID_HS QWEN36_GDN_STATE_INPLACE=$QWEN36_GDN_STATE_INPLACE QWEN36_FLA_SCAN_FID=${QWEN36_FLA_SCAN_FID:-<unset>} QWEN36_FLA_SCAN_FID_BY_LEN=$QWEN36_FLA_SCAN_FID_BY_LEN QWEN36_I4_SDPA_Q64=$QWEN36_I4_SDPA_Q64 QWEN36_I4_SDPA_EXP_COMPAT=$QWEN36_I4_SDPA_EXP_COMPAT QWEN36_PRELUDE_TRACE=$QWEN36_PRELUDE_TRACE -> $OUT =="

RUN_LOG="${OUT%.json}.log"
set +e
timeout 1800 python3 "$SCRIPT_DIR/bench_e2e_p150.py" \
  "${PROMPT_ARGS[@]}" --osl "$OSL" --runs "$RUNS" --out "$OUT" 2>&1 | tee "$RUN_LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "== exit status: $STATUS (log: $RUN_LOG) =="
exit "$STATUS"
