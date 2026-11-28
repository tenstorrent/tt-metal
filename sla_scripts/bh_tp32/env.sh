# Best known config for Llama-3.3-70B TP32 (models/demos/llama3_70b_galaxy) on one BH Galaxy, Nov 18 2026.
# Source it before run_server.sh / run_sla.sh / prefill_profile.sh:  source sla_scripts/bh_tp32/env.sh
# Prefetcher + chunked prefill (existing switches)
export LLAMA_BH_PREFETCHER=1 LLAMA_BH_GLOBAL_CB_TILES=512 LLAMA_BH_BATCHED_PREFILL=1 GALAXY_PREFILL_NUM_LINKS=2 LLAMA_GALAXY_CHUNKED_PREFILL=1
# Decode: shared RoPE tiles, ring collectives, 2-link attention all-gather (decode layer ~259 -> ~214 us)
export LLAMA_DECODE_ROPE_CACHE=1 LLAMA_BH_CCL_TOPOLOGY=ring LLAMA_BH_RS_WORKERS=1 LLAMA_BH_ATTN_AG_LINKS=2
# Prefill (one 4K chunk 8.1 -> ~5.3 ms device time; 80-layer logits PCC 0.992 vs all off). SDPA LoFi drops PCC to 0.96: keep HiFi2.
export LLAMA_BH_PREFILL_RS_WORKERS=4 LLAMA_BH_SDPA_FIDELITY=HiFi2 LLAMA_BH_PREFILL_WO_RS=1 LLAMA_BH_PREFILL_FF2_MINIMAL=1
export LLAMA_BH_PREFILL_MM_TUNED=1 LLAMA_BH_PREFILL_FF13_BF8=1 LLAMA_BH_PREFILL_MINIMAL_2K=1
# Ring SDPA with 64-row query chunks (4K layer 0.99 -> 0.28 ms SDPA) and the fused all-gather + FF2 matmul at 4K
# (all_gather_minimal_matmul_async, 456 -> 342 us). 80-layer logits PCC vs all off: 0.9945 at 4K, 0.9989 at 2K.
export LLAMA_BH_SDPA_Q_CHUNK=64 LLAMA_BH_PREFILL_FF2_AGMM=1
# Prefill levers ported from Wormhole (gtobar/prefill-levers): 1024-multiple padding, sequence-split residual add + RMSNorm,
# sequence-split QKV heads + RoPE, and for prefill over a cached prefix: ring SDPA on all 4 columns (CPF_RING) with a reusable
# prefix K|V replica. 80-layer last-token PCC vs these off: 0.995-0.999, top-1 unchanged (tests/test_cpf_port_pcc.py).
# vLLM (Nov 19): 60 RPM TTFT 0.62/1.09 -> 0.54/1.01 s, 40% prefix TTFT 0.77/2.88 -> 0.55/1.57 s, 240 RPM 1.26/8.93 -> 1.08/7.47 s.
# CPF ring warmup adds traces: run_server.sh's TRACE_REGION default is 700 MB.
export LLAMA_GALAXY_PREFILL_PAD=1024 LLAMA_GALAXY_PREFILL_SEQSPLIT=1 LLAMA_GALAXY_PREFILL_QKV_SEQSPLIT=1
export LLAMA_GALAXY_PREFILL_CPF_RING=1 LLAMA_GALAXY_PREFIX_SKIP=ring LLAMA_GALAXY_PREFILL_CPF_QKV_SEQSPLIT=1 LLAMA_GALAXY_PREFILL_CPF_REPLICA=1
# Seq-split prefill on BH: the norm output all-gather in bf8, and WO as one fused matmul + reduce-scatter
# (ttnn.experimental.matmul_reduce_scatter_prefill). 1-layer 4K 3.50 -> 3.31 ms; 80-layer PCC 0.996-0.999, top-1 unchanged;
# quick MMLU 81.9%. vLLM (Nov 20, vs the levers above alone): single 4K 0.29 -> 0.27 s, 60 RPM TTFT 0.54/1.01 -> 0.50/0.94 s,
# 40% prefix 0.55/1.57 -> 0.50/1.42 s, 240 RPM 1.08/7.47 -> 0.98/7.24 s (decode 22.5 -> 23.4), c32 4.38/8.36 -> 4.26/7.60 s.
export LLAMA_BH_PREFILL_NORM_BF8=1 LLAMA_BH_PREFILL_WO_MMRS=1
# vLLM step budget 8192 (run_server.sh): prompts up to 6K prefill in one pass instead of 4096 + a costly second chunk,
# and at most one new prompt per lane per prefill step (vllm-tt-plugin additional_config tt.max_new_prefills_per_step,
# a local plugin change) so a step does not hold every first token back. The 240 RPM tail was the opening burst of 32
# requests draining slowly through split prompts. vLLM (Nov 20, sla_cap1 vs sla_cpf3): 240 RPM TTFT 0.98/7.24 ->
# 0.58/3.00 s, decode 23.4 -> 25.7 tok/s/u; 32 x 4K at once 4.26/7.60 -> 3.99/7.60 s; 60 RPM 0.50/0.94 -> 0.43/0.76 s;
# 40% prefix 0.50/1.42 -> 0.46/1.04 s; cold 16K 1.81 -> 1.58 s. 8192 without the cap: 32 x 4K mean 5.1 s.
# Decode step after every prefill step (plugin default 2; Nov 22, il1_a / il1_b): decoding users then finish their
# outputs staggered instead of together, so the closed-loop 32 x 4K line's later rounds stop arriving as bursts.
# 32 x 4K 1.47 -> 1.13 / 1.16 s (later rounds 1.34 -> 0.80 s, round 0 1.74 -> 1.80 s); 32 x 16K, 60 RPM and 240 RPM
# unchanged. Every 8 steps: 1.70 s. Two decode steps per insertion: 1.18 s.
export MAX_BATCHED_TOKENS=8192 TT_CFG_EXTRA='"max_new_prefills_per_step": 1, "decode_interleave_prefill_steps": 1, "decode_interleave_decode_steps": 4'
# Gio plugin scheduler (lane_scheduler.py): hold a prefill step up to 50 ms while users that finished together are still
# re-arriving (8 ms quiet), use 1 decode step per prefill step instead of 4 while the next prompt is > 8K tokens, and
# prefill a single group that arrived during the last prefill step at once (FRESH_SKIP: breaks a one-step backlog).
export TT_PLUGIN_BUSY_GATHER_MS=${TT_PLUGIN_BUSY_GATHER_MS:-8} TT_PLUGIN_LONG_PROMPT_TOKENS=${TT_PLUGIN_LONG_PROMPT_TOKENS:-8192} TT_PLUGIN_FRESH_SKIP=${TT_PLUGIN_FRESH_SKIP:-1}
# DP4 x TP8 cold prefill (models/demos/llama3_70b_galaxy/tt/prefill_dp4tp8.py): TP32 prefill is communication-bound (per
# 4K layer ~1.7 ms all-gather / reduce-scatter vs ~0.95 ms matmul), so each mesh column prefills its own lane's prompt
# with TP8 instead: the four prompts of a prefill step run at once and the KV lands in the TP32 decode layout. Cold prompts
# >= 2048 tokens on >= 2 columns take it; single prompts and prefix hits stay on TP32. Step budget 16384 so 16K prompts
# prefill in one pass. vLLM (Nov 20, runs/dp4_v4 vs sla_cap1): 32 x 4K 3.99/7.60 -> 2.53/4.92 s, 32 x 16K 28.7/49 ->
# 12.8/21.1 s, 60 RPM and 240 RPM no regression; quick MMLU 81.84% (DP4 forced onto >= 1K prompts), needle 0 misses.
# Second weight copy ~6.7 GB per chip (cache dp4tp8_*; first start builds it). Opt-in precision change:
# LLAMA_BH_PREFILL_DP4_LOFI=ff2,wo,qkv (LoFi for those bf8 matmuls: 32 x 4K 2.45/4.54 s, quick MMLU 81.93%).
export LLAMA_BH_PREFILL_DP4=1 MAX_BATCHED_TOKENS=16384
# One-op rms_norm and bf16 accumulation for QKV / FF13 (K block 16 is the code default): with GIO=1 FLUSH=1
# TT_PLUGIN_IDLE_GATHER_MS=150 TT_MRS_BF8_TRANSPORT=1 LLAMA_BH_PREFILL_DP4_LOFI=ff2,wo,qkv
# LLAMA_BH_PREFILL_DP4_SDPA_FP32=0 (runs/dp4_opt1, Nov 21): 32 x 4K 1.80/3.59 s, 32 x 16K 8.22/14.35 s, quick MMLU
# 82.11%, needle 0 misses.
export LLAMA_BH_PREFILL_DP4_NORM=${LLAMA_BH_PREFILL_DP4_NORM:-rms} LLAMA_BH_PREFILL_DP4_FP32=${LLAMA_BH_PREFILL_DP4_FP32:-none}
# Shipped DP4 precision settings (every run since dp4_ship used them on the command line). The code also defaults to
# SDPA LoFi, QKV bf8 output and WO via all-to-all with fp32 accumulation from 8K tokens. Full MMLU with every prompt on DP4
# (14042 questions): 82.99% before those three, 83.13% with them (150 lost / 170 gained). Served (api8_a, Nov 22):
# 32 x 16K 6.80 s, 32 x 4K 1.47 s, 1 user 4K 0.42 s. Serving launch used for these numbers:
#   SERVE_CLI=1 SERVER_ARGS=--api-server-count=8 GIO=1 FLUSH=1 TT_PLUGIN_IDLE_GATHER_MS=150 (see README)
export TT_MRS_BF8_TRANSPORT=${TT_MRS_BF8_TRANSPORT:-1} LLAMA_BH_PREFILL_DP4_LOFI=${LLAMA_BH_PREFILL_DP4_LOFI-ff2,wo,qkv}
export LLAMA_BH_PREFILL_DP4_SDPA_FP32=${LLAMA_BH_PREFILL_DP4_SDPA_FP32:-0}
# matmul_reduce_scatter_prefill chunk splits for the DP4 WO / FF2 shapes ("Mt,Kt,Nt,n_m,n_n"): in the full model the
# decode path's resident L1 caps the plan budget and the planner's cost model then picks slow splits. Column axis,
# model-like budget (tests/test_mrs_chunks.py):
#   FF2 16K: auto 3.74 ms, (4,6) 2.87 ms
#   WO 16K: auto 1.80 ms, (3,4) 1.60 ms
#   FF2 8K: auto 1.93 ms, (2,6) 1.49 ms
#   WO 8K: auto 0.94 ms, (2,3) 0.83 ms
# 4K is best on auto. Nov 22, ranked by energy at the power cap with bfp4 W2 (the 16K default; tests/test_mrs_energy.py):
# FF2 16K (4,6) 233 mJ / 3.32 ms, (8,3) 224 mJ / 3.23 ms, (6,4) 224 mJ, (4,4) 225 mJ. 4K auto stays cheapest.
export TT_MRS_FORCE_CHUNKS=${TT_MRS_FORCE_CHUNKS:-"512,112,256,8,3;512,32,256,3,4;256,112,256,2,6;256,32,256,2,3"}
# Fabric: vLLM reads FABRIC (run_server.sh), pytest reads TT_FABRIC_OVERRIDE (conftest.py)
export FABRIC=${FABRIC:-FABRIC_1D_RING} TT_FABRIC_OVERRIDE=${TT_FABRIC_OVERRIDE:-FABRIC_1D_RING}
# vLLM async scheduling is on by default in run_server.sh (+13% decode vs off). MMLU 81.6% before and 82.9% after the
# full SLA bench with it on (Nov 19). Garbage output seen earlier came from chips left dirty by a failed or killed server,
# not from async: reset the Galaxy (tt-smi -glx_reset_auto) after any hang, hard kill or experimental-op crash.
# KV-cache tokens per DP lane (8 users per lane): 131,072 held exactly 8 x 16K prompts, so in 32 x 16K at once the first
# users stalled ~6.7 s for KV blocks. 163,840 (~20K per user) removes that stall; +25% KV DRAM (~0.7 GB per chip).
export LLAMA_BH_KV_TOKENS_PER_LANE=${LLAMA_BH_KV_TOKENS_PER_LANE:-163840}
