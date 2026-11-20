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
export MAX_BATCHED_TOKENS=8192 TT_CFG_EXTRA='"max_new_prefills_per_step": 1'
# Fabric: vLLM reads FABRIC (run_server.sh), pytest reads TT_FABRIC_OVERRIDE (conftest.py)
export FABRIC=${FABRIC:-FABRIC_1D_RING} TT_FABRIC_OVERRIDE=${TT_FABRIC_OVERRIDE:-FABRIC_1D_RING}
# vLLM async scheduling is on by default in run_server.sh (+13% decode vs off). MMLU 81.6% before and 82.9% after the
# full SLA bench with it on (Nov 19). Garbage output seen earlier came from chips left dirty by a failed or killed server,
# not from async: reset the Galaxy (tt-smi -glx_reset_auto) after any hang, hard kill or experimental-op crash.
