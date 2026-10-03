# DSV4.1-Flash decode: what GPU stacks fuse that we do not (research, 2026-10-03)

Scope: BH Galaxy 4x8, batch 16 (T=4 tokens/device), ~1.5 ms/layer. Our numbers come from `docs/superpowers/specs/2026-10-02-dsv41-layer-op-table.md` (L2 = 85 programs, L3 = 63 programs). Evidence tags: **[V]** = read in a source this session (page or local file), **[V-sum]** = a fetched web page that went through a summarizing model, so wording and numbers are second-hand and some summaries contained obvious slips (flagged), **[I]** = inference/estimate by me. No device runs.

## 0. Headline

* DeepSeek's own paper (arXiv 2609.19969, sec 3.2) says the "vast majority" of layers (CSA2 Reuse Mode) execute with **15 kernels in prefill and 11 in decode** [V-sum, the sentence was returned twice, by two fetches]. We run 63 programs (L3, reuse-mode-like) and 85 (L2, compressor owner). 76/85 are under 30 us, 45 are launch-bound (op table).
* The kernels named there: FlashMLA "fused-RoPE-attention-RoPE-cast", DeepGEMM **Mega-Gate, Mega-mHC, Mega-MoE**, TileKernels, DeepSelect TopK [V-sum]. The list of the 11 decode kernels is not in what I could retrieve (PDF was binary, HTML fetch returned only summary lines); my reconstruction in sec 2 is [I].
* Only 4 of 40 layers own a compressor (`kv_source_layer_ids` [2,8,14,20]), 8 own an indexer (`index_source_layer_ids`), 2 have Engram ([1,14]) [V, `/mnt/tt-data/ssinghal/deepseek-v41-flash/config.json`]. The op table's compressor candidate (#7) therefore saves ~26 us x 4 layers = 0.1 ms/token, not x40.
* Our two biggest non-fusion items stay bigger than any fusion: the moe_compute straggler (reduce_scatter 181-194 us in trace vs 48-53 eager) and the MoE expert streaming itself.

## 1. Sources consulted

| source | what I got | tag |
|---|---|---|
| arXiv 2609.19969 (V4.1-Flash) https://arxiv.org/html/2609.19969v1 | 15/11 kernel statement; Mega-mHC memory traffic; DSpark = 3 transformer blocks, window 128, 5 draft positions in parallel, confidence head selects verification length; Engram = multi-head hashing + context gating, embeddings prefetched from host by RDMA | [V-sum] |
| arXiv 2606.19348 (V4) https://arxiv.org/html/2606.19348v1 | MegaMoE: fine-grained comm/compute overlap, wave-based expert scheduling, 1.50-1.73x (1.96x latency-sensitive); TileLang host codegen; deterministic mHC reductions | [V-sum] |
| DeepGEMM README https://github.com/deepseek-ai/DeepGEMM | Mega MoE fuses EP dispatch + linear1 + SwiGLU + linear2 + EP combine (`fp8_fp4_mega_moe`); Mega-Gate, Mega-mHC, hash gate listed in 2026.09.10 news, no detail; MQA-logits indexer kernels (`fp8_fp4_paged_mqa_logits`) | [V-sum] |
| FlashMLA README https://github.com/deepseek-ai/FlashMLA | fused Q-norm + Q-RoPE + attention + inverse-RoPE (O-RoPE conjugate) + cast-to-FP8 in one kernel; `attn_sink` applied in-kernel as exp(lse)/(exp(lse)+exp(sink)); needs Q_b/Wv pre-permutation; 528 B/token V4.1 KV format | [V-sum] |
| TileKernels https://github.com/deepseek-ai/TileKernels | dirs `moe` (top-k routing), `quant` (fused SwiGLU+quant), `engram` (gating with fused RMSNorm), `mhc` (Sinkhorn, mix split/apply), `transform` (RoPE) | [V-sum] |
| vLLM PR #56962 / #56255 (Mega-mHC) https://github.com/vllm-project/vllm/pull/56962 | "shifted post + next pre + GEMM + Sinkhorn + RMSNorm" in ONE kernel (`mhc_shifted_post_pre`); table H=5120: 1 token 13.30 -> 11.68 us, 64 tokens 18.30 -> 12.64 us vs TileLang fused; needs hc_mult=4, H%1024==0; falls back to 2 TileLang kernels when Engram is present. (The summarizer called SM100 "H100" and mHC "multi-head-expert"; ignore those words.) | [V-sum] |
| vLLM PR #56344 (attention megakernel) https://github.com/vllm-project/vllm/pull/56344 | FlashMLA fused kernel = Q RoPE + sparse attn (SWA + compressed) + inverse RoPE + FP8 cast; GB200 decode s_q=1 22.7 us, "halves total time" vs split-KV + separate inv-RoPE/quant kernels; one query token per CTA, no split-KV | [V-sum] |
| vLLM RFC #56506 https://github.com/vllm-project/vllm/issues/56506 | on ROCm, mHC = 2.05 ms/step in 460 launches (eager-inflated); fused mHC post+pre collapses a seam to 5 launches (4 with all-reduce); aux-stream overlap proposed for shared expert as the cheap route; fold MXFP8 quant into RMSNorm prologue (1.49-1.77x on norm+quant pairs) | [V-sum] |
| vLLM tracking issue #56400 https://github.com/vllm-project/vllm/issues/56400 | backend list: FlashMLA, DeepGEMM Mega-Gate/Mega-mHC, DeepSelect, attention megakernel #56344, sparse indexer via DeepGEMM MQA logits #56254, Engram PRs #56219-56357 | [V-sum] |
| vLLM recipe https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash | env `FLASHINFER_MLA_SPARSE_DSV41`, `FLASHMLA_MEGA_ATTN_DSV41`, `--moe-backend deep_gemm_mega_moe`, DSpark config `{"method":"dspark","num_speculative_tokens":5,...,"enable_adaptive_verification"}` | [V-sum] |
| SGLang blog (PyTorch) https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0/ | full text read: #24775 mHC re-wired (mhc_pre on DeepGEMM, RMSNorm fused into mHC, dedicated fused `hc_head`), #25976 `mhc_fused_post_pre`, #24890 KV Compression V2 (c4, c128, online c128 kernels + fused norm/rope), #25052 W4A4 MegaMoE, #25810 prewarm MHC token buckets; roadmap "more fusion on small kernels" | [V] |
| SGLang PR #25976 https://github.com/sgl-project/sglang/pull/25976 | fuses previous hc_post + bf16 residual materialisation + pre-norm GEMM partials + RMS square-sum partials for small token batches (TileLang scalar-FMA kernel `mhc_fused_post_pre_fma_tilelang`); keeps the numerically sensitive `mhc_pre_big_fuse` finalisation separate; +3.35% total throughput on V4-Flash | [V-sum] |
| SGLang PR #39857 (AMD gfx950) https://github.com/sgl-project/sglang/pull/39857 | mHC boundary fusion with deferred coefficients; fused reduce-add gate (shared expert optional); fused WO-A MXFP8 (split-K batched GEMM); fused K norm-RoPE; fused decode glue (length fold + compression metadata + page table); all-reduce + mHC post. DSpark block 5: batch 1 real text 2.15x (random 4.12x), batch 8 1.52x, batch 32 1.03x | [V-sum] |
| LMSYS V4.1 day-0 blog https://www.lmsys.org/blog/2026-09-10-deepseek-v41/ | mHC: overlap mixing-coefficient computation with attention/FFN + fuse mixing-statistic reductions with Sinkhorn iterations; FP4 indexer: RoPE + FP4 quant + pack/cache write in one kernel; **ratio-2 decode pooling fused into one kernel**; BF16 single-token matvec kernel; Engram: gating reduces FP32 temporaries, n-gram hashing in one kernel; replay 1.56x on 8xH200 | [V-sum] |
| LMSYS V4 day-0 blog https://www.lmsys.org/blog/2026-04-25-deepseek-v4/ | Flash Compressor: whole compress pipeline in one on-chip pass, HBM round-trips 5 -> 2, >10x over naive; Lightning TopK: radix-select cluster reduction 100+ us -> ~15 us at small batch; FlashMLA runs SWA + C4/C128 extra attention in one call with shared metadata; mHC pre-GEMM split-K; fused mHC (RMSNorm + Sinkhorn + residual mixing, PDL); in-graph spec-decode metadata | [V-sum] |
| Local: v3_b1, ttnn experimental ops, our model code | see sec 4 | [V] |

Not retrievable: the DSV41 vLLM kernel sources themselves (no clone in the environment; only pages), DeepGEMM Mega-Gate / hash-gate API details, TensorRT-LLM DSV4 (search surfaced nothing specific; I did not find a source, so nothing is claimed about it), arXiv 2609.19969 sec 3.2 kernel list.

## 2. What a GPU decode layer looks like vs ours

Reconstruction of the 11 decode kernels of a Reuse-Mode layer [I, from the fused-kernel names in sec 1; the exact split is not verified]:
1. Mega-mHC (previous post + this pre + GEMM + Sinkhorn + RMSNorm) -> normed x
2. fused wq_a/wkv GEMM (single-token BF16 matvec / FP8 GEMM)
3. q-norm + kv-norm (+ quant)
4. wq_b GEMM
5. K norm-RoPE + KV/SWA cache write (FlashMLA-side or the "fused norm/rope V2")
6. (indexer layers only: indexer GEMM + RoPE/FP4 + MQA logits + DeepSelect TopK; Reuse-Mode layers skip)
7. FlashMLA fused: Q-RoPE + sparse attention with sink + inverse RoPE + cast
8. wo_a grouped GEMM
9. wo_b GEMM (+ all-reduce)
10. Mega-Gate (router GEMM + sqrtsoftplus + bias + top-k + normalise, [I] from the name)
11. Mega-MoE (dispatch + GEMM1 + SwiGLU(clamp) + GEMM2 + combine + shared expert), then the next layer's Mega-mHC picks up the post.
That is ~10-11 launches of which ~7 are GEMMs. Ours is 63-85 programs; the non-GEMM glue is the gap.

## 3. Block by block

For each block: GPU fusion | what we run now (op-table row numbers, L3 unless noted) | saving | TT feasibility | effort.

### 3.1 mHC (mixes / Sinkhorn / collapse / expand / RMSNorm)

* **GPU [V-sum]:** Mega-mHC is one kernel for post(prev) + pre(next) + GEMM + Sinkhorn + RMSNorm (11.7-12.6 us at H=5120, 1-64 tokens). SGLang first fused post+pre ("fused_post_pre"), then moved to DeepGEMM; the paper quotes single-pass traffic (2n+2)d vs (3n+2)d per token (one summarizer returned "4n+4" for the old number; treat the constants as unverified). The LMSYS blog says the mixing-coefficient computation is overlapped with attention/FFN: the post/comb coefficients are only needed at the expand, so the Sinkhorn is off the critical path [I interpretation].
* **Ours:** per mHC site: expand 27 + proj 21.7 + post 46.2 (ONE core, 20 Sinkhorn iterations) + collapse 12.5 + norm 12-15 = ~120 us, x2 sites = **~240 us/layer (16-17% of the layer)** (rows 0-3, 28-32, 42-46 incl. 17.5 us layer-entry wait that is not an op gap). Data is 4x5120 fp32 = 80 KB, so these are latency-bound, not bandwidth-bound [I].
* **Saving:** target ~65 us/site (expand+proj fused ~30, post ~30, collapse+norm ~15 -> 75; take 65-75): **110-150 us/layer, 4.4-6 ms/token** [I]. Beyond that, Sinkhorn off the critical path (only `pre` needed before collapse) saves another ~30-40 per site [I].
* **TT feasibility:** local building blocks exist [V]: our JIT kernels `tt/mhc_kernels/mhc_{proj,post,collapse,norm,expand}_*.cpp` (post is a copy of `deepseek_prefill/mhc_split_sinkhorn` compute); `ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` (prefill version, same maths); `attn_res_weighted_reduce_nc` (weighted reduce over dim 1: the collapse). Plan: (a) split post kernel: `pre` (sigmoid, cheap) emitted early, `comb`/`post` Sinkhorn emitted by a second core group that the collapse/norm cores do not wait for (inside one program the kernel time is the max, so the real win is when Sinkhorn is hidden behind the attention ops, which needs a persistent side kernel or a second command queue: high risk); (b) merge `expand(prev)` into `proj(next)` reader (cross-layer, row 6 in the op table): reader loads x_prev, comb, post, attn_out, forms x_new, writes it and feeds the K-chunk matmul; (c) merge `collapse` + `norm` (needs a cross-core sum of squares: one semaphore or compute collapse on 8 cores and norm in the same program). Sinkhorn per-iteration (2.3 us) can also be cut by keeping the 4x4 blocks in DST and using SFPU row/col reductions instead of 2 tile matmuls per iteration [I].
* **Effort 4, risk medium-high** (fp32 numerics, trace/program-cache constraints; SGLang explicitly keeps the sensitive finalisation separate).

### 3.2 Attention

**q/kv LoRA + norms (rows 4-10: wqkv linear 34, slice, rms_norm 13.6 on 1 core, linear 21, slice, rms_norm 7.2 on 1 core, concat)**
* GPU: separate small kernels but fused norm+quant, fused K-norm-RoPE (PR #39857 [V-sum]); FlashMLA lists Q-norm inside its fused kernel.
* Ours: slice x2 (1.7 each) + two single-core rms_norm (13.6 + 7.2) + concat 2.5 + gaps = ~31 us.
* Saving ~18-20 us/layer (op table #8) ~0.75 ms/token. TT: split `wqkv` into two matmuls (removes both slices; `ttnn.linear` is already used with custom program configs), then one JIT kernel "dual rmsnorm" (2 cores concurrently, 4x1280 and 4x512, gamma folded as a plain multiply) writing straight into the concat layout `nlp_create_qkv_heads_decode` consumes. Pre-scaling `wq_b` rows by q_norm gamma removes only the gamma multiply [I]. Effort 2. Risk low.

**Partial RoPE (q: rows 26-28; inverse RoPE on o: rows 33-35; compressor latent: 14-16, L2 only)**
* GPU: FlashMLA fuses Q-RoPE, attention, inverse-RoPE and cast in one kernel [V-sum]; K norm-RoPE fused; indexer RoPE + FP4 quant + cache write fused.
* Ours: `x*C + (x@P)*S` = multiply 3.1 + linear 17.6 (dense 512x512 pair-swap matmul, 8 cores) + addcmul 3.1 + gaps ~1.6, twice per layer (q/kv forward, o inverse) = ~58 us/layer. The 17.6 us matmul reads a 512 KB matrix per call; the real rotation is block-diagonal with 32x32 blocks on the last 64 dims only [I].
* **Not directly reusable [V]:** `ttnn.experimental.rotary_embedding_llama` TT_FATALs for head_dim > 256, and for head_dim > 128 requires `fp32_dest_acc_en=False` (`rotary_embedding_llama_device_operation.cpp:68-71`), all tensors bf16. `rotary_embedding_llama_fused_qk` is the same family. v3_b1 `micro_ops/rope` and `unified_kernels/rope.hpp` implement `out = in*cos + (in@trans_mat)*sin` as a micro-op on the b1 core layout [V].
* Saving 35-45 us/layer (1.4-1.8 ms/token) [I]. Options: (1) cheapest: apply `rotary_embedding_llama` to a 64-wide tail only (slice + rope + concat, but needs the row layout it expects; gain uncertain, ~10 us); (2) one JIT eltwise kernel doing 16 tiles x (mm with a 32x32 trans_mat, mul cos, mul sin, add) for q+kv rows in the `nlp_create_qkv_heads_decode` output layout, and the inverse variant for o; ~3-4 us each; (3) GPU-style: inverse RoPE as an epilogue of SDPA decode (modify `sdpa_decode` writer): effort 5. Recommend (2). Effort 3, risk low-medium.

**KV compress (compressor owners only: L2 rows 4-16, 4 layers per token)**
* GPU: ratio-2 decode pooling in one kernel [V-sum]; Flash Compressor single on-chip pass [V-sum].
* Ours: subtract, 3 slices, sigmoid, addcmul, typecast, rms_norm, copy + RoPE (3) = ~40 us + 8 us gaps.
* Saving ~26 us x 4 layers = 0.1 ms/token (op-table #7 x40 is overstated). Effort 2 (one eltwise JIT incl. RoPE). Low priority.

**Indexer (8 layers; not in the measured L2/L3 forward as ratio-2 L2 has no indexer step in the table)**
* GPU: DeepSelect/Lightning TopK ~15 us; DeepGEMM `fp8_fp4_paged_mqa_logits`; RoPE+FP4+pack fused [V-sum].
* Ours: `tt/indexer.py` already uses `ttnn.experimental.indexer_score_dsa` + `topk_large_indices` (a fused score op and a large-k top-k op) [V], but q-side linear + heads + RoPE (x@P) remain separate, and it is unmeasured. Action: measure first.

**KV cache write + SDPA with sink (rows 30-32)**
* GPU: SWA + compressed extra attention in ONE call, sink in-kernel [V-sum]. Ours already has that: single SDPA decode over a combined ring+compressed cache with additive mask and `attention_sink` [V, `attention.py` forward], 38 us on 4 cores. Two `paged_update_cache` calls (3.8 each + 1.4 gap) could be one when a compressed latent is written (L2: 2 calls, rows 30-31) [I]. SDPA at 38 us/4 cores is the outlier; 2 cores/head is ~7 us faster but blocked by L1 clash [V, comment in attention.py].

**Grouped o-proj (rows 36-39: nlp_concat_heads_decode 3.3, to_memory_config 1.9, wo_a 23.6, wo_b 21.6)**
* GPU: fused WO-A MXFP8 batched split-K GEMM [V-sum]; all-reduce fused with mHC post [V-sum].
* Ours: 4 programs + RS 20 + AG 23. Saving: concat + to_memory_config ~7 us if SDPA writes interleaved DRAM in concat-ready layout [I]. `deepseek/mla/matmul_wo` exists [V] but is a b1 (7168-hidden, 64-core) op, not our shape.
* CCL (rows 40-41: RS 20.4 + AG 23.4 = 44 us, 3% of layer): TT has `llama_reduce_scatter_matmul`, `matmul_reduce_scatter_async`, `all_gather_matmul_async`, `all_reduce_async`, `rms_allgather` (`fused_rms_minimal`) [V names]. Saving ~15-20 us; fusing the AG into the next mHC proj reader (read the gathered x from peers directly) would also drop AG [I]. Effort 4, risk medium.

### 3.3 Router (rows 47-69 in L2, 33-47 in L3)

* **GPU:** Mega-Gate (DeepGEMM) and DeepSelect TopK; hash gate for hash layers; TileKernels `moe` routing [V-sum, names only; Mega-Gate contents NOT verified, assumed router GEMM + activation + bias + top-k + normalise].
* **Ours:** L3: matmul 22 + add 6.3 + add 3.1 + topk(12 cores) 4.7 + **topk (1 core) 63** + typecast + reshape + eq 7.5 + reshape + 2 matmul 12.6 + div + to_layout + typecast + to_layout + 3 to_memory_config = ~133 kernel + ~12 gaps = ~145 us (L2 ~165; table says 123-164 removable). 63 us is the final single-core merge of `ttnn.topk` over 384.
* **TT, exists [V]:** `ttnn.experimental.deepseek.generalized_moe_gate` (gate + bias + top-k + normalise in one program on height-sharded input, `moe_gate_mm` for the matmul, `deepseek_moe_gate` micro-op in v3_b1, `topk_router_gpt` experimental op, and for hash layers `deepseek_prefill.moe_hash_gate`: tid2eid lookup + sqrtsoftplus + normalise + scale). `router.py` already has `DSV41_ROUTER=kernel` using the bf16 gate with the shifted-bias trick: "faster, ~95% exact sets" [V, docstring]. The blocker is exactness, not availability. `git status` shows `generalized_moe_gate.hpp` is being modified in this worktree.
* **Plan:** (a) use the fused gate for selection, then recompute weights in fp32 for the 6 chosen experts (one small matmul + div), accepting expert-set differences only where bf16 ranks tie (measure vs exact on real activations; DSpark verify with exact greedy is robust to routing noise only if the *backbone* path is the exact one, so keep an exactness flag); or (b) write a single-core fp32 JIT "top6 of 384" (6 passes of max + mask over 4x384 values is ~1500 elements; 63 us is the generic bitonic merge, not the data size) fused with sqrtsoftplus, bias, select and normalise: 1 matmul + 1 JIT op, ~30 us total.
* **Saving 100-120 us/layer (4-4.8 ms/token), effort 3 (a) or 4 (b), risk medium (exactness).** Also drops the 3 `_format_dispatch_inputs` `to_memory_config` (5 us) if the gate writes L1 RM weights/indices directly.

### 3.4 MoE (dispatch / combine / grouped GEMM / shared expert)

* **GPU:** Mega-MoE: dispatch + GEMM1 + SwiGLU + GEMM2 + combine in ONE kernel with wave scheduling overlapping NVLink and tensor cores [V-sum, README + paper]; "fused reduce-add gate" folds shared expert into the combine [V-sum]; aux-stream overlap for shared experts is the cheap alternative [V-sum, RFC].
* **Ours:** dispatch_metadata 10-15 + `moe_compute` 282-358 (median) + tilize_with_val_padding 24.5 + fast_reduce_nc 14.3 + reduce_scatter 181-194 (straggler wait) + allgather 25 + shared expert 111 (rows 73-84). moe_compute already fuses dispatch-receive, GEMM1/SwiGLU/GEMM2 and combine-send.
* **Tail (rows 75-76, 24.5 + 14.3 + gaps = ~40 us):** `deepseek_moe_post_combine_tilize` exists but is only used when `batch_per_device == TILE_SIZE (32)` and an NdShard config is available [V, `tt_moe_decode.py:955`]; we have T=4, so we fall back to `tilize_with_val_padding`. Saving ~25 us/layer (1 ms/token) by extending the combine-writer to emit tiled bf16 for T<32 or fusing tilize into `fast_reduce` (read RM, tilize in the reader). Effort 3, risk medium (it is a shared kernel; ttnn op + moe_compute program already modified in the worktree).
* **Shared expert (rows 79-83, 111 us/layer, 7.6%):** current: 65 (fused gate+up linear, 36 cores) + 2 slices + multiply + 38 linear. (a) Concurrency with moe_compute (it uses 32 cores of ~130; the shared expert uses 36-120): ideal saving ~99 us only if it truly overlaps on the *slowest* device; merely reordering before the reduce_scatter does NOT help the critical path (the slowest device pays the same total) [I]. TT has no ready op-level concurrency in one command queue; options: sub-devices/second CQ (high risk), or (b) treat the shared expert as an extra always-on expert inside moe_compute (`tt_moe_decode.py` already documents a "combined routed+shared per-device expert count" for weight packing [V, line ~533-541]); costs ~83 us/expert streaming on every device [V, roofline_model.py docstring: "moe_compute streams an expert in 83 us"] so the saving is only ~30 us and may worsen the straggler; (c) cheap fixes: remove the two slices and the multiply by using `ttnn.linear` with separate gate/up outputs and a fused `silu * up` activation on the second (`input_tensor_a_activations`, as already done in router.py): ~10 us. v3_b1 `fused_ops/shared_expert` (mcast + gate/up on 128 cores + gated reduce + down + residual add) and `gated_local_reduce_down_proj` are the reference for a 1-program version but are tied to the b1 7168/130-core layout [V].
* **Clamp [V]:** the checkpoint clamps gate<=10 and up in [-10,10] (`swiglu_limit`: config.json, yaml line 21); `moe_compute` SILU path has none, so routed-expert outputs differ from the reference when |activations| exceed 10. `moe_gpt/device/kernels/swiglu_sfpu.h` has clamping but with GPT-OSS formula `(up+1)*gate*sigmoid(alpha*gate)`. A DeepSeek variant is two SFPU clamps in `pack_compute_activation` (cheap, effort 2, risk low); this is a fidelity fix more than a speed one. Check the shared expert path too (model.py clamps there).
* **Straggler (not a fusion, biggest single item):** reduce_scatter 181-194 us in trace vs 48-53 eager means ~130-140 us/layer waiting for the busiest device (moe_compute max 463-522 vs median 282-358). GPUs hide this with wave scheduling and EPLB-style balancing (DeepSeek/SGLang: DeepEP Waterfill, PR #25391 [V]). Spec decode (more tokens/step) spreads expert load and raises per-token throughput in this regime [I].

### 3.5 Engram (layers 1 and 14 only; unmeasured)

* GPU: n-gram hashing in one kernel; gating (RMSNorm of h and key, dot, sign-sqrt, sigmoid, residual) in TileKernels `engram` with fused RMSNorm; fewer FP32 temporaries; embeddings prefetched from host by RDMA [V-sum]. vLLM Mega-mHC falls back to 2 TileLang kernels when Engram is present.
* Ours: host hashes + reads rows from the memory-mapped table, uploads `rows`; device does `kv = rows@wkv`, two RMS stats, dot, sign-sqrt, sigmoid, residual add [V, `tt/engram.py` docstring]. Also `tt/engram_hash_kernels/{engram_hash,engram_gather}.cpp` exist [V] (device hash + gather kernel work already started).
* Saving: unknown; the gate math is probably ~15 small eltwise ops; one JIT kernel could save ~40-60 us x 2 layers = ~0.1 ms/token [I]. Low priority. Measure first.

### 3.6 LM head + sampling (per token, not per layer; unmeasured)

* GPU: no specific kernel found in sources.
* TT [V]: v3_b1 `fused_ops/lm_head_sampling` (CCL broadcast + mcast + vocab matmul fused) and `micro_ops/sampling` + `unified_kernels/{argmax,sampling}.hpp`; ours: matmul + `to_layout` + `argmax` + all-gather + 2nd argmax in `device_head.py`. Saving maybe 50-150 us/token [I]; negligible vs the 61 ms. For DSpark the markov head (bigram bias: Embedding 129280x256 and `markov_embed @ Head^T`) is per-position sequential in the reference; it is a 5-step dependent chain, so launch count matters there more than in plain decode [I].

### 3.7 DSpark verify / speculative decode

* GPU [V-sum]: block size 5; in-graph metadata (device-side length prep, verify metadata, acceptance tracking: SGLang PR #39857, vLLM in-graph spec metadata); FlashMLA decode kernel handles several query tokens, falling back to a split-KV kernel below a batch threshold; `enable_adaptive_verification` uses the confidence head to pick verify length; real-text speedups 2.15x (b=1), 1.52x (b=8), 1.03x (b=32) on MI350X.
* Ours: spec design doc `2026-10-02-dsv41-spec-decode-design.md` [V]. At T=4 users/device and 6 verified rows per user the SDPA/matmul M dimension is 24 <= 32 tile rows [I], so verification fits in the same tile as today's decode; the cost is MoE expert activation (roofline: 60 ms @16, 69 @32, 85 @64 tokens [V, `MEASURED_MS`]) and the dependency chain, not launches. Fusion relevance: per-step glue fused into the trace (position tensors, masks, ring slot indices: already device-side via `st` tensors), and the verify step needs the 3-block drafter (~3 layers) + the markov head; everything in this document that cuts per-layer glue also cuts the drafter. No separate verify kernel is needed for greedy-exact acceptance (compare argmax per position on device, one small kernel) [I].

## 4. What tt-metal already has that we can reuse [V unless noted]

| item | path | reuse for |
|---|---|---|
| `generalized_moe_gate` (gate+bias+topk+norm, grouped or not) | `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate` | router (already selectable via `DSV41_ROUTER=kernel`) |
| `moe_gate_mm`, `deepseek_moe_gate` micro-op, `topk_router_gpt` | `.../deepseek/moe/moe_gate_mm`, `models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate`, `.../experimental/topk_router_gpt` | gate matmul feeding the fused gate |
| `moe_hash_gate` (tid2eid + sqrtsoftplus + norm) | `.../deepseek_prefill/moe_hash_gate` | hash-routed layers if the checkpoint has them (config has no hash-layer field that I found; check `router.py`) |
| `mhc_split_sinkhorn` | `.../deepseek_prefill/mhc_split_sinkhorn` (our `mhc_post_compute.cpp` copies its compute) | Sinkhorn building block |
| `attn_res_weighted_reduce_nc` | `.../deepseek_prefill/attn_res_weighted_reduce_nc` (BH-only, bf16) | weighted stream collapse |
| `deepseek_moe_post_combine_tilize` | `.../experimental/deepseek_moe_post_combine_tilize` | MoE tail; only batch_per_device == 32 today |
| `deepseek_moe_fast_reduce_nc_fused`, `deepseek_moe_reduce_scatter`, `moe_compute`, `moe_gpt` (SwiGLU+clamp SFPU) | `.../experimental/ccl/...`, `.../deepseek/...` | MoE; clamp reference |
| `rotary_embedding_llama(_fused_qk)` | `.../experimental/transformer/rotary_embedding_llama*` | head_dim<=256 only; tail-only use |
| `rotary_embedding_indexed` | `.../deepseek_prefill/rotary_embedding_indexed` | prefill (chunked) only |
| `indexer_score_dsa`, `topk_large_indices` | `.../experimental/indexer_score`, `.../topk_large_indices` | already used by `tt/indexer.py` |
| `matmul_decode` (L1 width-sharded decode matmul) | `.../experimental/matmul_decode` | wqkv/wq_b/wo for small M |
| `llama_reduce_scatter_matmul`, `matmul_reduce_scatter_async`, `all_gather_matmul_async`, `all_reduce_async`, `rms_allgather` | `.../experimental/ccl/...` | attention all-reduce + next norm |
| `fused_distributed_rmsnorm`, `dit_rms_norm_unary_fused`, `fused_rms_minimal` | `.../experimental/transformer/...`, `.../ccl/rms_allgather` | reference for multi-core rmsnorm on tiny M |
| v3_b1 `fused_ops/{pre_sdpa, kv_cache_branch, post_sdpa, attention_block, shared_expert, moe, gated_local_reduce_down_proj, lm_head_sampling, broadcast_rms, decoder_block}` and `unified_kernels/{rmsnorm,rope,matmul,mcast,gather,eltwise_*,gated_reduce,argmax,sampling}.hpp` | `models/demos/deepseek_v3_b1/` | patterns (UnifiedKernelDescriptor, mcast/gather/CB layout) for any multi-stage JIT fusion; they assume the b1 130-core grid and 7168-hidden shapes, so reuse the pattern and unified-kernel op headers, not the ops |
| `ttnn.generic_op` JIT kernels with ProgramDescriptor | `tt/mhc_kernels/*`, `tt/mhc_collapse.py` (`_kernel`, `_cb`, `_hash`) | all new fusions below |
| `fusion` experimental dispatch op | `.../experimental/fusion/` | present; not evaluated (I did not read it) |

## 5. Ranked table (per layer, L2/L3 average; x40 unless noted; targets are estimates [I])

Saving/layer is kernel + gap of the replaced ops minus a target; the verified part is the "before" column (op table), the target is mine.

| rank | fusion (GPU analog) | before us | saving/layer us | per token ms | effort (1-5) | risk | evidence |
|---|---|---|---|---|---|---|---|
| 1 | Router: gate + sqrtsoftplus + bias + top6 + normalise in 1-2 ops (Mega-Gate / DeepSelect) | 123-164 | 100-120 | 4.0-4.8 | 3 | medium (exact expert sets) | before [V]; GPU kernel names [V-sum]; TT ops exist [V] |
| 2 | mHC: expand(prev)+proj(next) merged, collapse+norm merged, faster/multicore-time Sinkhorn (Mega-mHC, fused_post_pre) | ~240 | 110-150 | 4.4-6.0 | 4 | medium-high (fp32 numerics) | GPU [V-sum]; our kernels [V] |
| 3 | Shared expert concurrent with moe_compute (aux stream / Mega-MoE shared) | 114 | 30-100 | 1.2-4.0 | 3-4 | high (no concurrency primitive; may worsen straggler) | GPU [V-sum]; TT inferred |
| 4 | Fused partial RoPE (q+kv forward, o inverse) as one JIT op (FlashMLA fused RoPE) | ~58 | 35-45 | 1.4-1.8 | 3 | low-medium | before [V]; llama-rope limits [V]; target [I] |
| 5 | MoE tail: tiled combine output + fast_reduce, drop 3 format shuffles (Mega-MoE combine) | ~45 | 25-30 | 1.0-1.2 | 3 | medium | before [V] |
| 6 | q/kv lora: split matmuls + dual multicore rmsnorm kernel, no slices/concat | 31 | 18-20 | 0.7-0.8 | 2 | low | before [V]; target [I] |
| 7 | Attention all-reduce: RS+AG -> fused all-reduce / feed mHC proj (all-reduce + mHC post) | 45 | 15-20 | 0.6-0.8 | 4 | medium | before [V]; GPU [V-sum] |
| 8 | Remove to_memory_config / layout shuffles (SDPA output layout, producer layouts) | 14 | 12-14 | 0.5-0.6 | 1-2 | low | before [V] |
| 9 | Compressor chain in 1 eltwise kernel (ratio-2 pool fused; Flash Compressor) | 38+ | 26 | 0.1 (4 layers) | 2 | low | GPU [V-sum] |
| 10 | Engram gate + hash on device in one kernel (2 layers) | unmeasured | ~50 | ~0.1 | 3 | low | GPU [V-sum] |
| n/a | SwiGLU clamp in moe_compute (fidelity fix, free in speed) | 0 | 0 | 0 | 2 | low | missing clamp [V] |
| n/a | Straggler/load balance of moe_compute (reduce_scatter wait) | ~130-140 wait | up to 100+ | up to 4+ | 4-5 | high | op table [V] |

Totals if 1,2,4,5,6,8 land: ~300-380 us/layer = 12-15 ms/token of 61-65 (-20 to -24%) [I]. This reaches ~50 ms/token, i.e. ~20 tok/s/u, which is the plain target only with the straggler also addressed.

## 6. Recommended next five (with pointers)

1. **Fused router** (rank 1). Start from `router.py` `_forward_kernel` / `generalized_moe_gate` (`ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/device/unified_kernels/generalized_moe_gate.hpp`, in the worktree diff). Quantify set mismatch vs `exact_fast2` on real layer inputs (`tests/probe_topk_fp32.py` exists). Fallback plan: single-core fp32 JIT top6 + normalise (model on `tt/mhc_kernels/mhc_post_compute.cpp` for the generic_op scaffolding). Output straight into the L1 RM layout `_format_dispatch_inputs` needs.
2. **mHC consolidation** (rank 2). Do it incrementally: (a) `collapse+norm` into one program (`mhc_collapse.py` already shares `_kernel/_cb`); (b) `expand` fused as the reader of the next `proj` (`mhc_expand_reader.cpp` + `mhc_proj_reader.cpp`); (c) separate `pre` from the Sinkhorn part of `mhc_post_compute.cpp` and cut per-iteration cost. Validate with `tests/mhc_ref_impl.py` (bit-for-bit claim in the kernel header).
3. **Fused RoPE JIT op** (rank 4). 16 tiles x (mm trans_mat 32x32, mul cos, mul sin, add); consumes the `nlp_create_qkv_heads_decode` output layout and replaces rows 26-28 and 33-35 (`attention.py` `_rope_heads`, `_rope_rows`). Reference: `models/demos/deepseek_v3_b1/unified_kernels/rope.hpp`. Cos/sin tables already full-width with identity on nope dims (`st["Ch","Sh","nSh"]`), so no partial-rotary logic is needed.
4. **MoE tail + format shuffles** (ranks 5, 8): extend `deepseek_moe_post_combine_tilize` (`ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize`) to batch_per_device < 32, or tilize inside `fast_reduce`; write gate outputs in dispatch-ready layouts. Add the DeepSeek swiglu clamp in `moe_compute/device/kernels/compute.cpp` in the same pass.
5. **q/kv lora dual-norm + split matmuls** (rank 6) as the low-risk warm-up, then decide on shared-expert concurrency (rank 3) only after prototyping a second-CQ or persistent-kernel overlap on one layer; keep the moe_compute straggler analysis in parallel since it is worth more than ranks 3-8 combined.

## 7. Verification status summary

* Verified by reading local files: all op-table numbers, `rotary_embedding_llama` limits, `post_combine_tilize` precondition, router `kernel` mode caveat, missing swiglu clamp, config layer counts, existence and signatures of the ttnn ops in sec 4, the v3_b1 fused-op docstrings.
* Read on the web but via a summarizing model (numbers may carry errors; the SGLang/PyTorch blog was read in full and is the only fully verbatim source): all GPU fusion descriptions, speedups, the 15/11 kernel claim, Mega-mHC microbenchmark table, DSpark numbers.
* Inferred by me: all "after" targets and savings, the 11-kernel reconstruction, Mega-Gate contents, the shared-expert-reorder critical-path argument, DSpark row-packing at T=4, Engram savings, LM-head savings.
* Not found: TensorRT-LLM DSV4 fusions; DeepGEMM/FlashMLA/TileKernels source for mHC/Mega-Gate (only README-level descriptions); the paper's explicit per-kernel list.
