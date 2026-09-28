# GLM-5.3-Flash: bring-up feasibility (initial findings)

Read-only study, 2026-09-28. No weights were downloaded, no device was used. Target: chunked prefill with the gated
bring-up framework (`models/demos/common/bringup`, overview `models/demos/common/bringup/docs/pipeline_design.html`)
on 4x Blackhole p150b (32 GB DRAM each, about 27 GiB usable per chip), mesh 1x4 or 2x2, FABRIC_2D, 56320 tokens in
5120-token chunks. Scratch copies of the fetched config, index and modeling files were in
`/tmp/claude-1211409778/-localdev-dnijemcevic-tt-metal2/667a454c-7ac4-4ac3-adac-4d20d7141327/scratchpad/glm53/`
(temporary; refetch from Hugging Face if gone).

## Verdict

Feasible only as a layer subset. The full model needs about 42 GiB per chip even with bfp4 experts. Layers 0-7 (3
dense + 5 MoE, 2 of them DSA) fit with bfp8 experts at about 9 GiB of experts per chip and cover every block type.
The repo has most of the pieces (the model is structurally close to Kimi-K3 plus GLM-5.2 plus DeepSeek-V4, all in
`models/demos/deepseek_v3_d_p`), but one piece is missing (indexer key pooling) and several need adapting to a 4-chip
mesh. Rough effort: 85-110 tasks, about 1.5-2x the MiMo-V2.6 bring-up (60 tasks).

## Checkpoint

| Repo | Revision | Notes |
|---|---|---|
| `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | FP8 e4m3, 128x128 block scales; 62 shards, 328 GB; not gated; MIT |
| `zai-org/GLM-5.3-Flash-BF16` | `a5b45eb41df6402735dedc900be14a42e8d5e538` | BF16, 120 shards, 643 GB; same config |

No modeling code ships with the checkpoint. It is in transformers as model type `glm5_next`, class
`Glm5NextForConditionalGeneration`, from the 5.17.0 wheel (also on main, commit `0291458166a6`). The repo's
`python_env` has 5.12.1, which lacks it: vendor the modeling file or use a second environment.

## Architecture

320B total, about 17B active (model card: 320B total, 18B active). The checkpoint also has a 24-block vision tower
(0.56B) and 1 MTP layer; both are skipped for prefill.

| Item | Value |
|---|---|
| Layers | 45 text layers + 1 MTP layer (layer 45). Hidden 4096, vocab 154880, untied LM head, context 1M |
| Attention mix | 34 KDA (Kimi Delta Attention) linear-attention layers; 11 DSA layers at 3, 7, ..., 43 (every fourth) |
| KDA layers | 64 heads x 128. q/k/v projections 4096 to 8192, each with a 4-tap causal depthwise conv1d and SiLU, L2 norm on q/k. Low-rank per-channel forget gate (f_a/f_b, rank 128) with dt_bias and A_log, bounded to -5*sigmoid; beta from b_proj; gated RMSNorm output (low-rank g_a/g_b), then o_proj |
| DSA layers (MLA) | q_lora 1536, kv_lora 512, 64 heads, qk 256 / v 256, all NoPE |
| DSA indexer | 32 heads x 128, no RoPE, keys through a LayerNorm with bias; keys pooled by 4 with a learned softmax; selects the top 512 pools (2048 tokens) plus up to 3 tail tokens; every DSA layer runs its own indexer |
| Position encoding | none in the text model |
| Residual (mHC) | 4 streams, residual [T, 4, 4096]. Two mHC blocks per layer, each with a [24, 16384] fp32 projection, sigmoid pre/post and a 20-iteration Sinkhorn on the 4x4 mixing matrix. The final head is an unweighted mean of the streams |
| MoE | layers 3-44: 288 experts, top-8, sigmoid scores with the noaux_tc correction bias (one group), normalised top-k, routed scale 2.5, 1 shared expert, expert width 2048. Layers 0-2 dense (12288). SwiGLU with clamp 10 |
| Norms | RMSNorm, eps 1e-5 |
| Stored formats (FP8 repo) | experts, dense MLP, shared expert and most DSA q/kv projections FP8; KDA, o_proj, router, indexer, embedding, LM head BF16 |

Parameters: 321.3B in total (routed experts 304.4B, KDA attention 4.68B, DSA attention 1.37B, embedding and LM head
1.27B, shared experts and router 1.1B, dense MLP 0.45B, MTP 7.4B, vision 0.56B). About 16.7B active for the text path
without MTP.

Memory for the full model, experts over 4 chips, everything else split 4 ways:

| Precision | Total | Per chip | Fits? |
|---|---|---|---|
| experts bfp4, rest bfp8 | 181 GB | 42.2 GiB | no |
| experts bfp8, rest bfp8 | 334 GB | 77.6 GiB | no |

One MoE layer's experts per chip: 0.95 GiB at bfp4, 1.79 GiB at bfp8. Non-expert weights are about 35 MB per layer.
State at 56k is small (whole model, before splitting): MLA latent cache 634 MB, indexer cache 317 MB, KDA recurrent
state 143 MB. The 4-stream residual is 168 MB per 5120-token chunk in bf16.

## How it differs from what the repo has

- GLM-5.1/5.2 (in `models/demos/deepseek_v3_d_p`, `PREFILL_MODEL=glm_5_1|glm_5_2`, adapters at
  `models/demos/common/prefill/adapter.py:310`) are pure DSA models with RoPE, q_lora 2048, qk 192+64 and 256
  experts. GLM-5.3 adds KDA layers, mHC and key pooling, drops RoPE and changes the MLA geometry. It is structurally
  closer to Kimi-K3 (KDA + NoPE MLA) than to GLM-5.2. Only the DSA/sparse-MLA plumbing of the GLM adapters carries
  over. Commit `3f7ed020577` (GLM-5.2 sc4 prefill) is serving plumbing only.
- The `deepseek_v3_d_p` modules take configurable SP/TP axes; nothing is hard-wired to Galaxy, but the full block and
  transformer tests run on 8x4 only.

## Components

| Component | Needs | Existing code | Status | Notes |
|---|---|---|---|---|
| Embedding | 154880x4096 | `models/demos/deepseek_v3_d_p/tt/tt_parallel_embedding.py`, or the Gemma/MiMo embedding | covered | tested on 1x4; vocab not a power of 2, mask the pad token with `ttnn.minimum` (known issue) |
| mHC pre/post/mixing + Sinkhorn | 4 streams, 20 iterations, fp32 | `models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py`, op `deepseek_prefill.mhc_split_sinkhorn` | needs adaptation | standalone, single device, never wired into a model; GLM's head is a plain mean (DeepSeek-V4's is weighted), different norm eps |
| KDA: projections, conv1d+SiLU, gates, chunked recurrence with state across chunks, gated RMSNorm | 64 heads, TP=4 | `models/demos/deepseek_v3_d_p/tt/kda/*`, ops in `ttnn/cpp/ttnn/operations/experimental/kda/*` | needs adaptation | low-rank gate and the -5 bound supported; tested on 2x4 / 1x8 / 8x4 only, never 1x4 or 2x2; the tuned 5120-token config is for Kimi-K3's 96 heads |
| MLA q/kv projections, norms, absorbed kv_b | qk 256 / v 256, latent 512, no RoPE | `models/demos/deepseek_v3_d_p/tt/mla/mla.py` | needs adaptation | cache built around a 576-wide latent (512 + 64 RoPE); GLM-5.3 has 512 (Kimi-K3, also NoPE, keeps the 64 columns); a 2x2 test exists |
| Sparse attention | top-k indices into the latent cache | `ttnn.transformer.sparse_sdpa`, head-to-sequence reshard at `mla.py:2006` | needs adaptation | needs 32+ heads per chip: 64/4 = 16, the existing reshard handles it; index width 2051 pads to 2176 with sentinels as a contiguous tail, so indices need compacting; Blackhole only (fine) |
| Indexer scores + top-k | scores over pools, top 512 pools | `tt/mla/indexer.py`, `experimental/indexer_score`, `topk_large_indices` | needs adaptation | current indexer applies RoPE (GLM-5.3 has none); causal mask is per token, pools need "pool end <= query position", probably a fork of the score op |
| Indexer key pooling + expand by 4 + tail tokens | learned softmax over 4 keys, pooled-key cache across chunks | none | missing | buildable from existing ops (reshape, softmax, weighted sum, index arithmetic) or a new op; candidate for the op-gen deferral |
| Router | sigmoid + bias, noaux_tc, 288 experts, top-8, fp32 | `models/demos/mimo_v2_6_d_p/tt/router.py` (fp32 scores + `ttnn.topk`) | covered, small change | the fused `moe_grouped_topk` is hard-coded to 256 experts; keep the bias in fp32 (known issue) |
| Routed experts (EP=4) | 72 experts per chip, clamped SwiGLU at 10 | `ttnn.bringup.dispatch/combine/offset_cumsum` + `ttnn.bringup.unified_routed_expert_moe` (`ClampedSiluGlu`); `models/demos/mimo_v2_6_d_p{,_2x2}/tt/experts.py` | covered, small change | check ClampedSiluGlu clamps gate and up as GLM does; the fused kernel's bf16 accumulation failed MiMo's norm-ratio check on some layers (use high_precision) |
| Shared expert, dense MLP | SwiGLU with clamp | `tt/moe/tt_shared_expert.py`, `tt_ffn.py`, MiMo dense MLP | covered | tested on 1x4 |
| Final norm + LM head | 154880 | host LM head (Gemma and MiMo precedent) or `tt_lm_head.py` | covered | |
| Chunked-prefill state | MLA latent + indexer/pooled keys + KDA recurrent and conv state | `update_padded_kv_cache` (1x4/2x2), KDA stateful path, `tt/kimi_k3/kda_state.py` | needs adaptation | the framework has only run GQA k/v state (`state.kind: kv`); mixed per-layer state is untested in goldens, swap tests and the K.1 serving contract |
| CPU reference | HF parity, chunked == one-shot at 56k | transformers `glm5_next` | needs adaptation | HF's indexer and eager attention build dense [S, kv] tensors (about 74 GB of attention scores, 100 GB of indexer scores at 56k), so R.2/R.3 need a gather-based reference in the hooks plus a per-layer loader (the BF16 checkpoint, 643 GB, exceeds the 503 GB of host RAM) |
| MTP, vision | | | out of scope | skip, as MiMo did |

## Fit and spec shape

- bfp8 experts: about 11-12 MoE layers fit per chip; bfp4: about 20.
- Proposed subset `layers: "0-7"`: 3 dense + 5 MoE, 2 of them DSA, about 9 GiB of experts per chip at bfp8. The
  minimum that covers every block type is 0-3. A subset result is never reported as the full model.
- Block types: `kda_dense` (layers 0-2), `dsa_moe` (3, 7, ..., 43), `kda_moe` (every other layer from 4 to 44). Each
  also includes the two mHC steps.

## Main risks

1. Key pooling, the pool-level causal mask and the index compaction for `sparse_sdpa`: new work, probably one fork.
2. KDA has never run on a 1x4 or 2x2 mesh.
3. mHC has never been in a model; its 4x-wide residual feeds every block; the Sinkhorn needs fp32.
4. First use of MLA plus recurrent state in the framework: goldens, state metrics and the serving contract.
5. The CPU reference at 56k needs a custom sparse reference and per-layer loading.
6. Accuracy of the fused MoE kernel with 288 experts (known norm-ratio issues).

## Not verified

- Whether the KDA and sparse-MLA modules run on 1x4 or 2x2.
- Whether ClampedSiluGlu matches GLM's clamps exactly.
- Whether `unified_routed_expert_moe` accepts 72 local experts.
- Whether `ttnn.topk` handles width 288.
- How the vLLM/SGLang implementations compare with HF (for example an FP8 indexer).
- Real performance.

Short device probes after the intake would settle the first four.
