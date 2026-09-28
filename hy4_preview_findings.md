# Tencent Hy4 Preview: bring-up feasibility (initial findings)

Read-only study, 2026-09-28. No weights were downloaded, no device was used. Target: chunked prefill with the gated
bring-up framework (`models/demos/common/bringup`, overview `models/demos/common/bringup/docs/pipeline_design.html`)
on 4x Blackhole p150b (32 GB DRAM each, about 27 GiB usable per chip), mesh 1x4 or 2x2, FABRIC_2D, 56320 tokens in
5120-token chunks. Scratch copies of the fetched config, tensor shapes (`shapes.json`) and the transformers 5.17
`hy_v4` modeling code were in
`/tmp/claude-1211409778/-localdev-dnijemcevic-tt-metal2/667a454c-7ac4-4ac3-adac-4d20d7141327/scratchpad/hy4/`
(temporary; refetch if gone).

## Verdict

Feasible as a layer subset. The full model needs about 104 GiB per chip even with bfp4 experts (a 32-chip Galaxy
would fit it at about 12 GiB per chip). Layers 0-5 cover all three block types and fit at about 15-17 GiB per chip on
2x2 with bfp8 experts. Every component already exists somewhere in the repo: the attention and MoE geometry is the
same as GLM-5.2's in `models/demos/deepseek_v3_d_p`, and the extras (output gate, sinks, clamped SwiGLU) are each
implemented for another model there. Nothing needs a new op, so nothing is a candidate for the op-gen deferral.
Rough effort: 70-85 tasks (MiMo-V2.6 was 60; GLM-5.3-Flash estimated 85-110).

## Checkpoint

- `tencent/Hy4-preview`, revision `705d81ee51566a186d645b74c974d642ef2828fe` (last modified 2026-08-28). Not gated,
  Apache-2.0, `model_type: hy_v4`, class `HYV4ForCausalLM`. A separate `Hy4-preview-FP8` repo exists.
- Read: config.json, README.md, tokenizer_config.json, generation_config.json, model.safetensors.index.json (131
  shards, 2006 tensors) and the headers of all 131 shards. Chat template: `<｜hy_start/middle/end:opensource｜>`
  markers; `reasoning_effort` high or no_think.
- Stored dtypes: BF16, plus F32 for the hyper-connection parameters, sinks and router bias.
- No `trust_remote_code` files. The class is in transformers 5.17.0 (`models/hy_v4/`). The repo's `python_env` has
  5.12.1, which lacks it (the config says 5.16.2, which is not on PyPI): vendor `modeling_hy_v4.py` or use a second
  environment.
- Checkpoint tensor names differ from the model code (`linear_gate`, `learnable_sink_param`,
  `hc_attn_layer.hc_pre.hc_fn`, ...); the mapping in transformers' `conversion_mapping.py` is rename-only, no permutes.

## Architecture

About 780B total (769.9B backbone + 10.1B MTP; routed experts about 744B), about 47-49B active per token.

| Item | Value |
|---|---|
| Layers | 78: layer 0 dense FFN (intermediate 18432), layers 1-77 MoE; 1 MTP layer (skipped for prefill) |
| Hidden / vocab / context | 6144 / 120832 / 1M; untied embeddings; the reference runs the LM head in fp32 |
| Attention | gated DeepSeek Sparse Attention (DSA) on MLA in every layer: 64 heads, q_lora 2048, kv_lora 512, qk head 192 NoPE + 64 RoPE = 256, v head 256, scale 256^-0.5 = 1/16 |
| Attention extras | a learnable sink per head (an extra softmax logit, not scaled); an elementwise output gate `sigmoid(gate_proj(h))`, gate_proj 6144 to 64x256 |
| Indexer | 32 heads x 128, top-2048 keys. Queries `wq_b(q_a latent)`; keys `LayerNorm(wk(h))` with bias; scores ReLU(q.k) weighted by `weights_proj`, scaled by 32^-0.5 * 128^-0.5; RoPE on the last 64 of 128 dims |
| Indexer sharing | only 21 "full" layers run the indexer ({0, 1, 5, 9, ..., 77}); the other 57 reuse the latest full layer's top-k and have no indexer weights |
| RoPE | default, theta 1e7, no YaRN, rotate_half in both MLA and indexer |
| MoE | 256 routed experts, top-8, 1 shared expert, intermediate 2048; sigmoid router + correction bias (noaux_tc), 1 group, normalised top-k, scale 2.827 |
| Expert activation | routed: `silu(min(g, 10)) * clamp(u, +-10)`; shared expert unclamped; gate and up stored fused as [256, 4096, 6144] |
| Residual (iHC) | 4 residual streams. Per sublayer, in fp32: `mixes = flat(4x6144) @ fn[8x24576]^T * rsqrt(mean(flat^2))`; `pre = sigmoid(.)` weights the sum of the 4 streams into the sublayer input; `post = 2*sigmoid(.)` sets `stream_j += post_j * y`. DeepSeek-V4's mHC without the Sinkhorn mixing matrix. A final head collapses the streams, then RMSNorm and the LM head |
| Multimodal | none |

Per MoE layer: 9.66B routed experts, 266M attention, 9.4M indexer (full layers only), 38M shared expert.

Memory per chip, experts over 4 chips, other weights split 4 ways:

| Expert format | Routed experts per MoE layer | Full model |
|---|---|---|
| bf16 | 4.51 GiB | 354 GiB |
| bfp8 | 2.39 GiB | 191 GiB |
| bfp4 | 1.27 GiB | 104 GiB |

State at 56320 tokens: MLA latent cache (576 values per token, bf16) 62 MiB per layer, 4.7 GiB for all 78 layers;
indexer key cache 13.75 MiB per full layer, 0.28 GiB for all 21. (The HF reference caches expanded K/V, 3.4 GiB per
layer; the device path would not.) The 4-stream residual is about 252 MB per 5120-token chunk in bf16.

## What the repo has

- No Hunyuan LLM code; only `models/experimental/hunyuan_image_3_0` (an image model).
- Closest match: `models/demos/deepseek_v3_d_p` running GLM-5.2. `reference/glm_5_2_config.py` matches Hy4 on hidden
  size, heads, every MLA dimension, the indexer dimensions and top-k, experts and top-k, and the layer count (78). It
  already implements indexer sharing (`indexer_types`, an explicit full/shared list). Differences: GLM-5.2 has 3
  dense layers of 12288 (Hy4: 1 of 18432), vocab 154880, RoPE theta 8e6, a different full/shared pattern. Its
  serving adapters raise NotImplementedError (`tt/runners/adapters/sparse_mla.py`); only reference-parity tests are
  wired.
- From sibling models in the same directory: Kimi-K3's output gate (`tt/mla/mla.py:_output_gate`, flag
  `mla_use_output_gate`); DeepSeek-V4's sinks in SDPA (`tt/mla/sliding_window_attention.py`), mHC
  (`tt/mhc/tt_mhc.py`, op `deepseek_prefill.mhc_split_sinkhorn`) and the `ClampedSiluGlu` activation at L=10 in
  `unified_routed_expert_ffn` (the bring-up fork has it too).
- Mesh evidence: production MLA tests are 8x4 only (`test_mla.py`), but the sparse-MLA tests have 4-chip shapes
  (`tests/sparse_mla/sparse_mla_mesh.py`: (1,4) and (2,2); the perf test uses QuietBox (2,2) with SP2xTP2). The
  prefill-block conftest wants FABRIC_2D_TORUS_X for 1x4 and FABRIC_2D for 2x2. MoE dispatch on 2x2 FABRIC_2D is
  proven by `models/demos/mimo_v2_6_d_p_2x2` (known issue 90: call `post_combine_reduce` directly). 2x2 FABRIC_2D is
  the configuration with evidence behind it.

## Components

| Component | Needs | Existing code | Status | Notes |
|---|---|---|---|---|
| Embedding | 120832x6144 | `models/demos/deepseek_v3_d_p/tt/tt_parallel_embedding.py` | covered | |
| iHC expand / pre-mix / post-residual | 4 streams, fp32 8-way linear over 24576, rsqrt norm, sigmoid, weighted sum, `post*y + res` | `models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py` | needs adaptation | drop the Sinkhorn and mixing matrix; magnitude 2 and eps; composite ops (matmul N=8, sigmoid, multiply, add) at HiFi4 with fp32 accumulation; a partial reduce across TP if the residual is split along hidden; the framework's residual tests need to cover 4 streams |
| iHC head + final norm | 4-way sigmoid collapse, then RMSNorm | same module + `ttnn.rms_norm` / `ttnn.bringup.rms_norm` | needs adaptation | small |
| q_a / kv_a projections and norms | standard MLA | `models/demos/deepseek_v3_d_p/tt/mla/mla.py` (ttMLA) | covered | permute the rope columns on the host (rotate_half to interleaved) |
| Indexer (full layers) | wq_b, wk + LayerNorm with bias, weights_proj, ReLU scores, top-2048 over up to 56k keys | `tt/mla/indexer.py` (TtIndexer), `ttnn.experimental.ring_indexer_score_dsa`, `topk_large_indices` | needs adaptation | RoPE on the last 64 dims: permute the wq_b/wk rows and the LayerNorm weight/bias on the host; the half-split path (`index_rope_interleave=False`) exists; keep index keys in bf16 (MiniMax-M3 lesson on top-k stability) |
| Indexer sharing | reuse top-k on 57 layers | GLM-5.2 `indexer_types` | covered | pass Hy4's explicit list |
| Sparse attention | top-k gather, absorbed 576/512, 64 heads, sinks | `ttnn.transformer.sparse_sdpa` (has `attention_sink`) | needs adaptation | the op supports sinks but ttMLA does not pass them; pass sink and an explicit scale of 1/16 (not the op's 576^-0.5 default); the first ~2048 tokens are effectively dense |
| Output gate + o_proj | `sigmoid(h @ Wg)` per head (16384) x attention output | Kimi-K3 `_output_gate` | covered / needs adaptation | combine with the sparse path |
| KV + index-key cache (chunked) | latent 576 + index keys, 5120-token chunks | `update_padded_kv_cache`, block-cyclic SP cache (GLM `MlaKvCaches`) | covered | test on SP=2 |
| Dense FFN (layer 0) | SwiGLU 18432 | `tt_ffn.py` / `ttnn.linear` | covered | |
| Router | sigmoid + bias, top-8 of 256, normalise, x2.827 | `models/demos/mimo_v2_6_d_p/tt/router.py` or `tt_moe_gate_prefill.py` | covered | use the SFPU add + `ttnn.topk` path (known issue: `moe_grouped_topk` selects on TF32 keys) |
| Routed experts | 256 experts over 4 chips, clamped SiLU-GLU at L=10 | DeepSeek dispatch/combine forks, `unified_routed_expert_ffn` with `ClampedSiluGlu`, `models/demos/mimo_v2_6_d_p_2x2/tt/experts.py` | covered | split the fused gate/up tensor on the host |
| Shared expert | SwiGLU 2048, no clamp | `tt_shared_expert.py` | covered | |
| LM head | 120832, fp32 in the reference | `tt_lm_head.py` | covered | last token only; bf16 or fp32 on device, or on the host |
| MTP | | | skip | not part of prefill |

The largest open item is passing sinks into the sparse attention, which is wiring, not a new op.

## Fit and spec shape

- Layers 0-5 cover every block type: `dense_full` (layer 0: dense FFN, own indexer), `moe_full` (layers 1 and 5: MoE,
  own indexer), `moe_shared` (layers 2-4: MoE, reused top-k).
- On 2x2 (experts over all 4 chips, attention SP2xTP2) one MoE layer costs about 2.7 GiB per chip with bfp8 experts,
  about 1.6 GiB with bfp4. Layers 0-5 with bfp8 experts: about 15-17 GiB per chip including embedding, LM head, KV
  and activations. Maximum depth: about 8 layers with bfp8 experts, 13-14 with bfp4.
- A subset result is never reported as the full model.

## Main risks

1. Integrating `deepseek_v3_d_p`'s ttMLA and TtIndexer, which are large, Galaxy-tuned and built around SPxTP, into the
   framework's component and swap steps. The sparse path on (2,2) has only run in isolated tests.
2. Indexer top-k precision (bf16 or bfp8 index keys, 56k-wide selection): the tests need a selection-overlap metric.
3. Sinks with DSA precision; the MiMo known issues show sinks can mask attention errors in whole-output metrics.
4. iHC precision: the reference forces fp32 across 24576-wide reductions.
5. CPU goldens at 56k: the HF code runs eager attention only, and 64xSxS scores at 56k would be about 812 GB. The
   reference needs a chunked sparse CPU path (the repo has `reference/cpu_deepseek_v32` and
   `tests/sparse_mla/sparse_mla_reference.py`); HF parity can only be checked at short lengths.
6. The reference needs transformers >= 5.17.
7. Download: layers 0-5 span 35 shards, about 415 GB as whole shards (about 105 GB fetching only the needed tensors
   with HTTP range requests). Disk (1.9 TB free) and RAM (503 GB) are fine.

## Not verified

- Whether ttMLA with the indexer runs end to end at (2,2) and (1,4) on FABRIC_2D (1x4 wants torus-X in the conftest).
- Real activation and dispatch-buffer memory at 5120-token chunks.
- Whether sparse attention's 128-wide chunking and its "multiple of 32 heads" requirement hold after q is
  redistributed across TP.
- How closely the device matches the reference.

## Key paths

- `models/demos/deepseek_v3_d_p/reference/glm_5_2_config.py`
- `models/demos/deepseek_v3_d_p/tt/mla/{mla.py,indexer.py,sliding_window_attention.py}`
- `models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py`
- `models/demos/deepseek_v3_d_p/tests/sparse_mla/sparse_mla_mesh.py`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sparse_sdpa.hpp`
- `ttnn/cpp/ttnn/operations/experimental/indexer_score/`
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp`
- `models/demos/mimo_v2_6_d_p_2x2/tt/experts.py`
- `models/demos/mimo_v2_6_d_p/tt/router.py`
