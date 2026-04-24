# Generalized MoE Module — Plan & Learnings

## Index

- [Goal](#goal)
- [1. Model Registry](#1-model-registry)
- [2. Process Notes & Learnings](#2-process-notes-learnings)
- [3. Cross-Model MoE Parameter Comparison](#3-cross-model-moe-parameter-comparison-all-verified)
- [4. Architecture Family Classification](#4-architecture-family-classification)
- [5. Generalization Implications](#5-generalization-implications)
- [6. Next Steps](#6-next-steps)
- [Appendix A: Model MoE Architectures (Detailed)](#appendix-a-model-moe-architectures-detailed)
  - [A.1 GPT-OSS](#a1-gpt-oss-moe-architecture-verified)
  - [A.2 DeepSeek V3](#a2-deepseek-v3-moe-architecture-verified)
  - [A.3 GLM-5](#a3-glm-5-moe-architecture-verified) — DeepSeek family
  - [A.4 Kimi K2.5](#a4-kimi-k25-moe-architecture-verified) — DeepSeek family
  - [A.5 Ling-1T](#a5-ling-1t-moe-architecture-verified) — DeepSeek family
  - [A.6 GLM-4.7](#a6-glm-47-moe-architecture-verified) — DeepSeek family (confirmed from modeling code)
  - [A.7 Qwen3 MoE](#a7-qwen3-moe-moe-architecture-verified) — Standard softmax
  - [A.8 Qwen3.5 MoE (397B)](#a8-qwen35-moe-moe-architecture-verified) — Standard softmax
  - [A.9 Qwen3.5 35B](#a9-qwen35-35b-a3b-moe-architecture-verified) — Standard softmax
  - [A.10 Qwen3-Omni 30B](#a10-qwen3-omni-30b-a3b-moe-architecture-verified) — Standard softmax
  - [A.11 Mistral Large 3](#a11-mistral-large-3-675b-moe-architecture-verified) — Custom
  - [A.12 Gemma 4 (optional)](#a12-gemma-4-26b-a4b-moe-architecture-verified-optional) — Custom
  - [A.13 DeepSeek OCR](#a13-deepseek-ocr-moe-architecture-verified) — Legacy/small
  - [A.14 DeepSeek V4 Flash](#a14-deepseek-v4-flash--moe-architecture-verified) — DS V4 family
  - [A.15 DeepSeek V4 Pro](#a15-deepseek-v4-pro--moe-architecture-verified) — DS V4 family
- [Appendix B: Existing TT-Metal Implementations](#appendix-b-existing-tt-metal-implementations-comparison-as-of-main437da8d3796)
- [Appendix C: Verification of unverified_moe_info.md](#appendix-c-verification-of-unverified_moe_infomd)

---

## Goal

Unify the two near-duplicate MoE implementations (`moe_compute` for DeepSeek, `moe_gpt` for GPT-OSS) into a single generalized module that supports multiple MoE model families.

---

## 1. Model Registry

All verified from config.json and/or modeling code.

| Model | HuggingFace ID |
|-------|---------------|
| DeepSeek V3 | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| GPT-OSS | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) / [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| GLM-4.7 | [zai-org/GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) |
| GLM-5 | [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) |
| Kimi K2.5 | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) |
| Qwen 3.5 (MoE) | [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| Qwen3 (MoE) | [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| DeepSeek OCR | [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) |
| Mistral Large 3 (675B) | [mistralai/Mistral-Large-3-675B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) |
| Ling-MoE 1T | [inclusionAI/Ling-1T](https://huggingface.co/inclusionAI/Ling-1T) |
| Qwen3-Omni 30B-A3B | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| Qwen3.5 35B-A3B | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Gemma 4 26B-A4B (optional) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| DeepSeek V4 Flash | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) / [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) |
| DeepSeek V4 Pro | [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) / [deepseek-ai/DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) |

### Notes
- **GLM-4.7** uses the same `glm4_moe` architecture as GLM-4.5 but with extended context (202752 vs 131072). Also has a Flash variant: `zai-org/GLM-4.7-Flash` (`glm4_moe_lite`).
- **GLM-4.5** MoE params match GLM-4.7 exactly (same `glm4_moe` arch, same hidden/expert/routing values). Not a separate plan entry.
- **Mistral Large 3 (675B)** is the correct model (not "Mistral 3.2 Large"). From `params.json`: hidden=7168, expert_hidden_dim=**4096**, 128 experts, 1 shared, K=**4**. Mistral Small 3.2 is a 24B dense model (no MoE).
- **Qwen n_shared_experts**: No Qwen config contains an explicit `n_shared_experts` or `num_shared_experts` field. Shared expert count is inferred: present if `shared_expert_intermediate_size > 0`, absent if field is 0 or missing. Marked "(inferred)" in the comparison table.
- **DeepSeek V4 Flash == Flash-Base config**: For both Flash and Pro, the non-base and Base variants have byte-for-byte identical config.json files. A single architecture section covers both. `inference/model.py` is present only in the non-base repos (Flash, Pro) — the Base repos contain no Python files.

---

## 2. Process Notes & Learnings

### Gathering MoE info from HuggingFace — what works
1. `hf download <model-id> config.json` — fastest way to get all architecture params. For multimodal models, MoE params are in `text_config` or `language_config`.
2. `hf download <model-id> modeling_*.py` — custom modeling code has the exact MoE block implementation (only needed for custom activations/routing).
3. Model card (README.md) — has high-level info but rarely MoE-specific dimensions.
4. Mistral uses `params.json` instead of `config.json`.
5. Some models (Qwen3, Qwen3.5) are in transformers proper — no custom modeling_*.py to download.

---

## 3. Cross-Model MoE Parameter Comparison (ALL VERIFIED)

| **Model** | **hidden_size** | **moe_intermediate** | **shared_expert_interm** | **n_routed_experts** | **n_shared_experts** | **K (top-k)** | **activation** | **scoring_func** | **topk_method** | **n_group/topk_group** | **scaling_factor** | **expert_bias** | **router_bias** | **first_k_dense** | **num_layers** | **parallel dense** | **base_arch** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GPT-OSS 120B** | 2880 | 2880 | — | 128 | 0 | 4 | custom GELU-gated | softmax | simple | —/— | 1.0 | Yes | Yes | all MoE | 36 | No | GPT-OSS |
| **DeepSeek V3** | 7168 | 2048 | 2048 | 256 | 1 | 8 | SiLU/SwiGLU | `sigmoid` | noaux_tc | 8/4 | 2.5 | No | correction | 3 | 61 | No | DS V3 |
| **DS V4 Flash** | 4096 | 2048 | 2048 | 256 | 1 | 6 | SiLU/SwiGLU (swiglu_limit=10, routed only) | `sqrtsoftplus` (code) | noaux_tc; hash L0–2 (code) | —/— | 1.5 | No | correction (code) | 0 | 43 | No | DS V4 |
| **DS V4 Pro** | 7168 | 3072 | 3072 | 384 | 1 | 6 | SiLU/SwiGLU (swiglu_limit=10, routed only) | `sqrtsoftplus` (code) | noaux_tc; hash L0–2 (code) | —/— | 2.5 | No | correction (code) | 0 | 61 | No | DS V4 |
| **GLM-5** | 6144 | 2048 | 2048 | 256 | 1 | 8 | SiLU/SwiGLU | `sigmoid` | noaux_tc | 1/1 | 2.5 | No | correction | 3 | 78 | No | DS V3-like |
| **Kimi K2.5** | 7168 | 2048 | 2048 | 384 | 1 | 8 | SiLU/SwiGLU | `sigmoid` | noaux_tc | 1/1 | 2.827 | No | correction | 1 | 61 | No | DS V3-like |
| **Ling 1T** | 8192 | 2048 | 2048 | 256 | 1 | 8 | SiLU/SwiGLU | `sigmoid` | group_limited_topk (code) | 8/4 | 2.5 | No | correction (bias-enabled) | 4 | 80 | No | DS V3-like |
| **GLM-4.7** | 5120 | 1536 | 1536 | 160 | 1 | 8 | SiLU/SwiGLU | `sigmoid` (code) | noaux_tc-style (code) | 1/1 | 2.5 | No | correction (code) | 3 | 92 | No | DS V3-like |
| **Qwen3 235B** | 4096 | 1536 | — | 128 | 0 (no field) | 8 | SiLU/SwiGLU | softmax (code) | simple top-k (code) | —/— | — | No | No | all MoE | 94 | No | Qwen3 |
| **Qwen3.5 397B** | 4096 | 1024 | 1024 | 512 | 1 (inferred) | 10 | SiLU/SwiGLU | softmax (code) | simple top-k (code) | —/— | — | No | No | all MoE | 60 | No | Qwen3.5 |
| **Qwen3.5 35B** | 2048 | 512 | 512 | 256 | 1 (inferred) | 8 | SiLU/SwiGLU | softmax (code) | simple top-k (code) | —/— | — | No | No | all MoE | 40 | No | Qwen3.5 |
| **Qwen3-Omni Thinker** | 2048 | 768 | — | 128 | 0 (inferred, size=0) | 8 | SiLU/SwiGLU | softmax (code) | simple top-k (code) | —/— | — | No | No | all MoE | 48 | No | Qwen3-Omni |
| **Qwen3-Omni Talker** | 1024 | 384 | 768 | 128 | 1 (inferred, size=768) | 6 | SiLU/SwiGLU | softmax (code) | simple top-k (code) | —/— | — | No | No | all MoE | 20 | No | Qwen3-Omni |
| **Mistral Large 3** | 7168 | 4096 | 4096 (from vLLM) | 128 | 1 | 4 | SiLU/SwiGLU (code) | softmax (code) | simple top-k (code) | 1/1 | 1.0 | No | No | 3 | 61 | No | Mistral |
| **Gemma 4 26B** | 2816 | 704 | — (parallel dense) | 128 | 0 (parallel dense) | 8 | **GELU/SwiGLU** | softmax | simple+per_expert_scale | —/— | per-expert learned | No | No (has learned scale) | all MoE | 30 | **Yes** | Gemma4 |
| **DS-OCR** | 1280 | 896 | 1792 | 64 | 2 | 6 | SiLU/SwiGLU | softmax (V2 default) | `greedy` | 1/1 | 1.0 (V2 default) | No | No | 1 | 12 | No | DS V2 |

**Notes on the table above:**
- Qwen3.5 models (397B, 35B) use a **sigmoid gate on shared expert output** before adding to routed output (see §A.8). This is a unique architectural detail for generalization.
- Qwen-family models (Qwen3, Qwen3.5, Qwen3-Omni) show "—" for scaling_factor because their configs have no `routed_scaling_factor` field. Their routing uses `norm_topk_prob` normalization instead, effectively equivalent to scaling=1.0 after normalization.
- DS V4 `sqrtsoftplus` scoring: `sqrt(F.softplus(logits))` (model.py L571) — a new function, neither sigmoid nor softmax. After top-k selection, weights are normalized to sum to 1 (since it's not softmax), then multiplied by `routed_scaling_factor`.
- DS V4 hash routing: for the first `num_hash_layers=3` layers, expert indices come from a learned lookup table `tid2eid[token_id]` (model.py L577), not from gate scores. Routing weights are still computed from gate scores for these layers. The score correction bias is absent in hash layers.
- DS V4 `swiglu_limit=10.0` clamps gate (max=10) and up (min=−10, max=10) in **routed** experts only (model.py L600–602). The shared expert is created without `swiglu_limit` (L627–628) and has no clamping.

---

## 4. Architecture Family Classification

Based on the verified configs, the models cluster into clear families:

### Family 1: DeepSeek V3-style (sigmoid + noaux_tc + SwiGLU)
- **DeepSeek V3**: The original. Group routing (8/4), scaling=2.5.
- **GLM-5**: Nearly identical MoE. No grouping (1/1), but same sigmoid+noaux_tc routing.
- **GLM-4.7**: Same sigmoid + grouped-top-k + bias-correction routing confirmed from modeling code (matches the noaux_tc algorithm, though the config has no `topk_method` field). n_group=1/topk_group=1 (single group). Uses fused gate_up_proj+chunk(2) layout.
- **Kimi K2.5**: Built on DS V3. 384 experts, scaling=2.827, no grouping.
- **Ling-1T**: Very close to DS V3. Same group routing (8/4), same scaling=2.5.

### Family 2: Standard softmax routing + SwiGLU (confirmed from modeling code)
- **Qwen3 (235B)**: Simple config, no grouping, no shared experts.
- **Qwen3.5 (397B)**: 512 experts, K=10, softmax routing.
- **Qwen3.5 (35B)**: Smaller Qwen3.5, 256 experts, K=8, same MoE arch.
- **Qwen3-Omni Thinker (30B)**: 128 experts, K=8, no shared experts, same Qwen3 family routing.
- **Qwen3-Omni Talker (30B)**: 128 experts, K=6, 1 shared expert (768 intermediate), same routing.

### Family 3: Custom architectures
- **GPT-OSS**: Unique activation (GELU-gated with clamping), has bias everywhere, fused interleaved gate_up, softmax routing.
- **Mistral Large 3**: SiLU/SwiGLU + softmax routing (confirmed from code), K=4, large expert_hidden_dim=4096. Implemented as DeepseekV3 subclass in vLLM. Separate w1/w2/w3 projections.
- **Gemma 4 (optional)**: GELU activation (not SiLU), parallel dense+MoE (not replacement), per-expert learned scaling, fused chunked gate_up. Unique parallel dense MLP + MoE architecture.

### Family 4: Legacy/Small
- **DeepSeek OCR**: Based on DS V2, greedy routing, very small (1280 hidden, 12 layers).

### Family 5: DeepSeek V4 (sqrtsoftplus + noaux_tc + hash layers)
- **DS V4 Flash**: 256 experts, K=6, all layers MoE, `sqrtsoftplus` scoring, `noaux_tc` for layers 3+, hash routing for layers 0–2, SwiGLU with clamping (limit=10) on routed experts only.
- **DS V4 Pro**: 384 experts, K=6, same architecture as Flash at larger scale (hidden=7168, intermediate=3072). scaling=2.5 (same as DS V3).

Key distinctions from DS V3: different scoring function (sqrtsoftplus vs sigmoid), hash routing replaces learned routing for first 3 layers, no expert grouping (no n_group/topk_group), all layers are MoE (no first_k_dense_replace), SwiGLU clamping on routed experts (not shared). Architecture class is `DeepseekV4ForCausalLM`.

---

## 5. Generalization Implications

### 5.1 Parameters that MUST be configurable
1. `hidden_size` — ranges from 1024 (Qwen3-Omni Talker) to 8192 (Ling-1T)
2. `moe_intermediate_size` — ranges from 384 to 4096 (Qwen3-Omni talker: 384, Mistral: 4096)
3. `n_routed_experts` — ranges from 64 to 512. All values are divisible by 8, 16, and 32 — ring distribution can always be done exactly across standard device counts.
4. `n_shared_experts` — 0, 1, or 2
5. `num_experts_per_tok` (K) — 4, 6, 8, or 10. Note: K=6 appears in DS-OCR, Qwen3-Omni Talker, DS V4 Flash, and DS V4 Pro (4 models). Neither existing TT-Metal implementation (K=8 or K=4) handles K=6, so new kernel work is needed if any of these are in scope for v1.
6. `shared_expert_intermediate_size` — usually `moe_intermediate * n_shared`, but Qwen3.5 has an explicit field. In all Qwen3/Qwen3.5 models, the shared expert is always 1 MLP with `intermediate_size = shared_expert_intermediate_size` (not N smaller experts). The Qwen3-Omni Talker's 768 = 2×384 is a single wider expert, not two experts.
7. `shared_expert_gate` (on/off) — Qwen3.5 (397B, 35B) applies `sigmoid(gate(x)) * shared_output` before adding to routed output (modeling code L788). This requires an additional weight: `shared_expert_gate` projection `[hidden_size, 1]`. Other models add shared expert output directly.
8. `routed_scaling_factor` — 1.0, 2.5, or 2.827
9. Expert bias (on/off)
10. Router bias / bias correction (on/off)
11. `moe_layer_freq` — all current models use 1 (every layer after `first_k_dense_replace` is MoE). Listed for completeness; if a future model uses freq > 1 (only every Nth layer is MoE), the interface needs to handle it.

### 5.2 Activation function strategy
- **14/16 models use SiLU/SwiGLU** (all confirmed from config or modeling code): `down(silu(gate(x)) * up(x))`; DS V4 Flash and Pro also apply clamping to gate/up in routed experts (`swiglu_limit=10.0`) but not in the shared expert
- **Gemma 4**: GELU/SwiGLU — same gate/up/down pattern but with `gelu_pytorch_tanh` instead of SiLU
- **GPT-OSS**: custom GELU-gated with clamping and shift — unique pattern (`(up+1) * gate * sigmoid(1.702*gate)`, limit=7.0)
- Recommendation: parameterize the activation function (SiLU vs GELU), implement GPT-OSS as a special variant; `swiglu_limit` clamping should be an optional configurable parameter

### 5.3 Routing strategy
- **DeepSeek V3 family (5 rows: DS V3, GLM-5, GLM-4.7, Kimi K2.5, Ling-1T)**: sigmoid scoring + group-based top-k + bias correction + scaling (GLM-4.7 confirmed from modeling code)
- **DeepSeek V4 family (2 rows: DS V4 Flash, DS V4 Pro)**: sqrtsoftplus scoring + simple top-k + bias correction + scaling; first `num_hash_layers=3` layers use hash routing (token-ID lookup table) instead of learned gate scores for expert selection
- **Standard softmax family (5 rows: Qwen3, Qwen3.5×2, Qwen3-Omni×2)**: softmax top-k (confirmed from modeling code)
- **GPT-OSS**: softmax top-k with router bias
- **Gemma 4**: softmax top-k with per-expert learned scaling
- **Mistral Large 3**: softmax top-k (confirmed from mistral-inference moe.py)
- **DS-OCR**: softmax scoring (V2 default) + greedy top-k selection
- Recommendation: implement sigmoid+group, sqrtsoftplus+simple, and softmax as three routing modes; hash routing is a compile-time variant of the first N layers

### 5.4 Expert projection layout
- **11/14 rows**: separate gate_proj + up_proj + down_proj (no bias)
- **GPT-OSS**: fused interleaved gate_up_proj + down_proj (with bias) — split via even/odd indices
- **GLM-4.7**: fused chunked gate_up_proj + down_proj (no bias) — split via `.chunk(2)` (same as Gemma 4)
- **Gemma 4**: fused chunked gate_up_proj + down_proj (no bias) — split via `.chunk(2)` (first half/second half)
- Recommendation: canonical internal layout is separate projections; fused layout handled at weight loading

### 5.5 Dense+MoE parallelism
- **13/14 rows**: MoE *replaces* the FFN (with optional shared expert added)
- **Gemma 4**: MoE runs *in parallel* with a full dense MLP, outputs summed — unique architecture
- Recommendation: support both modes (replacement vs parallel) as a config flag

### 5.6 Common tile sizes for ring distribution
Using tile_size=32:

| Model | W0/W1 tiles (hidden→intermediate) | W2 tiles (intermediate→hidden) |
|-------|-----------------------------------|-------------------------------|
| DeepSeek V3 | 7168/32=224 × 2048/32=64 | 2048/32=64 × 7168/32=224 |
| GPT-OSS | 2880/32=90 × 2880/32=90 | 2880/32=90 × 2880/32=90 |
| GLM-4.7 | 5120/32=160 × 1536/32=48 | 48 × 160 |
| GLM-5 | 6144/32=192 × 2048/32=64 | 64 × 192 |
| Kimi K2.5 | 224 × 64 | 64 × 224 |
| Qwen3.5 397B | 4096/32=128 × 1024/32=32 | 32 × 128 |
| Qwen3.5 35B | 2048/32=64 × 512/32=16 | 16 × 64 |
| Qwen3 235B | 128 × 1536/32=48 | 48 × 128 |
| Qwen3-Omni Thinker | 2048/32=64 × 768/32=24 | 24 × 64 |
| Qwen3-Omni Talker | 1024/32=32 × 384/32=12 | 12 × 32 |
| DS-OCR | 1280/32=40 × 896/32=28 | 28 × 40 |
| Mistral L3 | 7168/32=224 × 4096/32=128 | 128 × 224 |
| Ling-1T | 8192/32=256 × 64 | 64 × 256 |
| Gemma 4 26B | 2816/32=88 × 704/32=22 | 22 × 88 |
| DS V4 Flash | 4096/32=128 × 2048/32=64 | 64 × 128 |
| DS V4 Pro | 7168/32=224 × 3072/32=96 | 96 × 224 |

---

## 6. Next Steps

- [ ] Identify the minimal set of parameters needed for the unified MoE interface
- [ ] Design the unified operation interface (C++ struct for MoE config)
- [ ] Determine which routing/activation variants to support in first version
- [ ] Plan the kernel generalization (parameterize tile counts, activation, bias)
- [ ] Prototype the generalized module
- [ ] Test against existing DeepSeek and GPT-OSS implementations for regression

### Open design questions
- For Qwen3.5 397B/35B, should `n_shared_experts` be a first-class config param only when present in JSON, or always derived from `shared_expert_intermediate_size` in HF modeling code?
- For DeepSeek-OCR, is `greedy` routing (softmax scoring + greedy top-k) a first-class mode for the generalized op, or out of scope for v1?
- Should Qwen3-Omni Talker (different K=6, shared width 2× routed) be in-scope for the same primitive as the "main LLM" MoE, or a second profile? K=6 requires new kernel work — neither existing TT-Metal implementation handles it.

---

## Appendix A: Model MoE Architectures (Detailed)

### A.1 GPT-OSS — MoE Architecture (Verified)

**Source:** `openai/gpt-oss-120b` config.json + transformers `modeling_gpt_oss.py`
**Downloaded to:** `plans/hf_model_cards/gpt-oss-120b/`, `plans/hf_model_cards/gpt-oss-20b/`

#### Source References
- `hf_model_cards/gpt-oss-120b/config.json`: hidden_size (L11), intermediate_size (L14), num_local_experts (L59), num_experts_per_tok (L56), hidden_act (L10), router_aux_loss_coef (L81), swiglu_limit (L83), experts_per_token (L8), num_hidden_layers (L57)
- `hf_model_cards/gpt-oss-20b/config.json`: num_local_experts=32 (L47), num_hidden_layers=24 (L45) — differences from 120B
- `hf_model_cards/gpt-oss-120b/modeling_gpt_oss.py` (from `transformers/models/gpt_oss/`): GptOssExperts class (L69), gate_up_proj+bias (L75-76), down_proj+bias (L77-78), alpha=1.702 (L79), limit=7.0 (L80), custom activation `_apply_gate` (L82-86), expert forward (L90-110); GptOssTopKRouter class (L117), weight+bias (L123-124), softmax scoring (L129); GptOssMLP class (L134) combining router+experts (L137-138)


#### Downloaded Files
- `config.json` — `hf download openai/gpt-oss-120b config.json`
- `README.md` — `hf download openai/gpt-oss-120b README.md`
- `modeling_gpt_oss.py` — https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
- (20B) `config.json` — `hf download openai/gpt-oss-20b config.json`
#### Config Parameters (from config.json)

| Parameter | 120B | 20B | Description |
|-----------|------|-----|-------------|
| `hidden_size` | **2880** | **2880** | Model hidden dimension |
| `intermediate_size` | **2880** | **2880** | Per-expert intermediate dim (same as hidden!) |
| `num_local_experts` | **128** | **32** | Number of routed experts |
| `num_experts_per_tok` | **4** | **4** | Top-K experts selected per token |
| `hidden_act` | `silu` | silu | Config says silu but actual activation is custom (see below) |
| shared experts | **0** (none) | **0** (none) | No shared experts |
| `router_aux_loss_coef` | 0.9 | 0.9 | Load balancing loss coefficient |
| `swiglu_limit` | 7.0 | 7.0 | Clamping limit for activation |
| `num_hidden_layers` | 36 | 24 | Total transformer layers |
| Total / Active params | 117B / 5.1B | 21B / 3.6B | |

#### MoE Block Structure (from modeling_gpt_oss.py)

```
GptOssMLP
├── router: GptOssTopKRouter
│   ├── weight: Parameter(num_experts=128, hidden_dim=2880)
│   ├── bias: Parameter(128)                               # has bias, unlike DeepSeek
│   └── selection: simple softmax top-k (no grouping, no bias correction)
│
└── experts: GptOssExperts (128 routed experts, NO shared experts)
    ├── gate_up_proj: Parameter(128, 2880, 2*2880=5760)    # fused & interleaved gate+up
    ├── gate_up_proj_bias: Parameter(128, 5760)            # has bias!
    ├── down_proj: Parameter(128, 2880, 2880)
    ├── down_proj_bias: Parameter(128, 2880)               # has bias!
    ├── alpha: 1.702                                        # activation constant
    └── limit: 7.0                                          # clamping limit
```

#### Activation Function — NOT standard SwiGLU

GPT-OSS uses a **custom gated activation** that differs significantly from DeepSeek's SwiGLU:

```python
# GPT-OSS activation (interleaved gate/up layout):
gate, up = gate_up[..., ::2], gate_up[..., 1::2]   # de-interleave
gate = gate.clamp(max=7.0)
up = up.clamp(min=-7.0, max=7.0)
glu = gate * sigmoid(gate * 1.702)                   # GELU-like gating
output = (up + 1) * glu                              # shift-and-gate

# DeepSeek activation (separate gate/up):
output = silu(gate_proj(x)) * up_proj(x)             # standard SwiGLU
```

Key differences:
1. **Interleaved layout**: gate and up values alternate in the fused projection (even/odd indices)
2. **GELU-variant gating**: `gate * sigmoid(gate * 1.702)` approximates GELU, not SiLU
3. **Shift-and-gate**: `(up + 1) * glu` instead of simple multiplication
4. **Clamping**: both gate and up values are clamped to [-7, 7]

#### Router — Simple Softmax Top-K

```python
# GPT-OSS router:
logits = linear(hidden_states, weight, bias)         # [N, 128]
top_values, top_indices = topk(logits, k=4)
scores = softmax(top_values)                          # normalize over selected experts only

# DeepSeek router:
logits = linear(hidden_states, weight)                # [N, 256], no bias
scores = sigmoid(logits)                              # sigmoid, not softmax
# + group-based selection + bias correction + routed_scaling_factor
```

#### Forward Pass Flow

```
input x: [batch, seq_len, 2880]
    │
    ├──► GptOssTopKRouter(x) → router_scores [N, 4], router_indices [N, 4]
    │     1. logits = x @ weight.T + bias               # [N, 128]
    │     2. top-4 logits selected
    │     3. scores = softmax(top-4 values)              # [N, 4]
    │
    └──► GptOssExperts(x, indices, scores)
          1. For each active expert:
          2.   gate_up = tokens @ gate_up_proj[expert] + bias   # [tokens, 5760]
          3.   output = custom_activation(gate_up)               # [tokens, 2880]
          4.   out = output @ down_proj[expert] + bias           # [tokens, 2880]
          5.   weighted by routing score, accumulated
          → output [batch, seq_len, 2880]

(NO shared expert addition — output is purely from routed experts)
```

#### Weight Shapes (per MoE layer, 120B)

| Component | Shape | Has Bias | Notes |
|-----------|-------|----------|-------|
| Router weight | [128, 2880] | Yes [128] | Softmax scoring |
| Expert gate_up_proj | [128, 2880, 5760] | Yes [128, 5760] | Fused+interleaved gate & up |
| Expert down_proj | [128, 2880, 2880] | Yes [128, 2880] | |

---

### A.2 DeepSeek V3 — MoE Architecture (Verified)

**Source:** `deepseek-ai/DeepSeek-V3` config.json + modeling_deepseek.py
**Downloaded to:** `plans/hf_model_cards/deepseek-v3/`

#### Source References
- `hf_model_cards/deepseek-v3/config.json`: hidden_size (L17), moe_intermediate_size (L23), n_routed_experts (L26), n_shared_experts (L27), num_experts_per_tok (L30), hidden_act (L16), scoring_func (L58), topk_method (L61), n_group (L25), topk_group (L60), routed_scaling_factor (L57), first_k_dense_replace (L15), num_hidden_layers (L31), norm_topk_prob (L28), moe_layer_freq (L24)
- `hf_model_cards/deepseek-v3/modeling_deepseek.py`: MoEGate class (L393), gate weight+bias_correction (L408-414), sigmoid scoring (L429), group-based top-k (L437-461), scaling factor applied (L471), DeepseekV3MoE class (L475), expert MLP gate/up/down (L383-385), SwiGLU activation (L389), shared_experts init (L518), shared expert addition (L531)


#### Downloaded Files
- `config.json` — `hf download deepseek-ai/DeepSeek-V3 config.json`
- `README.md` — `hf download deepseek-ai/DeepSeek-V3 README.md`
- `modeling_deepseek.py` — `hf download deepseek-ai/DeepSeek-V3 modeling_deepseek.py`
- `configuration_deepseek.py` — `hf download deepseek-ai/DeepSeek-V3 configuration_deepseek.py`
#### Config Parameters (from config.json)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_size` | 7168 | Model hidden dimension |
| `intermediate_size` | 18432 | Dense FFN intermediate dim (non-MoE layers) |
| `moe_intermediate_size` | **2048** | Per-expert intermediate dim |
| `n_routed_experts` | **256** | Number of routed experts |
| `n_shared_experts` | **1** | Number of shared (always-on) experts |
| `num_experts_per_tok` | **8** (K=8) | Top-K experts selected per token |
| `hidden_act` | `silu` | Activation function |
| `first_k_dense_replace` | 3 | First 3 layers are dense (no MoE) |
| `moe_layer_freq` | 1 | Every layer after first 3 is MoE |
| `n_group` | 8 | Expert group count for grouped top-k |
| `topk_group` | 4 | Number of groups selected per token |
| `scoring_func` | `sigmoid` | Gating score function |
| `topk_method` | `noaux_tc` | Auxiliary-loss-free top-k selection |
| `routed_scaling_factor` | 2.5 | Multiplied into expert weights after normalization |
| `norm_topk_prob` | true | Normalize top-k probabilities to sum to 1 |
| `ep_size` | 1 | Expert parallelism size |
| `num_hidden_layers` | 61 | Total transformer layers |

#### MoE Block Structure (from modeling_deepseek.py)

```
DeepseekV3MoE
├── gate: MoEGate
│   ├── weight: Linear(hidden_size=7168, n_routed_experts=256)  # no bias
│   ├── e_score_correction_bias: Parameter(256)                  # noaux_tc method
│   ├── scoring: sigmoid
│   └── selection: group-based top-k (8 groups, select top-4 groups, then top-8 experts)
│
├── experts[0..255]: DeepseekV3MLP  (routed experts)
│   ├── gate_proj: Linear(7168 → 2048, no bias)
│   ├── up_proj:   Linear(7168 → 2048, no bias)
│   ├── down_proj: Linear(2048 → 7168, no bias)
│   └── activation: SiLU (SwiGLU pattern: down(silu(gate(x)) * up(x)))
│
└── shared_experts: DeepseekV3MLP  (1 shared expert)
    ├── gate_proj: Linear(7168 → 2048, no bias)   # intermediate = moe_intermediate_size * n_shared_experts
    ├── up_proj:   Linear(7168 → 2048, no bias)
    ├── down_proj: Linear(2048 → 7168, no bias)
    └── activation: SiLU (same SwiGLU pattern)
```

#### Forward Pass Flow

```
input x: [batch, seq_len, 7168]
    │
    ├──► MoEGate(x) → topk_idx [batch*seq, 8], topk_weight [batch*seq, 8]
    │     1. logits = x @ gate.weight.T                    # [N, 256]
    │     2. scores = sigmoid(logits)                       # [N, 256]
    │     3. scores_for_choice = scores + e_score_correction_bias
    │     4. Group into 8 groups of 32 experts each
    │     5. Per group: top-2 scores summed → group_scores  # [N, 8]
    │     6. Select top-4 groups
    │     7. Mask non-selected groups to -inf
    │     8. Select top-8 experts from remaining
    │     9. Gather original scores (not bias-corrected) for weights
    │     10. Normalize weights to sum=1, multiply by routed_scaling_factor=2.5
    │
    ├──► moe_infer(x, topk_idx, topk_weight)
    │     1. Count tokens per expert
    │     2. Sort tokens by expert assignment
    │     3. Dispatch to each expert sequentially
    │     4. Unsort and weighted-sum over K=8 experts per token
    │     → routed_output [batch, seq_len, 7168]
    │
    └──► shared_experts(x)
          → shared_output [batch, seq_len, 7168]

output = routed_output + shared_output
```

#### Weight Shapes (per MoE layer)

| Component | Shape | Count | Notes |
|-----------|-------|-------|-------|
| Gate weight | [256, 7168] | 1 | |
| Gate bias correction | [256] | 1 | noaux_tc only |
| Expert gate_proj | [2048, 7168] | 256 | Often stacked as [256, 2048, 7168] |
| Expert up_proj | [2048, 7168] | 256 | Often stacked as [256, 2048, 7168] |
| Expert down_proj | [7168, 2048] | 256 | Often stacked as [256, 7168, 2048] |
| Shared gate_proj | [2048, 7168] | 1 | |
| Shared up_proj | [2048, 7168] | 1 | |
| Shared down_proj | [7168, 2048] | 1 | |

---

### A.3 GLM-5 — MoE Architecture (Verified)

**Source:** `zai-org/GLM-5` config.json
**Downloaded to:** `plans/hf_model_cards/glm-5/`

#### Source References
- `hf_model_cards/glm-5/config.json`: hidden_size (L17), moe_intermediate_size (L26), n_routed_experts (L30), n_shared_experts (L31), num_experts_per_tok (L34), hidden_act (L15), scoring_func (L51), topk_method (L54), routed_scaling_factor (L50), n_group (L29), topk_group (L53), first_k_dense_replace (L14), num_hidden_layers (L35)


#### Downloaded Files
- `config.json` — `hf download zai-org/GLM-5 config.json`

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **6144** |
| `moe_intermediate_size` | **2048** |
| `n_routed_experts` | **256** |
| `n_shared_experts` | **1** |
| `num_experts_per_tok` (K) | **8** |
| `hidden_act` | `silu` |
| `scoring_func` | **sigmoid** |
| `topk_method` | **noaux_tc** |
| `routed_scaling_factor` | 2.5 |
| `n_group` / `topk_group` | 1 / 1 |
| `first_k_dense_replace` | 3 |
| `num_hidden_layers` | 78 |
| `num_nextn_predict_layers` | 1 |
| Architecture | `GlmMoeDsaForCausalLM` ("DSA" = DeepSeek Architecture) |

**Notes:** GLM-5's MoE module is essentially identical to DeepSeek V3 — same sigmoid+noaux_tc routing, same SwiGLU activation, same scaling factor. Key difference: no expert grouping (n_group=1).

---

### A.4 Kimi K2.5 — MoE Architecture (Verified)

**Source:** `moonshotai/Kimi-K2.5` config.json (multimodal; MoE params in `text_config`)
**Downloaded to:** `plans/hf_model_cards/kimi-k2.5/`

#### Source References
- `hf_model_cards/kimi-k2.5/config.json` (all in `text_config`): hidden_size (L50), moe_intermediate_size (L69), n_routed_experts (L72), n_shared_experts (L73), num_experts_per_tok (L79), hidden_act (L49), scoring_func (L144), topk_method (L157), routed_scaling_factor (L143), n_group (L71), topk_group (L156), first_k_dense_replace (L46), num_hidden_layers (L80), architecture=DeepseekV3ForCausalLM (L21)


#### Downloaded Files
- `config.json` — `hf download moonshotai/Kimi-K2.5 config.json`

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **7168** |
| `moe_intermediate_size` | **2048** |
| `n_routed_experts` | **384** |
| `n_shared_experts` | **1** |
| `num_experts_per_tok` (K) | **8** |
| `hidden_act` | `silu` |
| `scoring_func` | **sigmoid** |
| `topk_method` | **noaux_tc** |
| `routed_scaling_factor` | **2.827** |
| `n_group` / `topk_group` | **1 / 1** |
| `first_k_dense_replace` | **1** |
| `num_hidden_layers` | 61 |
| `num_nextn_predict_layers` | 0 |
| Base architecture | **DeepseekV3ForCausalLM** |

**Notes:** Kimi K2.5 is a multimodal model built directly on DeepSeek V3 architecture (config explicitly references `DeepseekV3ForCausalLM`). MoE-wise nearly identical to DS V3 but with 384 experts (vs 256), higher scaling factor (2.827 vs 2.5), no expert grouping, and fewer dense layers.

---

### A.5 Ling-1T — MoE Architecture (Verified)

**Source:** `inclusionAI/Ling-1T` config.json + modeling_bailing_moe_v2.py
**Downloaded to:** `plans/hf_model_cards/ling-1t/`

#### Source References
- `hf_model_cards/ling-1t/config.json`: hidden_size (L12), moe_intermediate_size (L20), num_experts (L25), num_shared_experts (L36), num_experts_per_tok (L22), hidden_act (L17), score_function (L51), moe_router_enable_expert_bias (L46), routed_scaling_factor (L47), n_group (L48), topk_group (L49), first_k_dense_replace (L16), num_hidden_layers (L11), norm_topk_prob (L21), use_bias (L32)
- `hf_model_cards/ling-1t/modeling_bailing_moe_v2.py`: BailingMoeV2Gate class (L287), expert_bias register_buffer (L302), sigmoid scoring (L338), expert_bias addition (L340), group_limited_topk method (L310-331), routed_scaling_factor applied (L346), shared_experts init with `moe_intermediate_size * num_shared_experts` (L362-364), shared expert addition (L390-391)


#### Downloaded Files
- `config.json` — `hf download inclusionAI/Ling-1T config.json`
- `modeling_bailing_moe_v2.py` — `hf download inclusionAI/Ling-1T modeling_bailing_moe_v2.py`
- `configuration_bailing_moe_v2.py` — `hf download inclusionAI/Ling-1T configuration_bailing_moe_v2.py`

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **8192** |
| `moe_intermediate_size` | **2048** |
| `num_experts` | **256** |
| `num_shared_experts` | **1** |
| `num_experts_per_tok` (K) | **8** |
| `hidden_act` | `silu` |
| `score_function` | **sigmoid** |
| `topk_method` | group_limited_topk (from modeling code L310-331) |
| `moe_router_enable_expert_bias` | **true** (L302, L340) |
| `routed_scaling_factor` | **2.5** |
| `n_group` / `topk_group` | **8 / 4** |
| `norm_topk_prob` | true |
| `first_k_dense_replace` | 4 |
| `num_hidden_layers` | 80 |
| `use_bias` | false (no bias on expert projections) |
| Shared expert intermediate | 2048 (= moe_intermediate × 1) |

**Notes:** Ling-1T is very similar to DeepSeek V3 in routing — same sigmoid scoring, same 8/4 group-based routing (`group_limited_topk` at L310-331), same scaling factor, same expert bias correction. Confirmed from `modeling_bailing_moe_v2.py`: sigmoid (L338), expert_bias (L302, L340), group_limited_topk (L310-331), routed_scaling_factor (L346). Shared expert uses `moe_intermediate_size * num_shared_experts = 2048`.

---

### A.6 GLM-4.7 — MoE Architecture (Verified)

**Source:** `zai-org/GLM-4.7` config.json + modeling_glm4_moe.py
**Downloaded to:** `plans/hf_model_cards/glm-4.7/`

#### Source References
- `hf_model_cards/glm-4.7/config.json`: hidden_size (L15), moe_intermediate_size (L21), n_routed_experts (L26), n_shared_experts (L27), num_experts_per_tok (L29), hidden_act (L14), routed_scaling_factor (L28), n_group (L24), topk_group (L25), first_k_dense_replace (L30), num_hidden_layers (L31), norm_topk_prob (L22)
- `hf_model_cards/glm-4.7/modeling_glm4_moe.py`: sigmoid scoring (L390), e_score_correction_bias (L300, L391), group-based top-k (L392-406), routed_scaling_factor applied (L411), fused gate_up_proj (L338), chunk(2) split (L360), expert forward (L355-363)


#### Downloaded Files
- `config.json` — `hf download zai-org/GLM-4.7 config.json`
- `modeling_glm4_moe.py` — https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **5120** |
| `moe_intermediate_size` | **1536** |
| `n_routed_experts` | **160** |
| `n_shared_experts` | **1** |
| `num_experts_per_tok` (K) | **8** |
| `hidden_act` | `silu` |
| `scoring_func` | sigmoid (from modeling code L390) |
| `topk_method` | noaux_tc-style (from modeling code L389-412) |
| `routed_scaling_factor` | 2.5 |
| `norm_topk_prob` | true |
| `n_group` / `topk_group` | 1 / 1 | Same algorithm as DeepSeek, collapses to single group |
| `first_k_dense_replace` | 3 |
| `num_hidden_layers` | 92 |
| `num_nextn_predict_layers` | 1 | Multi-token prediction like DeepSeek |
| `max_position_embeddings` | 202752 |
| Expert layout | Fused gate_up_proj + chunk(2) (L338, L360) — same as Gemma 4 |
| Shared expert intermediate | 1536 (= moe_intermediate × 1) |
| Architecture | `Glm4MoeForCausalLM` (`glm4_moe`) |

**Notes:** GLM-4.7 uses the same `glm4_moe` architecture as GLM-4.5 (identical MoE parameters) with extended context length. Config does not expose `scoring_func` or `topk_method`, but `modeling_glm4_moe.py` confirms **DeepSeek-style routing**: sigmoid scoring (L390), e_score_correction_bias (L391), group-based top-k with masking (L392-406), routed_scaling_factor (L411). With n_group=1/topk_group=1, the group logic collapses to a single group but the algorithm is the same. Expert projections use **fused gate_up_proj + chunk(2)** (L338, L360), not separate gate/up projections. A lighter Flash variant exists (`GLM-4.7-Flash`, arch `glm4_moe_lite`).

---

### A.7 Qwen3 MoE — MoE Architecture (Verified)

**Source:** `Qwen/Qwen3-235B-A22B` config.json + transformers `modeling_qwen3_moe.py`
**Downloaded to:** `plans/hf_model_cards/qwen3-moe/`

#### Source References
- `hf_model_cards/qwen3-moe/config.json`: hidden_size (L12), moe_intermediate_size (L19), num_experts (L22), num_experts_per_tok (L23), hidden_act (L11), num_hidden_layers (L24), norm_topk_prob (L20), intermediate_size (L14), router_aux_loss_coef (L30)
- `hf_model_cards/qwen3-moe/modeling_qwen3_moe.py` (from `transformers/models/qwen3_moe/`): Qwen3MoeTopKRouter class (L254), softmax scoring (L266), top-k selection (L267), norm_topk_prob (L268-269)


#### Downloaded Files
- `config.json` — `hf download Qwen/Qwen3-235B-A22B config.json`
- `modeling_qwen3_moe.py` — https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **4096** |
| `moe_intermediate_size` | **1536** |
| `num_experts` | **128** |
| `num_experts_per_tok` (K) | **8** |
| `hidden_act` | `silu` |
| `scoring_func` | softmax (from modeling code L266) |
| `norm_topk_prob` | true |
| `router_aux_loss_coef` | 0.001 |
| `intermediate_size` | 12288 (dense FFN dim) |
| `num_hidden_layers` | 94 |
| Shared experts | **None** (no shared expert fields in config) |

**Notes:** Qwen3 MoE uses **softmax routing** confirmed from modeling code (L266: `softmax(router_logits)` → top-k → optional normalization). No shared experts.

---

### A.8 Qwen3.5 MoE — MoE Architecture (Verified)

**Source:** `Qwen/Qwen3.5-397B-A17B` config.json (multimodal; MoE params in `text_config`) + transformers `modeling_qwen3_5_moe.py`
**Downloaded to:** `plans/hf_model_cards/qwen3.5-moe/`

#### Source References
- `hf_model_cards/qwen3.5-moe/config.json` (all in `text_config`): hidden_size (L16), moe_intermediate_size (L88), num_experts (L92), num_experts_per_tok (L93), shared_expert_intermediate_size (L98), hidden_act (L15), num_hidden_layers (L94), router_aux_loss_coef (L97)
- `hf_model_cards/qwen3.5-moe/modeling_qwen3_5_moe.py` (from `transformers/models/qwen3_5_moe/`): Qwen3_5MoeTopKRouter class (L754), softmax scoring (L765), top-k (L766), norm top-k (L767); Qwen3_5MoeSparseMoeBlock (L773), shared_expert init (L778), shared_expert_gate sigmoid gating (L788), output = expert + shared (L790)


#### Downloaded Files
- `config.json` — `hf download Qwen/Qwen3.5-397B-A17B config.json`
- `modeling_qwen3_5_moe.py` — https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **4096** |
| `moe_intermediate_size` | **1024** |
| `num_experts` | **512** |
| `num_experts_per_tok` (K) | **10** |
| `shared_expert_intermediate_size` | **1024** (same as moe_intermediate) |
| `hidden_act` | `silu` |
| `scoring_func` | softmax (from modeling code L765) |
| `router_aux_loss_coef` | 0.001 |
| `num_hidden_layers` | 60 |
| `mtp_num_hidden_layers` | 1 (multi-token prediction) |
| Shared expert gating | sigmoid gate on shared expert output (L788, unique to Qwen3.5) |

**Notes:** Qwen3.5 has 512 experts (most of any model here), K=10 (highest), and an explicit `shared_expert_intermediate_size` field. Uses **softmax routing** confirmed from modeling code (L765). Shared expert has a **sigmoid gate** (L788: `sigmoid(gate(x)) * shared_output`) before being added to routed output — this is different from DeepSeek-family models which add shared expert output directly.

---

### A.9 Qwen3.5 35B-A3B — MoE Architecture (Verified)

**Source:** `Qwen/Qwen3.5-35B-A3B` config.json (multimodal; MoE params in `text_config`)
**Downloaded to:** `plans/hf_model_cards/qwen3.5-35b/`

#### Source References
- `hf_model_cards/qwen3.5-35b/config.json` (all in `text_config`): hidden_size (L16), moe_intermediate_size (L68), num_experts (L72), num_experts_per_tok (L73), shared_expert_intermediate_size (L78), hidden_act (L15), num_hidden_layers (L74), router_aux_loss_coef (L77)


#### Downloaded Files
- `config.json` — `hf download Qwen/Qwen3.5-35B-A3B config.json`

| Parameter | Value | vs Qwen3.5-397B |
|-----------|-------|-----------------|
| `hidden_size` | **2048** | 4096 (half) |
| `moe_intermediate_size` | **512** | 1024 (half) |
| `num_experts` | **256** | 512 (half) |
| `num_experts_per_tok` (K) | **8** | 10 |
| `shared_expert_intermediate_size` | **512** | 1024 |
| `hidden_act` | `silu` | same |
| `router_aux_loss_coef` | 0.001 | same |
| `num_hidden_layers` | 40 | 60 |
| `mtp_num_hidden_layers` | 1 | 1 |
| Architecture | `Qwen3_5MoeForConditionalGeneration` | same |

**Notes:** Qwen3.5-35B is a smaller variant of Qwen3.5-397B with roughly halved MoE dimensions. Same MoE architecture — SwiGLU, softmax routing. K=8 (vs K=10 for the 397B), 256 experts (vs 512).

---

### A.10 Qwen3-Omni 30B-A3B — MoE Architecture (Verified)

**Source:** `Qwen/Qwen3-Omni-30B-A3B-Instruct` config.json
**Downloaded to:** `plans/hf_model_cards/qwen3-omni-30b/`

#### Source References
- `hf_model_cards/qwen3-omni-30b/config.json` — Thinker MoE (in `thinker_config.text_config`): hidden_size (L315), moe_intermediate_size (L334), num_experts (L340), num_experts_per_tok (L341), shared_expert_intermediate_size=0 (L372), hidden_act (L314), num_hidden_layers (L342), norm_topk_prob (L336), router_aux_loss_coef (L370)
- `hf_model_cards/qwen3-omni-30b/config.json` — Talker MoE (in `talker_config.text_config`): hidden_size (L52), moe_intermediate_size (L57), num_experts (L60), num_experts_per_tok (L61), shared_expert_intermediate_size=768 (L77), num_hidden_layers (L62)


#### Downloaded Files
- `config.json` — `hf download Qwen/Qwen3-Omni-30B-A3B-Instruct config.json`
This is an omni-modal model (text+vision+audio) with **two separate MoE modules**: a "thinker" (main LLM) and a "talker" (audio generation).

#### Thinker MoE (main LLM — `thinker_config.text_config`)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **2048** |
| `moe_intermediate_size` | **768** |
| `num_experts` | **128** |
| `num_experts_per_tok` (K) | **8** |
| `shared_expert_intermediate_size` | **0** (no shared experts) |
| `hidden_act` | `silu` |
| `norm_topk_prob` | true |
| `router_aux_loss_coef` | 0.001 |
| `num_hidden_layers` | 48 |
| `intermediate_size` (dense) | 768 (same as moe_intermediate) |
| Total / Active params | 30B / 3B |

#### Talker MoE (audio generation — `talker_config.text_config`)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **1024** |
| `moe_intermediate_size` | **384** |
| `num_experts` | **128** |
| `num_experts_per_tok` (K) | **6** |
| `shared_expert_intermediate_size` | **768** (2× moe_intermediate!) |
| `hidden_act` | `silu` |
| `norm_topk_prob` | true |
| `num_hidden_layers` | 20 |

**Notes:** The thinker is the main MoE module for generalization purposes. It uses the same `qwen3_omni_moe` architecture as the Qwen3 family — simple softmax routing, SiLU/SwiGLU, no group-based selection. Notable that the talker has a shared expert with intermediate size 2× the routed expert size — this is a pattern not seen in other models where shared = routed × n_shared.

---

### A.11 Mistral Large 3 (675B) — MoE Architecture (Verified)

**Source:** `mistralai/Mistral-Large-3-675B-Instruct-2512` params.json + README.md + mistral-inference code + vLLM implementation
**Downloaded to:** `plans/hf_model_cards/mistral-large-3/`

#### Source References
- `hf_model_cards/mistral-large-3/params.json`: dim/hidden_size (L2), hidden_dim=16384 (L4), moe.expert_hidden_dim (L12), moe.num_experts (L17), moe.num_experts_per_tok (L18), moe.num_shared_experts (L19), moe.first_k_dense_replace (L15), moe.route_every_n (L20), moe.routed_scale (L21), moe.num_expert_groups (L14), moe.num_expert_groups_per_tok (L16), n_layers (L22)
- `hf_model_cards/mistral-large-3/README.md`: "Granular Mixture-of-Experts" (L28), 675B total / 41B active (L28), 673B LM + 2.5B vision encoder (L43-44)
- `hf_model_cards/mistral-large-3/transformer_layers.py` (from `mistralai/mistral-inference`): FeedForward w1/w2/w3 SiLU activation (L101-106), MoeLayer init (L150-153)
- `hf_model_cards/mistral-large-3/moe.py` (from `mistralai/mistral-inference`): softmax routing (L27), simple top-k (L26)
- `hf_model_cards/mistral-large-3/vllm_mistral_large_3.py` (from `vllm-project/vllm`): MistralLarge3ForCausalLM subclasses DeepseekV3ForCausalLM (L8, L11), shared_experts weight remapping confirming w1/w2/w3 (L27-29), expert w1/w2/w3 (L30-32)


#### Downloaded Files
- `params.json` — `hf download mistralai/Mistral-Large-3-675B-Instruct-2512 params.json`
- `README.md` — `hf download mistralai/Mistral-Large-3-675B-Instruct-2512 README.md`
- `moe.py` — https://raw.githubusercontent.com/mistralai/mistral-inference/main/src/mistral_inference/moe.py
- `transformer_layers.py` — https://raw.githubusercontent.com/mistralai/mistral-inference/main/src/mistral_inference/transformer_layers.py
- `args.py` — https://raw.githubusercontent.com/mistralai/mistral-inference/main/src/mistral_inference/args.py
- `vllm_mistral_large_3.py` — https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/mistral_large_3.py

| Parameter | Value |
|-----------|-------|
| `dim` (hidden_size) | **7168** |
| `expert_hidden_dim` (moe intermediate) | **4096** |
| `shared_expert_intermediate` | **4096** (= expert_hidden_dim, confirmed from vLLM weight remapping) |
| `hidden_dim` (dense FFN intermediate) | **16384** |
| `num_experts` | **128** |
| `num_shared_experts` | **1** |
| `num_experts_per_tok` (K) | **4** |
| `first_k_dense_replace` | 3 |
| `route_every_n` | 1 |
| `routed_scale` | 1.0 |
| `num_expert_groups` | 1 |
| `n_layers` | 61 |
| `scoring_func` | softmax (from moe.py L27) |
| `activation` | SiLU/SwiGLU (from transformer_layers.py L106: `silu(w1(x)) * w3(x)`) |
| Expert layout | Separate w1/w2/w3 (gate/down/up), no bias |
| Total / Active params | 675B / 41B |

**Notes:** Mistral uses a custom config format (`params.json` not `config.json`). The MoE params are nested under `moe` key. Uses `routed_scale=1.0` (no scaling), single expert group (no group-based routing), softmax routing. vLLM implements Mistral Large 3 as a **subclass of DeepseekV3ForCausalLM** with weight name remapping (vllm_mistral_large_3.py L11), confirming shared experts use the same w1/w2/w3 structure as routed experts. Shared expert intermediate size is 4096 (same as expert_hidden_dim) — inferred from vLLM remapping using the same gate_proj/up_proj/down_proj pattern with no separate size config.

---

### A.12 Gemma 4 26B-A4B — MoE Architecture (Verified, Optional)

**Source:** `google/gemma-4-26B-A4B-it` config.json + transformers `modeling_gemma4.py`
**Downloaded to:** `plans/hf_model_cards/gemma-4-26b/`

#### Source References
- `hf_model_cards/gemma-4-26b/config.json` (all in `text_config`): hidden_size (L32), moe_intermediate_size (L70), intermediate_size (L35), num_experts (L72), top_k_experts (L92), hidden_activation (L31), enable_moe_block (L26), num_hidden_layers (L74), use_double_wide_mlp (L95)
- `hf_model_cards/gemma-4-26b/modeling_gemma4.py` (from `transformers/models/gemma4/`): Gemma4TextMLP dense class (L1016), gate/up/down proj (L1025-1027), SwiGLU activation (L1031); Gemma4TextExperts class (L1250), gate_up_proj fused (L1258), down_proj (L1259), chunk(2) split (L1280); Gemma4TextRouter class (L1289), learned scale (L1299), per_expert_scale (L1300), softmax scoring (L1307), topk (L1310), per-expert scaling applied (L1320); decoder layer init: mlp+router+experts (L1332-1349), parallel dense+MoE forward (L1382-1396)


#### Downloaded Files
- `config.json` — `hf download google/gemma-4-26B-A4B-it config.json`
- `modeling_gemma4.py` — https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **2816** |
| `moe_intermediate_size` | **704** |
| `intermediate_size` (dense MLP) | **2112** |
| `num_experts` | **128** |
| `top_k_experts` (K) | **8** |
| Shared experts | **None** |
| `hidden_activation` | **gelu_pytorch_tanh** |
| `enable_moe_block` | true |
| `num_hidden_layers` | 30 |
| Total / Active params | 26B / 4B |

#### MoE Block Structure (from modeling_gemma4.py)

```
Gemma4TextDecoderLayer (when enable_moe_block=true)
├── Dense MLP path (Gemma4TextMLP):
│   ├── gate_proj: Linear(2816 → 2112, no bias)
│   ├── up_proj:   Linear(2816 → 2112, no bias)
│   ├── down_proj: Linear(2112 → 2816, no bias)
│   └── activation: gelu_pytorch_tanh (SwiGLU pattern: down(gelu(gate) * up))
│
├── Router (Gemma4TextRouter):
│   ├── norm: RMSNorm (no scale)
│   ├── scale: Parameter(2816) — learned input scaling
│   ├── proj: Linear(2816 → 128, no bias)
│   ├── per_expert_scale: Parameter(128) — learned per-expert output scaling
│   └── selection: softmax → top-8 → renormalize → multiply by per_expert_scale
│
└── Experts (Gemma4TextExperts, 128 routed, NO shared):
    ├── gate_up_proj: Parameter(128, 2*704=1408, 2816) — fused gate+up (chunked, not interleaved)
    ├── down_proj: Parameter(128, 2816, 704)
    └── activation: gelu_pytorch_tanh (same SwiGLU: gelu(gate) * up)

output = dense_mlp_output + moe_output   (parallel dense + MoE, not sequential!)
```

#### Key Unique Features
1. **Parallel dense MLP + MoE**: Unlike all other models where MoE replaces the FFN, Gemma 4 runs a dense MLP and MoE experts **in parallel** and sums their outputs. This is architecturally unique.
2. **GELU activation (not SiLU)**: Uses `gelu_pytorch_tanh` in both dense and expert MLPs. Still SwiGLU *pattern* (gate + up + down), but with GELU instead of SiLU.
3. **Fused gate_up but chunked (not interleaved)**: `gate_up_proj` is split via `.chunk(2)` (first half = gate, second half = up), unlike GPT-OSS which interleaves (even/odd indices).
4. **Router has learned input+output scaling**: Pre-routing RMSNorm + learned scale, plus per-expert output scaling — more complex than simple softmax.
5. **No shared experts**: MoE output is combined with the parallel dense MLP instead.
6. **Unusual dimensions**: hidden=2816, intermediate=704 — not powers of 2.
7. **`use_double_wide_mlp`**: False in this model. Only affects KV-shared attention layers (doubles dense MLP intermediate_size for those layers) — has no effect on MoE expert dimensions. Irrelevant to MoE generalization.

#### Weight Shapes (per MoE layer)

| Component | Shape | Has Bias | Notes |
|-----------|-------|----------|-------|
| Dense gate_proj | [2112, 2816] | No | Parallel dense MLP |
| Dense up_proj | [2112, 2816] | No | Parallel dense MLP |
| Dense down_proj | [2816, 2112] | No | Parallel dense MLP |
| Router proj | [128, 2816] | No | |
| Router scale | [2816] | — | Learned input scaling |
| Router per_expert_scale | [128] | — | Learned output scaling |
| Expert gate_up_proj | [128, 1408, 2816] | No | Fused, chunked (not interleaved) |
| Expert down_proj | [128, 2816, 704] | No | |

---

### A.13 DeepSeek OCR — MoE Architecture (Verified)

**Source:** `deepseek-ai/DeepSeek-OCR` config.json (multimodal; MoE params in `language_config`)
**Downloaded to:** `plans/hf_model_cards/deepseek-ocr/`

#### Source References
- `hf_model_cards/deepseek-ocr/config.json` (in `language_config`): hidden_size (L29), moe_intermediate_size (L34), n_routed_experts (L36), n_shared_experts (L37), num_experts_per_tok (L39), topk_method (L47), n_group (L35), topk_group (L46), first_k_dense_replace (L28), num_hidden_layers (L40), architecture=DeepseekV2ForCausalLM (L19)
- `hf_model_cards/deepseek-ocr/configuration_deepseek_v2.py`: scoring_func default = `'softmax'` (L142, L185)
- `hf_model_cards/deepseek-ocr/modeling_deepseekv2.py`: MoEGate class (L400), scoring_func from config (L407), softmax scoring (L438-439), sigmoid scoring (L440-441), greedy topk_method (L448-451)


#### Downloaded Files
- `config.json` — `hf download deepseek-ai/DeepSeek-OCR config.json`
- `modeling_deepseekv2.py` — `hf download deepseek-ai/DeepSeek-OCR --include '*.py'`
- `configuration_deepseek_v2.py` — `hf download deepseek-ai/DeepSeek-OCR --include '*.py'`
- `modeling_deepseekocr.py` — `hf download deepseek-ai/DeepSeek-OCR modeling_deepseekocr.py`

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **1280** |
| `moe_intermediate_size` | **896** |
| `n_routed_experts` | **64** |
| `n_shared_experts` | **2** |
| `num_experts_per_tok` (K) | **6** |
| `scoring_func` | softmax (default from configuration_deepseek_v2.py L142) |
| `topk_method` | **greedy** |
| `n_group` / `topk_group` | 1 / 1 |
| Base architecture | **DeepseekV2ForCausalLM** (V2, not V3) |
| `num_hidden_layers` | **12** |
| `first_k_dense_replace` | 1 |
| Shared expert intermediate | 1792 (= 896 × 2) |

**Notes:** DeepSeek OCR is a very small model (~3.3B) based on DeepSeek-**V2** (not V3). Uses **softmax scoring** (V2 default from configuration_deepseek_v2.py L142 — config.json does not override) with **greedy top-k** selection.

---

### A.14 DeepSeek V4 Flash — MoE Architecture (Verified)

**Source:** `deepseek-ai/DeepSeek-V4-Flash` config.json + `inference/model.py` (Flash and Flash-Base share identical configs)
**Downloaded to:** `plans/hf_model_cards/deepseek-v4-flash/`, `plans/hf_model_cards/deepseek-v4-flash-base/`

#### Source References
- `hf_model_cards/deepseek-v4-flash/config.json`: hidden_size (L14), moe_intermediate_size (L21), n_routed_experts (L22), n_shared_experts (L23), num_experts_per_tok (L26), num_hash_layers (L27), num_hidden_layers (L28), scoring_func (L55), topk_method (L59), routed_scaling_factor (L54), swiglu_limit (L57), norm_topk_prob (L24)
- `hf_model_cards/deepseek-v4-flash/inference/model.py`: Gate class (L546–584) — scoring functions (L566–571), hash routing path (L558, L576–578), score correction bias (L561–562, L575), norm+scale (L581–583); Expert class (L587–606) — SwiGLU with clamping (L598–603); MoE class (L609–645) — shared expert no clamping (L627–628), forward (L630–645), shared expert addition (L644)

#### Downloaded Files
- `config.json` — `hf download deepseek-ai/DeepSeek-V4-Flash config.json --local-dir plans/hf_model_cards/deepseek-v4-flash`
- `inference/model.py` (+ supporting files) — `hf download deepseek-ai/DeepSeek-V4-Flash --include "*.py" --local-dir plans/hf_model_cards/deepseek-v4-flash`
- `README.md` — `hf download deepseek-ai/DeepSeek-V4-Flash README.md --local-dir plans/hf_model_cards/deepseek-v4-flash`
- (Base) `config.json` — `hf download deepseek-ai/DeepSeek-V4-Flash-Base config.json --local-dir plans/hf_model_cards/deepseek-v4-flash-base`

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **4096** |
| `moe_intermediate_size` | **2048** |
| `n_routed_experts` | **256** |
| `n_shared_experts` | **1** |
| `num_experts_per_tok` (K) | **6** |
| `num_hash_layers` | **3** (layers 0–2 use hash routing) |
| `scoring_func` | **sqrtsoftplus** (config + code L571: `F.softplus(scores).sqrt()`) |
| `topk_method` | **noaux_tc** (layers 3+); hash (layers 0–2) |
| `routed_scaling_factor` | **1.5** |
| `swiglu_limit` | **10.0** (applied to routed experts only; shared expert has no limit) |
| `norm_topk_prob` | true |
| `n_group` / `topk_group` | — / — (no grouping) |
| `num_hidden_layers` | 43 |
| `num_nextn_predict_layers` | 1 (multi-token prediction) |
| Architecture | `DeepseekV4ForCausalLM` |
| Expert projection layout | Separate w1/w2/w3 (gate/down/up), no bias |
| Shared expert intermediate | 2048 (= moe_intermediate_size; code L628: `Expert(args.dim, args.moe_inter_dim)`) |

#### MoE Block Structure (from inference/model.py)

```
MoE (model.py L609)
├── gate: Gate(layer_id, args)
│   ├── weight: Parameter(n_routed_experts=256, dim=4096)
│   ├── bias: Parameter(256, float32)          # score correction (non-hash layers only)
│   ├── tid2eid: Parameter(vocab_size, K=6, int32)  # hash layers 0–2 only
│   └── forward:
│       ├── scores = sqrtsoftplus(linear(x, weight))   # L571: sqrt(softplus(logits))
│       ├── if hash layer: indices = tid2eid[input_ids]  # L577: deterministic from token ID
│       ├── else:          indices = (scores + bias).topk(K=6)  # L579: learned + correction
│       ├── weights = original_scores.gather(indices)   # bias not included in weights
│       ├── weights /= weights.sum()                     # L582: normalize (non-softmax)
│       └── weights *= route_scale (1.5)                 # L583
│
├── experts: ModuleList[256 × Expert(dim=4096, inter_dim=2048, swiglu_limit=10.0)]
│   └── forward:
│       ├── gate = w1(x).float()
│       ├── up   = w3(x).float()
│       ├── gate = clamp(gate, max=10.0)        # L601
│       ├── up   = clamp(up, min=-10.0, max=10.0)  # L602
│       └── return w2(silu(gate) * up)
│
└── shared_experts: Expert(dim=4096, inter_dim=2048, swiglu_limit=0)
    └── forward: standard SwiGLU, no clamping
```

#### Scoring Function — `sqrtsoftplus`

```python
# DeepSeek V4 (model.py L571):
scores = F.softplus(logits).sqrt()   # sqrt(log(1 + exp(x)))

# DeepSeek V3 (for comparison):
scores = logits.sigmoid()            # 1 / (1 + exp(-x))
```

`sqrtsoftplus` is always positive and unbounded above; `sigmoid` is bounded to (0, 1). Both are then normalized across selected experts before scaling.

#### Hash Routing (layers 0–2)

```python
# Hash layers (model.py L558, L576–578):
self.tid2eid = Parameter(vocab_size=129280, K=6, dtype=int32)   # learned lookup
# ...
if self.hash:
    indices = self.tid2eid[input_ids]   # expert indices from token ID, not gate score
    # routing weights still come from gate scores (original_scores.gather)
    # bias correction parameter absent for hash layers (L558–562)
```

Each token is deterministically routed to the same K=6 experts based solely on its token ID. This replaces the score-based top-k for the first 3 layers. The `e_score_correction_bias` is absent in hash layers (`self.bias = None`, L558–562).

#### Weight Shapes (per MoE layer, 120B-class Flash)

| Component | Shape | Notes |
|-----------|-------|-------|
| Gate weight | [256, 4096] | No bias |
| Gate correction bias | [256] float32 | Score correction (non-hash layers only) |
| Gate tid2eid | [129280, 6] int32 | Hash routing LUT (hash layers only) |
| Routed expert w1 (gate) | [256, 2048, 4096] | |
| Routed expert w2 (down) | [256, 4096, 2048] | |
| Routed expert w3 (up) | [256, 2048, 4096] | |
| Shared expert w1/w2/w3 | same dims as above | No swiglu_limit |

---

### A.15 DeepSeek V4 Pro — MoE Architecture (Verified)

**Source:** `deepseek-ai/DeepSeek-V4-Pro` config.json + `inference/model.py` (identical to Flash codebase; Pro and Pro-Base share identical configs)
**Downloaded to:** `plans/hf_model_cards/deepseek-v4-pro/`, `plans/hf_model_cards/deepseek-v4-pro-base/`

#### Source References
- `hf_model_cards/deepseek-v4-pro/config.json`: hidden_size (L14), moe_intermediate_size (L21), n_routed_experts (L22), n_shared_experts (L23), num_experts_per_tok (L26), num_hash_layers (L27), num_hidden_layers (L28), scoring_func (L55), topk_method (L59), routed_scaling_factor (L54), swiglu_limit (L57), norm_topk_prob (L24)
- `hf_model_cards/deepseek-v4-pro/inference/model.py`: identical MoE code as Flash (same architecture, different dimensions)

#### Downloaded Files
- `config.json` — `hf download deepseek-ai/DeepSeek-V4-Pro config.json --local-dir plans/hf_model_cards/deepseek-v4-pro`
- `inference/model.py` (+ supporting files) — `hf download deepseek-ai/DeepSeek-V4-Pro --include "*.py" --local-dir plans/hf_model_cards/deepseek-v4-pro`
- `README.md` — `hf download deepseek-ai/DeepSeek-V4-Pro README.md --local-dir plans/hf_model_cards/deepseek-v4-pro`
- (Base) `config.json` — `hf download deepseek-ai/DeepSeek-V4-Pro-Base config.json --local-dir plans/hf_model_cards/deepseek-v4-pro-base`

| Parameter | Value | vs DS V4 Flash |
|-----------|-------|----------------|
| `hidden_size` | **7168** | 4096 (↑) |
| `moe_intermediate_size` | **3072** | 2048 (↑) |
| `n_routed_experts` | **384** | 256 (↑) |
| `n_shared_experts` | **1** | same |
| `num_experts_per_tok` (K) | **6** | same |
| `num_hash_layers` | **3** | same |
| `scoring_func` | **sqrtsoftplus** | same |
| `topk_method` | **noaux_tc** | same |
| `routed_scaling_factor` | **2.5** | 1.5 (↑, same as DS V3) |
| `swiglu_limit` | **10.0** | same |
| `norm_topk_prob` | true | same |
| `n_group` / `topk_group` | — / — | same |
| `num_hidden_layers` | 61 | 43 (↑, same as DS V3) |
| `num_nextn_predict_layers` | 1 | same |
| Architecture | `DeepseekV4ForCausalLM` | same |
| Expert projection layout | Separate w1/w2/w3, no bias | same |
| Shared expert intermediate | 3072 (= moe_intermediate_size) | 2048 |

**Notes:** DS V4 Pro is the larger variant of DS V4. MoE architecture is identical to Flash — same scoring function, same hash routing in first 3 layers, same clamped SwiGLU. Key dimension differences: hidden_size=7168 (matches DS V3 Pro), moe_intermediate_size=3072 (larger than DS V3's 2048), n_routed_experts=384 (more than DS V3's 256, same as Kimi K2.5), routed_scaling_factor=2.5 (same as DS V3).

---

## Appendix B: Existing TT-Metal Implementations — Comparison (as of main@5e955223a1d)

Based on code exploration of:
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/` (DeepSeek)
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/` (GPT-OSS)

#### Shared (~90%)
- Same kernel files: `dm0.cpp`, `dm1.cpp`, `tilize_reader.cpp`, `tilize_writer.cpp`, `tilize_compute.cpp`, `combine_dm1.cpp`, `compute.cpp`
- Same CB management patterns
- Same tilize/matmul/combine kernel invocation framework
- Common MoE utilities from `ttnn/operations/ccl/common/host/moe_utils.hpp`

#### Key Differences

| Aspect | moe_compute (DeepSeek) | moe_gpt (GPT-OSS) |
|--------|----------------------|-------------------|
| **layer_id** | Explicit parameter | Not present |
| **hidden_size** | Inferred from tensors | Explicit param (default 2880) |
| **Combine module** | Delegates to `SelectiveReduceCombineDeviceOperation` | Fused combine in MoEGPTMeshWorkloadFactory |
| **Core placement** | Hardcoded cores | Dynamic rectangle search avoiding matmul cores |
| **Ring tile counts** | W0/W1=224, W2=64 | W0/W1=90, W2=90 (symmetric) |
| **Activation** | Standard (SiLU/SwiGLU) or optional GPT-OSS style (f9ba5b323f4) | Custom gated activation (see §A.1 for details; `swiglu_sfpu.h`) |
| **Cross-device** | Full fabric support (topology, num_links, semaphores) | Simpler, cluster_axis only |
| **Semaphores** | Init + final barrier, explicit | Optional (fused mode only) |
| **Namespace** | `ttnn::experimental` | `ttnn::experimental` |

#### Implications for Generalization
- Ring tile counts and hidden_size must be parameterized (not hardcoded)
- Activation function must be selectable (SiLU/SwiGLU vs others) — moe_compute now has optional GPT-OSS activation (f9ba5b323f4)
- Core placement strategy needs to be configurable (fixed vs dynamic)
- Cross-device support should be optional but available
- Combine module approach needs unification

#### Recent Changes Since Initial Analysis
- **f9ba5b323f4** (MoE: Add optional GPT-OSS style clamped SwiGlu activation): Added optional `activation_type` parameter to moe_compute, reducing the activation gap with moe_gpt
- **d6cb8e9ec29** (MoE: Arbitrary experts per device): Refactored semaphore communication to L1 tensor, enabling more flexible expert-per-device configurations

---

## Appendix C: Verification of `unverified_moe_info.md`

This appendix records the accuracy of the original unverified parameter list (`plans/unverified_moe_info.md`) against verified config.json values. Kept for provenance; the unverified file may be discarded.

| Model | Values Correct? | Notes |
|-------|----------------|-------|
| DeepSeek V3 | ✓ All correct | |
| GPT-OSS | **Wrong** on hidden (2048→2880), intermediate (2048→2880), K (8→4) | |
| GLM-4.7 | ✓ Same arch as GLM-4.5, extended context | |
| GLM-5 | ✓ All correct (K was unspecified) | |
| Kimi K2.5 | ✓ All correct | |
| Qwen 3.5 | **Wrong** on hidden (2048→4096), intermediate (512→1024), experts (256→512) | |
| Qwen3 | **Wrong** on hidden (2048→4096), intermediate (768→1536) | |
| DeepSeek OCR | **Wrong** on hidden (4096→1280), intermediate (1407→896), shared_intermediate (5632→1792) | Was likely a different model |
| Mistral | **Wrong** on hidden (6144→7168), intermediate (2048→4096), K (8→4), name was wrong | |
| Ling-1T | **Wrong** on intermediate (1536→2048), shared experts (2→1) | |

**5 out of 10 models had significant errors. Always verify from config.json.**
