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
