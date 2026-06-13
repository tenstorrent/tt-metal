# NVIDIA Nemotron-3 Nano 30B-A3B-BF16 Architecture Analysis

## Model Overview

**HuggingFace ID:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
**Model Type:** `nemotron_h` — Mamba-2 / Transformer Hybrid Mixture-of-Experts (MoE) decoder
**Parameters:** 30B total, ~3.5B active per token
**HF Class:** `NemotronHForCausalLM`

---

## Model Family

**Mamba-2 + Transformer Hybrid MoE Decoder-only LLM**

This model interleaves three distinct layer types across 52 hidden layers:
- **Mamba-2 (SSM) layers** (23 total): State-space model blocks with selective scan
- **MoE Transformer layers** (23 total): Sparse MoE MLP + GQA Attention
- **Dense Attention layers** (6 total): GQA Attention + dense ReLU² MLP

The hybrid pattern (`hybrid_override_pattern`):
```
MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
```
Where `M` = Mamba-2, `E` = MoE Transformer, `*` = Dense Attention Transformer.

---

## Architecture Parameters (from config.json)

| Parameter | Value |
|---|---|
| `hidden_size` | 2688 |
| `num_hidden_layers` | 52 (23M + 23E + 6*) |
| `num_attention_heads` | 32 (Q heads) |
| `num_key_value_heads` | 2 (KV heads — GQA) |
| `head_dim` | 128 |
| `intermediate_size` | 1856 (dense MLP) |
| `moe_intermediate_size` | 1856 (per routed expert) |
| `moe_shared_expert_intermediate_size` | 3712 |
| `n_routed_experts` | 128 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 6 |
| `routed_scaling_factor` | 2.5 |
| `mamba_num_heads` | 64 |
| `mamba_head_dim` | 64 |
| `ssm_state_size` | 128 |
| `conv_kernel` | 4 |
| `expand` | 2 |
| `vocab_size` | 131072 |
| `max_position_embeddings` | 262144 |
| `rope_theta` | 10000 |
| `partial_rotary_factor` | 1.0 |
| `mlp_hidden_act` | `relu2` (squared ReLU) |
| `mamba_hidden_act` | `silu` |
| `norm_eps` | 1e-05 |
| `rescale_prenorm_residual` | true |
| `torch_dtype` | bfloat16 |

---

## Component Inventory

| Component | Kind | Depends On | Reference TTNN Impl |
|---|---|---|---|
| Embedding | embedding | — | `models/tt_transformers/tt/embedding.py` |
| LayerNorm | norm | — | `models/tt_transformers/tt/distributed_norm.py` |
| RoPE | other | — | `models/tt_transformers/tt/rope.py` |
| Mamba2Layer | other | LayerNorm | `models/demos/wormhole/mamba/tt/` |
| DenseAttention | attention | LayerNorm, RoPE | `models/tt_transformers/tt/attention.py` |
| DenseMLP | mlp | DenseAttention | `models/tt_transformers/tt/mlp.py` |
| MoEGate | other | LayerNorm | `models/demos/deepseek_v3/tt/moe_gate.py` |
| MoEExperts | mlp | MoEGate | `models/demos/deepseek_v3/tt/experts.py` |
| SharedExpert | mlp | MoEGate | `models/demos/deepseek_v3/tt/mlp/shared_expert.py` |
| MoEAttention | attention | LayerNorm, RoPE | `models/tt_transformers/tt/attention.py` |
| LMHead | linear | Embedding | `models/tt_transformers/tt/lm_head.py` |

---

## Similar Implementations

| Component | Reference Implementation | Similarity |
|---|---|---|
| Embedding | `models/tt_transformers/tt/embedding.py` | Identical vocabulary embedding lookup |
| LayerNorm | `models/tt_transformers/tt/distributed_norm.py` | Distributed LayerNorm with eps=1e-5 |
| RoPE | `models/tt_transformers/tt/rope.py` | RoPE with rope_theta=10000, partial_rotary_factor=1.0 |
| Mamba-2 SSM | `models/demos/wormhole/mamba/tt/mamba_ssm.py` | Closest Mamba-1 implementation; Mamba-2 variant needs adaptation (multi-head SSM) |
| Mamba-2 Conv | `models/demos/wormhole/mamba/tt/mamba_conv.py` | 1D depthwise conv with kernel_size=4 |
| GQA Attention | `models/tt_transformers/tt/attention.py` | GQA with KV-cache, RoPE (adapt: 32Q/2KV heads) |
| Dense MLP (ReLU²) | `models/tt_transformers/tt/mlp.py` | Same structure; activation is relu2 not silu |
| MoE Gate | `models/demos/deepseek_v3/tt/moe_gate.py` | Top-K gating with 128 experts, norm_topk_prob |
| MoE Experts | `models/demos/deepseek_v3/tt/experts.py` | Expert parallel MLP, adapt for ep=4, 32 experts/device |
| Shared Expert | `models/demos/deepseek_v3/tt/mlp/shared_expert.py` | Shared expert MLP, larger intermediate (3712 vs 1856) |
| MoE Decoder Block | `models/demos/deepseek_v3/tt/decoder_block/` | MoE block combining attention + MoE MLP |
| LM Head | `models/tt_transformers/tt/lm_head.py` | Vocab-parallel LM head |

---

## Key Differences from Reference Implementations

1. **Mamba-1 vs Mamba-2:** The existing TTNN Mamba implementation (`models/demos/wormhole/mamba/`) targets Mamba-1 (d_state=16, sequential SSM). Nemotron uses Mamba-2 with multi-head SSM (`mamba_num_heads=64`, `mamba_head_dim=64`, `ssm_state_size=128`), SSD (state-space duality) kernel, and structured state matrices. Significant adaptation required.

2. **ReLU² activation in MLP:** Both dense and MoE MLPs use `relu2` (squared ReLU = `x * relu(x)` or `relu(x)^2`), not SwiGLU/GELU as in most reference LLMs.

3. **Tiny KV heads (2 KV heads for 32 Q heads):** Extremely aggressive GQA — 16:1 Q-to-KV ratio. With 4 devices, each KV head must be replicated ×2 (kv_replication=2) for chip-local SDPA.

4. **MoE has `n_group=1`, `topk_group=1`:** No group-constrained routing (unlike DeepSeek-V3 which uses `n_group=8`). Routing is simple top-6-of-128 with `norm_topk_prob=True`.

5. **LayerNorm (not RMSNorm):** Uses LayerNorm with `norm_eps=1e-5`; most modern LLMs use RMSNorm.

6. **`rescale_prenorm_residual=True`:** Non-standard residual scaling applied before norm.

7. **Wormhole-targeted Mamba:** Existing Mamba TTNN implementation targets Wormhole; BlackHole support will need verification.

---

## Weight Mapping

| HuggingFace Key | Component | Notes |
|---|---|---|
| `model.embed_tokens.weight` | Embedding | Standard |
| `model.norm.weight` / `.bias` | LayerNorm | Final LayerNorm before LM head |
| `model.layers.{i}.norm.weight` | LayerNorm | Pre-block norm |
| `model.layers.{i}.mamba.*` | Mamba2Layer | SSM-specific weights (in_proj, conv1d, out_proj, A_log, D, dt_bias, norm) |
| `model.layers.{i}.self_attn.q_proj.weight` | DenseAttention / MoEAttention | Q projection |
| `model.layers.{i}.self_attn.k_proj.weight` | DenseAttention / MoEAttention | K projection |
| `model.layers.{i}.self_attn.v_proj.weight` | DenseAttention / MoEAttention | V projection |
| `model.layers.{i}.self_attn.o_proj.weight` | DenseAttention / MoEAttention | Output projection |
| `model.layers.{i}.mlp.gate_proj.weight` | DenseMLP | Dense layers only |
| `model.layers.{i}.mlp.up_proj.weight` | DenseMLP | Dense layers only |
| `model.layers.{i}.mlp.down_proj.weight` | DenseMLP | Dense layers only |
| `model.layers.{i}.block_sparse_moe.gate.weight` | MoEGate | Router |
| `model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight` | MoEExperts | Expert gate proj |
| `model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight` | MoEExperts | Expert down proj |
| `model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight` | MoEExperts | Expert up proj |
| `model.layers.{i}.block_sparse_moe.shared_expert.gate_proj.weight` | SharedExpert | |
| `model.layers.{i}.block_sparse_moe.shared_expert.up_proj.weight` | SharedExpert | |
| `model.layers.{i}.block_sparse_moe.shared_expert.down_proj.weight` | SharedExpert | |
| `lm_head.weight` | LMHead | Not tied to embedding |

---

## Implementation Order (Topological)

1. **Embedding** — simple lookup; reuse `tt_transformers/tt/embedding.py`
2. **LayerNorm** — `distributed_norm.py`; trivial adaptation for LayerNorm vs RMSNorm
3. **RoPE** — reuse `tt_transformers/tt/rope.py` with rope_theta=10000
4. **DenseAttention** — GQA with KV cache; reuse `tt_transformers/tt/attention.py` (adapt 32Q/2KV, kv_replication=2)
5. **DenseMLP** — reuse `tt_transformers/tt/mlp.py` with relu2 activation swap
6. **MoEGate** — adapt `deepseek_v3/tt/moe_gate.py` (128 experts, no group constraint)
7. **MoEExperts** — adapt `deepseek_v3/tt/experts.py` (EP=4, 32 experts/device, relu2)
8. **SharedExpert** — adapt `deepseek_v3/tt/mlp/shared_expert.py`
9. **MoEAttention** — same as DenseAttention (shared implementation)
10. **Mamba2Layer** — most novel; adapt `wormhole/mamba/tt/mamba_ssm.py` for Mamba-2 multi-head SSM; verify BlackHole compatibility
11. **LMHead** — reuse `tt_transformers/tt/lm_head.py`

---

## Tensor-Parallel Plan (4× BlackHole, 1×4 mesh)

**Device DRAM:** 31.88 GiB per chip
**Total model size:** ~58.7 GiB BF16 (MoE experts dominate at ~54.7 GiB)
**Per-device after TP/EP sharding:** ~14.68 GiB (within 50% headroom budget of 15.94 GiB)

Sharding is **required** — the model does not fit on a single BlackHole device.

| Component | Parallelism | Strategy | Notes |
|---|---|---|---|
| Embedding | Shard (vocab-parallel) | `ShardTensorToMesh(dim=-1)` | vocab 131072 ÷ 4 = 32768 rows/device |
| LayerNorm | Replicate | `ReplicateTensorToMesh` | Tiny weights; replicate for norm stability |
| RoPE | Replicate | `ReplicateTensorToMesh` | Position tables replicated |
| Mamba2Layer | Shard (TP=4) | `ShardTensorToMesh(dim=-1)` for in_proj | Shard along mamba_heads dim (64÷4=16/device); `in_proj` column-parallel, `out_proj` row-parallel + all_reduce |
| DenseAttention | Shard (TP=4) + KV-replication×2 | Column-parallel QKV, row-parallel O | q_heads=32÷4=8/device; kv_heads=2 < 4 → each KV head replicated on 2 devices; chip-local SDPA |
| DenseMLP | Shard (TP=4) | Standard column+row parallel | gate/up col-parallel, down row-parallel + all_reduce |
| MoEGate | Shard (TP=4) | `ShardTensorToMesh(dim=-1)` | hidden→128 experts sharded |
| MoEExperts | Shard (EP=4) | Expert parallelism | 128 experts ÷ 4 = 32 experts/device per layer; all-to-all dispatch + gather CCL |
| SharedExpert | Shard (TP=4) | Column+row parallel | intermediate 3712÷4=928/device |
| LMHead | Shard (vocab-parallel) | `ShardTensorToMesh(dim=-1)` | 131072÷4=32768 rows/device; concat logits via all_gather |

**KV-replication judgment:** With `kv_heads=2 < num_devices=4`, the planner outputs `kv_replication: 2` — each KV head's weights and cache are replicated onto 2 devices (the pair serving the Q-head group for that KV head). Q-heads shard 8/device normally; SDPA remains chip-local (no cross-device attention).

**MoE EP judgment:** With `n_group=1` (no group constraint), routing is pure top-6-of-128. At EP=4 each device owns 32 experts. Average active experts per device per token = 6/4 = 1.5, with possible imbalance. First-pass correctness with synchronous `ttnn.all_gather` dispatch; switch to async CCL in optimization phase.

**KV cache headroom warning:** With only ~1.26 GiB/device headroom below the 50% budget, paged KV cache is strongly recommended. For 6 dense attention layers at BF16: ~0.78 GiB at seq=8192, but ~3.1 GiB at seq=32768. Confirm maximum serving sequence length before finalizing cache design.

---

## Use cases

| Name | Input | Output | needs_ar | HF class | Metric | Threshold | Components used |
|---|---|---|---|---|---|---|---|
| text_generation | text | text | true | NemotronHForCausalLM | bleu | HF - 1.0 | Embedding, LayerNorm, RoPE, Mamba2Layer, DenseAttention, DenseMLP, MoEGate, MoEExperts, SharedExpert, MoEAttention, LMHead |

---

## Open Questions / Risks

1. **Mamba-2 TTNN kernel:** The existing `mamba_ssm.py` implements Mamba-1 selective scan via custom TTNN kernels (`prefix_scan`, `hc_sum_reduce`). Mamba-2 (SSD kernel) has a different mathematical structure. Either adapt the existing kernels or implement a new SSD kernel. This is the highest-risk component.

2. **BlackHole Mamba compatibility:** Existing Mamba implementation targets Wormhole. BlackHole hardware differences (memory layout, compute tiles) need verification.

3. **ReLU² activation:** Verify `relu2` is available as a fused TTNN op or implement as `ttnn.relu(x) ** 2` / `x * ttnn.relu(x)`.

4. **Hybrid layer routing:** The dispatch logic (selecting Mamba-2 vs MoE-attn vs dense-attn based on `hybrid_override_pattern`) is model-specific infrastructure to implement.

5. **KV cache for only 6 dense layers:** KV cache management must skip Mamba-2 layers (their state is the SSM hidden state, not KV pairs).
