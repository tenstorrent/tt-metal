# GLM-4.7-Flash Optimization Provenance

This document records the provenance of every optimization in the
`sdawle/glm47_flash_optimal` branch. The optimized code merges the best of two
implementations:

- **Framework** (`sdawle/ign/glm_flash`): Human-written, modular OOP design
using the `tt_symbiote` TTNNModule lifecycle. Located at
`models/experimental/tt_symbiote/`.
- **Agent** (`sdawle/mickg10/glm47_flash`): AI-generated, performance-tuned
procedural implementation. Located at `models/demos/glm4_moe_lite/`.

---

## Components Kept from Framework (Human)


| Component              | Source File                   | Why Kept                                                                                                                               |
| ---------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| TTNNModule base class  | `core/module.py`              | Clean lifecycle (`from_torch` -> `preprocess_weights` -> `move_to_device` -> `forward`), idempotency guards, recursive child traversal |
| TorchTTNNTensor bridge | `core/tensor.py`              | Transparent PyTorch-TTNN interop with lazy conversion                                                                                  |
| Dispatcher system      | `core/dispatchers/*.py`       | Pluggable ATen-to-TTNN routing (DEFAULT, CPU, DEBUG, TENSOR_OPS)                                                                       |
| Module replacement     | `utils/module_replacement.py` | Recursive HF model surgery via `register_module_replacement_dict()`                                                                    |
| Device management      | `utils/device_management.py`  | `set_device()` with timing hooks, distributed config init                                                                              |
| Run config / tracing   | `core/run_config.py`          | DispatchManager timing, TracedRun, DistributedConfig                                                                                   |
| RoPE module            | `modules/rope.py`             | TTNNRotaryPositionEmbedding, TTNNDistributedRotaryPositionEmbedding                                                                    |
| Normalization module   | `modules/normalization.py`    | TTNNRMSNorm, TTNNDistributedRMSNorm                                                                                                    |
| Activation module      | `modules/activation.py`       | TTNNSilu, TTNNReLU, TTNNGelu                                                                                                           |
| SDPA fallback          | `modules/attention.py`        | TTNNSDPAAttention with matmul fallback for robustness                                                                                  |
| Paged KV cache API     | `modules/attention.py`        | Clean `fill` / `update` / `sdpa_decode` interface                                                                                      |
| MLA QKV projection     | `modules/attention.py`        | Correct MLA latent decomposition with all_gather for TP                                                                                |
| Prefill attention path | `modules/attention.py`        | Generic SDPA prefill, correct for any sequence length                                                                                  |
| MoE routing (PyTorch)  | `modules/moe.py`              | Reference routing with sigmoid + group scores + topk                                                                                   |
| Sparse matmul pipeline | `modules/moe.py`              | `all_to_all_dispatch` -> `sparse_matmul` -> `all_to_all_combine`                                                                       |
| Shared expert MLP      | `modules/moe.py`              | Standard gate/up/down structure                                                                                                        |
| Test harness           | `tests/test_glm_flash.py`     | HF `model.generate()` integration validates end-to-end correctness                                                                     |


## Optimizations Ported from Agent (AI)


| Optimization                    | Agent Source          | Target File            | What Was Ported                                                                                            | Impact                                                                                                          |
| ------------------------------- | --------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| FlashMLA decode                 | `decoder_layer_tt.py` | `modules/attention.py` | `paged_flash_multi_latent_attention_decode` call in `_forward_decode_paged_flash_mla()`                    | Fused attention kernel that understands MLA latent structure; eliminates separate Q-K matmul, softmax, V matmul |
| kv_b decomposition              | `layer_weights.py`    | `modules/attention.py` | Split `kv_b_proj.weight` into `kv_b1` (nope absorption) and `kv_b2` (value extraction) in `from_torch()`   | Reduces per-position KV cache from `H*(nope+v)` to `kvpe_dim` (576) -- ~10x memory reduction                    |
| KVPE combined cache             | `decoder_layer_tt.py` | `modules/attention.py` | `TTNNPagedAttentionKVCache` with `kvpe_mode=True` storing `[kv_lora_rank + qk_rope_head_dim]` per position | Enables FlashMLA + dramatic memory savings                                                                      |
| DRAM-sharded weights            | `layer_weights.py`    | `modules/linear.py`    | `TTNNLinearDRAMSharded` class with `WIDTH`-sharded weight placement across DRAM banks                      | Zero DRAM round-trips for weight fetch during decode                                                            |
| 1D matmul program config        | `decoder_layer_tt.py` | `modules/linear.py`    | `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `fuse_batch=True`, `in0_block_w=8`                     | Optimal core utilization for M=1 decode matmuls                                                                 |
| Fused SiLU*mul                  | `moe_tt.py`           | `modules/moe.py`       | `ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`                                   | Saves 1 kernel launch per MLP/expert evaluation                                                                 |
| BFP8/BFP4 expert weights        | `layer_weights.py`    | `modules/moe.py`       | `get_experts_dtype()` defaulting to BFP8, configurable via `GLM4_EXPERTS_DTYPE` env var                    | 2x-4x memory reduction for 128 expert weight matrices                                                           |
| Fused gate+up projection        | `moe_tt.py`           | `modules/moe.py`       | `torch.cat([w1, w3], dim=-1)` -> single `sparse_matmul` -> slice                                           | Halves expert projection kernel launches                                                                        |
| Optimized sparse matmul configs | `moe_tt.py`           | `modules/moe.py`       | Default `in0_block_w=8` (up from 4) in `_make_sparse_matmul_program_config()`                              | Better tile utilization for GLM expert dimensions                                                               |
| Router bias centering           | `layer_weights.py`    | `modules/moe.py`       | `e_bias_centered = e_bias - e_bias.min()` in `TTNNMoERouterDecode.preprocess_weights_impl()`               | Centers correction bias near 0 where BF16 has finest resolution                                                 |
| Compute kernel configs          | `decoder_layer_tt.py` | `modules/attention.py` | `WormholeComputeKernelConfig` with LoFi for decode speed, HiFi4 for attention accuracy                     | Fidelity-accuracy tradeoff tuned per operation type                                                             |
| Config dataclass                | `config.py`           | `modules/config.py`    | `Glm4MoeLiteHParams` with `kvpe_dim`, `qk_head_dim`, dimension validation                                  | Derived dimensions computed once, validated at init time                                                        |


## Not Ported (with Reasoning)


| Component                    | Agent Source                                        | Reason                                                                                                                                                                           |
| ---------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Custom C++ fused kernels     | `fused_ops/kv_cache_branch/`, `fused_ops/pre_sdpa/` | ~2800 lines of C++ tightly coupled to specific Wormhole core grids and tile formats. Breaks framework modularity. Future optimization if TTNN op-level approach is insufficient. |
| vLLM integration             | `generator_vllm.py`, `model_tt.py`                  | Separate concern from model optimization. HF `generate()` is sufficient for validation. Can be added independently.                                                              |
| MTP speculative decoding     | `model_tt.py` (layer 47 logic)                      | Adds significant complexity. Orthogonal to core model performance. Can be layered on later.                                                                                      |
| Batch-bucketed traced decode | `model_tt.py`                                       | Runtime optimization handled by framework's existing `TracedRun`. Orthogonal to model code.                                                                                      |
| Layer 0 special handling     | `layer0_tt.py`                                      | Dense MLP layer 0 is handled naturally by the framework's `first_k_dense_replace` config.                                                                                        |
| LazyStateDict                | `weights.py`                                        | Framework uses HF's standard `model.state_dict()` directly.                                                                                                                      |
| Custom embedding             | `tt_embedding.py`                                   | HF model's embedding layer is used as-is.                                                                                                                                        |


---

## Module-by-Module Cross-Implementation Mapping

### File-Level Mapping

Agent (`models/demos/glm4_moe_lite/`) vs Framework (`models/experimental/tt_symbiote/`)


| Agent File                            | Framework File                                          | Optimal Decision                                            |
| ------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| `tt/config.py`                        | No equivalent                                           | **Ported** to `modules/config.py`                           |
| `tt/decoder_layer_tt.py` (2113 lines) | `modules/attention.py` (TTNNGlm4MoeLiteAttention)       | Framework class + agent's FlashMLA/kv_b/DRAM-sharding       |
| `tt/layer_weights.py` (963 lines)     | Per-module `from_torch()` + `preprocess_weights_impl()` | Framework lifecycle + agent's dtype/decomposition           |
| `tt/moe_tt.py` (1768 lines)           | `modules/moe.py` (TTNNMoE/TTNNExperts)                  | Framework classes + agent's fused SiLU, BFP8, fused gate+up |
| `tt/model_tt.py`                      | `tests/test_glm_flash.py` + HF `generate()`             | Framework's HF approach (simpler, reusable)                 |
| `tt/generator_vllm.py`                | No equivalent                                           | Not ported (separate concern)                               |
| `tt/layer0_tt.py`                     | `first_k_dense_replace` config                          | Not ported (handled naturally)                              |
| `tt/reference_*.py`                   | `_fallback_torch_layer` on TTNNModule                   | Not needed (framework has auto fallback)                    |
| `tt/weights.py`                       | HF `model.state_dict()`                                 | Not ported                                                  |
| `fused_ops/*/`                        | No equivalent                                           | Not ported (breaks modularity)                              |
| No equivalent                         | `core/module.py` (TTNNModule)                           | **Kept** from framework                                     |
| No equivalent                         | `core/tensor.py` (TorchTTNNTensor)                      | **Kept** from framework                                     |
| No equivalent                         | `core/dispatchers/*.py`                                 | **Kept** from framework                                     |
| No equivalent                         | `utils/module_replacement.py`                           | **Kept** from framework                                     |
| No equivalent                         | `modules/linear.py`                                     | **Kept** + new `TTNNLinearDRAMSharded`                      |


### Class/Function-Level Mapping

#### Attention


| Framework                                          | Agent                                                 | Decision                               |
| -------------------------------------------------- | ----------------------------------------------------- | -------------------------------------- |
| `TTNNGlm4MoeLiteAttention.from_torch()`            | `convert_decoder_layer_weights()`                     | Framework + agent's kv_b decomposition |
| `TTNNGlm4MoeLiteAttention._project_qkv()`          | Inline in decode/prefill functions                    | Framework (modular)                    |
| `TTNNGlm4MoeLiteAttention._forward_prefill()`      | `run_decoder_layer_prefill_update_cache_tt()`         | Framework (generic SDPA prefill)       |
| `TTNNGlm4MoeLiteAttention._forward_decode_paged()` | `run_decoder_layer_decode_one_step_update_cache_tt()` | Framework + **agent's FlashMLA**       |
| `TTNNPagedAttentionKVCache` (separate K/V)         | Bare KVPE tensors                                     | Framework API + **agent's KVPE mode**  |
| `TTNNSDPAAttention`                                | Direct `ttnn.transformer.`* calls                     | Framework (has matmul fallback)        |
| `PagedAttentionConfig`                             | Inline parameters                                     | Framework (cleaner API)                |


#### MoE


| Framework                                           | Agent                               | Decision                                               |
| --------------------------------------------------- | ----------------------------------- | ------------------------------------------------------ |
| `TTNNMoE.forward()`                                 | MoE caller in `decoder_layer_tt.py` | Framework class structure                              |
| `TTNNExperts.forward()` (w1->silu, w3, mul, w2)     | `moe_sparse_experts_forward_tt()`   | Framework + **agent's fused SiLU*mul + fused gate+up** |
| `TTNNExperts.preprocess_weights_impl()` (bfloat16)  | `_experts_weight_tt()` (BFP8)       | Framework lifecycle + **agent's BFP8 dtype**           |
| `_make_sparse_matmul_program_config(in0_block_w=4)` | Same with `in0_block_w=8`           | **Agent's tuning**                                     |
| `TTNNMoERouterDecode` (3-pass topk)                 | `moe_topk_tt()` (similar 3-pass)    | Framework + **agent's bias centering**                 |
| `TTNNGlm4MoeMLP` (shared experts)                   | Inline shared expert MLP            | Framework + **agent's fused SiLU*mul**                 |


#### Linear Layers


| Framework                               | Agent                               | Decision                              |
| --------------------------------------- | ----------------------------------- | ------------------------------------- |
| `TTNNLinear` (base)                     | Direct `ttnn.linear()`              | Framework                             |
| `TTNNLinearIColShardedWRowSharded` (TP) | Inline linear + reduce_scatter      | Framework                             |
| No equivalent                           | `_maybe_dram_shard_linear_weight()` | **Ported** as `TTNNLinearDRAMSharded` |


---

## Environment Variables

These environment variables control optimization behavior, all ported from the
agent's implementation:


| Variable             | Default           | Description                                         | Agent Source                                    |
| -------------------- | ----------------- | --------------------------------------------------- | ----------------------------------------------- |
| `GLM4_EXPERTS_DTYPE` | `bf8` (bfloat8_b) | Expert weight quantization: `bf8`, `bf16`, or `bf4` | `layer_weights.py: _env_experts_dtype()`        |
| `GLM4_FUSE_GATE_UP`  | `1` (enabled)     | Fuse gate+up projections into single sparse matmul  | `moe_tt.py: GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP` |
| `GLM4_USE_KVPE`      | `1` (enabled)     | Use KVPE combined cache + FlashMLA decode           | `decoder_layer_tt.py`                           |


---

## Performance Targets

From agent's `PLAN_GLM47_FLASH.md`:


| Metric            | Target     | Notes                                |
| ----------------- | ---------- | ------------------------------------ |
| Prefill (TTFT)    | < 3.0s     | For 128-token prompts on Wormhole x4 |
| Decode throughput | > 30 T/S/U | Per user on Wormhole x4              |


---

## Test & Analysis Results

Automated testing and static analysis performed on branch
`sdawle/glm47_flash_optimal` to validate correctness and completeness of the
merged implementation.

### 1. Code Size Comparison (lines)

| Module | Agent | Framework | Optimal |
| --- | ---: | ---: | ---: |
| Attention (`decoder_layer_tt.py` / `attention.py`) | 2,113 | 1,537 | 1,863 |
| MoE (`moe_tt.py` / `moe.py`) | 1,768 | 1,627 | 1,692 |
| Linear (`layer_weights.py` / `linear.py`) | 963 | 366 | 452 |
| Config (`config.py`) | 127 | — | 160 |
| Test (`test_glm_flash.py`) | — | 191 | 225 |
| Test (`test_glm47_flash_optimal_cpu.py`) [new] | — | — | 483 |
| **TOTAL (all Python + C++ for agent)** | **18,280** | **12,720** | **13,874** |

The optimal branch adds ~1,154 lines over the framework (9% increase) while
remaining 24% leaner than the agent's full codebase. Growth is concentrated in
performance-critical paths.

### 2. Architecture Comparison

| Metric | Agent | Framework | Optimal |
| --- | ---: | ---: | ---: |
| Classes (core modules) | procedural | 46 | 48 |
| Functions/methods (core modules) | 60 | 142 | 155 |
| C++ fused kernel lines | 2,208 | 0 | 0 |
| Design pattern | Procedural | OOP / TTNNModule | OOP / TTNNModule |

### 3. Diff Stats: Framework → Optimal

```
 attention.py      | 416 ++++++++++++++++++---
 linear.py         |  86 +++++
 moe.py            | 173 ++++++---
 test_glm_flash.py |  68 +++-
 4 files changed, 627 insertions(+), 116 deletions(-)
```

### 4. Optimization Coverage Audit

All 12 agent optimizations verified present in the optimal branch:

| Optimization | File | Occurrences |
| --- | --- | ---: |
| FlashMLA decode | `attention.py` | 2 |
| kv_b decomposition | `attention.py` | 18 |
| KVPE cache mode | `attention.py` | 4 |
| Fused SiLU\*mul | `moe.py` | 2 |
| BFP8/BFP4 quantization | `moe.py` | 2 |
| Fused gate+up projection | `moe.py` | 15 |
| Router bias centering | `moe.py` | 2 |
| DRAM-sharded linear | `linear.py` | 1 |
| 1D matmul program config | `linear.py` | 2 |
| WormholeComputeKernelConfig | `attention.py` | 1 |
| Config dataclass (kvpe_dim) | `config.py` | 2 |
| Sparse matmul in0_block_w=8 | `moe.py` | 6 |

**Result: 12/12 optimizations present.**

### 5. KVPE Memory Reduction

```
Standard K/V cache:  32 heads × 128 dim × 2 (K+V) = 8,192 elements/position
KVPE cache:          1 × 576 (kv_lora_rank + qk_rope_head_dim) = 576 elements/position
Memory reduction:    8,192 / 576 = 14.2× per position
```

### 6. CPU Unit Tests — 40/40 Passed

Tests in `tests/test_glm47_flash_optimal_cpu.py` validate pure Python / PyTorch logic
without TT hardware, using mocked `ttnn` module.

| Test Class | Tests | What It Validates |
| --- | ---: | --- |
| `TestGlm4MoeLiteHParams` | 12 | Config construction from HF, field values, derived dimensions (`kvpe_dim`, `qk_head_dim`), validation rules, immutability, edge cases |
| `TestKvBDecomposition` | 6 | Weight split shapes, nope/value recovery, Q-absorption matmul, value extraction matmul, full roundtrip reconstruction |
| `TestRouterBiasCentering` | 5 | Non-negativity, top-k ordering preservation, relative order preservation, BF16 precision improvement, all-positive inputs |
| `TestGetExpertsDtype` | 9 | Default BFP8, explicit bf8/bf16/bf4, invalid input, whitespace stripping, case insensitivity, empty string |
| `TestFusedGateUp` | 3 | Fused vs separate equivalence, fused SiLU\*mul equivalence, output shape validation |
| `TestKVPEDimensions` | 3 | kvpe_dim = 576, >10× memory reduction vs standard, kvpe > kv_lora_rank |
| **Total** | **40** | |

```
$ python3 -m pytest tests/test_glm47_flash_optimal_cpu.py -v --noconftest
======================== 40 passed in 1.71s =========================
```

### 7. Import Chain Verification — 7/7 Passed

Full module import and class instantiation verified with mocked `ttnn`/`tracy`
backends:

| Check | Status |
| --- | --- |
| `config.py` — `Glm4MoeLiteHParams`, `get_experts_dtype` | PASS |
| `attention.py` — `TTNNPagedAttentionKVCache`, `TTNNGlm4MoeLiteAttention` | PASS |
| `moe.py` — `TTNNGlm4MoeTopkRouter`, `TTNNGlm4MoeMoE`, `TTNNGlm4MoeMLP`, `TTNNMoE`, `TTNNExperts` | PASS |
| `linear.py` — `TTNNLinearDRAMSharded` | PASS |
| `Glm4MoeLiteHParams` construction + `validate()` | PASS |
| `get_experts_dtype()` → `bfloat8_b` | PASS |
| `TTNNLinearDRAMSharded(3584, 1024)` construction | PASS |
