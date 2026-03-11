# Status: GLM-4.7-Flash Modularization

## Current Phase: Phase 4 (Optimize)

## Phase 1: UNDERSTAND (done)

### Architecture Summary
- **Model**: GLM-4.7-Flash (47 layers, MLA attention, MoE)
- **Attention**: DeepSeek-style Multi-Latent Attention
  - Low-rank Q projection: x -> q_a (512) -> layernorm -> q_b (32 * 192) -> split nope/rope -> kv_b1 -> RoPE -> concat
  - Low-rank KV projection: x -> kv_a (576) -> split nope(512)/rope(64) -> layernorm(nope) -> RoPE(rope) -> concat -> cache
  - FlashMLA decode/prefill with paged KVPE cache
  - Output: kv_b2 -> head flatten -> w_o
- **MLP (dense layers 0..first_k_dense_replace-1)**: SwiGLU (gate + up + silu*mul + down)
- **MLP (MoE layers)**: shared expert (dense SwiGLU) + 8 routed experts (top-k routing)
- **Norm**: RMSNorm (pre-norm)
- **KV Cache**: Paged KVPE (fused nope+rope, 576 dims)

### Op Mapping
| PyTorch Op | TTNN Equivalent | Used In |
|-----------|----------------|---------|
| nn.Linear | ttnn.linear | All projections |
| nn.Embedding | ttnn.embedding | Token embedding |
| RMSNorm | models.common.rmsnorm.RMSNorm | Pre-norm |
| F.silu * up | ttnn.mul with SILU activation | MLP gate*up |
| RoPE | ttnn.experimental.rotary_embedding_llama | Q and KV rope dims |
| FlashMLA decode | ttnn.transformer.paged_flash_multi_latent_attention_decode | Attention |
| FlashMLA prefill | ttnn.transformer.flash_mla_prefill | Attention |
| KV cache update | ttnn.experimental.paged_update_cache | Per-step cache write |
| KV cache fill | ttnn.experimental.paged_fill_cache | Prefill cache fill |
| concat | ttnn.concat | Q/KV nope+rope merge |
| slice | ttnn.slice | Q/KV nope/rope split |
| top-k | ttnn.topk (via moe_tt) | MoE routing |
| all_reduce | ttnn.all_reduce | TP reduction |

### Modularity Problems Identified
1. `decoder_layer_tt.py` is a 2113-line monolith with 12+ nested helper functions
2. 25+ env vars parsed inline via `os.environ.get()` on every call
3. Attention, MLP, MoE, memory management all interleaved in one function
4. `model_tt.py` (2685 lines) mixes weight loading, trace capture, MTP, and vLLM plumbing
5. No separation between "what to compute" and "how to shard/place in memory"

## Phase 2: PROFILE BASELINE (done)

- [x] Profile single layer 0 decode — 106 ops, 4533 us (refactored path)
- [x] Profile dense layers (pre-refactoring) — 1316 ops, 54.3 ms
- [x] Profile full model decode (4 tokens) — completed on 4x Wormhole (1.98 tok/s, 504.6 ms/token)
- [x] Record baselines in baseline.yaml
- [x] Identify top-5 bottleneck ops

## Phase 3: MODULARIZE (done)

- [x] Extract runtime_config.py
- [x] Extract linear_helpers.py
- [x] Extract attention_decode.py
- [x] Extract mlp_decode.py
- [x] Wire into decoder_layer_tt.py (2113 -> 1098 lines)
- [x] Extract mtp_forward.py and decode_trace_state.py
- [x] Hardware validation: layer 0 decode PCC > 0.999, MoE layer 1 PASS

## Phase 4: OPTIMIZE (in progress)

### Full Model Baseline (4x Wormhole)
- Decode: **1.98 tok/s** (504.6 ms/token)
- Device kernel: 44.2 ms/device (8.8% of latency)
- Host dispatch overhead: **91.2%**

### Top Optimization Targets (from full model profile)
1. **Host dispatch overhead (91.2%)**: Enable metal trace capture/replay — single biggest win
2. **Data movement ops (24.5% of kernel)**: FillPad 10.5%, Permute 4.9%, Repeat 4.5%, Clone 3.4%, Transpose 3.3%
3. **Layout conversion (9.9% of kernel)**: TilizeDeviceOperation — pre-tilize inputs
4. **MoE ops (8.6% of kernel)**: SparseMatmul 5.7% + ExpertRemap 2.9% — fuse kernels
5. **Binary/Unary element-wise (9.2% of kernel)**: 328 ops, 4084 us — may be inherent

### Detailed Report
- `experiments/glm4_full_model_profile_report.md`
- `experiments/glm4_full_model_ops_profile.csv` (7,496 ops, all devices)
- `experiments/glm4_full_model_ops_summary.csv` (per-op summary)

## Phase 5: INTEGRATE — pending
