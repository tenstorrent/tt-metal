# DeepSeek V3 Prepare Weights — Design Document

Design doc for the **prepare_weights** feature: building device-resident, fused (blitz decode) weights from a HuggingFace-style state dict. The intended flow is: **load full HF tensors → prepare_weights (key mapping, transpose, split) → blitz_decode_weights (shard onto single device or across mesh)**. Update this doc as the feature evolves.

---

## 1. Motivation

- **Decouple weight preparation from the model class.** Weight loading and fusion live in a separate, reusable path so we can load and fuse once (e.g. at startup) and pass a single `DeepSeekV3Weights` into the model, and test with synthetic or real checkpoints without tying to socket I/O.

- **Single pipeline from full HF to sharded device weights.** We take **full HuggingFace state dict tensors** (logical model shapes), run **prepare_weights** for key mapping, transposes, and splits (e.g. `kv_b_proj` → `kv_b1` / `kv_b2`), then **blitz_decode_weights** fuses and **shards** them onto the target device(s). On a single device, blitz produces one shard per layer; on a mesh (e.g. 4×2), blitz shards across the mesh (MLA-TP along 2, MoE shared TP=8, etc.). No expansion or pre-slicing in prepare_weights for mesh—blitz does all sharding.

- **Support both dense and MoE layers.** DeepSeek V3 has 61 layers: first 3 dense, remaining 58 MoE. The prepared type is a union so call sites can use `isinstance(layer, DeepSeekV3MoELayerWeights)` and access MoE-only fields when present.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  state_dict (HF convention, full logical shapes)                        │
│  model.layers.{i}.self_attn.*, model.layers.{i}.mlp.*, ...              │
│  e.g. q_b (24576, 1536), o_proj (7168, 16384), kv_b (32768, 512), ...   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  prepare_weights(state_dict, device, num_layers, first_k_dense_replace)  │
│  • Key mapping & transpose (HF → blitz K,N convention)                   │
│  • _split_kv_b_proj(kv_b_proj) → kv_b1, kv_b2                            │
│  • No expansion: pass full logical tensors through                       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BlitzDecodeWeights(device)                                              │
│  • Fuses groups into shared L1 (q_ab+kv_a, o_proj+gate_mm+norms,          │
│    kv_b1+kv_b2, shared gate/up)                                          │
│  • Shards full logical tensors: single device = one shard;               │
│    mesh (e.g. 4×2) = MLA-TP=2, MoE shared TP=8, etc.                     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DeepSeekV3Weights                                                       │
│  .layers: list[DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights]  │
│  Each layer holds OverlappedTensor views into fused device buffers.      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Files and Roles

| File | Role |
|------|------|
| **prepare_weights.py** | Entry point and types. `prepare_weights()` takes a state dict and device; does key mapping, transpose, and `_split_kv_b_proj`; calls `BlitzDecodeWeights` per layer. No TP expansion—caller passes full logical (mesh) or per-device slice (single-device tests). Defines `DeepSeekV3Weights`, `DeepSeekV3DenseLayerWeights`, `DeepSeekV3MoELayerWeights`. |
| **blitz_decode_weights.py** | Weight fusion and **sharding**. Fuses weight groups into shared L1; when device is a mesh, shards full logical tensors across the mesh (mla_tp, moe_tp). Exposes `BlitzDecodeWeights`, `OverlappedTensor`; `get_tt_o_proj_and_gate_mm_weights` requires a gate_mm tensor (dense layers pass dummy zeros). |
| **model.py** | Host I/O and prefill/decode orchestration; intended to accept optional `DeepSeekV3Weights`. |
| **tests/unit_tests/test_prepare_weights.py** | Unit tests: synthetic state dicts with **full logical HF shapes** for 4×2 mesh tests, and **per-device slice shapes** for single-device tests; verifies prepare_weights + blitz produce correct types and shapes. |

---

## 4. Data Types

### 4.1 Layer weight containers

- **DeepSeekV3DenseLayerWeights** — Layers `0 .. first_k_dense_replace - 1`. Attention: `q_a_proj`, `q_b_proj`, `kv_a_proj`, `kv_b1_proj`, `kv_b2_proj`; output + norms: `o_proj`, `attn_norm`, `q_norm`, `kv_norm`, `ffn_norm`. No gate_mm or shared expert.

- **DeepSeekV3MoELayerWeights** — Layers `first_k_dense_replace .. num_layers - 1`. Same as dense plus `gate_mm`, `shared_gate_proj`, `shared_up_proj`.

- **DeepSeekV3Weights** — `layers: list[DeepSeekV3LayerWeights]` where `DeepSeekV3LayerWeights = DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights`.

All weight fields are **OverlappedTensor** views (see blitz_decode_weights): `tensor_shape`, `shard_shape`, `core_range_set`, `dtype`, `tile_shape`, `byte_offset` into a shared fused L1 buffer.

### 4.2 Blitz fusion groups (reference)

| Fusion method | Outputs | Notes |
|---------------|--------|--------|
| `get_tt_q_ab_proj_and_kv_a_proj_weights` | q_a_proj, q_b_proj, kv_a_proj | WIDTH_SHARDED; q_b shuffled for Qnope/Qrope. |
| `get_tt_o_proj_and_gate_mm_weights` | o_proj, gate_mm, attn_norm, q_norm, kv_norm, ffn_norm | gate_mm required (dense: dummy zeros); six outputs. |
| `get_tt_kv_b12_proj_weights` | kv_b1_proj, kv_b2_proj | HEIGHT_SHARDED. |
| `get_tt_gate_up_proj_weights` | shared_gate_proj, shared_up_proj | HEIGHT_SHARDED; MoE only. |

---

## 5. State Dict Convention and Sharding Model

**Design principle:** We load **full HuggingFace tensors** (logical model shapes). **prepare_weights** only transforms (transpose, split); it does not expand or slice for TP. **blitz_decode_weights** receives full logical shapes and **shards** them onto one device or across the mesh.

### 5.1 HF shapes (full logical)

Real HuggingFace checkpoints use these **full logical** shapes (one row per weight). Linear weights are `(out_features, in_features)`.

| Weight | HF key (under `model.layers.{i}.`) | HF shape (full logical) | After prepare_weights (.T / split) |
|--------|-------------------------------------|--------------------------|------------------------------------|
| q_a_proj | self_attn.q_a_proj.weight | (1536, 7168) | (7168, 1536) |
| q_b_proj | self_attn.q_b_proj.weight | **(24576, 1536)** | (1536, 24576) |
| kv_a_proj | self_attn.kv_a_proj_with_mqa.weight | (576, 7168) | (7168, 576) |
| kv_b_proj | self_attn.kv_b_proj.weight | **(32768, 512)** | split → kv_b1 (16384, 512), kv_b2 (512, 16384) |
| o_proj | self_attn.o_proj.weight | **(7168, 16384)** | (16384, 7168) |
| gate_mm | mlp.gate.weight | (256, 7168) | (7168, 256); MoE only |
| attn_norm | input_layernorm.weight | (7168,) | (1, 7168) |
| q_norm | self_attn.q_a_layernorm.weight | (1536,) | (1, 1536) |
| kv_norm | self_attn.kv_a_layernorm.weight | (512,) | (1, 512) |
| ffn_norm | post_attention_layernorm.weight | (7168,) | (1, 7168) |
| shared_gate_proj | mlp.shared_experts.gate_proj.weight | **(2048, 7168)** | (7168, 2048); MoE only |
| shared_up_proj | mlp.shared_experts.up_proj.weight | **(2048, 7168)** | (7168, 2048); MoE only |

### 5.2 Who shards what

- **Single device:** Caller may pass either (a) full logical state dict, or (b) a **per-device slice** (e.g. q_b (12288, 1536), o_proj (7168, 8192), kv_b (16384, 512), shared (256, 7168)) for testing. blitz then places that single shard on the device. Tests use (b) for single-device and (a) for 4×2.
- **Mesh (e.g. 4×2):** Caller passes **full logical** state dict. prepare_weights passes it through unchanged (after transpose/split). blitz_decode_weights shards: MLA-TP=2 (q_b, o_proj, kv_b1, kv_b2), MoE shared TP=8 (shared_gate_proj, shared_up_proj), rest replicated.

So: **full HF tensors → prepare_weights (transform only) → blitz (shard to device/mesh).**

### 5.3 Per-device logical shapes (after blitz sharding, 4×2)

| Weight | Full logical (post prepare_weights) | Per-device logical (4×2, MLA-TP=2, MoE-TP=8) |
|--------|--------------------------------------|-----------------------------------------------|
| q_b_proj | (1536, 24576) | (1536, 12288) per device |
| o_proj | (16384, 7168) | (8192, 7168) per device |
| kv_b1_proj | (16384, 512) | (8192, 512) per device |
| kv_b2_proj | (512, 16384) | (512, 8192) per device |
| shared_gate_proj / shared_up_proj | (7168, 2048) | (7168, 256) per device |
| q_a_proj, kv_a_proj, gate_mm, norms | — | Replicated (full shape per device) |

---

## 6. kv_b_proj Split

HF stores one `kv_b_proj`; blitz expects two matrices. **prepare_weights** splits via `_split_kv_b_proj`.

- **Supported HF shapes:** (16384, 512) per-device and (32768, 512) full logical. `out_features = num_heads * 256` (256 = qk_nope_head_dim + v_head_dim).
- **Algorithm:** `num_heads = out_features // 256`; reshape to `(num_heads, 256, 512)`; first 128 dims → kv_b1 (no transpose); last 128 → reshape then `.T` → kv_b2.
- **Result:** (16384, 512) → kv_b1 (8192, 512), kv_b2 (512, 8192); (32768, 512) → kv_b1 (16384, 512), kv_b2 (512, 16384).

Constants: `qk_nope_head_dim=128`, `v_head_dim=128`, `kv_lora_rank=512`.

---

## 7. Dense vs MoE and gate_mm

- **Dense layers** (`i < first_k_dense_replace`): No `mlp.gate` or `mlp.shared_experts.*` in the state dict. We pass a **dummy** gate_mm `torch.zeros(7168, 256, dtype=torch.bfloat16, device=...)` to `get_tt_o_proj_and_gate_mm_weights` and ignore the returned gate_mm OverlappedTensor when building `DeepSeekV3DenseLayerWeights`. We do not call `get_tt_gate_up_proj_weights`.

- **MoE layers** (`i >= first_k_dense_replace`): Real `gate_mm` and `shared_gate_proj` / `shared_up_proj` from the state dict; all six o_proj+gate_mm+norms and both shared expert OverlappedTensors are stored in `DeepSeekV3MoELayerWeights`.

---

## 8. Testing

- **test_prepare_dense_layer_single_layer(device)** — Synthetic state dict with **per-device** shapes (e.g. q_b (12288, 1536), o_proj (7168, 8192)); one dense layer; asserts type and tensor_shape. Skips if not slow dispatch or grid < 12×10.

- **test_prepare_moe_layer_single_layer(device)** — Same for one MoE layer (per-device shapes); asserts MoE fields and shared_gate_proj / shared_up_proj tensor_shape (7168, 256). Skips as above and if grid would require >120 compute banks for GATE_UP.

- **test_prepare_dense_layer_single_layer_4x2(bh_2d_mesh_device)** — **Full logical** state dict (q_b (24576, 1536), o_proj (7168, 16384), kv_b (32768, 512)); 4×2 mesh; one dense layer. Verifies prepare_weights + blitz shard correctly.

- **test_prepare_moe_layer_single_layer_4x2(bh_2d_mesh_device)** — Full logical state dict including shared (2048, 7168); one MoE layer on 4×2. Same sharding checks.

- **test_prepare_real_weights** — Placeholder for real checkpoint testing.

---

## 9. Multi-Device Sharding (blitz_decode_weights)

**BlitzDecodeWeights** is responsible for sharding full logical tensors onto the device or mesh.

- **Single device:** One `device`; blitz expects either full logical or a single shard (e.g. from tests). Fused tensors are placed on that device’s grid (e.g. 12×10 minimum).

- **Mesh (e.g. 4×2):** `BlitzDecodeWeights(device)` with a MeshDevice sets `mla_tp=2`, `moe_tp=8`. It expects **full logical** input shapes (e.g. q_b (1536, 24576), o_proj (16384, 7168)). It uses mesh mappers to shard: MLA weights (q_b, o_proj, kv_b1, kv_b2) along TP dimension 2; MoE shared expert (gate_proj, up_proj) across all 8 devices (TP=8); q_a, kv_a, gate_mm, norms replicated. No expansion in prepare_weights—blitz does all sharding.

Pipeline parallelism (layers split across devices) and further tensor parallelism variants are out of scope for this doc; see code and overlap specs in blitz_decode_weights.

---

## 10. Integration Summary

- **HF state dict:** Full logical shapes (see §5.1). Single checkpoint; no pre-slicing by the loader.
- **prepare_weights:** Key mapping, transpose, `_split_kv_b_proj`. Passes tensors through to blitz **without** TP expansion. Single-device tests may use per-device slice state dicts; mesh tests use full logical.
- **blitz_decode_weights:** Fuses weight groups and **shards** full logical tensors: single device = one shard, mesh = MLA-TP=2, MoE shared TP=8. Requires gate_mm in `get_tt_o_proj_and_gate_mm_weights` (dense: dummy zeros). Returns OverlappedTensors used by the fused decode ops.

---

## 11. Future Work / Open Points

- **Integrate with DeepSeekV3:** Pass `DeepSeekV3Weights` into the model and wire OverlappedTensors to Pre-SDPA, Post-SDPA, MoE, shared expert ops.
- **Real checkpoint test:** Implement `test_prepare_real_weights` with a real HF checkpoint; validate shapes and optionally numerical consistency.
- **Config-driven constants:** Move `num_layers`, `first_k_dense_replace`, and kv_b split constants into a small config to stay in sync with the reference model.
- **Lazy / on-demand loading:** Optionally prepare weights for a subset of layers or stream from checkpoint.

---

## 12. Changelog

- **Initial:** prepare_weights API, DeepSeekV3Weights / Dense / MoE types, state dict mapping, kv_b split, unit tests, placeholder for real weights.
- **Multi-device / blitz sharding:** Documented flow: full HF tensors → prepare_weights (transform only) → blitz (shard to device/mesh). Removed TP expansion from prepare_weights; blitz does all sharding. Tests: single-device use per-device slice, 4×2 use full logical.
- **Design doc refactor:** Cleaned stale “current blitz assumes” and “expand for TP” wording; centered doc on full HF → prepare_weights → blitz sharding; simplified §5 (one table for full logical, one for per-device after sharding), §6 (kv_b supports both 16384 and 32768), §8 (test descriptions), §9–§10 (sharding model and integration).
