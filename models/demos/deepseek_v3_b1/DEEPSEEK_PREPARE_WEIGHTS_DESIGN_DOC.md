# DeepSeek V3 Prepare Weights — Design Document

Design doc for the **prepare_weights** feature: building device-resident, fused (blitz decode) weights from a HuggingFace-style state dict. Update this doc as the feature evolves.

---

## 1. Motivation

- **Decouple weight preparation from the model class.** The host-side `DeepSeekV3` class in `model.py` handles H2D/D2H sockets and prefill/decode orchestration. Weight loading and fusion should live in a separate, reusable path so that:
  - We can load and fuse weights once (e.g., at startup or from a checkpoint) and pass a single `DeepSeekV3Weights` object into the model.
  - The same pipeline can be tested with synthetic state dicts or real checkpoints without tying either to the socket I/O layer.

- **Reuse the blitz decode fusion pipeline.** `blitz_decode_weights.py` already fuses multiple weight tensors into shared L1 buffers (weight overlapping) to reduce fragmentation and buffer count. We want a single entry point that:
  - Takes a standard state dict (e.g., from HuggingFace).
  - Applies the correct key mapping, transposes, and any splits (e.g., `kv_b_proj` → `kv_b1` / `kv_b2`).
  - Calls `BlitzDecodeWeights` for every layer and returns a typed container (`DeepSeekV3Weights`) that the rest of the stack can consume.

- **Support both dense and MoE layers.** DeepSeek V3 has 61 layers: the first 3 are dense (standard MLP), the remaining 58 are MoE (routing gate + shared expert + routed experts). The prepared type is a union so call sites can use `isinstance(layer, DeepSeekV3MoELayerWeights)` and access `gate_mm`, `shared_gate_proj`, `shared_up_proj` only when present.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  state_dict (HF convention)                                              │
│  model.layers.{i}.self_attn.*, model.layers.{i}.mlp.*, ...              │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  prepare_weights(state_dict, device, num_layers=61,                     │
│                   first_k_dense_replace=3)                               │
│  - Key mapping & transpose (HF → blitz K,N convention)                   │
│  - _split_kv_b_proj(kv_b_proj) → kv_b1, kv_b2                            │
│  - Per layer: BlitzDecodeWeights.get_tt_*() fusion calls                 │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DeepSeekV3Weights                                                       │
│  .layers: list[DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights]  │
│  Each layer holds OverlappedTensor views into fused device buffers.      │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DeepSeekV3(..., weights: DeepSeekV3Weights)  [future]                   │
│  Fused ops (Pre-SDPA, Post-SDPA, MoE, …) read from OverlappedTensors    │
│  via fused_tensor + byte_offset + core_range_set.                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Files and Roles

| File | Role |
|------|------|
| **prepare_weights.py** | Entry point and types. Defines `prepare_weights()`, `DeepSeekV3Weights`, `DeepSeekV3DenseLayerWeights`, `DeepSeekV3MoELayerWeights`, and helpers (`_key`, `_split_kv_b_proj`). Imports `BlitzDecodeWeights` and `OverlappedTensor` from `blitz_decode_weights`. |
| **blitz_decode_weights.py** | Weight overlapping (fusion) implementation. Exposes `BlitzDecodeWeights` and configs; `get_tt_o_proj_and_gate_mm_weights(..., gate_mm_weights=None)` supports dense layers when `gate_mm_weights` is omitted. |
| **model.py** | Host I/O and prefill/decode orchestration. Intended to accept an optional `DeepSeekV3Weights` once the real decoder pipeline is wired; no weight loading in this doc’s scope. |
| **tests/unit_tests/test_prepare_weights.py** | Unit tests: random-weight state dicts for one dense and one MoE layer, plus a skip-marked placeholder for real checkpoint tests. |

---

## 4. Data Types

### 4.1 Layer weight containers

- **DeepSeekV3DenseLayerWeights**  
  Used for layers `0 .. first_k_dense_replace - 1`. Contains:
  - Attention: `q_a_proj`, `q_b_proj`, `kv_a_proj`, `kv_b1_proj`, `kv_b2_proj`
  - Output + norms: `o_proj`, `attn_norm`, `q_norm`, `kv_norm`, `ffn_norm`  
  No `gate_mm` or shared expert (dense layers use a standard MLP, not MoE).

- **DeepSeekV3MoELayerWeights**  
  Used for layers `first_k_dense_replace .. num_layers - 1`. Same as dense plus:
  - `gate_mm` (routing gate)
  - `shared_gate_proj`, `shared_up_proj` (shared expert gate/up)

- **DeepSeekV3LayerWeights**  
  Type alias: `DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights`.

- **DeepSeekV3Weights**  
  Top-level container: `layers: list[DeepSeekV3LayerWeights]`.

All weight fields are **OverlappedTensor** views (see `blitz_decode_weights.py`): they reference a shared `fused_tensor` and carry `tensor_shape`, `shard_shape`, `core_range_set`, `dtype`, `tile_shape`, and `byte_offset` so kernels can address each sub-weight inside the fused L1 buffer.

### 4.2 Blitz fusion groups (reference)

| Fusion method | Outputs | Notes |
|---------------|--------|--------|
| `get_tt_q_ab_proj_and_kv_a_proj_weights` | q_a_proj, q_b_proj, kv_a_proj | WIDTH_SHARDED; q_a packed, q_b shuffled for Qnope/Qrope, kv_a shard-reordered. |
| `get_tt_o_proj_and_gate_mm_weights` | o_proj, [gate_mm], attn_norm, q_norm, kv_norm, ffn_norm | gate_mm optional for dense; UINT32 raw container. |
| `get_tt_kv_b12_proj_weights` | kv_b1_proj, kv_b2_proj | HEIGHT_SHARDED. |
| `get_tt_gate_up_proj_weights` | shared_gate_proj, shared_up_proj | HEIGHT_SHARDED, block-sharded; MoE only. |

---

## 5. State Dict Convention and Transformations

We assume keys under `model.layers.{layer_idx}.` as follows. Linear weights are stored in HF as `(out_features, in_features)`; we transpose to `(K, N)` for blitz.

| Blitz / prepare_weights | HF state_dict key (under `model.layers.{i}.`) | Shape (HF) | Transform |
|------------------------|------------------------------------------------|------------|-----------|
| q_a_proj               | self_attn.q_a_proj.weight                      | (1536, 7168) | .T → (7168, 1536) |
| q_b_proj               | self_attn.q_b_proj.weight                      | (12288, 1536) | .T → (1536, 12288) |
| kv_a_proj              | self_attn.kv_a_proj_with_mqa.weight            | (576, 7168)  | .T → (7168, 576) |
| kv_b1, kv_b2           | self_attn.kv_b_proj.weight                     | (16384, 512) | See §6; **kv_b1 not transposed**, kv_b2 transposed |
| o_proj                 | self_attn.o_proj.weight                        | (7168, 8192) | .T → (8192, 7168) |
| gate_mm                | mlp.gate.weight                                | (256, 7168)  | .T → (7168, 256); MoE only |
| attn_norm              | input_layernorm.weight                         | (7168,)      | unsqueeze(0) → (1, 7168) |
| q_norm                 | self_attn.q_a_layernorm.weight                  | (1536,)      | unsqueeze(0) → (1, 1536) |
| kv_norm                | self_attn.kv_a_layernorm.weight                 | (512,)       | unsqueeze(0) → (1, 512) |
| ffn_norm               | post_attention_layernorm.weight                 | (7168,)      | unsqueeze(0) → (1, 7168) |
| shared_gate_proj       | mlp.shared_experts.gate_proj.weight             | (256, 7168)  | .T → (7168, 256); MoE only |
| shared_up_proj         | mlp.shared_experts.up_proj.weight               | (256, 7168)  | .T → (7168, 256); MoE only |

---

## 6. kv_b_proj Split

HuggingFace stores a single `kv_b_proj`; the blitz pipeline expects two matrices. **All linear weights are transposed for blitz except kv_b1.**

- **kv_b1_proj:** (8192, 512) — 64 heads × 128 (qk_nope head dim), HEIGHT_SHARDED on the Qnope grid. **Not transposed**: use HF slice as (8192, 512).
- **kv_b2_proj:** (512, 8192) — 64 heads × 128 (v head dim); **transposed** from HF (8192, 512) to (512, 8192). Blitz then reshapes for HEIGHT_SHARDED placement.

Constants: `num_heads=64`, `qk_nope_head_dim=128`, `v_head_dim=128`, `kv_lora_rank=512`, so HF shape is `(num_heads * 256, 512) = (16384, 512)`.

**Algorithm (`_split_kv_b_proj`):**

1. Reshape (16384, 512) to (64, 256, 512) — (num_heads, head_dim, kv_lora_rank).
2. kv_b1: slice `[:, :128, :]` → (64, 128, 512); reshape to (8192, 512). **No transpose.**
3. kv_b2: slice `[:, 128:, :]` → (64, 128, 512); reshape to (8192, 512); then `.T` → (512, 8192).

This aligns with `models/demos/deepseek_v3/tt/mla/mla1d.py` (wkv_b1 / wkv_b2) and keeps b1 in (out, in) and b2 in (K, N) for blitz.

---

## 7. Dense vs MoE and Optional gate_mm

- **Dense layers** (`i < first_k_dense_replace`): No `mlp.gate` or `mlp.shared_experts.*`. We call `get_tt_o_proj_and_gate_mm_weights(o_proj, None, attn_norm, q_norm, kv_norm, ffn_norm)` so that the 8 gate_mm cores are omitted from the fused buffer; the returned list has 5 elements (no gate_mm OverlappedTensor). We do not call `get_tt_gate_up_proj_weights`.

- **MoE layers** (`i >= first_k_dense_replace`): We pass `gate_mm` and call `get_tt_gate_up_proj_weights(shared_gate, shared_up)`; the layer is stored as `DeepSeekV3MoELayerWeights` with all fields populated.

---

## 8. Testing

- **test_prepare_dense_layer(device)**  
  Builds a synthetic state dict for a single dense layer (random bfloat16), calls `prepare_weights(..., num_layers=1, first_k_dense_replace=1)`, and asserts type and `tensor_shape` for all fields. Skips if device grid is smaller than 12×10.

- **test_prepare_moe_layer(device)**  
  Same idea for two layers (one dense, one MoE); asserts the second layer is `DeepSeekV3MoELayerWeights` and checks shapes including `gate_mm`, `shared_gate_proj`, `shared_up_proj`.

- **test_prepare_real_weights(device)**  
  Placeholder: `@pytest.mark.skip` with a note that real checkpoint testing will be added later.

---

## 9. Multi-device and full-pipeline extension (blitz_decode_weights)

The current blitz decode weights path is built around a **single logical device** (one chip). The full decoder pipeline (Pre-SDPA, Flash MLA, Post-SDPA, MoE, etc.) is written to run on a **MeshDevice**: fused ops take mesh tensors and use `ttnn.get_device_tensors(...)` to get one tensor per device, then launch per-device work with CCL for cross-device sync. To support that pipeline end-to-end, blitz_decode_weights will need to be extended as follows.

### 9.1 Current behavior (single device)

- **BlitzDecodeWeights** takes a single `device` and stores it as `self._device`.
- **Grid validation** uses `self._device.compute_with_storage_grid_size()` and asserts the grid is at least as large as the core ranges in each overlap config (e.g. 12×10 for q_ab + kv_a). This is the **per-device** grid.
- **Placement** uses `ttnn.from_torch(..., device=self._device, mesh_mapper=ttnn.ReplicateTensorToMesh(self._device))`. So:
  - If `device` is a **single Device**: one fused tensor is created on that device; core ranges (0,0)–(11,7), etc. are that device’s logical cores.
  - If `device` is a **MeshDevice**: `ReplicateTensorToMesh` replicates the same tensor to **every** device in the mesh. Each device gets the same logical core layout and the same weight data. Fused ops that call `get_device_tensors(weight_mesh)` then get one weight tensor per device, which is what they use today (e.g. `matmul_weights_tensors_per_device[device_idx]`). So **replicated** weights on a mesh already match how the fused ops consume weights.

So today:

- **Single device:** fully supported; all fusion groups live on that device.
- **Mesh with replication:** supported in principle (pass a MeshDevice; each chip gets a full copy of the fused weights). Not yet exercised in tests; grid validation is still per-device and must hold on every device in the mesh.

What is **not** supported today:

- **Pipeline parallelism:** each device only holds weights for “its” layers (e.g. device 0: layers 0–15, device 1: layers 16–31). That would require preparing weights per stage and using a mesh mapper (or separate calls) so each device only gets the tensors for its layer range, instead of replicating the full model.
- **Tensor parallelism:** sharding a single layer’s weights across multiple devices (e.g. splitting q_a_proj along K or N across 4 chips). Core ranges and shard specs are currently single-device; supporting this would mean defining per-device core ranges and a mesh mapper that shards the weight tensor across devices (e.g. `ShardTensorToMesh` on a dimension), and ensuring fused ops and CCL (e.g. all-reduce after matmul) are aware of the sharding.

### 9.2 Extensions needed for the full decoder pipeline

1. **Explicit MeshDevice support and testing**
   - When `device` is a MeshDevice, keep using `ReplicateTensorToMesh(self._device)` so each device gets a full copy.
   - Ensure grid validation uses the per-device grid (same as today) and that every device in the mesh has a large enough grid (e.g. 12×10). If the API returns a single grid for the mesh, validate that grid; otherwise validate per device if the runtime exposes it.
   - Add a test that builds weights with a small mesh (e.g. 2×1) and runs at least one fused op (e.g. Pre-SDPA) to confirm replicated weights work with `get_device_tensors`.

2. **Pipeline parallelism (layers split across devices)**
   - **Option A:** `prepare_weights` (or a variant) accepts a **stage_id** and **num_stages** and only produces weights for layers in that stage (e.g. stage 0: layers 0–15, stage 1: 16–31). Each stage is prepared with `BlitzDecodeWeights(single_device)` for that device, and the runner only passes that stage’s `DeepSeekV3Weights` to the device. No change to BlitzDecodeWeights internals; only the orchestration (who calls prepare_weights and with which layers) changes.
   - **Option B:** `prepare_weights` takes a MeshDevice and a “layer → device” mapping and uses a mesh mapper that places each layer’s fused tensors only on the assigned device (e.g. no replication of other layers). That would require a custom mesh mapper or multiple `from_torch` calls per layer with device placement constrained to one mesh coordinate.

3. **Tensor parallelism (weight sharding across devices)**
   - Today each fusion group (q_ab+kv_a, o_proj+gate_mm+norms, kv_b1+kv_b2, gate+up) is a single logical tensor sharded over cores on **one** device. To shard a group across devices (e.g. q_a_proj rows split across 4 devices), we’d need:
     - A **mesh mapper** that shards the fused tensor along the chosen dimension across mesh devices (e.g. `ShardTensorToMesh` on a logical dimension that corresponds to “which device”).
     - **Per-device core ranges** so that each device’s shard is placed on the right cores locally, and the overall ShardSpec / memory config describes the cross-device sharding.
     - Fused ops (and possibly kernels) to be aware of sharded weights: e.g. matmul with sharded weights followed by an all-reduce or other CCL. This is a larger change and may require new op variants or config flags.

4. **Full decoder weight coverage**
   - Blitz currently fuses only a subset of decoder weights (q_a/q_b/kv_a, o_proj+gate_mm+norms, kv_b1/kv_b2, shared expert gate/up). The full pipeline also uses (and may want to fuse or overlap):
     - **down_proj** (and expert down_projs in MoE).
     - **Expert weights** (gate_proj, up_proj, down_proj per expert).
     - **Embedding** and **lm_head** (if they are to live on device and be overlapped with other buffers).
   - Extending blitz for these would mean new overlap configs and new `get_tt_*` methods (and corresponding updates to `prepare_weights` and the layer weight dataclasses), following the same pattern: define core ranges and shard layout, then stitch or pack into a single buffer per core (or per device in a multi-device setup).

### 9.3 Summary

| Scenario | Today | Extension |
|----------|--------|-----------|
| Single device | Supported | — |
| Mesh, replicated weights (full model on each device) | Supported in principle | Validate + test with MeshDevice |
| Pipeline parallelism (layers per device) | Not supported | prepare_weights per stage or layer→device mapping + mapper |
| Tensor parallelism (weight shard across devices) | Not supported | ShardTensorToMesh + per-device core ranges + op/CL support |
| Full decoder (down_proj, experts, embed, lm_head) | Partial (only current fusion groups) | New overlap configs and get_tt_* in blitz_decode_weights |

---

## 10. Adapting prepare_weights to branch `bliu/deepseek`

On branch **bliu/deepseek**, `BlitzDecodeWeights` adds **tensor parallelism (TP)** and renames configs. Below is what must change in **prepare_weights.py** when that branch is used.

### 10.1 HF state dict is unchanged; TP/DP are TTNN placement only

The HuggingFace state dict and checkpoint stay the same regardless of TP/DP. We always load **full-model** weights (e.g. q_b (1536, 12288), o_proj (8192, 7168), etc.). The **TP/DP factors** (e.g. mla_tp=2, moe_tp=8 on a 4×2 mesh) refer to how the **TTNN tensors** (the fused device buffers) are **sharded across the decoder mesh** — i.e. a runtime/placement concern, not a different checkpoint layout. So **prepare_weights** always passes the same shapes to blitz (the full-model shapes derived from the HF state dict). BlitzDecodeWeights is responsible for slicing/sharding those full-model tensors and placing them on the mesh when the device is multi-device.

### 10.2 Branch changes in blitz_decode_weights.py

- **Config renames:** `QAB_KVA_PROJ_OverlapConfig` → `QAB_KVA_PROJ_SingleDeviceOverlapSpec`, and similarly for the other three (constants use `*_SINGLE_DEVICE_OVERLAP_SPEC`). Logic is equivalent; only names differ.
- **BlitzDecodeWeights.__init__:** Sets `self.mla_tp` and `self.moe_tp` from the device (single device → 1/1; 4×2 mesh → 2/8). These drive how the **output** TTNN tensors are sharded (e.g. `ShardTensor2dMesh`), not the **input** shapes from the caller.
- **Intended input shapes (full-model from HF):** Callers (e.g. prepare_weights) always pass **full-model** weights: q_b (1536, 12288), o_proj (8192, 7168), kv_b1 (8192, 512), kv_b2 (512, 8192), gate/up (7168, 256). When the device is a mesh, blitz should **accept these same shapes**, slice them per TP index internally, and produce the sharded mesh tensor. (Note: the current branch code may still expect TP-concatenated shapes like (1536, 12288×mla_tp); aligning with “HF stays the same” would mean changing blitz to accept (1536, 12288) and do the slicing inside.)
- **Dense layers:** On the branch, `get_tt_o_proj_and_gate_mm_weights` **requires** `gate_mm_weights` (no `None`). So either the branch adds optional `gate_mm` for dense, or prepare_weights passes a dummy (e.g. zeros) of shape (7168, 256) for dense layers.
- **GATE_UP:** `GATE_UP_PROJ_SingleDeviceOverlapSpec` uses different core range groupings and `reshuffle_block_to_height_sharded(weights, core_range_set)`; no change to prepare_weights call signature for gate/up.

### 10.3 Required changes in prepare_weights.py

Because **HF stays the same** and TP/DP only affect how TTNN tensors are sharded on the mesh, prepare_weights continues to load the same state dict and pass **the same full-model shapes** to BlitzDecodeWeights for both single-device and 4×2 mesh:

- **No TP-concatenation in prepare_weights.** We do not build (1536, 12288×mla_tp) or (7168, 256×moe_tp); we always pass (1536, 12288), (7168, 256), etc. from the HF state dict (after transpose/split as today). Blitz is responsible for sharding those full-model tensors across the mesh when the device is multi-device.

- **Dense layers and gate_mm:** If the branch does not support `gate_mm_weights=None`, then for dense layers either add optional `gate_mm` on the branch, or in prepare_weights pass a zero tensor of shape (7168, 256) when `i < first_k_dense_replace`.

- **Branch blitz alignment:** If the current branch API still expects TP-concatenated input shapes (e.g. q_b (1536, 12288×mla_tp)), then the branch should be updated to accept full-model shapes and perform the TP slicing internally, so that prepare_weights can remain agnostic of mla_tp/moe_tp and always pass HF-derived shapes.

### 10.4 Summary

- **HF state dict:** Unchanged; single full-model checkpoint.
- **prepare_weights:** Always passes full-model shapes (same as today); no mesh- or TP-specific logic.
- **BlitzDecodeWeights:** Uses mla_tp/moe_tp to shard the **output** TTNN tensors across the 4×2 decoder mesh; should accept the **same** full-model input shapes and do any per-TP slicing internally.

---

## 11. Future Work / Open Points

- **Integrate with DeepSeekV3:** Pass `DeepSeekV3Weights` into `DeepSeekV3.__init__` and wire OverlappedTensors to Pre-SDPA, Post-SDPA, MoE, and shared expert ops (each op already takes `ttnn.Tensor`; kernels use `byte_offset` where the fused buffer is a raw container).
- **Real checkpoint test:** Implement `test_prepare_real_weights` using a real HuggingFace checkpoint (or a small fixture) and validate numerical consistency or golden outputs.
- **Config-driven constants:** Move `num_layers`, `first_k_dense_replace`, and kv_b split constants (e.g. `_NUM_HEADS`) into a small config or the existing DeepSeek V3 config so they stay in sync with the reference model.
- **Lazy / on-demand loading:** If needed, consider a variant that prepares weights for a subset of layers or streams from checkpoint instead of loading the full state dict up front.

---

## 12. Changelog

- **Initial:** Added prepare_weights API, DeepSeekV3Weights / Dense / MoE types, state dict mapping, kv_b split, optional gate_mm in blitz_decode_weights, and unit tests with random weights plus placeholder for real weights.
- **Multi-device / full pipeline:** Added §9 (Multi-device and full-pipeline extension): current single-device assumption, MeshDevice + replication behavior, and required extensions for pipeline parallelism, tensor parallelism, and full decoder weight coverage.
- **bliu/deepseek branch:** Added §10 (Adapting prepare_weights to branch bliu/deepseek): TP-aware shapes, mla_tp/moe_tp, and required prepare_weights.py changes for single-device vs 4×2 mesh.
