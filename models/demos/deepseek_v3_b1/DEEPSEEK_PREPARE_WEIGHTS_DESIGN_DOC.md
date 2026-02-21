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

### 5.4 How HF weights map onto MoE and MLP implementation

DeepSeek V3 has **two layer types**: the first few layers are **dense** (one MLP, no routing); the rest are **MoE** (router + shared expert + 256 routed experts). The table below maps HF state-dict keys to what **prepare_weights** produces and which **fused ops** in this directory consume them.

**Layer types in HF:**

- **Dense layers** (e.g. `i = 0, 1, 2`): `model.layers.{i}.mlp` has a single “expert” with `gate_proj`, `up_proj`, `down_proj` (no `gate`, no `shared_experts`, no `experts`).
- **MoE layers** (e.g. `i = 3..60`): `model.layers.{i}.mlp` has `gate` (router), `shared_experts.gate_proj/up_proj/down_proj`, and `experts.{0..31}.gate_proj/up_proj/down_proj`.

**Implementation in this directory:**

| HF concept | HF keys (under `model.layers.{i}.`) | prepare_weights / blitz | Fused op that uses it |
|------------|-------------------------------------|--------------------------|------------------------|
| **Attention** (all layers) | self_attn.q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj + norms | Yes → OverlappedTensors in q_ab_kv_a, kv_b12, o_proj_gate_mm_norms | Pre-SDPA, Post-SDPA (not covered in this table in detail). |
| **Router** (MoE only) | mlp.gate.weight | Yes → gate_mm in o_proj_gate_mm_norms | **MoeOp**: gate matmul (step 2); expects gate_mm on L1. |
| **Shared expert gate/up** (MoE only) | mlp.shared_experts.gate_proj.weight, up_proj.weight | Yes → gate_up fusion (shared_gate_proj, shared_up_proj) | **SharedExpertOp**: fused gate/up matmul (steps 2–4); reads gate_up L1 buffer. |
| **Shared expert down** (MoE only) | mlp.shared_experts.down_proj.weight | Yes → plain tensor shared_down_proj (TP-sharded) | **SharedExpertOp** + **DownProj**: down_proj matmul (step 7); reads shared_down_proj. |
| **Dense MLP** (dense layers only) | mlp.gate_proj, mlp.up_proj, mlp.down_proj | **No** (Phase 2) | **MlpOp** (test_mlp.py): one “expert” gate/up/down; today tests use synthetic weights, not HF. |
| **Routed experts** (MoE only) | mlp.experts.{e}.gate_proj, up_proj, down_proj (e=0..31) | **No** (Phase 3) | **MoeOp** (MoeRoutedExpertOp): gate_proj, up_proj, down_proj per expert from **DRAM**; today tests use create_expert_matmul_tensors (synthetic). |

**Summary:**

- **Dense layers:** HF has one set of `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`. The **MlpOp** path (test_mlp.py) implements that single-expert MLP; **prepare_weights** does not yet load these three (Phase 2).
- **MoE layers:** HF has (1) router `mlp.gate`, (2) shared expert `mlp.shared_experts.{gate,up,down}_proj`, (3) 256 routed experts `mlp.experts.{e}.{gate,up,down}_proj`. We **do** load (1) and (2) via prepare_weights → gate_mm, gate_up fusion, and shared_down_proj. The **MoeOp** uses gate_mm and the **SharedExpertOp** uses gate_up + shared_down_proj. Routed expert weights (3) are **not** in prepare_weights yet; the MoE op expects them in DRAM (e.g. from create_expert_matmul_tensors in tests).

So: **HF dense MLP** → MlpOp (one gate/up/down); **HF MoE** → MoeOp (router + SharedExpertOp for shared expert + DRAM experts for routed). prepare_weights currently feeds only attention + router + shared expert; dense MLP and routed experts are future work.

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

## 11. Serialization and Deserialization

### 11.1 Motivation

`prepare_weights` is expensive: it transposes, splits, fuses, and shards every layer through blitz. For production we want to run it **once offline** and serialize the result so that at model runtime we can load pre-prepared weights directly onto the device(s) without re-doing the fusion. The runtime path should be: **load manifest → load fused tensors → reconstruct `DeepSeekV3Weights` → run model**.

### 11.2 Serialization primitives

- **`ttnn.dump_tensor(file_name, tensor)`** — saves a `ttnn.Tensor` to `.tensorbin` (FlatBuffer metadata + binary data). Works for host and device tensors.
- **`ttnn.load_tensor(file_name, device=device)`** — loads a `.tensorbin` and optionally places on device/mesh.

### 11.3 Key insight: fused tensors are shared

Each fusion group (q_ab+kv_a, o_proj+gate_mm+norms, kv_b12, gate_up) produces **one fused `ttnn.Tensor`** in L1 that multiple `OverlappedTensor` views reference via `byte_offset`. For example, q_a_proj, q_b_proj, and kv_a_proj all point to the same fused buffer with different `byte_offset`, `tensor_shape`, `shard_shape`, etc.

So we serialize **one `.tensorbin` per fusion group per layer**, not one per OverlappedTensor field. For 61 layers this gives:

| Layer type | Fusion groups | Files per layer |
|-----------|---------------|-----------------|
| Dense (0–2) | q_ab_kv_a, o_proj_gate_mm_norms, kv_b12 | 3 |
| MoE (3–60) | q_ab_kv_a, o_proj_gate_mm_norms, kv_b12, gate_up | 4 |

Total: 3×3 + 58×4 = **241 `.tensorbin` files** + 241 sidecar `.json` files.

### 11.4 Directory layout

There is **no top-level manifest**. Each layer directory is fully independent and can be prepared, regenerated, or loaded in isolation. Each layer has a `manifest.json` with layer-level metadata and all fusion group field metadata, plus one `.tensorbin` per fusion group.

```
<weights_dir>/
  layer_000/                            # dense layer — independently loadable
    manifest.json                       # layer metadata + all field metadata
    q_ab_kv_a.tensorbin
    o_proj_gate_mm_norms.tensorbin
    kv_b12.tensorbin
  layer_001/
    manifest.json
    ...
  layer_003/                            # MoE layer
    manifest.json
    q_ab_kv_a.tensorbin
    o_proj_gate_mm_norms.tensorbin
    kv_b12.tensorbin
    gate_up.tensorbin                   # MoE only
  ...
  layer_060/
    manifest.json
    ...
```

**Why no top-level manifest:**
- Each layer is fully independent — can be regenerated without touching other layers.
- Parallel writes: different processes prepare different layers with no coordination.
- Per-layer loading: pipeline parallelism loads only its layers by directory name.
- No single point of failure; adding or replacing a layer is just replacing its directory.

### 11.5 Per-layer manifest (`layer_NNN/manifest.json`)

Each layer's manifest contains both global metadata (for validation and self-description) and all fusion group field metadata (for reconstructing OverlappedTensors).

```json
{
  "version": 1,
  "created_time": "2026-02-20T15:30:00Z",
  "hf_model_name": "deepseek-ai/DeepSeek-V3",
  "hf_state_dict_name": "model-00001-of-000XX.safetensors",
  "device_mesh_shape": [4, 2],
  "layer_idx": 0,
  "layer_type": "dense",
  "fusion_groups": {
    "q_ab_kv_a": {
      "tensorbin": "q_ab_kv_a.tensorbin",
      "fields": {
        "q_a_proj": {
          "tensor_shape": [7168, 1536],
          "shard_shape": [7168, 16],
          "core_range_set": [[[0, 0], [11, 7]]],
          "dtype": "BFLOAT8_B",
          "tile_shape": [32, 32],
          "byte_offset": 0
        },
        "q_b_proj": {
          "tensor_shape": [1536, 24576],
          "shard_shape": [1536, 256],
          "core_range_set": [[[0, 0], [11, 7]]],
          "dtype": "BFLOAT8_B",
          "tile_shape": [32, 32],
          "byte_offset": 57344
        },
        "kv_a_proj": {
          "tensor_shape": [7168, 576],
          "shard_shape": [7168, 72],
          "core_range_set": [[[0, 0], [0, 7]]],
          "dtype": "BFLOAT8_B",
          "tile_shape": [32, 32],
          "byte_offset": 0
        }
      }
    },
    "o_proj_gate_mm_norms": {
      "tensorbin": "o_proj_gate_mm_norms.tensorbin",
      "fields": {
        "o_proj": { "..." : "..." },
        "gate_mm": { "..." : "..." },
        "attn_norm": { "..." : "..." },
        "q_norm": { "..." : "..." },
        "kv_norm": { "..." : "..." },
        "ffn_norm": { "..." : "..." }
      }
    },
    "kv_b12": {
      "tensorbin": "kv_b12.tensorbin",
      "fields": {
        "kv_b1_proj": { "..." : "..." },
        "kv_b2_proj": { "..." : "..." }
      }
    }
  }
}
```

MoE layers include an additional `"gate_up"` fusion group with `shared_gate_proj` and `shared_up_proj`.

**Manifest fields:**

| Field | Purpose |
|-------|---------|
| `version` | Schema version for forward compatibility. |
| `created_time` | ISO-8601 timestamp of when this layer was prepared. |
| `hf_model_name` | HF model identifier (e.g. `deepseek-ai/DeepSeek-V3`). |
| `hf_state_dict_name` | Name/pattern of the HF checkpoint files used as input. |
| `device_mesh_shape` | Mesh shape weights were prepared for (e.g. `[4, 2]`). |
| `layer_idx` | Layer index in the model. |
| `layer_type` | `"dense"` or `"moe"`. |
| Per-field: `tensor_shape`, `shard_shape`, `core_range_set`, `dtype`, `tile_shape`, `byte_offset` | Everything needed to reconstruct an `OverlappedTensor` from its fused tensor. |

### 11.6 API

**Per-layer (primary):** Serialization is defined per layer. Use these for single-layer save/load, incremental or parallel preparation, and replacing one layer.

```python
def save_layer(
    layer: DeepSeekV3LayerWeights,
    path: str | Path,
    layer_idx: int,
    *,
    hf_model_name: str,
    hf_state_dict_name: str,
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Serialize one layer to <path>/layer_{layer_idx:03d}/."""

def load_layer(
    path: str | Path,
    device,
    layer_idx: int,
) -> DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights:
    """Deserialize one layer from <path>/layer_{layer_idx:03d}/."""
```

**Full-model (convenience):** Thin wrappers for “save/load all layers”.

```python
def save_weights(weights, path, *, hf_model_name, hf_state_dict_name, device_mesh_shape=(1,1)) -> None:
    """Loop over weights.layers and call save_layer for each."""

def load_weights(path, device, num_layers: int = 61) -> DeepSeekV3Weights:
    """Load layer_000/ .. layer_{num_layers-1}/ via load_layer; return DeepSeekV3Weights(layers=...)."""
```

Layer type (dense vs MoE) comes from each layer’s manifest; `load_weights` does not take `first_k_dense_replace`.

### 11.7 Serialization algorithm (`save_layer`)

1. Create `path` and `path/layer_{layer_idx:03d}/`.
2. Collect all `OverlappedTensor` fields from the layer; group by `id(field.fused_tensor)`.
3. For each unique fused tensor: determine group name, call `ttnn.dump_tensor(layer_dir / f"{group_name}.tensorbin", fused_tensor)`, record per-field metadata.
4. Write `layer_dir / manifest.json` with layer metadata + all fusion group field metadata.

`save_weights` is a loop: `for i, layer in enumerate(weights.layers): save_layer(layer, path, i, ...)`.

### 11.8 Deserialization algorithm (`load_layer`)

1. Read `{path}/layer_{layer_idx:03d}/manifest.json`. Validate version.
2. For each fusion group: load fused tensor with `ttnn.load_tensor(..., device=device)`, reconstruct each field’s `OverlappedTensor` from metadata.
3. Build and return one `DeepSeekV3DenseLayerWeights` or `DeepSeekV3MoELayerWeights` (from manifest `layer_type`).

`load_weights` is: `layers = [load_layer(path, device, i) for i in range(num_layers)]`; return `DeepSeekV3Weights(layers=layers)`.

### 11.9 core_range_set serialization

`ttnn.CoreRangeSet` is serialized as a list of `[[start_x, start_y], [end_x, end_y]]` pairs. On deserialization, reconstruct via `ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(ex, ey)) for (sx, sy), (ex, ey) in ...])`.

### 11.10 Compatibility and validation

- **Mesh shape mismatch:** Each layer manifest contains `device_mesh_shape`. If it doesn't match the runtime device, the loader can raise an error.
- **Version check:** Loader rejects manifests with `version` > supported.
- **Independent regeneration:** To re-prepare a single layer, delete its `layer_NNN/` directory and call `save_layer(layer, path, idx, ...)` for that layer only.
- **Per-layer loading:** Pipeline parallelism calls `load_layer(path, device, i)` only for its assigned layer indices.

### 11.11 Testing

**test_save_load_dense_layer_single_layer(device, tmp_path):**

1. Build one dense layer with `prepare_weights` (synthetic state dict, per-device shapes).
2. Call `save_layer(weights.layers[0], tmp_path, 0, hf_model_name=..., hf_state_dict_name=...)`.
3. Verify on disk: `tmp_path/layer_000/manifest.json` and 3 `.tensorbin` files.
4. Call `deallocate_weights(weights)` then `layer = load_layer(tmp_path, device, 0)`.
5. Assert `isinstance(layer, DeepSeekV3DenseLayerWeights)` and that every OverlappedTensor field matches original (tensor_shape, shard_shape, dtype, tile_shape, byte_offset, core_range_set); same fusion group must share `id(fused_tensor)`.

**test_save_load_moe_layer_single_layer(device, tmp_path):**

Same pattern for one MoE layer: `save_layer` then `load_layer`; verify 4 tensorbins and `DeepSeekV3MoELayerWeights` with gate_mm, shared_gate_proj, shared_up_proj.

Both tests use `tmp_path` and skip unless slow dispatch + grid >= 12×10 (MoE also requires ≥128 cores).

### 11.12 End-to-end flow

```
[Offline / one-time]
  HF checkpoint --> prepare_weights(state_dict, device) --> DeepSeekV3Weights
                                                                  |
                                                          save_weights(weights, path)
                                                                  |
                                                                  v
                                              <weights_dir>/layer_NNN/manifest.json
                                              <weights_dir>/layer_NNN/{group}.tensorbin

[Runtime / every startup]
  load_weights(path, device) --> DeepSeekV3Weights --> model forward pass
```

---

## 12. Future Work / Open Points

- **Integrate with DeepSeekV3:** Pass `DeepSeekV3Weights` into the model and wire OverlappedTensors to Pre-SDPA, Post-SDPA, MoE, shared expert ops.
- **Real checkpoint test:** Implement `test_prepare_real_weights` with a real HF checkpoint; validate shapes and optionally numerical consistency.
- **Config-driven constants:** Move `num_layers`, `first_k_dense_replace`, and kv_b split constants into a small config to stay in sync with the reference model.
- **Lazy / on-demand loading:** Optionally prepare weights for a subset of layers or stream from checkpoint.

---

## 13. Changelog

- **Initial:** prepare_weights API, DeepSeekV3Weights / Dense / MoE types, state dict mapping, kv_b split, unit tests, placeholder for real weights.
- **Multi-device / blitz sharding:** Documented flow: full HF tensors → prepare_weights (transform only) → blitz (shard to device/mesh). Removed TP expansion from prepare_weights; blitz does all sharding. Tests: single-device use per-device slice, 4×2 use full logical.
- **Design doc refactor:** Cleaned stale “current blitz assumes” and “expand for TP” wording; centered doc on full HF → prepare_weights → blitz sharding; simplified §5 (one table for full logical, one for per-device after sharding), §6 (kv_b supports both 16384 and 32768), §8 (test descriptions), §9–§10 (sharding model and integration).

- **Serialization / deserialization plan (§11):** Added design for `save_weights` / `load_weights`: one `.tensorbin` per fusion group per layer + `manifest.json` with OverlappedTensor metadata, enabling offline weight preparation and fast runtime loading.
