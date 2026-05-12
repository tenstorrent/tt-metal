---
name: architecture-mapping
description: Identify existing TTNN implementations that share architectural blocks with the target model, enabling code reuse and faster bring-up. Use when starting a new model bring-up, analyzing model architecture, or mapping HuggingFace blocks to TTNN equivalents.
---

# SKILL: Architecture Mapping

## Purpose
Identify existing TTNN implementations that share architectural blocks with the target model, enabling code reuse and faster bring-up. Also produces a device-specific parallelization plan, memory budget, and bottleneck analysis before any code is written.

## Step-by-Step Process

### 0. Clarify Target Device

**Before any analysis, confirm the target hardware if not already stated.**

Ask: *"What is the target device? (e.g. N150, N300, T3K, TG, Galaxy)"*

Device properties used in the rest of this skill:
| Device | Chips | Mesh shape | DRAM/chip | Peak BF16 TFLOPS | DRAM BW | CCL BW |
|--------|-------|-----------|-----------|-----------------|---------|--------|
| N150 | 1 | 1×1 | 12 GB | 236 | 768 GB/s | — |
| N300 | 2 | 1×2 | 24 GB | 236×2 | 768 GB/s | ~50 GB/s |
| T3K | 8 | 1×8 | 12 GB | 236×8 | 768 GB/s | ~160 GB/s ring |
| TG / Galaxy | 32 | 8×4 | 12 GB | 236×32 | 768 GB/s | ~500 GB/s ring |

Record `n_devices`, `mesh_shape`, `dram_per_device`, `ring_bw` — these feed into steps 6 and 7.

---

### 1. Examine ALL Config and Modeling Files

**CRITICAL: Read EVERY file in the HuggingFace model snapshot — both JSON configs and Python source files. Do not skip any.**

```bash
# List everything in the snapshot
ls ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/*/*.json
ls ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/*/*.py
```

Read each file completely — not just the first few lines.

#### JSON config files — read all of them:
| File | What to extract |
|------|----------------|
| `config.json` | All sub-configs: `text_config`, `vit_config`, `adapter_config`, etc.; `max_position_embeddings`; all dims, heads, vocab |
| `preprocessor_config.json` | `pooling_size`, `patch_size`, `max_crops`, `overlap_margins`, normalization mean/std |
| `video_preprocessor_config.json` | `pooling_size` (often different from image!), `num_frames`, `max_fps`, `frame_sample_mode` |
| `processor_config.json` | Token flags: `use_col_tokens`, `use_frame_special_tokens`, `use_single_crop_start_token` |
| `tokenizer_config.json` | Special token IDs and their string forms |
| `generation_config.json` | `eos_token_id`, `bos_token_id`, `pad_token_id` |

**Values in configs are ground truth — never hardcode anything that is defined there.**

#### Python source files — read all of them fully:

| File | What to look for |
|------|-----------------|
| `modeling_{model}.py` | Forward pass, mask construction, fused weight layouts, non-standard ops, feature injection |
| `image_processing_{model}.py` | Crop/tile pipeline, pooling index construction, patch flattening |
| `video_processing_{model}.py` | Frame sampling, per-frame processing, grid construction |
| `processing_{model}.py` | Token string construction, `token_type_ids` assembly, special token placement |
| `configuration_{model}.py` | Config class defaults — note which fields are overridden by `config.json` |

**Why reading the Python source files is mandatory:**
- Masking logic (causal vs bidirectional vs token-type-based) is **only in the code**, not in configs
- Fused weight layouts (e.g. `att_proj = Q+K+V` concatenated) are **only visible in the forward pass**
- Gate/value ordering in fused MLP weights **may be reversed** vs standard implementations — verify in `.chunk()` or `.split()` calls
- Non-standard ops (scatter-add, boolean index-select, batched gather) must be identified early
- The preprocessing pipeline reveals what tensors the model actually receives at runtime

```python
# Read all configs at once
import json
from pathlib import Path

model_dir = Path("~/.cache/huggingface/hub/models--org--model/snapshots/xxx/")
for f in sorted(model_dir.glob("*.json")):
    print(f"\n{'='*60}\n{f.name}\n{'='*60}")
    print(json.dumps(json.load(open(f)), indent=2))
```

---

### 2. Preprocessing Policy

**Check whether HF provides a complete preprocessor (ImageProcessor, VideoProcessor, Processor class).**

If yes: **use it as-is**. Do not reimplement preprocessing in TTNN.
- HF preprocessors handle crop tiling, pooling index construction, normalization, frame sampling, tokenization
- They produce the exact tensors (`pixel_values`, `image_token_pooling`, `video_grids`, `token_type_ids`, etc.) the model expects
- Reimplementing introduces subtle bugs (e.g. wrong pooling_size for video vs image, off-by-one in overlap margins)

Record in ARCHITECTURE.md:
```markdown
## Preprocessing Policy
Use HF `{ModelNameImageProcessor}` and `{ModelNameVideoProcessor}` as-is.
Model inputs produced by the processor:
- `pixel_values`: [n_crops, n_patches, pixels_per_patch]
- `image_token_pooling`: [n_pooled, pool_h*pool_w]
- `image_grids`: [n_images, 4]  # [glob_h, glob_w, hr_h, hr_w]
- `token_type_ids`: [B, S]  # 1=image token, 0=text
```

---

### 3. Identify Model Family
Classify the target model:
- **Decoder-only LLM**: Llama, Falcon, Qwen, DeepSeek, Mistral
- **Encoder-only**: BERT, SentenceBERT, SqueezeBERT
- **Vision-Language (VLM)**: Qwen-VL, Molmo, LLaVA
- **Vision**: ViT, CLIP vision encoder
- **Speech/Audio (ASR)**: Whisper, Wav2Vec2
- **Text-to-Speech (TTS)**: SpeechT5, Qwen3-TTS, VITS, Bark

---

### 4. Complete Component Inventory

**Create a complete inventory of ALL model components from the weight files.**

```python
import json
with open("model.safetensors.index.json") as f:
    idx = json.load(f)
prefixes = {}
for k in idx["weight_map"]:
    p = ".".join(k.split(".")[:3])
    prefixes[p] = prefixes.get(p, 0) + 1
for p, c in sorted(prefixes.items()):
    print(f"{p}: {c} tensors")
```

Create a table listing EVERY neural network component:

```markdown
| Component | Weight Prefix | Tensor Count | Required For |
|-----------|--------------|--------------|--------------|
| Text Decoder | model.transformer.blocks.* | 288 | Text generation |
| Vision Encoder | model.vision_backbone.image_vit | 403 | Image/video encoding |
| Adapter | model.vision_backbone.image_pooling_2d | 8 | Feature pooling |
```

**CRITICAL RULES:**
1. List ALL components — not just the "main" transformer
2. Note what each component is REQUIRED FOR (omitting any component breaks end-to-end output)
3. For TTS: Speech Tokenizer Decoder is required to produce audio — list it explicitly
4. Do not proceed to Reference phase until this inventory is COMPLETE

---

### 5. Map Architectural Blocks and Identify Non-Standard Ops

For each block, find TTNN equivalents:

| Block Type | Reference Implementations |
|------------|--------------------------|
| GQA Attention | `models/demos/qwen3_vl/tt/attention.py`, `models/demos/llama3_70b_galaxy/tt/llama_attention.py` |
| MHA Attention | `models/demos/bert/tt/`, `models/demos/metal_BERT_large_11/tt/mha.py` |
| MQA Attention | `models/demos/falcon7b_common/tt/falcon_attention.py` |
| RoPE | `models/demos/qwen3_vl/tt/rope.py`, `models/demos/deepseek_v3/tt/rope.py` |
| RMSNorm | `models/common/rmsnorm.py` |
| LayerNorm | `models/demos/qwen3_vl/tt/vision_layernorm.py` |
| SwiGLU MLP | `models/tt_transformers/tt/mlp.py` |
| GELU MLP | `models/demos/qwen3_vl/tt/vision_mlp.py` |
| Vision Attention (no RoPE) | `models/demos/qwen3_vl/tt/vision_attention.py`, `models/demos/qwen25_vl/tt/vision_attention.py` |
| Cross-Attention / Patch Merger | `models/demos/qwen3_vl/tt/patch_merger.py` |
| MoE | `models/demos/deepseek_v3/tt/moe.py` |
| KV-Cache / Paged Attention | `models/demos/llama3_70b_galaxy/tt/` |

**Also inventory every non-standard operation** found while reading the modeling source:

| Operation | HF code | TTNN approach | Risk |
|-----------|---------|---------------|------|
| Scatter-add | `x[bool_mask] += feats` | Padded-dense add; precomputed scatter index | High |
| Batched gather | `feats[batch_idx, idx]` | `ttnn.gather` with precomputed flat indices | High |
| Boolean index-select | `out[valid.flatten()]` | Keep padded; return count | Medium |
| Bicubic interpolation | `F.interpolate(..., bicubic)` | CPU-only; upload to device | Low |
| Variable-length output | Ragged tensors | Pad to max known size | Medium |
| float32 attention | `float32_attention=True` | `MathFidelity.HiFi4` compute kernel | High |

Flag any op with no direct TTNN equivalent as HIGH risk — design the reformulation before implementation starts.

---

### 6. Attention Masking Analysis

Read the `modeling_{model}.py` forward pass and answer:
- Is attention purely causal, or are some tokens bidirectional?
- What drives the mask — `token_type_ids`, position ranges, sliding windows?
- Is it an additive bias (`-inf`) or a boolean gate?
- Does masking differ between prefill and decode?
- Does HF use `flex_attention` (lazy, not materialized) — and what is the TTNN equivalent?

**Compute mask size at max ISL for each relevant dtype:**

```python
max_ctx = config["text_config"]["max_position_embeddings"]  # or equivalent key
n_q_per_dev = n_q // n_devices  # after tensor-parallel sharding

for label, bpe in [("bfloat16", 2), ("bfloat8_b", 1), ("bfloat4_b", 0.5)]:
    size_2d = max_ctx * max_ctx * bpe / 1e6
    size_4d = n_q_per_dev * max_ctx * max_ctx * bpe / 1e6
    headroom = dram_per_device * 1024 - weights_mb  # approximate
    ok = "✓" if size_4d < headroom * 0.3 else "✗ too large"
    print(f"{label}: 2D={size_2d:.0f}MB  4D[1,{n_q_per_dev},S,S]={size_4d:.0f}MB  {ok}")
```

**Rule**: use `bfloat4_b` for masks at high ISL — reduces size 4× vs BF16. A full `[S,S]` bfloat4_b mask is typically feasible up to S ≈ 64K on T3K.

---

### 7. Device Parallelization Plan

**Using the device confirmed in Step 0, produce a per-component parallelism plan.**

#### 7a. Divisibility Checks

```python
# All of these must divide evenly across n_devices:
for name, val in [("n_q", n_q), ("n_kv", n_kv), ("intermediate_size", I)]:
    ok = val % n_devices == 0
    print(f"{name}={val}: {val//n_devices} per device {'✓' if ok else '✗ PROBLEM'}")

# Non-standard head dims: pad to next tile multiple
import math
tile = 32
padded_head_dim = math.ceil(head_dim / tile) * tile
if padded_head_dim != head_dim:
    print(f"head_dim {head_dim} → pad to {padded_head_dim} for TTNN tile alignment")
```

Common issues:
- `n_kv < n_devices`: cannot shard KV heads → replicate or restructure
- Non-standard `head_dim` (e.g. 72): pad to next multiple of 32
- Non-standard `intermediate_size`: check `nearest_multiple(dim, tile_size * n_devices)`

#### 7b. Component Parallelism Strategies

For each target device, adapt these general strategies:

| Component | General strategy | Mesh mapper | Notes |
|-----------|-----------------|------------|-------|
| Text attention Q/K/V | Column-parallel | `ShardTensor2dMesh(dims=(None,-1))` | `n_q/n_dev` Q-heads per device |
| Text attention output | Row-parallel | `ShardTensor2dMesh(dims=(None,-1))` | ReduceScatter after matmul |
| Text MLP (gate+up) | Column-parallel | `ShardTensor2dMesh(dims=(None,-1))` | Both halves shard together |
| Text MLP (down) | Row-parallel | `ShardTensor2dMesh(dims=(None,-1))` | ReduceScatter after matmul |
| Norms, QK-norm | Replicated | `ReplicateTensorToMesh` | Per-head weights |
| Vision ViT weights | Replicated | `ReplicateTensorToMesh` | Full ViT on each device |
| Vision ViT input (multi-crop/video) | Data-parallel | `ShardTensorToMesh(dim=0)` | `n_crops/n_dev` or `n_frames/n_dev` |
| Pooling / Adapter | Replicated | `ReplicateTensorToMesh` | After AllGather of ViT output |
| Embedding | Replicated | `ReplicateTensorToMesh` | Sparse vocab lookup |
| LM head | Replicated | `ReplicateTensorToMesh` | Or vocab-parallel for large vocab |

CCL operations (reference `tt_transformers/tt/ccl.py`):
- Ring topology when `n_devices >= 8`; Linear otherwise
- AllGather: reconstruct full hidden state before column-parallel matmul
- ReduceScatter: after row-parallel matmul output
- AllGather after ViT to reconstruct full image features before pooling

#### 7c. Memory Budget

**The formula below assumes the parallelization plan from 7b is actually implemented.**
If any linear in TTNN code uses `ReplicateTensorToMesh` instead of `ShardTensor2dMesh`,
its weight bytes do NOT divide by `n_dev` and the budget is wrong. The fact that
4-layer PCC tests pass does not prove the budget — always recompute per actually-shipped
mesh_mapper choice before scaling layer count. (Qwen3.6-galaxy shipped MLP/attention
replicated and OOMed at layer 52/64 on Blackhole Galaxy.)

```python
n_dev = n_devices
fp16 = 2  # bytes

# Per-device weight memory — ASSUMES MLP+attention are TP-sharded over n_dev.
# If you plan to replicate, drop the /n_dev from that line and re-check budget.
text_mb  = n_layers * (n_q*hd*H + n_kv*hd*H + H*H + 2*I*H + I*H) * fp16 / 1e6 / n_dev
vit_mb   = vit_params * fp16 / 1e6                   # replicated (typically small)
emb_mb   = vocab_size * H * fp16 / 1e6               # embedding replicated
lm_head_mb = vocab_size * H * fp16 / 1e6 / n_dev     # LM head MUST shard if vocab > 128K
kv_mb    = (n_kv//n_dev) * max_seq * hd * fp16 * n_layers / 1e6

total_mb = text_mb + vit_mb + emb_mb + lm_head_mb + kv_mb
budget   = dram_per_device * 1024  # MB
print(f"Weights+KV: {total_mb:.0f}MB / {budget}MB  (headroom: {budget-total_mb:.0f}MB)")
assert total_mb < budget * 0.6, "Exceeds 60% DRAM budget — revisit parallelism"
# 60%, not 85%: leaves headroom for prefill activations, KV growth, CCL scratch.

# REPLICATED-DRY-RUN: compute what budget would be if every linear is replicated.
# If this is < budget, replication is a valid bring-up shortcut. If not, replication
# is forbidden from the start.
text_repl_mb = n_layers * (n_q*hd*H + n_kv*hd*H + H*H + 2*I*H + I*H) * fp16 / 1e6
print(f"Weights if every linear REPLICATED: {text_repl_mb:.0f}MB  "
      f"(must be < {budget*0.6:.0f}MB to allow replicated bring-up)")
```

---

### 8. Bottleneck Analysis

**Run before any implementation to flag risks and calibrate expectations.**

Use device specs from Step 0. Substitute actual values for the model.

#### 8a. Compute

```python
peak_tflops = 236   # Wormhole B0 per chip
mfu = 0.45          # realistic MFU for matmuls

# SDPA — O(S²), dominates at ISL > 16K
sdpa_flop = 4 * (n_q // n_devices) * S_max * S_max * head_dim
t_sdpa_ms = sdpa_flop * n_layers / 1e12 / (peak_tflops * mfu) * 1e3

# MLP
mlp_flop = 2 * (2*S_max*H*(I//n_devices) + S_max*(I//n_devices)*H)
t_mlp_ms = mlp_flop * n_layers / 1e12 / (peak_tflops * mfu) * 1e3

# Vision ViT (data-parallel)
vit_flop = (n_crops_or_frames // n_devices) * vit_blocks * (
    6*vit_s*vit_h**2 + 4*vit_s*vit_h*vit_i)
t_vit_ms = vit_flop / 1e12 / (peak_tflops * mfu) * 1e3

print(f"SDPA: {t_sdpa_ms:.0f}ms | MLP: {t_mlp_ms:.0f}ms | ViT: {t_vit_ms:.0f}ms")
```

Always report SDPA at **both** a typical ISL and max ISL — the difference is often 100×.

#### 8b. CCL and Memory Bandwidth

```python
dram_bw = 768    # GB/s per device
ring_bw  = 160   # GB/s effective ring (T3K); adjust per device from Step 0

# CCL (4 collectives per text layer)
hidden_bytes = S_max * H * 2
t_ccl_ms = 4 * hidden_bytes * n_layers / 1e9 / ring_bw * 1e3

# AllGather after ViT
ag_size  = (n_crops_or_frames // n_devices) * vit_s * vit_feature_dim * 2
t_ag_ms  = ag_size / 1e9 / ring_bw * 1e3

# Decode throughput (memory-bandwidth bound)
kv_read  = (n_kv // n_devices) * S_max * head_dim * 2 * n_layers
mlp_read = (H*(2*I//n_devices) + (I//n_devices)*H) * 2 * n_layers
lm_read  = vocab_size * H * 2
t_decode_ms = (kv_read + mlp_read + lm_read) / 1e9 / dram_bw * 1e3
print(f"Decode: {t_decode_ms:.1f}ms/token  ({1000/t_decode_ms:.0f} tok/s)")
```

#### 8c. Risk-Ranked Bottleneck Table

Produce before writing any TTNN code:

```markdown
| Rank | Bottleneck | Est. impact | Bound by | Mitigation |
|------|-----------|-------------|----------|------------|
| 1 | Text SDPA at max ISL | Xms/pass | Compute | Flash-attn; check typical ISL |
| 2 | Non-standard scatter/gather ops | Impl blocker | — | Reformulate before impl |
| 3 | float32 attention in ViT | Kernel risk | — | Verify HiFi4 SDPA kernel |
| 4 | Text MLP at max ISL | Xms/pass | Compute | bfp4_mlp quantization |
| 5 | CCL at max ISL | Xms/pass | Ring BW | AG-matmul fusion; prefetcher |
| 6 | Decode at full KV cache | Xms/tok | DRAM BW | Acceptable; vocab-parallel lm_head if needed |
```

---

### 9. Weight Naming and Fused Weight Layout

Map HF weight names to TTNN equivalents. **Explicitly document all fused weights and their split logic.**

```python
# Standard:
"model.layers.{i}.self_attn.q_proj.weight" → "attention.wq.weight"
"model.layers.{i}.mlp.gate_proj.weight"    → "feed_forward.w1.weight"

# Fused QKV — verify split dimensions from forward pass:
# att_proj [Q_dim+K_dim+V_dim, H] → split along dim 0
#   wq = att_proj[:Q_dim], wk = att_proj[Q_dim:Q_dim+K_dim], wv = att_proj[Q_dim+K_dim:]

# Fused gate+up — verify ordering from .chunk() or .split() in forward pass:
# ff_proj [2*I, H]: standard = gate first → act(ff[:I]) * ff[I:]
#                  reversed  = value first → act(ff[I:]) * ff[:I]   ← verify!
```

---

### 10. Document in ARCHITECTURE.md

Create `models/demos/{model_name}/ARCHITECTURE.md` covering all sections:

```markdown
# {Model Name} Architecture Analysis

## Model Family
## Preprocessing Policy          ← HF-managed or custom; exact tensor names and shapes
## Complete Component Inventory  ← all components, tensor counts, required-for
## Sub-Component Details         ← per-component configs (from JSON), weight shapes, forward pass logic
## CRITICAL: Masking Mechanism   ← mask type, construction code, mask size table at max ISL
## Visual Feature Injection      ← how visual features enter the text stream (replace/add/concat)
## Model Forward Pass Flow       ← end-to-end data flow diagram
## Similar TTNN Implementations  ← reference files, key differences
## Key Differences from References
## Weight Mapping                ← HF key → TTNN key; fused split logic
## {Device} Parallelization Strategy  ← component-by-component plan, memory budget
## Attention Mask at High ISL    ← dtype choice, size table, construction strategy
## Performance Bottleneck Analysis    ← estimated latencies, non-standard op risks
## Implementation Order          ← phased relay-race plan
```

---

## Common Architectural Patterns

### Attention
- **GQA**: K,V heads grouped, fewer than Q heads (Llama, Qwen) — check `n_kv % n_devices == 0`
- **MHA**: All heads equal (BERT, ViTs) — divisibility usually fine
- **QK-norm (Qwen3-style)**: RMSNorm per head AFTER reshape to `[B, S, n_heads, head_dim]`; weight shape `[head_dim]`

### Position Encoding
- **RoPE**: Llama, Qwen, Mistral — check `rope_theta` and `rope_scaling` in config
- **Absolute learned**: ViT-style — check `image_num_pos`, may need bicubic interpolation
- **ALiBi**: Falcon

### Fused Weight Patterns
- **Separate Q/K/V**: standard
- **Fused QKV**: split at load time; read split dimensions from config (not hardcode)
- **Fused gate+up**: check `.chunk(2)` vs `.split([...])` in forward pass — gate/value order varies

### VLM Feature Injection
- **Replacement**: image tokens in `input_ids` replaced with embeddings (Qwen-VL)
- **Additive**: `x[is_image_patch] += image_features` (Molmo2) — needs scatter-add
  reformulation: pre-compute scatter index at model init; at runtime build a zero tensor
  same shape as x and add features at patch positions. Never do boolean indexing inside
  the TTNN forward. Risk: **HIGH**.
- **Concatenation**: visual prefix prepended to text sequence (LLaVA)

### VLM Attention: Causal + Image-Bidirectional Mask

Many VLMs use a **causal + image-bidirectional** mask driven by `token_type_ids`:
- Text tokens: strictly causal (attend to past only)
- Image/vision tokens: attend to ALL other image tokens regardless of position

Implementation pattern (see `models/demos/molmo2/tt/prefill_mask.py`):

```python
def build_vlm_prefill_mask(seq_len, token_type_ids, mesh_device, dtype=ttnn.bfloat8_b):
    q_idx  = torch.arange(seq_len).unsqueeze(1)
    kv_idx = torch.arange(seq_len).unsqueeze(0)
    causal_block   = kv_idx > q_idx
    is_image_q     = (token_type_ids == 1).unsqueeze(2)
    is_image_kv    = (token_type_ids == 1).unsqueeze(1)
    image_override = is_image_q & is_image_kv
    block = causal_block & ~image_override
    mask  = torch.where(block, float("-inf"), 0.0)
    return ttnn.from_torch(mask.unsqueeze(1), dtype=dtype, ...)
```

Pass as `attn_mask=mask` to SDPA with `is_causal=False`.

**CRITICAL — `token_type_ids` must mark ALL vision tokens, not just patch tokens:**
- Patch tokens (e.g. `image_patch_id` = 151938 for Molmo2)
- Frame boundary markers (e.g. `<im_start>` = 151936, `<im_end>` = 151937)
- Omitting frame markers gives them causal-only attention → ~30 pp accuracy drop

Find model-specific image token IDs before implementing:

```python
out = proc(text="<|video|>", videos=[frames], return_tensors="pt")
tti = out["token_type_ids"][0].tolist()
image_ids = sorted(set(int(out["input_ids"][0][i]) for i, t in enumerate(tti) if t == 1))
print("All image token IDs (type=1):", image_ids)
```

### VLM ViT Parallelism on T3K

**T3K ViT is tensor-parallel (TP8) on the hidden/head dimension**, not data-parallel on crops.
Data-parallel on crops is only practical when n_crops ≥ n_devices.

| Component | Strategy | Mapper | Key constraint |
|-----------|----------|--------|----------------|
| ViT QKV, W1 | Column-parallel | `ShardTensorToMesh(dim=3)` | n_local_heads = n_heads // 8 |
| ViT Wo, W2 | Row-parallel | `ShardTensorToMesh(dim=2)` | `ttnn.all_reduce(cluster_axis=1)` after each |
| ViT norms, biases | Replicated | `ReplicateTensorToMesh` | — |
| Input pixel values | Replicated | `ReplicateTensorToMesh` | All devices process same input |

**ViT AllBroadcast is the dominant T3K bottleneck**: n_blocks × 2 all_reduce calls = many
synchronous CCL barriers. Cannot be eliminated with TP; replicated weights OOM for large
n_frames inputs (intermediate activations scale as `[n_frames, n_heads_full, S_patch, head_dim]`).

Weight loading for column-parallel QKV requires per-device head interleaving — simple
`cat([wq,wk,wv])` is **wrong** (gives device i only the Q slice):

```python
cols = n_local_heads * padded_head_dim
qkv_chunks = [
    torch.cat([wq[:, i*cols:(i+1)*cols], wk[:, i*cols:(i+1)*cols], wv[:, i*cols:(i+1)*cols]], dim=-1)
    for i in range(num_devices)
]
wqkv = torch.cat(qkv_chunks, dim=-1)   # ShardTensorToMesh(dim=3) slices correctly
```

### VLM Video Frame Sampling

Frame sampling must match the HF VideoProcessor's algorithm **exactly**, including
sampling rate, strategy, and `frames_indices` metadata used for per-frame timestamp
computation. Mismatch → wrong timestamps → model misinterprets temporal ordering.
Register a model-specific vLLM video backend that replicates HF sampling exactly
(see tt-inference-server skill).

### TTS Models

```
Text → [Encoder] → [Decoder] → [Code Predictor] → [Audio Codec Decoder] → Waveform
```

All stages including the Audio Codec Decoder are required for end-to-end output.

| Stage | Components |
|-------|-----------|
| Codebook lookup | Token IDs → embeddings (RVQ) |
| Pre-transformer | Concatenated embedding processing |
| Upsampler | ConvTranspose1d / ConvNeXt temporal upsampling |
| Conv decoder | Final waveform synthesis |
