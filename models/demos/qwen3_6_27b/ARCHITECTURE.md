# Qwen3.6-27B — Architecture Analysis (BH Galaxy, DP=1)

> Target device: **Blackhole Galaxy (BH GLX)**, 32 chips, mesh `(8, 4)` rows×cols, **DP=1** — full TP across the entire mesh.
> Aim: maximum decode L1 residency. Reuse fused prefill/decode kernels from `llama3_70b_galaxy` and the Qwen variant config in `llama3_70b_galaxy/tt/qwen_model_config.py`. The **only new block** is Gated DeltaNet (linear attention).

---

## 1. Model family

`Qwen3_5ForConditionalGeneration` (`model_type = qwen3_5`) — a **hybrid linear/full-attention VLM**.

- **VLM** — image+video preprocessing → Qwen3-VL-style ViT → vision-merger → text decoder.
- **Hybrid text decoder**, 64 layers, pattern `3×linear → 1×full`, repeating 16 times: **48 Gated-DeltaNet layers + 16 Gated-Attention (GQA) layers**.
- **MTP head** (1 transformer block + 2 norms + a fused fc) for speculative decoding.

The text backbone is Qwen3-Next-style (Mamba2-flavored linear attention with conv1d short range + SSM long range), with:
- `attn_output_gate = True` — output-gate fused into `q_proj`.
- `partial_rotary_factor = 0.25` — RoPE applied to only 64 of 256 head dims.
- `mrope_interleaved = True`, `mrope_section = [11, 11, 10]` — Qwen-VL MRoPE.

---

## 2. Preprocessing policy

Use HF `Qwen3VLProcessor` / `Qwen2VLImageProcessorFast` / `Qwen3VLVideoProcessor` as-is.

Model inputs produced by the processor:
- `input_ids`: `[B, S]`
- `attention_mask`: `[B, S]`
- `pixel_values`: variable-length flattened patches `[N_patches, 3 × patch_size² × temporal_patch_size] = [N_patches, 1536]` for video, `[N_patches, 768]` for image.
- `image_grid_thw` / `video_grid_thw`: `[N_visuals, 3]` — temporal/H/W patch grid per visual.
- Vision/image/video token IDs: `vision_start=248053`, `vision_end=248054`, `image=248056`, `video=248057`.

Frame markers must be marked as vision tokens in any custom mask (causal-with-image-bidirectional) — see §5.

---

## 3. Complete component inventory

| Component | Weight prefix | Approx. params | Required for |
|---|---|---:|---|
| Vision ViT (27 blocks) | `model.visual.blocks.*` | ~430 M | Image/video understanding |
| Patch embed + pos embed | `model.visual.patch_embed`, `model.visual.pos_embed` | ~3 M | Visual tokenization |
| Vision Merger (2× FC) | `model.visual.merger.*` | ~45 M | ViT→LM hidden projection |
| Text embed tokens | `model.language_model.embed_tokens` | 1.27 B | Token embedding |
| Text decoder layers (64) | `model.language_model.layers.*` | ~24.0 B | Generation |
| ↳ 48× Gated-DeltaNet | `layers.{i}.linear_attn.*` | ~5.6 B (~116 M each) | Linear-attn layers |
| ↳ 16× Gated-Attention (GQA) | `layers.{i}.self_attn.*` | ~2.0 B (~125 M each) | Full-attn layers |
| ↳ 64× SwiGLU MLP | `layers.{i}.mlp.*` | ~17.1 B (~267 M each) | All layers |
| Final norm | `model.language_model.norm` | <1 M | Pre-LM head |
| LM head | `lm_head` | 1.27 B | Token generation |
| **MTP block** | `mtp.layers.0.*`, `mtp.{fc,norm,pre_fc_norm_*}` | ~270 M | Speculative decoding (optional) |

All listed components are **required** for end-to-end correctness except MTP (optional but planned).

---

## 4. Configuration constants (ground truth from `config.json`)

```
text_config:
  hidden_size H = 5120
  intermediate_size I = 17408
  num_hidden_layers = 64
  num_attention_heads (Q) = 24    head_dim = 256
  num_key_value_heads = 4         (GQA, group_size = 24/4 = 6)
  partial_rotary_factor = 0.25    (RoPE applied to 64 dims of each head)
  rope_theta = 10_000_000
  mrope_interleaved = True
  mrope_section = [11, 11, 10]
  rms_norm_eps = 1e-6
  attn_output_gate = True          (gate fused in q_proj)
  full_attention_interval = 4      (every 4th layer is full attention)
  layer_types = [linear x3, full] x 16
  max_position_embeddings = 262_144
  vocab_size = 248_320
  mtp_num_hidden_layers = 1
  mtp_use_dedicated_embeddings = False

  # Gated DeltaNet
  linear_num_value_heads = 48     linear_value_head_dim = 128   (V/Q live in 48×128 space)
  linear_num_key_heads   = 16     linear_key_head_dim   = 128   (K lives in 16×128 space)
  linear_conv_kernel_dim = 4
  mamba_ssm_dtype = float32

vision_config:
  depth = 27, hidden = 1152, intermediate = 4304, num_heads = 16
  patch_size = 16, temporal_patch_size = 2, spatial_merge_size = 2
  num_position_embeddings = 2304, out_hidden_size = 5120 (→ matches text H)
```

### Weight shapes (verified from safetensors)

| Tensor | Shape | Note |
|---|---|---|
| `embed_tokens.weight` | `[248320, 5120]` | |
| `lm_head.weight` | `[248320, 5120]` | `tie_word_embeddings = False` |
| **Gated-DeltaNet (per layer)** | | |
| `linear_attn.in_proj_qkv.weight` | `[10240, 5120]` | Concat: Q(6144)+K(2048)+V(2048) along dim 0 |
| `linear_attn.in_proj_z.weight` | `[6144, 5120]` | output gate `z` (= n_v_heads × v_dim) |
| `linear_attn.in_proj_a.weight` | `[48, 5120]` | α — forget gate (one per V-head) |
| `linear_attn.in_proj_b.weight` | `[48, 5120]` | β — discretization β |
| `linear_attn.conv1d.weight` | `[10240, 1, 4]` | depthwise conv1d (groups=10240) on QKV |
| `linear_attn.A_log` | `[48]` | log of SSM decay A |
| `linear_attn.dt_bias` | `[48]` | discretization step bias |
| `linear_attn.norm.weight` | `[128]` | GroupRMSNorm (per-head, head_dim=128) |
| `linear_attn.out_proj.weight` | `[5120, 6144]` | `n_v_heads × v_dim` → `H` |
| `input_layernorm` / `post_attention_layernorm` | `[5120]` | RMSNorm |
| **Gated-Attention (per layer)** | | |
| `self_attn.q_proj.weight` | `[12288, 5120]` | **= 24 × 256 × 2 (q **and** o-gate fused)** |
| `self_attn.k_proj.weight` | `[1024, 5120]` | 4 × 256 |
| `self_attn.v_proj.weight` | `[1024, 5120]` | 4 × 256 |
| `self_attn.o_proj.weight` | `[5120, 6144]` | `n_q × head_dim` → `H` |
| `self_attn.q_norm.weight` | `[256]` | QK-norm, per-head-dim |
| `self_attn.k_norm.weight` | `[256]` | same |
| **MLP (per layer)** | | |
| `mlp.gate_proj.weight` / `up_proj.weight` | `[17408, 5120]` | SwiGLU |
| `mlp.down_proj.weight` | `[5120, 17408]` | |
| **MTP** | | |
| `mtp.layers.0.*` | (full-attn block, same shapes as above) | |
| `mtp.pre_fc_norm_embedding.weight` / `pre_fc_norm_hidden.weight` | `[5120]` | RMSNorms |
| `mtp.fc.weight` | `[5120, 10240]` | `[h_norm ‖ emb_norm] → H` |
| `mtp.norm.weight` | `[5120]` | final norm |
| **ViT** | | |
| `visual.patch_embed.proj.weight` | `[1152, 3, 2, 16, 16]` | 3D conv, T=2, H=W=16 |
| `visual.pos_embed.weight` | `[2304, 1152]` | |
| `visual.blocks.*.attn.qkv.weight` | `[3456, 1152]` | fused QKV (3×1152) |
| `visual.merger.linear_fc1` / `fc2` | `[4608, 4608]` / `[5120, 4608]` | |

---

## 5. Critical: masking mechanism

### Full-attention layers (GQA)
- **Causal** for text tokens.
- **Image-bidirectional** for vision tokens (image and video patches as well as `vision_start`/`vision_end` markers) — same pattern as Molmo2, Qwen3-VL.
- Build once at prefill from `token_type_ids`; **must** mark all of `{image_token_id=248056, video_token_id=248057, vision_start=248053, vision_end=248054}` as vision (omitting markers costs ~30 pp accuracy — known regression).
- Use `bfloat4_b` mask at large ISL (`262_144` max → `262144² × 0.5 B / 16 = 2.1 GB` per shard; only feasible by sharding mask along query axis = `n_q_per_dev`).

### Gated-DeltaNet layers
- No SDPA-style mask: linear attention is recurrent. Causality is built into the recurrence; no `S×S` mask materialized.
- Vision tokens have no special bidirectional path through Gated-DeltaNet — that's a known modeling choice in Qwen3-Next-style hybrids; image semantics flow through full-attn layers only.

### MTP block
- Same masking as full-attn layers; runs on the speculatively-extended sequence.

---

## 6. Gated DeltaNet — base sources

Updated from initial draft: there are **two in-flight kernel implementations** and **one user-side reference + weight remap already on disk**. Gated DeltaNet is no longer "from scratch."

| Piece | Source branch | File(s) | Status |
|---|---|---|---|
| Golden PyTorch reference (full hybrid stack) | `ssinghal/qwen3.5-27B` (your prior work, shelved) | `models/demos/qwen3_5/reference/{gated_delta_net,model}.py` | Standalone, math correct, PCC self-consistent. **Take as-is.** |
| HF→TT weight remap including `wq_gate` split | `ssinghal/qwen3.5-27B` | `models/tt_transformers/tt/load_checkpoints.py` (`convert_hf_to_meta_qwen3_5`, `split_qwen3_5_attn_gate`, `map_hf_to_meta_keys_qwen3_5`) | Direct use for 3.6 (config keys identical) |
| Config plumbing (`linear_attention_pattern`, `attn_output_gate`, `partial_rotary_factor`, `rope_dim`, linear_*) | `ssinghal/qwen3.5-27B` | `models/tt_transformers/tt/model_config.py` | Direct use |
| Recurrent (decode) DeltaNet TTNN kernel | `ign/deltnet_kernel_fusion` | `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py` (recurrent path) | **PCC ≥ 0.999 vs FLA up to T=32K on N300**; single-device. Lift and add row-axis sharding. |
| Chunked (prefill) DeltaNet TTNN kernel | `ign/deltnet_kernel_fusion` | same file (chunked path) + `tt/fused_chunked_delta_rule_placeholder.py` (trace-safe constants) | "fused" name misleading — Python op-loop, not a fused kernel. Acceptable for bring-up; optimize later. |
| BH GLX scaffold (CCL tuning, fused-AGM K-dim fix, KV-head replication, batch=64) | `changh95/Qwen3.6-35B-A3B_bh_lb` | `models/tt_transformers/tt/{model_config.py, attention.py, ccl.py, mlp.py}` | Cherry-pick into tt_transformers |
| Layer-type dispatch in `tt/model.py` | NEW — neither branch handles hybrid correctly | — | Branch on `layer_types[i] in {"linear_attention","full_attention"}` |

### Algorithm (per layer, single step `t`)

Let `H = 5120`, `n_v = 48`, `n_k = 16`, `d_k = d_v = 128`, `conv_k = 4`.

```
x_t : [B, H]
# 1. Input projections (4 parallel matmuls — fuse as one for prefill)
qkv = in_proj_qkv @ x_t                  # [B, 10240]  → split Q[6144], K[2048], V[2048]
z   = in_proj_z   @ x_t                  # [B, 6144]   gate
α   = sigmoid(in_proj_a @ x_t)           # [B, 48]     per-head forget
β   = silu   (in_proj_b @ x_t)           # [B, 48]     per-head update

# 2. Depthwise short conv (replaces last conv_k tokens with a learned mix)
qkv = conv1d(qkv, w=conv1d.weight, groups=10240, kernel=4)   # uses persistent conv state

# 3. Per-head reshape + RMS norm
Q = silu(qkv[:6144].view(B, n_v, d_v))   # treat Q as living in V-head space
K = silu(qkv[6144:8192].view(B, n_k, d_k))
V = silu(qkv[8192:].view(B, n_k, d_v))
Q = group_rms_norm(Q, w=norm.weight)     # weight shape [128] per head_dim
K = group_rms_norm(K, w=norm.weight)
V = group_rms_norm(V, w=norm.weight)

# 4. Discretized SSM step (DeltaNet-style)
dt = softplus(β + dt_bias)               # [B, 48]
A  = -exp(A_log)                         # [48]     (decay)
g  = exp(dt * A)                         # [B, 48]  per-head decay this step
β_step = dt * α                          # [B, 48]  per-head update strength

# Expand K from 16 heads to 48 via GQA broadcast (group=3)
K48 = K.repeat_interleave(3, dim=1)      # [B, 48, 128]
V48 = V.repeat_interleave(3, dim=1)      # [B, 48, 128]

# 5. State update (rank-1 outer product, DeltaNet rule)
#    state: [B, 48, d_k=128, d_v=128]  — persistent KV state
state = state * g[:,:,None,None] + β_step[:,:,None,None] * outer(K48, V48)

# 6. Output read
y = (state * Q[:,:,:,None]).sum(dim=2)   # [B, 48, d_v=128]
y = y.reshape(B, 6144)

# 7. Output gate + projection
y = y * silu(z)                          # gated by z
out = out_proj @ y                       # [B, 5120]
```

### Notes / risks

| Item | Notes / risk |
|---|---|
| `mamba_ssm_dtype = float32` in config | Run SSM state update in FP32 accumulation. TTNN: `MathFidelity.HiFi4` compute kernel; KV-state buffer `float32`. |
| State size per layer per request | `B × n_v × d_k × d_v × 4 B = 1.5 MB` per request (FP32). For DP=1, that's 48 layers × 1.5 MB = **72 MB total state** — sits in DRAM, paged into L1 per-layer during decode. |
| Conv1d short state | `B × 10240 × (conv_k-1) × 2 B = 60 KB` per layer — L1-resident easily. |
| Prefill mode | Recurrence over S tokens; no SDPA. Implement as **chunked scan** (`chunk_size = 64` or 128 — tune): chunked outer-product accumulation amortizes per-token overhead. Reference math is identical; chunking only changes how the outer products are blocked. |
| Decode mode | True single-step recurrence — small matmuls + outer product + state update. **Highly L1-friendly** (working set < 2 MB per device after sharding). |
| Existing TTNN ops needed | `ttnn.conv1d` (✓ exists), `ttnn.matmul`, `ttnn.silu`, `ttnn.softplus`, `ttnn.exp`, `ttnn.layer_norm`/`rms_norm`. **No new kernel required** for the recurrence — it's outer products and reductions. |
| Reference implementation | Will live in `reference/functional.py` as `gated_deltanet_step` and `gated_deltanet_chunked_prefill`, validated against HF `transformers` (≥4.57) implementation of `Qwen3_5LinearAttention`. |

### Risk: MEDIUM — kernel exists, mesh sharding is the open question
The math and the single-device TTNN kernels are validated (`ign/deltnet_kernel_fusion`, PCC ≥ 0.999 to T=32K on N300). The open risks are:
1. **Row-axis TP sharding** of the 48 V-heads / 16 K-heads across 8 mesh rows (6 V-heads, 2 K-heads per row, K/V dim=128 unsharded).
2. **BH compute kernel configs** — kernel source uses `WormholeComputeKernelConfig`; need `is_blackhole()`-conditional equivalents.
3. **Recurrent state persistence** in the model wrapper — sized `[B, 6, 128, 128]` per chip per layer = 375 KB; lives in DRAM, paged into L1 each layer.
4. **Trace-safe constants** (tril/triu/eye builders): the `build_fused_chunked_delta_rule_constants` pattern from the kernel branch must be invoked at model construction.

First milestone: run the existing kernel on a single BH chip at Qwen3.6 shapes (16 K-heads × 128, 48 V-heads × 128) with **real Qwen3.6 layer-0 weights** loaded via `convert_hf_to_meta_qwen3_5` (from `ssinghal/qwen3.5-27B`) and validate PCC against the reference in the same branch. See QUALIFICATION_PLAN.md Test A.

---

## 7. Output-gated Gated-Attention (full-attn layer)

The full-attn layers are **standard GQA + QK-norm + partial RoPE**, with one twist:

```
q_full = q_proj @ x         # [B, S, 12288]  → reshape [B, S, n_q=24, 2, head_dim=256]
Q, g   = q_full.unbind(dim=-2)              # Q: [B,S,24,256]  gate: [B,S,24,256]

# Standard SDPA path (with QK-norm + partial-RoPE on first 64 dims)
Q, K  = qk_norm(Q, K)                       # weight shape [256]
Q, K  = apply_mrope(Q, K, freqs, rotary_dim=64)
attn  = SDPA(Q, K, V, mask=causal+vision_bidir)

# Output gate: per-head sigmoid gating of attention output before o_proj
attn  = attn * sigmoid(g)                   # element-wise

out   = o_proj @ attn.reshape(B, S, 24*256) # → [B, S, H]
```

**Implementation reuse:** `llama3_70b_galaxy/tt/llama_attention.py` + `qwen_model_config` provides the QK-norm + partial-RoPE path. The output-gate adds **one extra `sigmoid + mul`** between SDPA and `o_proj` — a 30-line patch to that file, conditional on `attn_output_gate` flag. The fused `q_proj` (double-width) is already representable as a wider matmul output that we then split.

---

## 8. Reuse map — every block to an existing file

| Block | Primary source | New work |
|---|---|---|
| Hybrid PyTorch reference (golden) | `ssinghal/qwen3.5-27B`: `models/demos/qwen3_5/reference/*.py` | Add vision encoder ref (from HF Qwen3VL), add MTP ref |
| HF→TT weight remap (incl. `wq_gate`, linear_attn keys) | `ssinghal/qwen3.5-27B`: `tt_transformers/tt/load_checkpoints.py` | Drop-in (config keys identical for 3.5 ↔ 3.6) |
| `is_qwen35` / hybrid config plumbing | merge of `ssinghal/qwen3.5-27B` and `changh95/Qwen3.6-35B-A3B_bh_lb` `model_config.py` | Take user branch's keys, layer on changh95's BH GLX num-splits + KV-replication |
| Text RMSNorm (distributed) | `tt_transformers/tt/distributed_norm.py` (via `llama3_70b_galaxy`) | none |
| Gated-Attention (16 layers) | `ssinghal/qwen3.5-27B`: `models/demos/qwen3_5/tt/attention.py` (Q+gate split, on-device QK-norm + partial-RoPE, paged_update_cache + sdpa_decode) | **Add mesh sharding** (cols=4 heads, rows=8 dims); +MRoPE call |
| **Gated-DeltaNet TTNN kernel (recurrent + chunked)** | `ign/deltnet_kernel_fusion`: `models/experimental/gated_attention_gated_deltanet/tt/*.py` | **Add row-axis TP sharding** (V-heads/8, K-heads/8); BH compute configs; recurrent state cache |
| SwiGLU MLP | `tt_transformers/tt/mlp.py` (via llama3_70b_galaxy) + changh95 AG-links retune | none |
| MRoPE [11,11,10] | `qwen3_vl/tt/rope.py` | Port to tt_transformers, gated by `mrope_interleaved` flag |
| ViT (27 blocks) | `qwen3_vl/tt/vision_{attention,block,layernorm,mlp}.py`, `patch_merger.py` | none — shapes match (depth=27, hidden=1152, merger 4608→5120) |
| Patch / pos embed | `qwen3_vl/tt/vision_block.py` + standard `ttnn.embedding` | Wire up `model.visual.patch_embed.proj` 3D conv |
| Embedding (text) | `llama3_70b_galaxy/tt/llama_embedding.py` | none |
| LM head | `llama3_70b_galaxy/tt/lm_head.py` | **Make vocab-parallel** across 32 chips (see §9 memory) |
| CCL (AG/RS, fused matmul) | `tt_transformers/tt/ccl.py` + `changh95` retuning + `llama3_70b_galaxy/tt/llama_ccl.py` prefetcher | none |
| Fused all-gather matmul for `n_heads*head_dim != dim` | `changh95/Qwen3.6-35B-A3B_bh_lb` commit `fc90c3ac` | Direct cherry-pick — exact case for our 24×256 vs 5120 |
| Layer-type dispatch in decoder | NEW — `tt/decoder.py` branches on `layer_types[i]` | New code, ~30 lines |
| MTP head | `deepseek_v3/tt/mtp.py` | Adapt for Qwen output-gated attention shapes |
| Generator + vLLM | `llama3_70b_galaxy/tt/generator{,_vllm}.py` (text), `qwen3_vl/tt/generator_vllm.py` (VLM path) | merge: VLM prep + hybrid decoder |
| BH GLX device wiring | `qwen_model_config.py` (`BH_GLX` branch already present) + `changh95` num-splits dict | Add Qwen3.6-27B params: n_q=24, n_kv=4, head_dim=256, intermediate=17408, layer_types |

**One TTNN block needs new code (`tt/linear_attention.py` wrapping `ign/deltnet_kernel_fusion` kernels with mesh sharding). Everything else is composition of existing files + cherry-picks.**

---

## 9. BH Galaxy parallelization plan (DP=1, full TP=32)

Mesh shape `(rows=8, cols=4)`. Two TP axes:
- **`cluster_shape.cols = 4`** — shard attention heads (Q, K, V, output) across cols.
- **`cluster_shape.rows = 8`** — shard MLP intermediate and DeltaNet V-heads across rows.

### Divisibility checks ✓

| Param | Value | Shard | Per device | OK? |
|---|---:|---|---:|---|
| H (residual) | 5120 | 4 cols (`dim_tp_factor=4`) | 1280 | ✓ (40 tiles) |
| n_q heads | 24 | 4 cols | 6 | ✓ |
| n_kv heads | 4 | 4 cols | 1 | ✓ |
| head_dim | 256 | — | 256 | ✓ (8 tiles) |
| intermediate I | 17408 | 8 rows (`intermediate_dim_tp_factor=8`) | 2176 | ✓ (68 tiles, ≠ multiple of 24 — see padding note) |
| n_v heads (DeltaNet) | 48 | 8 rows | 6 | ✓ |
| n_k heads (DeltaNet) | 16 | 8 rows | 2 | ✓ |
| vocab | 248_320 | replicated LM head, vocab-parallel optional | — | pad to 248_352 (= 32 × 7761) for 32-row split |

`intermediate=17408` per device = `2176`. Padding to nearest 24-core × 32-tile multiple → `2304` (overhead 5.9%, acceptable). The `intermediate_dim_per_tp_padded_24_cores` constant in `qwen_model_config.py` will need to be regenerated for Qwen3.6-27B.

### Per-block strategy

| Block | Strategy | Mapper | CCL |
|---|---|---|---|
| Gated-Attention Q+gate, K, V proj | Column-parallel across **cols** | `ShardTensor2dMesh(dims=(None,-1))` on col axis | AG before matmul (reuse llama prefetcher) |
| Gated-Attention o_proj | Row-parallel across **cols** | same | RS after, then AG on row axis to reconstruct residual |
| MLP gate+up proj | Column-parallel across **rows** | `ShardTensor2dMesh(dims=(None,-1))` on row axis | AG (already fused into prefetcher in `llama_mlp.py`) |
| MLP down proj | Row-parallel across **rows** | same | RS+AG (trace-safe) |
| Gated-DeltaNet `in_proj_*` (qkv, z, a, b) | Column-parallel across **rows** (V-heads dim) | shard along V-head axis | AG before |
| Gated-DeltaNet conv1d | Local (depthwise, groups = sharded width) | inherits prior sharding | none |
| Gated-DeltaNet SSM state | Sharded along V-heads (rows=8) → 6 heads/row | host-side reshape | none (state stays local; A_log/dt_bias replicated on row) |
| Gated-DeltaNet out_proj | Row-parallel across **rows** | — | RS+AG |
| ViT (replicated weights, data-parallel tokens) | Replicate weights, shard patches along seq dim | `ReplicateTensorToMesh` for W; patches split | AG of ViT features to reconstruct full sequence |
| Vision Merger | Replicated | `ReplicateTensorToMesh` | none |
| Embedding | Replicated | `ReplicateTensorToMesh` | none |
| LM head | Replicated (or vocab-parallel for memory) | `ReplicateTensorToMesh` | none |
| RMSNorms | Distributed RMSNorm | `distributed_norm.py` reuse | RS+AG inside norm |
| MTP block | Same as Gated-Attention | — | same |

### Memory budget (per chip, BF16 weights)

| Bucket | Approx. size / chip |
|---|---:|
| Text decoder weights (sharded) | 24 B / 32 ≈ **0.75 GB** |
| ViT weights (replicated, all chips) | ~0.86 GB |
| Embed + LM head (replicated) | 1.27 B × 2 × 2 ≈ **5.08 GB** (LARGEST contributor — must compress or vocab-parallel) |
| KV cache (full-attn 16 layers, S=64K, BF16 — typical) | `16 × 1 × 64K × 256 × 2 ≈ 524 MB` / chip |
| DeltaNet state (48 layers × FP32, sharded) | `48 × 1.5 MB / 8 ≈ 9 MB` / chip |
| Activations (decode) | < 5 MB |
| **Total decode footprint** | **~7.3 GB / chip** |

> BH chip DRAM is comfortably > 7.3 GB; this fits. **Risk: embedding + LM head at BF16 dominate.** Recommend BF8/INT8 for `embed_tokens` and `lm_head`, or vocab-parallel split of LM head across 32 chips (248320/32 = 7760 → 1 chip ≈ 40 MB at BF16). Vocab-parallel is the right call for this mesh.

### Decode L1 residency plan

> Goal: keep the per-layer decode hot loop fully in L1, with only weight streaming hitting DRAM.

| Tensor (per layer, per chip, post-sharding) | Size | Lives in |
|---|---:|---|
| Residual hidden state `[B=1, 1, H/4 = 1280]` BF16 | 2.5 KB | L1 |
| QKV projection output (Gated-Attn) | (6+1+1)×256×2 = 4 KB | L1 |
| Attention output post-gate `[1, 24/4, 256]` | 3 KB | L1 |
| MLP intermediate sharded (`I/8 = 2176`) | 4.3 KB | L1 |
| Distributed-norm scratch | < 32 KB | L1 |
| DeltaNet QKVZ projection outputs `[1, (6144+6144+2048+2048)/8]` | 4.1 KB | L1 |
| DeltaNet conv1d state `[1, 10240/8, 3]` | 7.5 KB | L1 (small) |
| **DeltaNet SSM state `[1, 6, 128, 128]` FP32** | **375 KB** | **DRAM, paged-in per layer** — too big for "always L1"; OK to stream |
| Weights | 0.75 GB / chip | DRAM, streamed |

**Conclusion:** every transient activation, every gate output, every MLP intermediate is small enough to stay sharded in L1. The SSM state for Gated-DeltaNet is the **only** sizeable per-layer working tensor and must page from DRAM (1 MB/chip per active layer, well under DRAM bandwidth budget). This matches the user requirement "decode as much on L1 as possible."

---

## 10. Risk-ranked bottleneck table (BH GLX)

| Rank | Bottleneck | Est. impact | Bound by | Mitigation |
|---|---|---|---|---|
| 1 | **DeltaNet row-axis TP sharding** — `ign/deltnet_kernel_fusion` is single-device; need to shard 48 V-heads / 16 K-heads across 8 mesh rows with correct CCL of state and outputs | Block-level perf and correctness | Implementation | Start with 1-chip kernel PCC (Test A), then extend `tt/linear_attention.py` to wrap the kernel under `ShardTensor2dMesh(dims=(None,row_axis))`. CCL only at `out_proj` (row-parallel ReduceScatter) |
| 2 | **BH GLX CCL is 1-link Linear** (vs WH 6U 4-link Ring) — `GALAXY_NUM_LINKS=1` | ~3-5× more CCL time per layer vs WH 6U | Ring/Fabric BW | Cherry-pick `changh95` CCL retuning (`chunks_per_sync=1`, `num_workers_per_link=1`, AG links 2→1); fused-AGM K-dim fix from same branch |
| 3 | **Embedding + LM head at BF16 = 5 GB/chip** if replicated | OOM risk | DRAM | Vocab-parallel LM head across 32 chips (248320/32 ≈ 7760 → 40 MB/chip); quantize embed (BF8/INT8) |
| 4 | Full-attn SDPA at S=262K | `bfloat4_b` mask required | Compute + memory | Standard `tt_transformers` SDPA config + changh95 SDPA LOFI per-layer BFP4 table |
| 5 | Output-gate adds extra `sigmoid + mul` per full-attn layer | 16 layers × ~5% overhead | Compute | Already in `ssinghal/qwen3.5-27B/tt/attention.py`; lift verbatim |
| 6 | ViT replicated weights (≈0.86 GB/chip) | Always-on cost | DRAM | Acceptable; only large at warm-up |
| 7 | MTP head not in critical decode path (speculative) | Latency win, but second decoder | Compute | Defer to Phase 2; keep speculative off until DeltaNet stable |
| 8 | MRoPE interleaved sections must match HF exactly | Accuracy regression risk | Numerical | Port `qwen3_vl/tt/rope.py` MRoPE; unit-test against HF for arbitrary T/H/W grids |
| 9 | Recurrent state cache management (per-layer FP32 state, 375 KB/chip) | DRAM BW during decode | DRAM | Persistent DRAM allocation, paged-in per layer; see §9 |

---

## 11. Implementation order (relay race)

1. **Architecture** (this doc) ✓
2. **Composition lock-in** ✓ — see `QUALIFICATION_PLAN.md` for the four-test gate.
3. **Pull in base material** (single PR, no code changes yet):
   - Cherry-pick `ssinghal/qwen3.5-27B`: `models/demos/qwen3_5/reference/*`, `models/tt_transformers/tt/load_checkpoints.py` (qwen3_5 paths), config_model.py linear_attn plumbing, `models/tt_transformers/model_params/Qwen3.5-27B/` → rename to `Qwen3.6-27B/`.
   - Cherry-pick `ign/deltnet_kernel_fusion`: `models/experimental/gated_attention_gated_deltanet/` (entire dir).
   - Cherry-pick `changh95/Qwen3.6-35B-A3B_bh_lb`: `is_qwen35`, CCL retunings, fused-AGM K-dim fix, KV replication, batch=64, custom-RoPE hook in `tt_transformers`.
4. **Test A — DeltaNet kernel PCC at Qwen3.6 shapes** (single chip, BH). Loaded with real Qwen3.6 layer-0 weights via the qwen3_5 remap. Pass: PCC > 0.99 against the user-branch reference for both recurrent and chunked paths at T ∈ {1, 32, 256, 4K, 32K}.
5. **TTNN — small parts** (after Test A passes):
   - `tt/linear_attention.py` wrapping `ign/deltnet_kernel_fusion` kernels under `ShardTensor2dMesh` for row-axis V/K-head sharding.
   - Layer-type dispatch in `tt_transformers/tt/decoder.py` (`if layer_types[i] == "linear_attention": …`).
   - MRoPE port from `qwen3_vl/tt/rope.py` into `tt_transformers/tt/rope.py`, gated by `mrope_interleaved` flag.
   - Lift `ssinghal/qwen3.5-27B/tt/attention.py` gated-attention (already on-device QK-norm + partial-RoPE) and add mesh sharding.
   - Vocab-parallel LM head (memory plan §9).
6. **Test C — mesh CCL** (parallel with #5): validate fused-AGM K-dim fix and CCL retunings on 2-chip BH subset, then 8×4 BH GLX.
7. **Test D — 4-layer hybrid slice end-to-end PCC** (3 linear + 1 full + MLPs + norms) at real Qwen3.6 weights.
8. **Full text decoder** — 64 layers, end-to-end, no vision.
9. **VLM frontend** — wire `qwen3_vl/tt/vision_*.py` ViT + merger; image and video paths.
10. **MTP head** — adapt `deepseek_v3/tt/mtp.py`; off by default.
11. **Server** — extend `qwen3_vl/tt/generator_vllm.py` with hybrid-decoder paths; register in `tt-inference-server`.

---

## 12. Open questions to track

- BH GLX exact per-chip DRAM and L1 numbers — pull from a `tt-smi -d` capture at session start and back-fill in §9 budget.
- Whether the existing `llama_ccl.py` prefetcher supports BH 1-link Linear topology end-to-end (config flags suggest yes; needs profiling).
- Confirm `mtp_use_dedicated_embeddings = False` semantics — MTP reuses `embed_tokens` and `lm_head`; verify in the reference run.
- Whether `chunked_prefill` for Gated-DeltaNet can reuse `ttnn.matmul` outer-product fusion or needs a fused op (likely NOT, but profile after first PCC pass).

---

## 13. Vision encoder — full detail

Qwen3.6 vision encoder is structurally identical to Qwen3-VL ViT — direct reuse from `models/demos/qwen3_vl/tt/vision_*.py`.

### Shapes (verified)

| Component | Shape | Notes |
|---|---|---|
| `patch_embed.proj.weight` | `[1152, 3, 2, 16, 16]` | 3D conv: `temporal_patch_size=2`, `H=W=16`. 3 in-channels (RGB), 1152 out. |
| `patch_embed.proj.bias` | `[1152]` | |
| `pos_embed.weight` | `[2304, 1152]` | absolute learned position table, max patches = 2304 = (48×48 grid) |
| `blocks[0..26].norm1.weight` / `.bias` | `[1152]` / `[1152]` | LayerNorm pre-attn (with bias!) |
| `blocks[i].attn.qkv.weight` | `[3456, 1152]` | fused QKV: 3 × 1152 along dim 0 |
| `blocks[i].attn.qkv.bias` | `[3456]` | non-zero — must be loaded |
| `blocks[i].attn.proj.weight` / `.bias` | `[1152, 1152]` / `[1152]` | output proj with bias |
| `blocks[0..26].norm2.weight` / `.bias` | `[1152]` / `[1152]` | LayerNorm pre-MLP |
| `blocks[i].mlp.linear_fc1.weight` | `[4304, 1152]` | GELU MLP first layer |
| `blocks[i].mlp.linear_fc1.bias` | `[4304]` | |
| `blocks[i].mlp.linear_fc2.weight` | `[1152, 4304]` | |
| `blocks[i].mlp.linear_fc2.bias` | `[1152]` | |
| `merger.norm.weight` / `.bias` | `[4608]` / `[4608]` | LayerNorm (4× hidden after spatial-merge reshape) |
| `merger.linear_fc1.weight` / `.bias` | `[4608, 4608]` / `[4608]` | |
| `merger.linear_fc2.weight` / `.bias` | `[5120, 4608]` / `[5120]` | projects to text hidden dim H |

### Forward pass

```
pixel_values [N_patches, 1536]  # 1536 = 3 × 16 × 16 × 2 (RGB × patch × temporal)
       │
       ▼  patch_embed (3D conv)
hidden [N_patches, 1152]
       │
       │  + pos_embed[pos_ids]  (indexed)
       ▼
27× ViT block:
   norm1 → attn (fused QKV, MHA-16, no RoPE) → +residual
   norm2 → mlp (GELU 1152→4304→1152)        → +residual
       │
       ▼  spatial merger: [N_patches, 1152] → [N_patches/4, 4608] (2×2 spatial reshape)
merger.norm → fc1 → GELU → fc2
       │
       ▼
features [N_merged, 5120]   # ready for injection at image_token positions
```

### Parallelism on BH GLX (DP=1)

| Component | Strategy | Mapper | Rationale |
|---|---|---|---|
| `patch_embed.proj` | Replicated | `ReplicateTensorToMesh` | Tiny weights (~110K params); input is shard-on-dim-0 patches |
| `pos_embed` | Replicated | `ReplicateTensorToMesh` | Sparse lookup |
| ViT attention QKV / proj | Replicated weights, **data-parallel patches** | `ShardTensorToMesh(dim=0)` on input | N_patches/(32) per chip; weights ≈ 0.86 GB total replicated |
| ViT MLP fc1/fc2 | Replicated weights, data-parallel patches | same | same |
| Block residuals | Local to each chip | — | All within one chip's patch slice |
| Merger | Replicated weights, AllGather inputs first | `ReplicateTensorToMesh` | After AllGather of `[N_merged, 4608]` |

### Bottleneck

- 27 blocks × O(N_patches² × 1152) compute. At N_patches = 1024 (typical image): ~0.7 GFLOPs/block, all 27 blocks ≈ 20 GFLOPs. At BH 580 TFLOPS peak × 0.4 MFU = 232 TFLOPS → 0.09 ms. Negligible vs text decode.
- AllGather after ViT to reconstruct full feature seq: `N_merged × 5120 × 2 B / 160 GB/s ≈ 0.2 ms`. Acceptable.

### Vision tests

T1.6, T2.7, T2.8, T4.1, T4.2 in TEST_PLAN.md.

---

## 14. MTP head — full detail

Multi-Token Prediction (MTP) head appended to the base model. One transformer block + 4 utility tensors. Optional (off by default until base text stable).

### Weight layout

| Tensor | Shape | Role |
|---|---|---|
| `mtp.layers.0.input_layernorm.weight` | `[5120]` | RMSNorm — same shape as base layer norms |
| `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight` | same as a full-attn layer | One full Gated-Attention block |
| `mtp.layers.0.self_attn.{q,k}_norm.weight` | `[256]` each | QK-norm |
| `mtp.layers.0.post_attention_layernorm.weight` | `[5120]` | |
| `mtp.layers.0.mlp.{gate,up,down}_proj.weight` | same as a full-attn MLP | |
| `mtp.pre_fc_norm_embedding.weight` | `[5120]` | RMSNorm on next-token embedding |
| `mtp.pre_fc_norm_hidden.weight` | `[5120]` | RMSNorm on last base hidden state |
| `mtp.fc.weight` | `[5120, 10240]` | concat-and-project: `[h_norm ‖ emb_norm] → H` |
| `mtp.norm.weight` | `[5120]` | final RMSNorm before reusing base LM head |

### Forward pass (speculative)

```
base_hidden = base_decoder(input_ids)   # last hidden state from layer 63
                                        │
                                        ▼
        h_norm = mtp.pre_fc_norm_hidden(base_hidden)
        e_norm = mtp.pre_fc_norm_embedding(embed_tokens(next_token))
        fused  = mtp.fc(concat([h_norm, e_norm], dim=-1))     # [B, S, H]
                                        │
                                        ▼
        x = mtp.layers.0(fused, mask=...)                     # one gated-attn + MLP
                                        │
                                        ▼
        x = mtp.norm(x)
        logits = lm_head(x)                                   # reuses base lm_head
```

### Reuse + integration

- The MTP transformer block is shape-identical to a base Gated-Attention block → reuse the same `tt/attention.py` and `tt/mlp.py` code paths (different weight prefix).
- `tt/mtp.py` template comes from `models/demos/deepseek_v3/tt/mtp.py`. The structural difference is DeepSeek MTP has its own LM head while Qwen MTP reuses `lm_head` (per config `mtp_use_dedicated_embeddings = False`).
- Off by default; turn on via `--speculative` CLI flag; speculative acceptance must be measured vs HF baseline (T5.3).

### MTP tests

T1.8, T2.10, T5.1, T5.2, T5.3 in TEST_PLAN.md.

---

## 15. Performance methodology

Per CLAUDE.md, optimization phase requires Tracy profiling, decode tracing, and an xlsx output via `run_block_profiles.sh`. Below is the scope plan, not numbers yet.

### Trace capture

**Decode trace** (per CLAUDE.md `/optimization`):
- Capture trace at step 1 of decode; replay for steps 2..N. **SAFE here**: this model has no variable-length inputs at decode (`B=1`, `S=1` per step). Decode trace is the right call.
- **`build_fused_chunked_delta_rule_constants` MUST run before trace capture** — see TEST_PLAN T6.1. Trace cannot include host-side tensor construction.
- Recurrent SSM state must be a persistent device buffer, **not** allocated inside the traced function — allocate at model construction, write in place.

**Prefill trace**: NOT used (variable sequence length).

### Tracy profile per block

```bash
python -m tracy -p -v -r tests/profile/profile_block.py --block <name>
```

Per-block scripts to create in `tests/profile/`:
- `profile_deltanet_recurrent.py`
- `profile_deltanet_chunked.py`
- `profile_gated_attention_decode.py`
- `profile_gated_attention_prefill.py`
- `profile_mlp.py`
- `profile_distributed_norm.py`
- `profile_vit_block.py`
- `profile_merger.py`
- `profile_mtp.py`
- `profile_lm_head.py`
- `profile_ccl_ag.py`
- `profile_ccl_rs.py`

`run_block_profiles.sh` collates all 12 into a single xlsx with: op name, mean ms, max ms, FLOPs, % bound (DRAM / CCL / compute), MFU.

### Targets

| Metric | Target | Source |
|---|---|---|
| Decode throughput, B=1 | **≥ 25 tok/s** | Tested by T6.3 |
| Prefill TTFT, S=512 | **≤ 200 ms** | T6.4 |
| Prefill TTFT, S=8K | **≤ 2 s** | T6.4 |
| DRAM utilization | < 85% | T6.5 |
| L1 utilization (decode hot) | < 70% | T6.5 |
| MFU on text MLP at S=4K | > 0.4 | profile |
| MFU on DeltaNet recurrent step | > 0.3 (memory-bound regime) | profile |

### Optimization levers if perf misses

1. **Decode tok/s low**:
   - First check trace overhead (T6.1); without trace expect 5-10× regression.
   - Reduce CCL count per layer (DeltaNet has 1 CCL at `out_proj`, gated-attn has 2 — AG before QKV + RS after o_proj). DeltaNet residual+norm CCL is shared with MLP.
   - Quantize MLP weights to BFP4 (per `changh95`'s per-layer config in `performance_decoder_config.json`).
2. **Prefill slow at long S**:
   - Increase DeltaNet chunk size (cs=128 or 256) if numerical drift permits.
   - Move SDPA mask from BF16 → BFP4 at long S (already planned).
3. **DRAM pressure**:
   - Vocab-parallel LM head (planned).
   - Quantize embedding to BF8/INT8 (40% saving).
   - Quantize ViT weights to BF8 (saves ~0.4 GB/chip).

---

## 16. Server integration (tt-inference-server)

### `generator_vllm.py` skeleton

Composition: text path from `llama3_70b_galaxy/tt/generator_vllm.py`, VLM frontend from `qwen3_vl/tt/generator_vllm.py`. Merge into `models/demos/qwen3_6_27b/tt/generator_vllm.py`.

Key wiring:
- `model_spec.py`: register `Qwen/Qwen3.6-27B`, dispatcher = our generator class.
- `model_spec.py`: enable image AND video modalities; register custom video backend (frame sampling must match HF `Qwen3VLVideoProcessor` exactly — see §2).
- `mtp_use_dedicated_embeddings = False` → MTP head registered as a child of the main model, sharing tokenizer/embeddings.

### Decode trace warning

Per `/tt-inference-server` skill: "decode traces with variable-length inputs bake SDPA config at capture S". Our model is not affected (decode is always S=1), but the recurrent DeltaNet state **must** be reset between requests — see T7.1.

### Server tests

T7.1 through T7.7 in TEST_PLAN.md. Completion criteria per CLAUDE.md:
- Server starts and handles requests without errors or device hangs (T7.7)
- Server accuracy ≥ demo accuracy (within 2-3 pp) (T7.6)
- 105-video (or equivalent) test suite (T7.6)
- `tt-smi -r` not needed between requests (T7.7)

---

## 17. TDD test gates — summary

The authoritative test contract is `TEST_PLAN.md`. Each block listed in §3 (component inventory) has explicit RED + GREEN specifications and a position in a dependency-ordered execution graph.

**Iron rule for the relay**: no production code lands in `models/demos/qwen3_6_27b/{reference,tt}/` without a corresponding test in `tests/` that was **observed failing for the documented RED reason** before the production code was written. Phase order (Architecture → Reference → TTNN → Debug → Opt → Server) is enforced by the test graph: every Phase 3 test depends on a passing Phase 2 test; every Phase 4 test depends on Phase 3.

| Phase | # tests | Hardware | Gate to next phase |
|---|---:|---|---|
| 1 — Reference | 12 | CPU only | All Phase 1 tests pass + golden tensors checkpointed |
| 2 — TTNN single-chip | 10 | 1× BH | All Phase 2 tests pass at PCC > 0.99 |
| 3 — TTNN mesh | 10 | BH GLX 8×4 | T3.8 (4-layer mesh) + T3.9 (full text) pass |
| 4 — VLM | 7 | BH GLX | T4.6 (VLM image E2E) + T4.7 (video) pass |
| 5 — MTP | 3 | BH GLX | T5.3 acceptance rate within 5pp of HF |
| 6 — Perf | 6 | BH GLX | T6.3 ≥ 25 tok/s + T6.5 memory < 85% |
| 7 — Server | 7 | BH GLX | T7.6 accuracy + T7.7 100-req stability |

Total: 55 tests across the relay. None of them may be added retroactively after their production code lands.

---

## 18. Deliverables checklist (this directory)

| File | Status |
|---|---|
| `ARCHITECTURE.md` (this file) | ✓ Locked composition + complete spec |
| `QUALIFICATION_PLAN.md` | ✓ Branch landscape + Test A protocol |
| `TEST_PLAN.md` | ✓ 55-test TDD contract |
| `BRINGUP_LOG.md` | Initialized — see file |
| `reference/` | Phase 1 — cherry-pick from `ssinghal/qwen3.5-27B` first |
| `tt/` | Phase 2 onwards |
| `tests/reference/` | Phase 1 tests |
| `tests/ttnn/` | Phase 2-5 tests |
| `tests/perf/` | Phase 6 tests |
| `tests/profile/` | Per-block profile scripts |
| `tests/server/` | Phase 7 tests |
| `demo/` | End-of-relay |
| `README.md` | Final phase — per CLAUDE.md completion criteria |
