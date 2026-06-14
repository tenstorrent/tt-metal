<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax-M3 — Missing-Op Discovery

Device-free analysis. For each of the 23 buildable components, the concrete ttnn ops its
TTNN forward needs were diffed against ttnn's **actual** inventory
(`dir(ttnn)`, `dir(ttnn.experimental)`, `dir(ttnn.transformer)`, `dir(ttnn.experimental.ccl)`)
and against how the SAME op is realized in the reference TTNN models
(`gpt_oss`, `deepseek_v3`, `llama3_70b_galaxy`, `qwen3_vl`, `tt_transformers`).

**Verdict: 2 ops must be hand-authored — exactly the two expected, no surprises.**
- `sparse_lightning_attention` → **tt-lang** (truly missing: learned indexer + block-sparse SDPA).
- `vision_rope_3d` → **ttnn-compose-with-custom-host-table** (the on-device apply reuses
  `rotary_embedding_llama`; the 3-axis/spatial-merge cos/sin table is a host builder; head_dim=80
  is a *padding workaround* already solved in `qwen3_vl`, not a true op gap).

The other 21 components are compositions of existing ttnn primitives (`ttnn-compose`).
`mtp_head` is out of scope (declared in config, **no weights** in this checkpoint).

## ttnn inventory — confirmations

| Need | Present in ttnn? | How / where |
|------|------------------|-------------|
| matmul / linear | yes | `ttnn.matmul`, `ttnn.linear` |
| rms_norm | yes | `ttnn.rms_norm` (+ `rms_norm_pre/post_all_gather`) |
| layer_norm (bias) | yes | `ttnn.layer_norm` |
| embedding | yes | `ttnn.embedding` |
| softmax / sigmoid / gelu / silu / erf / clamp | yes | `ttnn.*` (gelu erf-mode; `ttnn.erf`) |
| topk / sort / argmax | yes | `ttnn.topk`, `ttnn.sort`, `ttnn.argmax` |
| SDPA (dense/causal/paged/windowed/flash-MLA) | yes | `ttnn.transformer.scaled_dot_product_attention`, `paged_*`, `windowed_*`, `flash_mla_prefill`, ... |
| rotary embedding (1D llama/NeoX/HF) | yes | `ttnn.experimental.rotary_embedding_llama` / `_hf` / `_fused_qk` |
| create/concat qkv heads (incl. ViT) | yes | `ttnn.experimental.nlp_create_qkv_heads`, `nlp_create_qkv_heads_vit`, `nlp_concat_heads` |
| conv2d / conv1d | yes | `ttnn.conv2d`, `ttnn.conv1d` |
| **conv3d** | **NO** | `dir(ttnn).conv3d == False` (only `Conv3dConfig`, used by `tt_dit`). Not needed — see patch_embedding. |
| CCL (all_gather / reduce_scatter / all_reduce / all_to_all) | yes | `ttnn.all_gather`, `ttnn.reduce_scatter`, `ttnn.all_reduce`, `ttnn.experimental.ccl.*` incl. `deepseek_moe_reduce_scatter`, `all_to_all_dispatch_metadata` |
| MoE expert dispatch | yes | `ttnn.experimental.moe_expert_token_remap`, `moe_routing_remap` (gpt_oss `Experts`) |
| **rotary with 3-axis banded table** | **NO** | rotary ops are 1D only → `vision_rope_3d` builds its table host-side |
| **learned block-sparse / lightning indexer attention** | **NO** | no SDPA variant takes a per-query learned top-k-block selection → `sparse_lightning_attention` |

## Per-component ttnn-op needs

| Component | kind | ttnn ops needed | Backend | Gap reason (if any) |
|-----------|------|-----------------|---------|---------------------|
| rms_norm | norm | `rms_norm` (Gemma 1+w) | compose | — |
| vision_layernorm | norm | `layer_norm` (+bias) | compose | — |
| embedding | embedding | `embedding` | compose | — |
| rope | other | `rotary_embedding_llama` over 0:64 slice + passthrough 64:128 (deepseek MLA partial-rope) | compose | partial rope = compose; 64 tile-aligned |
| **vision_rope_3d** | other | host 3-axis/smerge table builder + `rotary_embedding_llama` (padded apply) | **ttnn-compose*** | 3-axis banded table has no 1D-rotary analogue; head_dim 80→96 pad (qwen workaround); 78-rot/2-pass |
| patch_embedding | conv | `linear` over flattened [C·tps·p·p]=1176 patch (no bias) | compose | conv3d ABSENT but ==linear (kernel==stride); host-resident OK |
| qk_norm | norm | `rms_norm` per-head over head_dim 128 | compose | — |
| gqa_attention | attention | `linear`, `nlp_create_qkv_heads`, `rms_norm`, `rotary_embedding_llama`, `scaled_dot_product_attention`, `nlp_concat_heads`, `all_reduce` | compose | KV ×8 replication via placement |
| **sparse_lightning_attention** | attention | indexer (`linear`,`rms_norm`,rope,matmul,amax,scatter+inf,`topk`,mask-expand) + GQA SDPA | **tt-lang** | learned block selector + block-sparse mask not expressible by any ttnn SDPA |
| vision_attention | attention | `linear(+bias)`, `nlp_create_qkv_heads_vit`, `scaled_dot_product_attention` (non-causal), `nlp_concat_heads` | compose | head_dim 80→96 pad (qwen3_vl) |
| swigluoai_mlp | mlp | `linear`, `clamp`, `mul`, `sigmoid`, `add` | compose | exact gpt_oss prefill sequence |
| moe_gate | linear | `linear`, `sigmoid`, `add`(bias, selection-only), `topk`, normalize, ×2.0 | compose | deepseek `DeepseekMoeGateOp` (demo composition) |
| moe_experts | mlp | gpt_oss `Experts` (dispatch + swiglu-oai + `all_reduce`) | compose | expert-parallel ~4/chip |
| shared_expert | mlp | swiglu_oai (bare, no scale) | compose | — |
| vision_mlp | mlp | `linear(+bias)`, `gelu(erf)` | compose | — |
| dense_decoder_layer | decoder | rms_norm + gqa_attention + swiglu_oai + residual `add` | compose | — |
| moe_decoder_layer | decoder | rms_norm + **sparse_lightning_attention** + moe_gate + moe_experts + shared_expert + `add` | compose** | body adds nothing new; depends on the authored attn |
| vision_encoder_layer | decoder | vision_layernorm + vision_attention + vision_mlp + `add` | compose | depends on vision_rope_3d |
| patch_merge_mlp | mlp | `reshape`, `linear(+bias)`, `gelu(erf)` | compose | host-resident OK |
| multimodal_projector | mlp | `linear(+bias)`, `gelu(erf)` | compose | host-resident OK |
| vision_encoder | other | patch_embedding + vision_rope_3d + 32×encoder_layer + projector + merge | compose** | composite; depends on vision_rope_3d |
| final_norm | norm | `rms_norm` | compose | — |
| lm_head | linear | `linear`, `all_gather` | compose | column-parallel |
| mtp_head | decoder | — | **out of scope** | declared in config, **no weights** in checkpoint |

\* `vision_rope_3d`: device apply is a compose, but the custom 3-axis/spatial-merge cos/sin **host
table builder** + 78-rot/2-passthrough handling is genuinely new code (just not a device kernel).
\*\* `moe_decoder_layer` / `vision_encoder` are composites whose own bodies add no new op; they
transitively depend on an authored op (`sparse_lightning_attention` / `vision_rope_3d`).

## Missing ops to author

### 1. `sparse_lightning_attention`  — backend: **tt-lang** (truly missing)
- **Used by:** `sparse_lightning_attention`, `moe_decoder_layer` (layers 3..59, 57 layers).
- **Why missing:** A learned *lightning indexer* (`index_q_proj`/`index_k_proj`, 4 heads × dim 128,
  per-head `index_q_norm`/`index_k_norm`, partial rope) scores `idx_q·idx_kᵀ`, **max-pools** over
  each 128-key block then over index heads (`amax`/`amax`), force-boosts the local block(s) to `+inf`,
  **top-16** selects blocks, and builds a **block-selection AND token-causal** additive mask that
  gates a normal GQA SDPA. No ttnn SDPA variant accepts a per-query *learned top-k-block* index set
  (`scaled_dot_product_attention`, `paged_*`, `windowed_*`, `flash_mla_*` take only dense/causal/
  paged/sliding-window masks). The fused indexer (`matmul → amax-over-block → amax-over-head →
  +inf local scatter → topk-block with −1 left-packing → repeat_interleave block→token mask`) has no
  ttnn analogue.
- **Tile alignment:** all dims (head_dim 128, index_head_dim 128, block_size 128, rotary_dim 64) are
  multiples of 32. S_k is `-inf`-padded to a 128 multiple before block reshape (already tile-aligned).
  The difficulty is the **dynamic gather**, not alignment.
- **Validate against:** `reference/golden/sparse_lightning_attention.pt` ←
  `sparse_lightning_attention_forward` (+ `_lightning_indexer_block_indices`, `_build_block_mask`).
- **Bring-up fallback:** run **full dense-causal GQA** (`gqa_attention`) on these layers for parity
  (the indexer only *prunes* keys, so dense GQA is a strict superset, PCC-equivalent when no block is
  dropped — see ARCHITECTURE.md and the use-case `hybrid_notes`). Author the real op for long-context
  efficiency afterward.
- **Spec:** `sparse_lightning_attention.reference.notes` (full tt-lang spec in git notes).

### 2. `vision_rope_3d`  — backend: **ttnn-compose** (custom host table builder; NOT a device kernel)
- **Used by:** `vision_rope_3d`, `vision_attention`, `vision_encoder_layer`, `vision_encoder`.
- **Why "missing" but compose:** 3D RoPE over (T,H,W): a shared `inv_freq` × per-axis coordinate,
  banded `[T13|H13|W13|T13|H13|W13]` (rot_dim 78), 2 tail channels never rotated, NeoX half-split,
  with spatial-merge (m=2) coordinate reordering and temporal `repeat_interleave`. ttnn rotary ops are
  **1D only** — no 3-axis banded table. **But** `qwen3_vl/tt/rope.py` proves the cos/sin **table is
  built host-side** (`compute_gather_cos_sin` → torch) and only the **apply** runs on device. So the
  banded/coord/passthrough logic is a host-side table builder; the device apply reuses
  `rotary_embedding_llama`. The new artifact is the table builder + honoring 78-rot/2-pass — not a
  device kernel.
- **Tile alignment:** head_dim 80 is **not** a multiple of 32 → pad to `padded_head_dim = 96`
  (`ceil(80/32)·32`), **exactly** the documented workaround in `qwen3_vl/tt/vision_attention.py`
  (it pads q/k/v/out weights *and* the rope cos/sin mats to 96; in-file comment: *"workaround until
  rotary embeddings support sub-tile head dims"*). Pad the 78-wide rotated band to 96 with cos=1/sin=0
  in the 78..96 tail so pad + the 2 genuine passthrough channels are identity. **head_dim=80 is a
  solved tile-alignment workaround, not a true op gap.**
- **Validate against:** `reference/golden/vision_rope_3d.pt` ←
  `build_vision_rope_3d` + `vision_rope_3d_forward`.
- **Spec:** `vision_rope_3d.reference.notes` (TT-LANG SPEC).

## Notes / surprises
- **conv3d is genuinely absent from ttnn** (`dir(ttnn).conv3d == False`; only `Conv3dConfig` exists,
  used by `tt_dit`). It does **not** become a missing op: `patch_embedding`'s Conv3d has kernel==stride
  and exactly tiles each flattened 1176-wide patch, so it is mathematically a bias-free **linear**
  (reference.notes: "==linear on flat patch"), and is host-resident-allowed.
- **swiglu-oai is NOT a new op** — the exact `clamp(max=7) → ·σ(α·g) → (up.clamp(±7)+1)·glu → down`
  sequence is already in `gpt_oss/tt/experts/prefill.py`.
- **moe_gate is NOT a ttnn primitive** — `ttnn.moe` is a specialized expert-0-weight op, unrelated to
  routing. The deepseek router is a **demo-level composition** (`DeepseekMoeGateOp`: `linear`+`sigmoid`
  +`add`(bias)+`topk`+scaling). MiniMax differs only in values (scaling 2.0 vs 2.5; no grouped routing).
- **No third gap found.** Every other component maps cleanly onto existing primitives + the
  TP/CCL ops (`all_reduce`/`all_gather`/`reduce_scatter`/`all_to_all`, `deepseek_moe_reduce_scatter`)
  already shipping in ttnn and exercised by the reference demos.

**Compose-OK count: 21.  Ops to author: 2 (`sparse_lightning_attention` tt-lang; `vision_rope_3d`
host-table-builder over existing rotary).  Out of scope: 1 (`mtp_head`, no weights).**
