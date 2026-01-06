
# ViT in TT-NN (Blackhole)
Authors: Vishal Shenoy, Mohamed Bahnas

## Contents
- [ViT in TT-NN (Blackhole)](#vit-in-tt-nn-blackhole)
  - [Contents](#contents)
  - [1. Overview](#1-overview)
  - [2. Blackhole-specific execution strategy (what changes vs. Grayskull)](#2-blackhole-specific-execution-strategy-what-changes-vs-grayskull)
    - [2.1 Blackhole hardware notes relevant to TT-NN](#21-blackhole-hardware-notes-relevant-to-tt-nn)
    - [2.2 Core-grid strategy used by the Blackhole ViT demo](#22-core-grid-strategy-used-by-the-blackhole-vit-demo)
    - [2.3 Shape padding assumptions (ViT seqL padding)](#23-shape-padding-assumptions-vit-seql-padding)
  - [3. ViT TT-NN Optimization Techniques (Blackhole)](#3-vit-tt-nn-optimization-techniques-blackhole)
    - [3.1 Sharding on all relevant ops (and minimizing inter-op data movement)](#31-sharding-on-all-relevant-ops-and-minimizing-inter-op-data-movement)
    - [3.2 Matmul sharding variants used in ViT (Blackhole)](#32-matmul-sharding-variants-used-in-vit-blackhole)
    - [3.3 Attention-specific optimizations (Blackhole)](#33-attention-specific-optimizations-blackhole)
    - [3.4 LayerNorm + compute kernel configuration](#34-layernorm--compute-kernel-configuration)
    - [3.5 Weight preprocessing (Blackhole demo)](#35-weight-preprocessing-blackhole-demo)
  - [4. ViT TT-NN Code Structure (Blackhole)](#4-vit-tt-nn-code-structure-blackhole)
    - [4.1 Top-level modules](#41-top-level-modules)
    - [4.2 Embeddings module](#42-embeddings-module)
    - [4.3 Encoder module](#43-encoder-module)
    - [4.4 Encoder layer module (one layer)](#44-encoder-layer-module-one-layer)
  - [5. ViT Encoder Layer TT-NN Deep Dive (Blackhole)](#5-vit-encoder-layer-tt-nn-deep-dive-blackhole)
    - [5.1 Input](#51-input)
    - [5.2 Sharding parametrization](#52-sharding-parametrization)
    - [5.3 Layer Normalization (LayerNorm)](#53-layer-normalization-layernorm)
    - [5.4 Multi-Head Self-Attention](#54-multi-head-self-attention)
      - [5.4.1 Q,K,V Generation (Fused Linear)](#541-qkv-generation-fused-linear)
      - [5.4.2 Reshard (120-core grid ↔ variable grid)](#542-reshard-120-core-grid--variable-grid)
      - [5.4.3 Split into Q/K/V + heads](#543-split-into-qkv--heads)
      - [5.4.4 Attention scores (Q×Kᵀ) + scale](#544-attention-scores-qk--scale)
      - [5.4.5 Softmax (in-place)](#545-softmax-in-place)
      - [5.4.6 Context (P×V)](#546-context-pv)
      - [5.4.7 Concatenate heads + Self-output Linear](#547-concatenate-heads--self-output-linear)
      - [5.4.8 Reallocate/defragmentation notes](#548-reallocatedefragmentation-notes)
    - [5.5 Add and Norm](#55-add-and-norm)
    - [5.6 Feed-Forward Network (FFN/MLP)](#56-feed-forward-network-ffnmlp)
    - [5.7 Output](#57-output)
  - [6. Conclusion](#6-conclusion)
  - [7. References](#7-references)

## 1. Overview

The Vision Transformer (ViT) treats an image as a sequence of patch embeddings and applies standard Transformer blocks:

LayerNorm → Multi-Head Self-Attention → residual → LayerNorm → MLP/FFN → residual

The TT-NN implementation focuses on:
- Keeping intermediate tensors in **sharded L1** when possible (minimizing data movement between ops)
- Using **matmul sharding + multicast** patterns that fit the device grid and ViT’s shapes
- Using lower precision datatypes (e.g. **BFLOAT8_B**) where acceptable for performance

This document is intentionally **Blackhole-specific**: it highlights the places where the optimized flow differs from the earlier Grayskull-oriented writeup, especially around **core-grid selection**, **resharding**, and Blackhole-relevant memory/NoC constraints.

## 2. Blackhole-specific execution strategy (what changes vs. Grayskull)

### 2.1 Blackhole hardware notes relevant to TT-NN

This report uses **repo-local** tech reports for Blackhole-relevant hardware constraints and software behavior:

- `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md`
- `tech_reports/memory/allocator.md`
- `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`

Key Blackhole differences that can influence TT-NN program configs and data movement:

- **Compute grid size**: Blackhole exposes a larger Tensix grid; the bring-up guide lists **13×10 compute-available** cores (within a 14×10 total Tensix grid).
- **DRAM topology**: Blackhole devices have **8 DRAM banks (~4GB each)** (vs. 12×~1GB on Wormhole); this affects allocator behavior and large-weight placement decisions.
- **NoC alignment + multicast**: Blackhole has different NoC read alignment (64B) and supports additional multicast shapes (rectangular/strided/L-shaped). The bring-up guide also notes some older-kernel patterns required extra NoC flushes on BH during bring-up.

### 2.2 Core-grid strategy used by the Blackhole ViT demo

The Blackhole ViT demo uses a **fixed 10×12 grid** (120 cores) for many block-sharded ops, and a **variable grid** derived from batch size for some attention internals and for the classifier.

Why `grid_x = 12` is convenient for ViT-Base:
- For ViT-Base, `hidden_size = 768`, so `dim_t = 768/32 = 24` tiles.
- With `grid_x = 12`, each core in X naturally maps to `dim_t__x = 24/12 = 2` tiles.

The demo also includes explicit comments indicating that for Blackhole `grid_x` can be reduced (e.g. 6/4/3) for lower latency regimes, but the current optimized path defaults to `grid_x = 12`.

### 2.3 Shape padding assumptions (ViT seqL padding)

The demo pads the ViT token sequence length to a tile multiple:

- Patch tokens: `seqL = 14×14 = 196`
- Padded tokens: `seqL_padded = 224` (tile height is 32)
- `seqL_t = seqL_padded/32 = 7`

Many sharded program configs are expressed in tiles and use these derived values.

## 3. ViT TT-NN Optimization Techniques (Blackhole)

This section summarizes the concrete optimization techniques used in the Blackhole ViT demo implementation (`ttnn_optimized_sharded_vit_bh.py`). Many techniques are shared with the earlier Grayskull writeup (sharding, mcast, fused QKV), but **Blackhole adds explicit grid/reshard/reallocate strategy** that is important to capture.

### 3.1 Sharding on all relevant ops (and minimizing inter-op data movement)

- **Goal**: keep activations sharded in **L1** across consecutive ops to avoid costly inter-core movement and DRAM round-trips.
- **Primary memory configs used**:
  - `ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG` for most MLP + projection-style ops
  - `ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG` for attention BMM-style ops (QKᵀ and PV)

**Blackhole-specific note (core-grid + reshard strategy)**:
- The demo uses a stable **10×12 grid** for many block-sharded ops (call it the “120-core grid”).
- It then **reshards** tensors onto a **variable grid** (derived from batch size) where needed, and sometimes uses `ttnn.reallocate()` to avoid fragmentation/defragmentation issues in attention.
- This makes the optimized BH flow explicitly a **multi-grid pipeline**, not “one grid everywhere”.

### 3.2 Matmul sharding variants used in ViT (Blackhole)

#### 3.2.1 Reuse + Multicast (2D Block sharded)
Used for the large “projection-like” matmuls where we want to keep inputs/outputs block-sharded:
- **QKV projection** (`hidden_states @ Wqkv`)
- **Self-output projection** (post-attention linear)
- **FFN/MLP** matmuls (FF1 with fused GELU, FF2)

Implementation pattern:
- `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig`
- Block sharding is arranged on a 2D core grid, and inputs are multicast along one grid dimension to reduce redundant reads.

The conceptual sharding diagrams from the original report are still useful for Blackhole:

![Sharding Concept](images/sharding_concept.png)
![Sharding Example](images/sharding_example.png)
![Mcast Block](images/block_mcast.png)

#### 3.2.2 Reuse (BMM) with Height sharding (1D)
Used in attention:
- **Q × Kᵀ** matmul to form attention scores
- **P × V** matmul to form the context

Implementation pattern:
- `ttnn.MatmulMultiCoreReuseProgramConfig`
- Tensors are **height sharded** so each core processes a slice of \((batch \times heads \times seq)\).

![BMM Height](images/bmm_height.png)
![Mcast Height](images/height_mcast.png)

### 3.3 Attention-specific optimizations (Blackhole)

Compared to a high-level “Transformer attention” description, the BH demo is explicit:

- **Fused QKV linear**: Q/K/V are generated with one `ttnn.linear` using pre-packed fused weights.
- **Explicit scaling**: attention scores are multiplied by \(1/\sqrt{head\_size}\) via `ttnn.mul_`.
- **In-place softmax**: uses `ttnn.softmax_in_place` with a sharded softmax program config.
- **Reshard + reallocate points**:
  - After the QKV linear, the result is **resharded** onto the variable core grid before split-heads + BMM.
  - In some batch regimes the code uses `ttnn.reallocate()` on `value` and/or on `self_output` to avoid allocator fragmentation/defragmentation behavior in attention.

These steps are part of the optimized “contract” and should be treated as required for the BH path, not incidental.

### 3.4 LayerNorm + compute kernel configuration

LayerNorm is executed in block-sharded form:
- `ttnn.LayerNormShardedMultiCoreProgramConfig`

The BH demo also passes a tuned compute kernel config for LayerNorm:
- `math_fidelity=HiFi2`, `math_approx_mode=True`, `packer_l1_acc=True`

### 3.5 Weight preprocessing (Blackhole demo)

The Blackhole demo performs preprocessing to match TT-NN’s expected packed/tiled formats:

- **Patch embedding projection weights** are padded (e.g., channels padded to 4), permuted, reshaped, and stored as tiled BFLOAT8_B.
- **QKV weights/bias** are concatenated into a fused QKV tensor and preprocessed (`preprocess_linear_weight`, `preprocess_linear_bias`) to match the QKV matmul program configuration.
- **Classifier weights/bias** are padded from 1000 → 1152 classes for tiling and sharding convenience.

These preprocessing steps are key for end-to-end performance since they:
- avoid runtime transposes/packing,
- keep compute operating on well-aligned tile shapes,
- and ensure mcast/shard shapes match program configs.

## 4. ViT TT-NN Code Structure (Blackhole)

> Draft placeholder: This section will be rewritten to point to Blackhole entrypoints and show BH-specific config setup (`update_model_config`) and model functions (`vit`, `vit_encoder`, `vit_layer`, `vit_attention`).

### 4.1 Top-level modules
> Draft placeholder

### 4.2 Embeddings module
> Draft placeholder

### 4.3 Encoder module
> Draft placeholder

### 4.4 Encoder layer module (one layer)
> Draft placeholder

## 5. ViT Encoder Layer TT-NN Deep Dive (Blackhole)

> Draft placeholder: This section will mirror the Grayskull deep dive but will be updated to match Blackhole:
> - fixed 10×12 grid for block-sharded ops
> - variable grid reshard inside attention
> - explicit scaling + in-place softmax
> - `reallocate()` notes

### 5.1 Input
> Draft placeholder

### 5.2 Sharding parametrization
> Draft placeholder

### 5.3 Layer Normalization (LayerNorm)
> Draft placeholder

### 5.4 Multi-Head Self-Attention
> Draft placeholder

#### 5.4.1 Q,K,V Generation (Fused Linear)
> Draft placeholder

#### 5.4.2 Reshard (120-core grid ↔ variable grid)
> Draft placeholder

#### 5.4.3 Split into Q/K/V + heads
> Draft placeholder

#### 5.4.4 Attention scores (Q×Kᵀ) + scale
> Draft placeholder

#### 5.4.5 Softmax (in-place)
> Draft placeholder

#### 5.4.6 Context (P×V)
> Draft placeholder

#### 5.4.7 Concatenate heads + Self-output Linear
> Draft placeholder

#### 5.4.8 Reallocate/defragmentation notes
> Draft placeholder

### 5.5 Add and Norm
> Draft placeholder

### 5.6 Feed-Forward Network (FFN/MLP)
> Draft placeholder

### 5.7 Output
> Draft placeholder

## 6. Conclusion

> Draft placeholder: Summarize Blackhole-specific aspects (grid strategy, reshard/reallocate, BFLOAT8_B usage) and point to next optimizations.

## 7. References

- ViT paper: https://arxiv.org/pdf/2010.11929
- HuggingFace ViT docs: https://huggingface.co/docs/transformers/en/model_doc/vit
- TT-NN sharding & layouts: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_layouts/tensor_layouts.md
- Blackhole bring-up notes (repo): `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md`
- Allocator / memory banks (repo): `tech_reports/memory/allocator.md`
- Matrix engine / fidelity notes (repo): `tech_reports/matrix_engine/matrix_engine.md`
- GEMM FLOPS and BH grid reference (repo): `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`
