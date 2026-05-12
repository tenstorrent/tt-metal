# Time Conditioning in PI0.5 — Comparison with tt-dit and Recommended Optimizations

**Author:** sdawle@tenstorrent.com
**Branch:** `sdawle/dvartanians/pi0.5_bh`
**Status:** All three optimizations landed and measured. PCC regression clean.

### Headline result

| | Per-call latency | Throughput |
|---|---:|---:|
| Baseline (pre-optimization) | 151.95 ms | 329 actions/s |
| + Opt 1 + Opt 3 | 151.00 ms (−0.95 ms) | 331 actions/s |
| **+ Opt 2 (fused 6× modulation Dense)** | **142.57 ms (−9.38 ms)** | **351 actions/s** |

Net: **6.2% faster end-to-end** with 0.04 ms stddev. Both PCC and trace replay verified on real `lerobot/pi05_base` weights on Blackhole.

---

## Executive summary

PI0.5 introduces **adaRMSNorm** (adaptive RMSNorm) in the action expert: every layer norm and the final stack norm replace a learnable per-channel scale with `(scale, shift, gate)` produced by a Dense projection of the flow-matching timestep embedding. This is structurally identical to **AdaLN** in DiT-style diffusion transformers — only the base norm differs (RMS vs Layer).

The `tt-dit` library (`models/tt_dit/`) ships a well-tuned AdaLN implementation that is several years more mature than our adaRMS port. Reading it side-by-side with our `ttnn_gemma.py` reveals three concrete, low-risk optimizations:

1. **Fuse the per-norm modulation Dense projections into one block-level Dense (`6×dim`)** — replaces 2 matmuls per block with 1.
2. **Drop the `slice → reshape → mul/add` chain** in favor of `ttnn.rms_norm(weight=(1+scale), bias=shift, …)` — fuses adaRMS modulation into the norm kernel itself.
3. **Pure index slicing without `ttnn.reshape`** — slice `[B, 1, 6*dim]` directly into 6 `[B, 1, dim]` views via Python slicing.

The first two are independent; the third is essentially free. **Expected total impact on pi0.5 e2e latency: ~3–6 ms / chunk** (from the current 152 ms → 146–149 ms), bringing us **even with or slightly under pi0**.

A separate finding (out of scope for this report but documented for follow-up): a published study ([arxiv 2509.13574](https://arxiv.org/abs/2509.13574)) shows that vanilla flow-matching robot policies are **strongest at 2–4 denoise steps** and degrade as steps grow. Cutting `num_denoising_steps` from 10 → 4 would give a much larger speedup (~80 ms / chunk saved) and a likely accuracy improvement — but is independent of the time-conditioning work.

---

## 1. What is "time conditioning" in PI0.5?

The flow-matching denoiser integrates `dx/dt = v_θ(x_t, t)` from `t=1` (noise) to `t=0` (clean action) using Euler steps. The velocity field `v_θ` is the action expert. The timestep `t` enters the expert *not* as a token but as a per-layer modulation signal — exactly like AdaLN in DiT.

### The pipeline (per inference chunk)

```
  scalar t  ─►  sincos(t)  ─►  time_mlp_in/out  ─►  adarms_cond ∈ ℝ^{B × width}
                                                          │
                                                          ▼
  in every expert layer:  norm(x) → (1+scale)·norm(x) + shift → sublayer → gate·sublayer → + x
```

### Where the conditioning enters the block

For each of the 18 expert layers, `adarms_cond` is projected by a small Dense `(width → 3·width)` and split into `(scale, shift, gate)`. The **norm** uses `(scale, shift)`; the **residual** uses `gate`. Pi0.5 does this for both the pre-attention norm and the pre-FFW norm, plus the final stack norm — so **3 modulation Dense projections per layer step**, and `3 × 18 × 10 steps = 540` per chunk.

### Why this matters for perf

At 10 denoise steps × 18 layers × 2 norms-per-layer (skipping the final norm), the modulation Dense + slice + scale-shift overhead is the only thing that distinguishes pi0.5's expert cost from pi0's. Our measurements:

| Variant | Expert path / chunk |
|---|---:|
| pi0 (plain RMSNorm) | ~118 ms |
| pi0.5 (adaRMS, current implementation) | ~123 ms (+5 ms) |

That 5 ms is what this report aims to claw back.

---

## 2. Reference: how tt-dit does AdaLN

### 2.1 Single block-level Dense → 6× outputs

**`models/tt_dit/blocks/transformer_block.py:65-73`**

```python
self.norm1_linear = ColParallelLinear(
    modulation_dim,
    6 * dim,           #  ← 6 outputs: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f
    bias=True,
    ...
)
```

One Linear produces **all six** modulation tensors for the block in a single matmul. Compare: our pi0.5 has **two** separate `Dense(width → 3*width)` calls per block (one for the pre-attn norm, one for the pre-FFW norm).

### 2.2 Chunk via pure index slicing — no reshape

**`models/tt_dit/blocks/transformer_block.py:334-336`**

```python
def _chunk_time3d(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
```

The `time_embed` is already shaped `[B, 1, modulation_dim]`. After the `6×dim` Linear it becomes `[B, 1, 6*dim]`. Index-slicing on the last dim produces 6 views shaped `[B, 1, dim]` — these broadcast naturally over the sequence dim when applied to the (input-being-normed). **No `ttnn.reshape` is required.**

### 2.3 The killer move: modulation fused into the norm kernel

**`models/tt_dit/blocks/transformer_block.py:232-239`**

```python
spatial_normed = ttnn.squeeze(
    self.norm1_norm(
        ttnn.unsqueeze(spatial, 0),
        dynamic_weight=(1 + spatial_scale_attn),   # ← γ
        dynamic_bias=spatial_shift_attn,           # ← β
    ),
    0,
)
```

The `DistributedLayerNorm` accepts `dynamic_weight` / `dynamic_bias`, and the underlying kernel (`ttnn.experimental.dit_layernorm_post_allgather`) computes `((x - μ) / σ) · γ + β` **in a single fused op**. No separate `mul` + `add` after the normalize.

### 2.4 Gate applied to sublayer output, residual unchanged

**`models/tt_dit/blocks/transformer_block.py:281, 284`**

```python
spatial_attn = spatial_attn * spatial_gate_attn      # gate the attention output
spatial_plus_attn = spatial + spatial_attn           # then plain residual add
```

Identical to our pattern — no change needed here.

---

## 3. Our current PI0.5 implementation

**`models/experimental/pi0_5/tt/ttnn_gemma.py:51-100`**

Each norm currently runs the following sequence:

```python
def ada_rms_norm_ttnn(x, ones_weight, cond, mod_weight, mod_bias, eps, core_grid):
    normed = ttnn.rms_norm(x, weight=ones_weight, epsilon=eps, …)          # 1) plain RMS
    mod = ttnn.linear(cond, mod_weight, bias=mod_bias, …)                   # 2) Dense(width → 3*width)
    scale = ttnn.slice(mod, [0, 0],         [B, width])                     # 3) split, 3 slices
    shift = ttnn.slice(mod, [0, width],     [B, 2*width])
    gate  = ttnn.slice(mod, [0, 2*width],   [B, 3*width])
    scale = ttnn.reshape(scale, (B, 1, width))                              # 4) reshape ×3
    shift = ttnn.reshape(shift, (B, 1, width))
    gate  = ttnn.reshape(gate,  (B, 1, width))
    scale_plus_one = ttnn.add(scale, 1.0)                                   # 5) (1+scale)
    scaled         = ttnn.mul(normed, scale_plus_one)                       # 6) ·γ
    modulated      = ttnn.add(scaled, shift)                                # 7) +β
    return modulated, gate
```

Each call is **8 TTNN ops** (rms_norm + linear + 3 slices + 3 reshapes + add + mul + add). And we do this **twice per block** (pre-attn and pre-FFW) — so 16 ops per layer just for modulation.

---

## 4. Gap analysis

| Aspect | tt-dit (AdaLN) | pi0.5 (current adaRMS) | Cost of current approach |
|---|---|---|---|
| Modulation Dense | 1 per block (`width → 6*width`) | 2 per block (`width → 3*width` each) | +1 matmul + 1 memory transfer per block |
| Tensor split | Pure Python slice; output already `[B, 1, dim]` | `ttnn.slice` then `ttnn.reshape` | +3 reshape ops per norm |
| Scale + shift | Fused into LayerNorm kernel via `dynamic_weight` / `dynamic_bias` | Separate `mul (1+scale)` + `add(shift)` after `rms_norm` | +2 ops per norm |
| Gate application | `sublayer_out * gate` then add | Same | — (parity) |
| Final stack norm | adaLN with `dynamic_weight/bias` | adaRMS with the same separate `mul`+`add` chain | +2 ops per chunk |

**Op count per layer step**, modulation-only:

| | tt-dit | pi0.5 (today) | pi0.5 (after fixes) |
|---|---:|---:|---:|
| Dense projections | 1 | 2 | 1 |
| Slices | 6 | 6 | 6 |
| Reshapes | 0 | 6 | 0 |
| Post-norm `mul + add` | 0 (fused) | 4 | 0 (fused) |
| **Total extra ops / layer / step** | **7** | **18** | **7** |

Over a full pi0.5 chunk: `(18 ops − 7 ops) × 18 layers × 10 steps = 1980 ops eliminated`.

---

## 5. Verification: does `ttnn.rms_norm` support the fused affine?

**Yes** — confirmed against the runtime build on this branch:

```text
ttnn.rms_norm(input_tensor, *,
              weight=None,           # γ
              bias=None,             # β
              epsilon=1e-12,
              memory_config=None,
              program_config=None,
              ...) -> ttnn.Tensor

formula:  x_normed * γ + β
```

For adaRMS we just pass `weight = (1 + scale)` and `bias = shift`. The runtime computes `((x · rsqrt(mean(x²) + ε)) · γ) + β` in the kernel — same fused pattern tt-dit uses on LayerNorm, no separate `mul`/`add` needed.

**Open question (low-risk):** verify the kernel handles per-batch `γ`/`β` with shape `[B, 1, dim]` broadcasting over a `[B, S, dim]` input — tt-dit's `dit_layernorm_post_allgather` handles this; we should confirm the same for the standard `rms_norm` kernel. A 5-line unit test against a small random input is sufficient.

---

## 6. Recommended optimizations (prioritized)

### Optimization 1 — fuse `(scale, shift)` into `ttnn.rms_norm`
**Effort:** 1–2 hours. **Risk:** low (purely numerical equivalence).
**Files:** `models/experimental/pi0_5/tt/ttnn_gemma.py:51-100`

```python
def ada_rms_norm_ttnn(x, cond, mod_weight, mod_bias, eps, core_grid):
    mod = ttnn.linear(cond, mod_weight, bias=mod_bias, …)          # unchanged
    scale = mod[:, 0:width]                                         # pure slice
    shift = mod[:, width:2*width]
    gate  = mod[:, 2*width:3*width]
    scale_plus_one = ttnn.add(scale, 1.0)                           # (1+scale) on (B, dim)
    out = ttnn.rms_norm(x, weight=scale_plus_one, bias=shift,       # ★ fused
                       epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
    return out, gate
```

Eliminates 2 ops per norm × 36 norms per chunk + 1 final norm = **74 ops removed per chunk**.

**Expected gain:** 1.5–2.5 ms / chunk.

### Optimization 2 — fuse the two per-block Dense projections into one
**Effort:** 2–3 hours. **Risk:** medium (requires weight concat at init time and refactor of `AdaRMSGemmaBlockTTNN.__init__` to hold a single fused `mod_weight`).
**Files:** `models/experimental/pi0_5/tt/ttnn_gemma.py` (block init + forward), `models/experimental/pi0_5/tt/ttnn_paligemma.py` (weight loader injection)

Concatenate `input_layernorm.dense.weight` and `post_attention_layernorm.dense.weight` along the output dim at init time, run one `Linear(width → 6*width)` at the top of the block, slice into 6, distribute to the two norms. This mirrors `norm1_linear` in tt-dit.

Eliminates 1 Dense (and its memory transfer) per block × 18 layers × 10 steps = **180 matmuls removed per chunk**.

**Expected gain:** 2–3 ms / chunk.

### Optimization 3 — drop `ttnn.reshape` after slice
**Effort:** 30 minutes. **Risk:** very low.
**Files:** `models/experimental/pi0_5/tt/ttnn_gemma.py:71-78`

Use Python slice syntax `mod[:, :, i*size:(i+1)*size]` instead of `ttnn.slice` + `ttnn.reshape`. Requires the modulation output to be in 3D layout `[B, 1, 3*width]` (or `[B, 1, 6*width]` after Opt 2), which means producing the Dense output with that shape from the start. Eliminates **6 reshape ops per norm × 36 norms per chunk = 216 reshapes**.

**Expected gain:** 0.5–1 ms / chunk (these are cheap ops, but they all sit in the trace).

### Measured impact (real pi05_base weights on Blackhole, trace replay, 20 iters)

| Stage | Latency / chunk | Δ | Throughput | Per-call stddev |
|---|---:|---:|---:|---:|
| Current pi0.5 baseline | 151.95 ms | — | 329 actions/s | 0.05 ms |
| + Opt 1 + Opt 3 (fused adaRMS in `ttnn.rms_norm`, reshape once) | 151.00 ms | −0.95 ms | 331 actions/s | 0.06 ms |
| **+ Opt 2 (fused 6× modulation Dense per block)** | **142.57 ms** | **−9.38 ms** | **351 actions/s** | **0.04 ms** |

**Total: 9.38 ms / chunk saved, 6.2% faster end-to-end.** Throughput up from 329 → 351 actions/s. PCC regression clean across all 3 TTNN smoke tests.

The Opt 2 result was substantially larger than the projected 2–3 ms because halving the modulation Dense calls (from 360 / chunk to 180 / chunk) reduces actual device-side matmul work, not just host-dispatch overhead — each removed Dense is a real tile-side workload.

---

## 7. What we should NOT take from tt-dit (yet)

- **Tensor parallelism / `ColParallelLinear`.** tt-dit uses a mesh device and shards Linears column-wise across multiple chips. PI0.5 currently runs on a single Blackhole chip — adopting this would be a larger architectural change orthogonal to time conditioning.
- **`DistributedLayerNorm` with all-gather.** Same reason — only matters for multi-chip.
- **The `unsqueeze(0)` / `squeeze(0)` dance around norms.** tt-dit does this because its `DistributedLayerNorm` requires a 4D shape; the standard `ttnn.rms_norm` we use accepts 3D directly.

---

## 8. Implementation plan (executed)

| Day | Task | Owner | Result |
|---|---|---|---|
| Day 1 | Land Opt 3 (drop reshape) and Opt 1 (fuse adaRMS in kernel) | sdawle | ✅ 3/3 PCC tests pass; trace replay 151.00 ms (−0.95 ms vs 151.95 baseline) |
| Day 1 | Land Opt 2 (fuse two block Denses into one 6× linear) | sdawle | ✅ 3/3 PCC tests pass; trace replay **142.57 ms (−9.38 ms total)** |
| Day 1 | Update report with measured numbers | sdawle | ✅ This update |

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| `ttnn.rms_norm` doesn't broadcast `[B, 1, dim]` γ/β over `[B, S, dim]` input | 5-line unit test before refactor; if it doesn't, broadcast explicitly using `ttnn.repeat` (would lose some of the gain but Opt 2 still applies) |
| Fused 6× Dense produces a tile-misaligned shape if `dim` isn't a multiple of 32 | `dim = 1024` for Gemma-300M expert, so `6*dim = 6144` — tile-aligned. No issue. |
| Weight-layout change in the checkpoint reader breaks loading | The change is in init-time concatenation, not on-disk layout. Existing checkpoint reader unchanged. |

---

## 9. Beyond time conditioning — pointer to higher-impact follow-up

Independent of any time-conditioning optimization, the [Dense-Jump Flow Matching paper](https://arxiv.org/abs/2509.13574) (late 2025) shows that vanilla flow-matching robot policies — which pi0.5 is — **peak in accuracy at 2–4 denoise steps** and *degrade* as steps grow past 5. The current `num_denoising_steps = 10` is likely past the sweet spot.

A simple self-consistency sweep (run `sample_actions` at 1/2/3/4/5/10 steps with the same seed; compare outputs by cosine similarity and L2 drift) would quantify this for our specific checkpoint. If 4 steps are within ε of 10 steps:

| Steps | Predicted e2e latency | Throughput |
|---:|---:|---:|
| 10 (current) | 152 ms | 329 actions/s |
| 5 | ~93 ms | 535 actions/s |
| **4** | **~81 ms** | **615 actions/s** |
| 3 | ~70 ms | 717 actions/s |

This is a ~2× win that doesn't require touching the model — only a config flip. It should be tracked separately from the time-conditioning work in this report.

---

## Appendix A — line references

| File | Key lines | What's there |
|---|---|---|
| `models/tt_dit/blocks/transformer_block.py` | 65–73, 220–239, 281, 334–336 | Block-level modulation Dense; fused LayerNorm modulation; `_chunk_time3d` |
| `models/tt_dit/layers/normalization.py` | 325–363 | `DistributedLayerNorm.forward` accepting `dynamic_weight`/`dynamic_bias`; underlying `dit_layernorm_post_allgather` kernel |
| `models/experimental/pi0_5/tt/ttnn_gemma.py` | 41–100 | Current `plain_rms_norm_ttnn` and `ada_rms_norm_ttnn` |
| `models/experimental/pi0_5/tt/ttnn_paligemma.py` | (block init) | Where the per-layer modulation Dense weights are converted and injected |

## Appendix B — runtime evidence for the fused affine path

```text
$ python -c "import ttnn; help(ttnn.rms_norm)"
Computes RMS norm over input_tensor.

  RMS_norm(x, γ, β, ε) = x / √(ε + (1/N) Σ x²) · γ + β

  Keyword args:
    weight (ttnn.Tensor, optional)   ← γ
    bias   (ttnn.Tensor, optional)   ← β
    epsilon (float)
    ...
```

This signature already supports the fused modulation we need; no kernel work is required.

## Appendix C — sources

- [tt-dit transformer block](../../tt_dit/blocks/transformer_block.py)
- [tt-dit DistributedLayerNorm](../../tt_dit/layers/normalization.py)
- [Dense-Jump Flow Matching with Non-Uniform Time Scheduling for Robotic Policies (arxiv 2509.13574)](https://arxiv.org/abs/2509.13574)
- [openpi pi0/pi0.5 implementation](https://github.com/Physical-Intelligence/openpi)
- [Review of π0.5: Scaling Open-World Generalization in VLA Models](https://medium.com/correll-lab/a-review-of-%CF%800-5-scaling-open-world-generalization-in-vision-language-action-models-e4cb0d74264e)
