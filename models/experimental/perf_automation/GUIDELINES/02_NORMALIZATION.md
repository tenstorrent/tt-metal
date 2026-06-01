# 02 · Normalization — LayerNorm, GroupNorm, RMSNorm

Normalization is the most numerically sensitive op in a transformer and a frequent
bottleneck. This file combines the LayerNorm findings from BGE-M3 (24-layer BERT,
the hardest precision case), ViT (12-layer, fully sharded), the GroupNorm findings
from Swin-L + DyHead, and the RMSNorm / distributed-norm patterns from the LLM
(Llama / DeepSeek) implementations.

> RMSNorm vs LayerNorm: RMSNorm skips the mean subtraction and the beta shift -
> `y = x * rsqrt(mean(x^2) + eps) * gamma`. It needs only `E[x^2]` (one reduction) where
> LayerNorm needs both `E[x]` and `E[x^2]`. Everything below about sharding, fidelity,
> fp32 accumulation, and residual fusion applies identically to both.

---

## 1. The math (so you know what you're trading)

LayerNorm / RMSNorm per row of `H` elements:
1. `Σ x_i` (mean) — *LayerNorm only; RMSNorm skips the mean*
2. `Σ (x_i − mean)²` (variance) — RMSNorm uses `Σ x_i²`
3. `1 / sqrt(var + ε)` (rsqrt)
4. `y = γ · (x − mean) · rsqrt + β`

The **reductions** are the bottleneck on tile hardware: they touch every input tile and
the rsqrt is a per-row scalar that must broadcast back. This is also why normalization
is the op most vulnerable to low-precision accumulation error.

GroupNorm (Swin-L DyHead) reduces over `(H·W / num_groups)` spatial elements per group
instead of over the hidden dim — same reduction structure, different axis.

---

## 2. Interleaved vs sharded vs distributed — the central decision

| Variant | Input layout | Program config | Use when |
|---|---|---|---|
| **Interleaved** | DRAM-interleaved | default (`None`) | activation doesn't fit L1 (large batch / long seq / LLM prefill) |
| **Sharded** | block-sharded L1 | `LayerNormShardedMultiCoreProgramConfig` | activation fits in the core-grid L1 shards |
| **Distributed** | sharded **along embedding dim across devices** | pre/post all-gather ops | multi-device, hidden dim fractured across chips (LLMs) |

**Interleaved-vs-sharded by phase (LLM):** for the *non-distributed* norm, interleaved
input parallelizes across `seq_len` -> optimal for **prefill** (long seq); width-sharded
input splits across the embedding dim -> optimal for **decode** (`seq_len=1`).

- **ViT**: sharded LN everywhere — the entire 12-layer encoder stays block-sharded in L1.
- **BGE-M3 batch 1**: sharded LN (activation fits 64-core L1).
- **BGE-M3 batch 32**: interleaved LN (activation is 32 MB — sharded path hits the L1 clash).
- **Swin-L stages**: interleaved LN; sharded LN was *slower* once reshard overhead was counted (see §8).

**Rule:** shard when the activation fits and the producer/consumer can share the shard
grid. Otherwise interleaved. Don't force sharding at large batch — you'll hit the clash.

---

## 3. Sharded program config — every knob

```python
ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(grid_x, grid_y),
    subblock_w=block_w,        # math chunk width; = block_w for one math iter/row (best)
    block_h=M_tiles // grid_y, # rows of tiles per shard — MUST divide evenly
    block_w=N_tiles // grid_x, # cols of tiles per shard — MUST divide evenly
    inplace=False,             # reuse input shard for output
    # The following are LN-kernel specific (BGE-M3 exposes them; ViT does not set them):
    # welford=False, legacy_reduction=True, legacy_rsqrt=True,
)
```

### Layout math
For `[B·S, H]` with `M_tiles = B·S/32`, `N_tiles = H/32` on `gx × gy`:
- `block_h = M_tiles / gy` must be integer.
- `block_w = N_tiles / gx` must be integer.
- `subblock_w ≤ block_w` and divides it cleanly.

Worked examples:
- **ViT 224-seq** `[10, 224, 768]`: `seqL_t=7`, `dim_t=24`, grid 12×10 → `block_h=7`, `block_w=2`, `subblock_w=2`.
- **ViT high-res 512²** `seqL=1024`, batch 1, grid 8×8 → `block_h = (1·32)/8 = 4`.
- **BGE-M3 B1** `[1, 512, 1024]`: `M_t=16`, `N_t=32`, grid 8×8 → `block_h=2`, `block_w=4`, `subblock_w=4`.

### Knob effects (BGE-M3 data)
| Knob | Safe value | Failure mode if changed |
|---|---|---|
| `inplace` | False | True breaks PCC at batch 32 (input/output CB collide) |
| `welford` | False (unless recip tensor set up) | True fails out-of-box without precomputed reciprocals |
| `legacy_reduction` | True | False (one-pass) drops 24-layer PCC to 0.89 |
| `legacy_rsqrt` | True | False is "faster" but +180 µs in practice at B32 |

ViT (12 layers) does **not** set the legacy/welford knobs and uses `math_approx_mode=True`
— acceptable at its depth. BGE-M3 (24 layers) needs the conservative settings.

---

## 3b. Distributed norm — sharded across devices on the embedding dim

When the hidden dim is fractured across devices (LLMs), no single device has the full row
to reduce over. The distributed norm is a three-step pattern:

```python
# 1. local partial stats on each device's shard
stats = ttnn.rms_norm_pre_all_gather(x_shard)          # [1,1,batch, TILE_W*num_stats]
#    num_stats=1 for RMSNorm (E[x^2]); =2 for LayerNorm (E[x], E[x^2])
# 2. gather stats across devices (moves only the tiny stats tensor, not the activation)
gathered = ttnn.all_gather(stats, dim=3, cluster_axis=1, mesh_device=mesh, topology=...)
# 3. global normalize using the gathered stats
y = ttnn.rms_norm_post_all_gather(x_shard, epsilon=eps, weight=g_shard, stats=gathered, ...)
```

The all-gather moves only the tiny stats tensor (one column per device), not the
activation - that's what makes distributed norm cheap. Use bf8b for the CCL where PCC
allows (see 08 section 7). Reference: `models/tt_transformers/tt/distributed_norm.py`.

---

## 3c. The DRAM weight-layout trick (norm weights)

Norm weights (gamma, beta) in TILE layout need padding to TILE_HEIGHT, wasting DRAM
bandwidth. Wrap them into TILE_WIDTH sticks in ROW_MAJOR instead - no padding, done once
at init, zero runtime cost:

```python
gamma = gamma.view(1, 1, embedding_dim // TILE_WIDTH, TILE_WIDTH)
ttnn_gamma_rm = ttnn.as_tensor(gamma, layout=ttnn.ROW_MAJOR_LAYOUT,
                               dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

---

## 4. Fidelity and fp32 accumulation — the firm rule

```python
ln_compute = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,   # NOT LoFi
    math_approx_mode=False,                   # False at depth ≥ ~16; True OK at ≤ ~12 (ViT)
    fp32_dest_acc_en=True,                    # MUST be True — False drops PCC to ~0.89
    packer_l1_acc=True,
)
```

- **HiFi2 is the floor.** LoFi LN passes single-layer (1.0) but fails 24-layer (~0.91).
  HiFi4→HiFi2 is a safe, large win (BGE-M3 B32: ~−870 µs LN bucket). HiFi2→LoFi is **not** safe for LN.
- **`fp32_dest_acc_en=True` is mandatory** for LN/GroupNorm/RMSNorm. The variance reduction loses precision in fp16 DST → PCC ~0.89.
  - *Contrast with matmul, where False is preferred.* Normalization is the exception.
- ViT-BH uses `fp32_dest_acc_en=False` for LN at 12 layers / 224 seq and it passes —
  **depth and sequence length set the tolerance.** Always full-model-PCC gate.

---

## 5. The bf8b normalization precision bug — and the fix

**Symptom (BGE-M3, the campaign's hardest bug):** bf8b activations through 24 sharded-LN
calls dropped full-model PCC to **~0.50** (random output) while single-layer PCC stayed
at 0.9999. bf8b tiles share one exponent per 16 elements, so the variance reduction
accumulates per-block quantization error that compounds over depth.

**The fix — typecast bf8b → bf16 *inside* the I→S reshard, fused into one op:**

```python
if x.dtype == ttnn.bfloat8_b and sharded_memcfg is not None:
    x_sharded = ttnn.interleaved_to_sharded(
        x, sharded_memcfg,
        output_dtype=ttnn.bfloat16,   # ← KEY: LN now sees bf16 tiles, no compounding
    )
else:
    x_sharded = ttnn.to_memory_config(x, sharded_memcfg)
norm_out = ttnn.layer_norm(x_sharded, ...)
```

Symmetric on the way out (downstream matmul wants bf8b):
```python
y_bf8b = ttnn.sharded_to_interleaved(norm_out, dram_cfg, output_dtype=ttnn.bfloat8_b)
```

Result: 24-layer PCC restored 0.50 → 0.95, MTEB −0.011 → 0.847. Net cost ~100 µs,
recovered by dropping the surrounding matmuls to LoFi (now safe because LN protects its
own precision).

**Generalize:** any reduction op fed bf8b that compounds over depth — fold the bf16 cast
into the reshard that precedes it. Single op, no extra dispatch.

---

## 6. Residual fusion — the most under-used feature

`ttnn.layer_norm` accepts `residual_input_tensor`. This makes `add + LN` one op:

```python
# Before (2 ops):  y = ttnn.add(x, r);  y = ttnn.layer_norm(y, ...)
# After  (1 op):   y = ttnn.layer_norm(x, residual_input_tensor=r, ...)
```

Mathematically identical (no PCC change), removes one `add` per LN call. Use it
**everywhere** you have `LN(x + r)`:
- Position-embedding fold into the embedding LN: `LN(word, residual=position)`.
- Encoder block residual: `LN_attn(attn_out, residual=x)`, `LN_mlp(mlp_out, residual=y1)`.

ViT keeps the residual `add` explicit but block-sharded (so no reshard); BGE-M3 folds it
into the LN. Both avoid the round-trip — pick whichever matches your layout.

---

## 7. Sharded handoff — chain LN output into the next op

Sharded LN output can feed the next op without a reshard, **if** the next op accepts the
same shard config:

| Next consumer | Approach | Result |
|---|---|---|
| Next sharded LN (same block) | pass sharded output as its residual | win (−19 µs BGE-M3) |
| Next sharded LN (next block) | cross-block sharded handoff | win (−53 µs BGE-M3) |
| Sharded-in0 matmul | pass sharded output as in0 | **often a loss** — many matmuls reshard internally (+106 µs) |

ViT threads block-sharded tensors through the *entire* encoder so LN→matmul→LN never
round-trips. This works because ViT's matmul program configs are written to consume the
exact block-sharded layout. **Test the handoff: if the downstream matmul reshards anyway,
the handoff is a regression.**

---

## 8. GroupNorm (Swin-L DyHead) — the spatial-reduction cousin

GroupNorm normalizes over `(H·W / groups)` spatial elements. Findings from the
Swin-L DyHead campaign (Wormhole):

- **`use_welford=False` was the single biggest GroupNorm win** (557 µs → 239 µs per call,
  −57%). Welford's running-mean reduction is more numerically careful but much slower;
  at large spatial sizes (≥200 elements/group) plain bf16 accumulation is stable enough.
- **Fidelity walk HiFi3 → HiFi2 → LoFi** all passed PCC (>0.98 gate) and saved time.
- **`fp32_dest_acc_en`** could go False here (unlike LayerNorm) because GroupNorm at large
  spatial size is less depth-compounding than 24-layer LN.
- **Core-grid utilization is a trap**: `_pick_grid` requires `grid_y` to divide
  `ceil(H·W/32)`. For prime tile counts (e.g. `Ht=13`) only a 1×8 grid fits → 8 of 64
  cores active. Padding to a composite tile count to fill the grid *lost PCC* (mean over
  padded zeros) — not worth it.

**Generalize:** for any norm, the welford/legacy/approx knobs trade precision for speed.
Turn off the careful-but-slow variant first, gate on full-model PCC, and don't over-pad
to fill the grid if it pollutes the reduction.

---

## 9. The bandwidth ceiling — know when to stop

After program-config and fidelity tuning, LN is **bandwidth-bound**: "read shard, reduce,
write shard." BGE-M3 B1 LN floors at ~9.6 µs/call. Doubling cores doesn't help; the op is
at its memory-bandwidth ceiling. **Don't keep chasing LN once it's sharded + HiFi2 + fused
residual + fused dtype cast.** If your norm is still slow it's almost certainly:
- running interleaved when it should be sharded,
- running HiFi4 when HiFi2 suffices,
- running with an unfused residual add or dtype cast.

---

## 10. What didn't work (with data)

| Attempt | Result | Why |
|---|---|---|
| Width-sharded LN | large regression | shard math doesn't fit the reduction pattern |
| `inplace=True` at large batch | PCC fail | input/output CB collide |
| LoFi LN (any depth ≥ 16) | 24-layer PCC 0.91 | reduction error compounds |
| one-pass reduction (`legacy_reduction=False`) | PCC 0.89 | loses bf8b precision |
| `fp32_dest_acc_en=False` on 24-layer LN | PCC 0.89 | fp16 variance accumulation |
| sharded LN at large batch | L1 clash | per-core shard + CBs > 1.57 MB |
| sharded LN → sharded-in0 matmul handoff | +106 µs | matmul reshards internally |
| GroupNorm welford=True | 2.3× slower | careful reduction unnecessary at large spatial |
| GroupNorm grid over-padding (prime Ht) | PCC drop | mean over padded zeros |

---

## 11. Quick reference

| Question | Answer |
|---|---|
| Shard or interleave? | Shard if activation fits L1 and consumer shares grid; else interleave |
| Fidelity | HiFi2 (floor). Never LoFi for LN at depth ≥ ~16. |
| `fp32_dest_acc_en` | **True** for LN/RMSNorm (opposite of matmul) |
| `math_approx_mode` | False at depth ≥ ~16; True acceptable at ≤ ~12 (ViT) |
| bf8b input | fuse bf16 cast into the I→S reshard |
| Residual | fuse via `residual_input_tensor` |
| GroupNorm welford | False (big win, gate PCC) |
| When to stop | once sharded + HiFi2 + fused residual + fused cast → it's bandwidth-bound |
