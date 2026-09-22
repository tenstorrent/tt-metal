# Per-op cells WITH the patches applied

> **Base note — do not compare these against `../per_op/` cell-by-cell.** That set is
> {3 chunk widths} x {first chunk, depth 57,344} at **one** configuration (Asif's branch +
> `GEMMA4_PREFILL_L1_ACT=1`, no patches). This set is {1-2 widths} x {first chunk only} x
> {4 flag configurations}. Different axes. Lining up a cell from each and taking the
> difference is the base-mixing that produced a retracted regression earlier in this work.

Captured 2026-09-21 on sha `ece1fb6c066`, mesh 8x4, ctx 256k, `chunk_idx 0`, `layer_type=both`,
`GEMMA4_PREFILL_L1_ACT=1` throughout. Floor captures only: both patches are per-chunk floor
fixes and `slope` moved <=0.3% at every width, so depth captures would be near-duplicates.

| file prefix | chunk | `GEMMA4_NORM_SHARD` | `GEMMA4_ATTN_MM_CFG` |
|---|---|---|---|
| `a_base_c2048` | 2048 | off | off |
| `b_norm_c2048` | 2048 | **on** | off |
| `c_attn_c2048` | 2048 | off | **on** |
| `d_both_c2048` | 2048 | **on** | **on** |
| `e_attn_c8192` | 8192 | off | **on** |

## What they show

**Layer totals (Device Time, ms/layer) and the whole-model delta vs `a_base`:**

| config | global | sliding | whole-model | e2e says |
|---|---|---|---|---|
| `a_base` | 2.546 | 2.330 | — | — |
| `b_norm` | 2.267 | 2.047 | **−16.93 ms** | −16.7 |
| `c_attn` | 2.537 | 2.360 | **+1.40 ms** | +1.5 |
| `d_both` | 2.257 | 2.080 | **−15.39 ms** | −15.2 |

**Per-op reconciles with the end-to-end fit to 1.4%, 7% and 1.3%.** That is what licenses the
op-level attribution below.

### Exp 4 (norm shard) — all three pre-registered predictions held

```
LayerNorm              0.463 -> 0.121 ms/layer   3.8x      -20.51 ms whole-model
InterleavedToSharded   0.000 -> 0.032            new        +1.93 ms
ShardedToInterleaved   0.000 -> 0.031            new        +1.85 ms
                                                  net      -16.73 ms  (e2e: -16.7)
```
CCL ops moved less than the 0.15 ms reporting floor — the control held. The two reshards give
back **18%** of the gross win, against ~22% predicted.

### Exp 2 (attention core_grid) — the sign flip is ONE op, and it flips

Per-layer, patched vs unpatched, at both widths:

| matmul | chunk 2048 | chunk 8192 |
|---|---|---|
| `_x2048x5376` | 0.086 -> 0.121 (**+41%**) | 0.164 -> 0.113 (**−31%**) |
| `_x5376x4096` | 0.179 -> 0.175 (−2%) | 0.226 -> 0.250 (+11%) |
| `_x4096x5376` | 0.153 -> 0.149 (−2%) | 0.280 -> 0.175 (−37%) |
| `_x5376x4608` | 0.185 -> 0.181 (−2%) | 0.249 -> 0.183 (−27%) |

At chunk 2048 the three larger projections are flat within noise and **the smallest one is 41%
slower** — x50 sliding layers = +1.77 ms, the entire regression. At 8192 everything helps.
`_x2048x5376` is half the MACs of the next smallest at this width (2.82 G vs 5.6-6.3 G), so
this reads as over-parallelisation: a full 120-core grid costs more in dispatch than it saves
on an op that small. Same class of result as the RMSNorm finding — more cores is not faster.

## ⚠️ Unresolved: does Exp 2 still cost anything once Exp 4 is applied?

Two instruments disagree, both near their noise floors:

- **per-op**, this set: `d_both` − `b_norm` = **+1.54 ms**
- **end-to-end**, same day: Exp 2 on 111.0 ms vs gated off 111.4 ms = **−0.4 ms**

A ~1.9 ms discrepancy. A size gate (`rows/device >= 1024`) was written and measured to settle
it; it moved nothing outside ±0.5 ms at any width, so it was **reverted rather than shipped on
ambiguous data**. Exp 2 remains committed and env-gated off.

**To resolve:** run the two e2e configurations back to back in one session, several repeats,
rather than an hour apart. The +1.5 ms regression that motivated the gate was measured with
Exp 2 *alone*, on a base without Exp 4 — it may simply not exist in the shipping combination.
