# Stage: 04-fused-msda

- source commit: working tree on `ctr-mmicic/optimizer/bevformer-msda-fused-2026-08-27` (parent
  [`8a48cf6bde2`](https://github.com/tenstorrent/tt-metal/commit/8a48cf6bde2))
- config: `nuscenes_base`, 100×100, N150
- layer profile: **489.5 ms kernel / 14.0 ms gap / 503.5 ms wall**, 129 device ops (+8), CSV
  `generated/profiler/reports/2026_08_27_23_24_32/`
- **−191.6 ms kernel (−28.1%), −190.6 ms layer wall (−27.5%)** against
  [stage 03](03-constant-uploads-cached.md) re-measured in the same session
  (`2026_08_27_23_03_17`: 681.1 ms kernel / 13.0 ms gap / 694.1 ms wall, 121 ops)
- PCC: **0.999611** (baseline 0.999608, gate 0.997) — unchanged

**No encoder-wall number.** Stages 02 and 03 quote a median over 11 timed iterations, but the
encoder harness in the tree runs `DEVICE_PERF_ITERS = 1` and has no wall-clock timing loop, so that
methodology is not reproducible from the repo. Only the layer harness was run. `PERF.md`'s own rule
— encoder ≈ 6 × layer — puts this at roughly −1.14 s of encoder wall, but that is arithmetic, not a
measurement.

## What this change was

The hand-rolled `multi_scale_deformable_attn_ttnn` decomposition is replaced by
`ttnn.experimental.multi_scale_deformable_attn`
([PR #52380](https://github.com/tenstorrent/tt-metal/pull/52380)), which fuses `grid_sample` with
the weighted sum over sampling points into one kernel.

A new `_fused_msda_level()` shapes one pyramid level for the op — value → `(N, H, W, D)`, grid →
`(N, Q*P, 1, 2)`, attn → `(N, Q, P)`, all ROW_MAJOR bfloat16 INTERLEAVED, which the device op
enforces with `TT_FATAL` rather than converting. `multi_scale_deformable_attn_ttnn` calls it per
level and `ttnn.add`s the results. The 47-line per-level `grid_sample` chain, the `ttnn.stack`, the
`mul` and the `sum` are deleted outright.

### Multi-level is exact, and it is on device

The candidates doc assumed multi-level SCA would need "a per-level call plus a host-side weighted
sum". It does not. `attention_weights` is softmaxed jointly over `L*P` and thereafter only summed,
so the joint weighted sum decomposes exactly into a sum of per-level weighted sums:

```
sum_{l,p} w[l,p] · v[l,p]  ==  sum_l ( sum_p w[l,p] · v[l,p] )
```

Each fused call computes one inner sum; the levels combine with an L-way `ttnn.add` on device. No
renormalization, no host round-trip, no approximation — and the measured PCC moved by 3e-6.

## Where the time went

| op | stage 03 | stage 04 | Δ |
|---|---:|---:|---:|
| GridSample | 116.0 ms | **0.0 ms** | −116.0 |
| Concat | 115.5 ms | **0.0 ms** | −115.5 |
| ReshapeView | 157.0 ms | 77.1 ms | −79.9 |
| Permute | 105.4 ms | 62.5 ms | −43.0 |
| Slice | 29.1 ms | 13.2 ms | −15.9 |
| FillPad | 4.7 ms | 0.0 ms | −4.7 |
| Reduce | 4.5 ms | 0.0 ms | −4.5 |
| BinaryNg | 85.7 ms | 81.5 ms | −4.2 |
| TilizeWithValPadding | 14.6 ms | 17.1 ms | +2.6 |
| Transpose | 0.1 ms | 7.9 ms | +7.8 |
| UntilizeWithUnpadding | 28.9 ms | 42.4 ms | +13.5 |
| **MSDAOperation** | — | **167.6 ms** | +167.6 |

Per module region:

| region | stage 03 | stage 04 | Δ |
|---|---:|---:|---:|
| TSA (`num_levels=1`) | 90.7 ms | 78.7 ms | −12.0 |
| SCA (`num_levels=4`) | 531.9 ms | 352.0 ms | −179.9 |

## The fused op is not a faster sampler

This matters for anyone reading the win as "the fused kernel is fast". It is not:

| | old `GridSample` | new `MSDAOperation` |
|---|---:|---:|
| TSA | 16.8 ms | **24.4 ms** |
| SCA (4 levels) | 99.2 ms | **143.3 ms** |

The op is **45% more expensive** than the sampling it subsumes. Every bit of the −191.6 ms comes
from deleting the tail it makes unnecessary — the `stack` (115.5 ms), the reshape that followed it
(74.9 ms), and the `mul`/`sum`/`FillPad` reduction. VADv2 reaches the same conclusion from the other
side: `models/experimental/vadv2/tt/tt_utils.py` gates its fused path behind `N*Q >= 1024` because
below that the decomposition wins.

No such threshold was added here. Copying an unmeasured constant from another model would be worse
than omitting it, and BEVFormer's smallest tested shape (30×30 TSA, `N*Q = 7200`) clears it 7×.

## PCC coverage

Full `tests/pcc/` sweep on this change: **32 passed, 1 failed** in 242.8 s.

| suite | result |
|---|---|
| `test_encoder.py` | 5/5 pass — 6-layer encoder, `nuscenes_base`/`_fast`/`_tiny`, `carla_base`/`_tiny` |
| `test_layer.py` | 4/4 pass |
| `test_temporal_self_attention.py` | 6/6 pass |
| `test_ms_deformable_attention.py` | 3/3 pass |
| `test_point_sampling_3d_2d.py` | 7/7 pass |
| `test_spatial_cross_attention.py` | 7/8 — see below |
| `test_layer_perf.py` gate | 0.999611 |

The encoder tests matter most here: they exercise all six layers end to end at PCC ≥ 0.995–0.997,
so the per-level fused path and the cross-level `ttnn.add` hold over the full stack, not just in the
single-layer harness.

### The 200×200 SCA failure is pre-existing

`test_spatial_cross_attention_forward[…-1-200-200-…]` fails with

```
TT_FATAL Out of Memory: Not enough space to allocate 2969567232 B DRAM buffer across 12 banks
```

**Verified not a regression**: stashing this change and re-running the same test produces the
identical failure, byte-for-byte the same 2969567232 B, on the stage-03 tree (jobs `012` vs `013`).

This is the same DRAM ceiling that [DEAD_ENDS entry 3](DEAD_ENDS.md#3-a-static-bound-on-max_len)
hit when bounding `max_len`. The candidates doc hoped candidate 2 would relieve it — **it does not**.
The allocation is in the sampling-location math upstream of the fused op, which this change does not
touch.

## What is left

SCA is now 352.0 ms, split:

| span | what | kernel |
|---|---|---:|
| pre-core | linears, softmax, **`div` by `offset_normalizer` (23.9 ms)**, reference-point add | 105.9 ms |
| core | 4 × `MSDAOperation` = **143.3 ms**, plus ~103 ms of per-level layout prep | 246.1 ms |

1. **`MSDAOperation`, 143.3 ms** — now the single largest op in the layer, and per the table above
   it is slower per sample than the `GridSample` it replaced. This is an upstream question: worth a
   standalone microbenchmark and, if it reproduces, an issue against the op.
2. **Per-level layout prep, ~103 ms** — `Untilize`/`Transpose`/`Slice`/`Permute`/`Tilize` ×4. Some
   is genuinely per-level; the tilize↔untilize churn around each call is not obviously necessary.
3. **The `div`, 23.9 ms** — [candidate 3](../perf_optimization_candidates.md#candidate-3--tile-padding-waste),
   removable outright by folding the normalizer into the `sampling_offsets` Linear weight.

## Effect on the backlog

- **[Candidate 4](../perf_optimization_candidates.md#candidate-4--the-msda-concat) is deleted, not
  deferred.** The 113.5 ms `Concat` no longer appears in the profile. Sequencing 4 before 2, as the
  ordering section had it, would have been wasted work.
- **[Candidate 3](../perf_optimization_candidates.md#candidate-3--tile-padding-waste) rescopes from
  ~60 ms to ~24 ms.** Its two `Permute` sites were inside the decomposition and are gone; only the
  pre-core `div` survives.
- **1b and [candidate 5](../perf_optimization_candidates.md#candidate-5--trace-capture) are
  unchanged.** The DRAM ceiling that sank 1b is still there, as the 200×200 failure shows.
