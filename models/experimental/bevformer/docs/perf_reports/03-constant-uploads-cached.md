# Stage: 03-constant-uploads-cached

- harness: encoder wall clock, same as [stage 02](02-rebatch-plan-hoisted.md)
- config: `nuscenes_base`, 100×100, 6 layers, N150, 11 timed iterations after one warm-up
- encoder wall: **4234.5 ms median** (−56.4 ms, **−1.3%**), 4217.7 ms min (−48.3 ms, −1.1%)
- cumulative from the stage-01 tree: **−151.0 ms, −3.4%**
- PCC: 26 tests pass across the whole `tests/pcc/` suite; no numerical change

Median and minimum agree here (−56.4 vs −48.3 ms) and the spread is much tighter than stage 02's
(max 4295.2 ms against 4394.2 ms). This is the more trustworthy of the two encoder-level numbers.

## What this change was

**Tensors that never change stop being rebuilt.** Five sites, all of them uploading or deriving
something fixed by config or by the BEV grid, all of them inside a per-layer or per-call path:

| Site | Was rebuilt | Now |
|---|---|---|
| `bev_reference_points` ([tt_encoder.py](../../tt/tt_encoder.py)) | per layer — 6× per forward | cached on the layer, keyed on the source tensor |
| `offset_normalizer` ([tt_ms_deformable_attention.py](../../tt/tt_ms_deformable_attention.py)) | per deformable-attention call — 12× per forward | cached on the module, keyed on `spatial_shapes` |
| `level_start_index` ([tt_temporal_self_attention.py](../../tt/tt_temporal_self_attention.py)) | per layer — 6× per forward | cached on the module, keyed on contents |
| `reference_points_3d` ([tt_encoder.py](../../tt/tt_encoder.py)) | per forward | cached on the encoder, keyed on `(bev_h, bev_w, bs)` |
| `to_torch(spatial_shapes)` ([tt_spatial_cross_attention.py](../../tt/tt_spatial_cross_attention.py)) | per layer, **a device readback** | skipped when the shapes are on device |

Two of the caches are keyed on **tensor identity** rather than contents: the encoder hands every
layer the same `reference_points_3d` object, and caching that object at the encoder is what makes
the layers' derived uploads hit across forwards as well as across layers. The two changes reinforce
each other; either alone would only have covered layers 2–6 of a single forward.

The `spatial_shapes` entry is not a cache. It was a `ttnn.to_torch` — a full device sync — run once
per layer to check an invariant that holds by construction. It now runs only when the shapes are
already on host, which is the path every current caller takes, so nothing is actually skipped today;
what is removed is a sync that would appear the moment a caller passed shapes on device.

## Read this as a steady-state win

The benchmark warms up before timing, so every timed iteration hits warm caches. That is the honest
framing of the result: **this is what a second and subsequent frame costs**, and the first forward
still pays for all of it. For a model that runs over a sequence, which BEVFormer does, that is the
number that matters — but it is not a one-shot latency improvement, and it should not be quoted as
one.

The cached device tensors stay resident for the module's lifetime. At these shapes that is a few
hundred KB across all six layers, small enough not to matter and worth remembering if the grid grows.

## Candidate 1 is complete

| | | |
|---|---|---|
| 1a | rebatch and scatter-back on device | −2171.9 ms layer wall (−71%) |
| 1b | static `max_len` bound | [rejected](DEAD_ENDS.md#3-a-static-bound-on-max_len) |
| 1c | rebatch plan once per forward | −94.6 ms encoder wall |
| 1d | constant uploads cached | −56.4 ms encoder wall |

The host-round-trip work is done. What remains of the per-layer host cost is dispatch, not
transfers, and the profile has moved decisively to kernel time —
[candidate 4](../perf_optimization_candidates.md#candidate-4--the-msda-concat) (one 115 ms concat)
and [candidate 2](../perf_optimization_candidates.md#candidate-2--fused-msda) (623 ms of the 682 ms
kernel) are what is left worth having.
