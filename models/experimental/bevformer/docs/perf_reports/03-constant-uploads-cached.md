# Stage: 03-constant-uploads-cached

| | |
|---|---|
| commit | [`7d6f4433bb3`](https://github.com/tenstorrent/tt-metal/commit/7d6f4433bb38e0782d0e9ec468354bb3bb384134) |
| candidate | [1d](../perf_optimization_candidates.md#1d-per-call-constant-uploads) |
| config | `nuscenes_base`, 100×100, N150 |
| encoder | **4234.5 ms median** (−56.4 ms, **−1.3%**); 4217.7 ms min (−48.3 ms, −1.1%) — 6 layers, 11 timed iterations, same harness as [stage 02](02-rebatch-plan-hoisted.md) |
| layer | **681.7 ms kernel** / 40.5 ms gap / 722.2 ms wall, **121 ops (−25)**, CSV `generated/profiler/reports/2026_08_27_13_47_55/` |
| steady-state gap | **8.9 ms** once [region-entry cost](../PERF.md#the-gap-column-carries-region-entry-cost) is separated (`2026_08_27_20_44_43`) |
| cumulative | **−151.0 ms encoder wall (−3.4%)** from the stage-01 tree |
| PCC | 26 tests pass across `tests/pcc/`; no numerical change |

Median and minimum agree here (−56.4 vs −48.3 ms) and the spread is much tighter than stage 02's
(max 4295.2 vs 4394.2 ms) — the more trustworthy of the two encoder-level numbers. Unlike stage 02
this one **is** visible in the layer harness: the caches sit on the layer and on the
deformable-attention module, not on the encoder, so 25 removed dispatches show up per layer.

## What changed

**Tensors that never change stop being rebuilt.** Five sites, all deriving or uploading something
fixed by config or by the BEV grid, all inside a per-layer or per-call path:

| Site | Was rebuilt | Now |
|---|---|---|
| `bev_reference_points` ([tt_encoder.py](../../tt/tt_encoder.py)) | per layer — 6× per forward | cached on the layer, keyed on the source tensor |
| `offset_normalizer` ([tt_ms_deformable_attention.py](../../tt/tt_ms_deformable_attention.py)) | per deformable-attention call — 12× per forward | cached on the module, keyed on `spatial_shapes` |
| `level_start_index` ([tt_temporal_self_attention.py](../../tt/tt_temporal_self_attention.py)) | per layer — 6× per forward | cached on the module, keyed on contents |
| `reference_points_3d` ([tt_encoder.py](../../tt/tt_encoder.py)) | per forward | cached on the encoder, keyed on `(bev_h, bev_w, bs)` |
| `to_torch(spatial_shapes)` ([tt_spatial_cross_attention.py](../../tt/tt_spatial_cross_attention.py)) | per layer, **a device readback** | skipped when the shapes are on device |

Two caches are keyed on **tensor identity** rather than contents: the encoder hands every layer the
same `reference_points_3d` object, and caching that object at the encoder is what makes the layers'
derived uploads hit across forwards as well as across layers. Either change alone would only have
covered layers 2–6 of a single forward.

The `spatial_shapes` entry is not a cache. It was a `ttnn.to_torch` — a full device sync — run once
per layer to check an invariant that holds by construction. What is removed is a sync that would
appear the moment a caller passed shapes on device.

## Read this as a steady-state win

Every timed iteration hits warm caches, so this is **what a second and subsequent frame costs**; the
first forward still pays for all of it. For a model running over a sequence, which BEVFormer does,
that is the number that matters — but it is not a one-shot latency improvement. The cached device
tensors stay resident for the module's lifetime, a few hundred KB across six layers at these shapes.
Worth remembering if the grid grows.

## Candidate 1 is complete

| | | |
|---|---|---|
| 1a | rebatch and scatter-back on device | −2344.7 ms layer wall (−76%) |
| 1b | static `max_len` bound | [rejected](DEAD_ENDS.md#3-a-static-bound-on-max_len) |
| 1c | rebatch plan once per forward | −94.6 ms encoder wall |
| 1d | constant uploads cached | −56.4 ms encoder wall |

8.9 ms of steady-state gap against 682 ms of kernel, so **kernel is 99% of layer wall clock**.
[Candidate 4](../perf_optimization_candidates.md#candidate-4--the-msda-concat) (one 115 ms concat) and
[candidate 2](../perf_optimization_candidates.md#candidate-2--fused-msda) (623 ms of the 683 ms
kernel) are what is left worth having.
