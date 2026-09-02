# Stage: 06-sca-key-permute-deleted

- source commit: [`a933e1059d9`](https://github.com/tenstorrent/tt-metal/commit/a933e1059d96ffb98c68f78d6965a6e92d9b35f4)
- candidate: [5a](../perf_optimization_candidates.md#5a-delete-the-key-permute)
- config: `nuscenes_base`, 100×100, N150
- layer profile: **438.4 ms kernel**, 126 device ops (−1), CSV
  `generated/profiler/reports/2026_09_01_23_20_11/`
- baseline for the delta: **456.5 ms / 127 ops**, `2026_09_01_22_47_34` — a same-session
  re-measure of [stage 05](05-offset-normalizer-folded.md) on unchanged code, which reproduced
  its 456.8 ms to 0.3 ms and its per-op table exactly
- **−18.1 ms kernel (−4.0%)**; **−242.4 ms (−34.7%)** cumulative against the stage-03 layer
- PCC: **0.999611**, unchanged from stage 05 — the deleted tensor had no consumer

## What this change was

`TTSpatialCrossAttention.forward` built `key_reshaped` — a `permute(2, 0, 1, 3)` of the
`[num_cams, L, bs, embed_dims]` camera features — and passed it to the deformable attention as
`key=`.

`TTMSDeformableAttention.forward` has no `key` parameter. The argument landed in `**kwargs` and was
never read: deformable attention samples `value` at the sampling locations and has no key path at
all. The permute ran every layer and its output was dropped.

The encoder makes it worse. [`tt_encoder.py:197-198`](../../tt/tt_encoder.py#L197-L198) sets
`value = key` when `value` is not given, so in the encoder path the two permutes at
[sca:349-352](../../tt/tt_spatial_cross_attention.py#L349-L352) were **the same tensor permuted
twice** — 92 MB moved, once for nothing.

Deleted. `L` still comes off `key.shape`, which is the only remaining use of the argument.

## Where the time went

| op | stage 05 (re-measured) | stage 06 | Δ |
|---|---:|---:|---:|
| PermuteDeviceOperation | 62.7 ms (11 inst) | **43.4 ms (10 inst)** | **−19.3** |
| everything else | 393.8 ms | 395.0 ms | +1.2 |

The permute delta is exactly the 19.32 ms of profiler row 469 in the stage-05 CSV — the predicted
number, to 0.02 ms. The +1.2 ms elsewhere is run-to-run noise (`BinaryNg` 48.2 → 49.3 on identical
code paths), which is why the layer figure reads −18.1 rather than −19.3.

## Layer profile now

| Op | inst | ms | % of kernel |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.9 | 38.3 |
| ReshapeViewDeviceOperation | 20 | 77.4 | 17.7 |
| BinaryNgDeviceOperation | 17 | 49.3 | 11.2 |
| PermuteDeviceOperation | 10 | 43.4 | 9.9 |
| UntilizeWithUnpaddingDeviceOperation | 17 | 42.4 | 9.7 |
| TilizeWithValPaddingDeviceOperation | 6 | 17.1 | 3.9 |
| SliceDeviceOperation | 13 | 13.2 | 3.0 |
| ScatterDeviceOperation | 1 | 10.5 | 2.4 |
| TransposeDeviceOperation | 12 | 8.0 | 1.8 |
| MatmulDeviceOperation | 11 | 4.7 | 1.1 |

## The lesson

**A tensor passed into `**kwargs` is not a tensor that is used.** This cost 19 ms per layer for the
lifetime of the module and no profile ever pointed at it, because the profiler names the op
(`Permute`) and not the fact that its output is dead. It was found by reading the callee's signature
after the [candidate 5](../perf_optimization_candidates.md#candidate-5--data-movement-vs-compute)
classification put a 19.3 ms row next to an identical 19.3 ms row and asked why there were two.

Check the other `**kwargs` pass-throughs in this model on the same grounds.
