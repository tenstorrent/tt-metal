# Why encoder PCC dropped after the deterministic camera rig

## Scope

Two encoder PCC parametrizations began failing after the camera geometry in the
BEVFormer tests stopped being random:

```
FAILED test_encoder.py::test_bevformer_encoder_forward[...nuscenes_tiny...0.996...] - PCC: 0.994607
FAILED test_encoder.py::test_bevformer_encoder_forward[...carla_tiny...0.995...]    - PCC: 0.994937
```

The commit that changed the geometry:

- `752990330b96edf4e0ca0c2125592d932f1998cb` — *Use deterministic lidar2img
  matrices in BEVFormer tests* (branch `ctr-mmicic/bev-former`, 2026-08-25).
  Replaces `torch.randn(4, 4)` projection matrices with an approximate nuScenes
  camera rig in `config/encoder_config/camera_rig.py`; consumed by
  `tests/pcc/test_encoder.py`, `tests/layer_common.py` and
  `tests/profile/test_encoder_profile.py` through `img_metas_for_dataset`.

The commit changed no model or kernel code. It changed the operating point the
tests measure.

## Conclusion first

1. The commit is not the defect. It removed a degenerate operating point that was
   hiding a pre-existing TT/reference gap.
2. The gap is **not** in point sampling, **not** in bilinear interpolation, and
   **not** in `MSDeformableAttention`. It is in the **`TTSpatialCrossAttention`
   wrapper** — the accumulate / count-divide / output-projection tail that runs
   after the inner deformable attention.
3. The wrapper injects a diffuse per-query relative error of **~9.5% for
   `num_levels=1` configs and ~5.0% for `num_levels=4`**, on essentially every
   attended query. A faithful bfloat16 replay of the same arithmetic accounts for
   only ~1.3% of it, so most of it is implementation, not storage dtype.
4. Thresholds should not be re-baselined yet. The wrapper is a fixable candidate
   and re-baselining would freeze the loss in.

## How the geometry changes the operating point

`lidar2img` enters the encoder only through `point_sampling_3d_to_2d`, which
produces `reference_points_cam` and `bev_mask`. Those decide where
`F.grid_sample` reads. `grid_sample` uses `padding_mode="zeros"`, so a sample
location outside the feature map returns zero on both paths — free agreement
that exercises no interpolation.

Random 4x4 matrices put nearly every valid sample at the extreme image border:

| config | geometry | valid_frac | max_len | valid coords in border band (<0.02 or >0.98) | sampling locations inside [0,1] |
|---|---|---|---|---|---|
| nuscenes_tiny | random | 0.1505 | 2847 | **0.9361** | 0.346 |
| nuscenes_tiny | rig | 0.1875 | 2472 | **0.0177** | 0.811 |
| carla_tiny | random | 0.1506 | 2850 | **0.9478** | — |
| carla_tiny | rig | 0.2052 | 2146 | **0.0148** | — |
| nuscenes_base | random | 0.1507 | 2852 | **0.9752** | 0.296 |
| nuscenes_base | rig | 0.1875 | 2472 | **0.0177** | 0.806 |

Per-camera coverage also went from degenerate to balanced — `nuscenes_base`
random `[248, 2603, 780, 1975, 2852, 2102]` versus rig
`[1579, 1913, 1919, 1812, 1799, 2472]`.

So the rig raised the fraction of sample locations that are actually interpolated
from ~30–35% to ~81%. The uniform-random control (`torch.rand` reference points,
the operating point the standing `test_ms_deformable_attention.py` uses) sits at
98%, i.e. the rig is much closer to the standing sub-op test than the old random
matrices were.

## Bottom-up ladder

Each level holds weights and feature inputs fixed and varies only the rig.
Harness: `tests/pcc/test_geometry_divergence.py` (diagnostic, no thresholds).

### L1 — `point_sampling_3d_to_2d` — diverges, but geometry-independent

| config / geometry | `reference_points_cam` PCC | `bev_mask` exact | mask mismatches | ref max_len | TT max_len |
|---|---|---|---|---|---|
| nuscenes_tiny / random | 0.998902 | no | 310 / 240k | 2847 | 2849 |
| nuscenes_tiny / rig | 0.998933 | no | 367 / 240k | 2472 | 2478 |
| carla_tiny / random | 0.999576 | no | 111 | 2850 | 2852 |
| carla_tiny / rig | 0.999942 | no | 229 | 2146 | 2155 |
| nuscenes_base / random | 0.998902 | no | 316 | 2852 | 2851 |
| nuscenes_base / rig | 0.998931 | no | 404 | 2472 | 2484 |

TT point sampling does not reproduce `bev_mask` bit-exactly under either
geometry, and `max_len` differs by up to 12 queries — the reference decides 2472
where TT decides 2484. This is a boundary-comparison effect (`depth > eps` and
the `0 < x,y < 1` tests evaluated in reduced precision), TT is consistently the
more permissive side, and the magnitude is the same before and after the commit.
**Not the cause of the regression**, but a standing discrepancy worth its own
fix: it makes the rebatch length device-dependent.

Measured directly: with the rig, a mask disagreement is harmless; with random
matrices it is not.

| config / geometry | SCA PCC, shared mask | SCA PCC, TT-native mask | disagreeing queries | PCC over disagreeing rows |
|---|---|---|---|---|
| nuscenes_tiny / random | 0.996641 | 0.996247 | 84 | **0.949564** |
| nuscenes_base / random | 0.999327 | 0.998893 | 84 | **0.946521** |
| nuscenes_tiny / rig | 0.995474 | 0.995477 | 79 | 0.996450 |
| nuscenes_base / rig | 0.998743 | 0.998744 | 89 | 0.999007 |

Under the rig a flipped query lands at an image edge where the sampled feature is
nearly identical either way. Under random matrices the flipped query's coordinate
is far out of range, so including it injects garbage. The rig therefore made the
mask discrepancy *less* damaging, not more.

### L2 — `MSDeformableAttention` in isolation — essentially clean

Fed the reference points the rig actually produces, rebatched exactly as SCA
does. `uniform` is the standing sub-op test's operating point.

| config | geometry | num_levels | PCC | sampling locations inside [0,1] |
|---|---|---|---|---|
| nuscenes_tiny | random | 1 | 0.999959 | 0.346 |
| nuscenes_tiny | rig | 1 | 0.999728 | 0.811 |
| nuscenes_tiny | uniform | 1 | 0.999727 | 0.983 |
| nuscenes_base | random | 4 | 0.999980 | 0.296 |
| nuscenes_base | rig | 4 | 0.999861 | 0.806 |
| nuscenes_base | uniform | 4 | 0.999858 | 0.976 |

`rig` and `uniform` agree to the fifth decimal, so the geometry did not move MSDA
to a worse regime than the standing test already covers. The interpolation itself
is fine; PCC 0.9997 cannot produce an encoder at 0.9946.

### L3 — `SpatialCrossAttention` — the gap appears here

Both paths given the identical reference mask, so indexing is identical and the
gap is arithmetic only.

| config | geometry | num_levels | PCC | row rel-err p50 | p90 | p99 | max | rows > 0.1 |
|---|---|---|---|---|---|---|---|---|
| nuscenes_tiny | random | 1 | 0.996641 | 0.07438 | 0.11903 | 0.15021 | 0.20258 | 2420 |
| nuscenes_tiny | rig | 1 | **0.995474** | **0.09507** | 0.11533 | 0.13384 | 0.16260 | 3767 |
| nuscenes_base | random | 4 | 0.999327 | 0.03435 | 0.04976 | 0.06363 | 0.08347 | 0 |
| nuscenes_base | rig | 4 | **0.998743** | **0.05073** | 0.05753 | 0.06337 | 0.07477 | 0 |

The error is **diffuse, not a tail**: p50 ≈ p90 ≈ p99. Mean relative error on
attended rows is 0.095 (tiny/rig) versus 0.0024 on unattended rows — i.e. the
error is entirely in the queries that go through the attention path, and it is
uniform across them.

`num_levels` is the discriminator: 1 level gives ~9.5% row error, 4 levels ~5.0%,
at identical geometry. With one level every sample lands on the largest, highest
frequency feature map; with four the result is an average over three additional
downsampled, smoother maps.

### L3b — inside SCA: wrapper, not inner attention

Reference rebatch / scatter / count-divide / output-projection executed on host in
float32, with only the inner deformable attention on TT.

| config | geometry | inner attention PCC | hybrid output PCC (host fp32 wrapper) | hybrid, bfloat16-rounded wrapper | full TT SCA PCC |
|---|---|---|---|---|---|
| nuscenes_tiny | random | 0.999948 | 0.999990 | 0.999989 | 0.996641 |
| nuscenes_tiny | rig | 0.999650 | **0.999916** | 0.999915 | **0.995474** |
| nuscenes_base | random | 0.999976 | 0.999995 | 0.999994 | 0.999327 |
| nuscenes_base | rig | 0.999827 | **0.999959** | 0.999958 | **0.998743** |

Host wrapper self-consistency (reference inner, host wrapper) is 1.000000, so the
replay is faithful.

**This is the isolation.** With the TT inner attention feeding a host wrapper the
output is 0.999916; with the TT wrapper it is 0.995474 — a **53x larger error**
from the wrapper alone. Rounding the hybrid's wrapper arithmetic to bfloat16
changes it by 1e-6, so bfloat16 *storage* explains none of the difference.

The wrapper's TT tail, per `tt/tt_spatial_cross_attention.py`:

- `slots_torch[...] += queries_output_torch[...]` — host scatter-add, both
  operands bfloat16 (they arrive via `ttnn.to_torch` of bfloat16 tensors)
- `ttnn.div(slots, count_expanded)` — count is 1 or 2 for 99%+ of queries, so
  exact in binary floating point
- `ttnn.linear(slots, output_proj.weight, bias=...)` — bfloat16 activations, no
  `compute_kernel_config` passed, so default math fidelity and accumulation
- `ttnn.add(slots, inp_residual)`

The division cannot lose 9.5%, and the scatter is one rounding. The output
projection matmul is the only step with the arithmetic depth to produce it, and
it is the only step running with default fidelity settings.

### L4 — `TemporalSelfAttention` — cannot be affected, not run

TSA is called with `reference_points=reference_points_3d[:, :, 0, :2]` and
`spatial_shapes=bev_shape` (`reference/encoder.py:146`). Those come from the BEV
grid, not from `lidar2img`; neither `reference_points_cam` nor `bev_mask` reaches
it. The profile signposts confirm it: TSA drives the deformable attention at
`10000 - 10000` queries/keys regardless of geometry, while the SCA call moves with
`max_len`. No device run was spent on this level.

### L5 — single layer

| config | random | rig |
|---|---|---|
| nuscenes_tiny | 0.997323 | 0.996338 |
| carla_tiny | 0.998015 | 0.996498 |
| nuscenes_base | 0.998461 | 0.998647 |

### L6 — encoder, per layer

| config | geometry | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| nuscenes_tiny | random | 0.997323 | 0.996482 | 0.996228 | | | |
| nuscenes_tiny | rig | 0.996338 | 0.995026 | 0.994639 | | | |
| carla_tiny | random | 0.998015 | 0.997186 | 0.997327 | | | |
| carla_tiny | rig | 0.996498 | 0.994988 | 0.994968 | | | |
| nuscenes_base | rig | 0.998647 | 0.998104 | 0.997949 | 0.997881 | 0.997973 | 0.997940 |

`nuscenes_base` / random is missing: with `return_intermediate=True` the larger
random-geometry `max_len` (2852 vs 2472) pushes the 6-layer, 4-level case over
DRAM. That is a limit of this diagnostic harness, not of the product path. The
L5 row above covers base under both geometries.

The curve is flat after layer 1 — the whole gap is present in the first layer and
does not accumulate with depth. That is why 3-layer `tiny` fails while 6-layer
`base` passes: depth is not the variable, `num_levels` is.

A storage-precision bound (same reference graph, inputs and weights rounded to
bfloat16, float32 arithmetic) gives 0.999997–0.999998 at every layer of every
config — three orders of magnitude below the observed gap.

## Old vs new, end to end

| config | encoder PCC, random | encoder PCC, rig | standing threshold | verdict |
|---|---|---|---|---|
| nuscenes_tiny | 0.996228 | 0.994639 | 0.996 | was passing on margin, now fails |
| carla_tiny | 0.997327 | 0.994968 | 0.995 | now fails by 3e-5 |
| nuscenes_base | — (L5: 0.998461) | 0.997940 (L5: 0.998647) | 0.997 | passes either way |

## Recommended next experiment

Instrument `TTSpatialCrossAttention` to PCC each stage of its tail against the
host reference at the same point: `slots` after the scatter, after
`ttnn.div`, after `ttnn.linear(output_proj)`, and after the residual add. The
prediction from L3b is that the loss is concentrated in the output projection.

Then the direct fix hypothesis: pass an explicit `compute_kernel_config` to that
`ttnn.linear` with higher math fidelity and fp32 destination accumulation, and
re-measure. If the SCA PCC moves from 0.9955 toward the hybrid's 0.9999, the
encoder gap closes without touching the geometry and the thresholds can go **up**,
not down.

Secondary, independent of the above: make TT `point_sampling_3d_to_2d` agree with
the reference on `bev_mask`. `max_len` differing between host and device makes the
spatial-path tensor shapes device-dependent, which also affects what the profile
harness measures.

## Do not re-baseline yet

The two failing thresholds are 1.4e-3 and 3e-5 under the line. Both are inside
the error the SCA wrapper contributes. Re-baselining now would encode a wrapper
deficiency as the expected accuracy of the model.

## Note found in passing

`model_config.num_points_in_pillar` (2 for `tiny`, 4 for `base`) is dead: the
pillar depth actually used is `dataset_config.z_cfg["num_points"]`, which is 4 for
every preset. `get_encoder_kwargs()` passes both, and `z_cfg` wins. Any reasoning
that treats `tiny` as a 2-point-pillar configuration is wrong.

## Reproducing

```
pytest models/experimental/bevformer/tests/pcc/test_geometry_divergence.py -sv -k point_sampling
pytest models/experimental/bevformer/tests/pcc/test_geometry_divergence.py -sv -k ms_deformable
pytest models/experimental/bevformer/tests/pcc/test_geometry_divergence.py -sv -k spatial_cross
pytest models/experimental/bevformer/tests/pcc/test_geometry_divergence.py -sv -k stage_isolation
pytest models/experimental/bevformer/tests/pcc/test_geometry_divergence.py -sv -k layer_divergence
pytest models/experimental/bevformer/tests/pcc/test_encoder_layerwise.py -sv
```

Run the levels as separate invocations; several TT modules instantiated in one
process exhaust DRAM.
