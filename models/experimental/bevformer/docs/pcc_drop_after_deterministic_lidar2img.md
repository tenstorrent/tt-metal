# Why encoder PCC dropped after the deterministic camera rig

**Closed.** Two encoder parametrizations began failing when the test camera geometry stopped being
random — `nuscenes_tiny` at 0.994607 against a 0.996 threshold, `carla_tiny` at 0.994937 against
0.995. The geometry commit was not the defect: it removed a degenerate operating point that was
hiding a pre-existing parameter-plumbing bug.

| | |
|---|---|
| exposed by | [`752990330b9`](https://github.com/tenstorrent/tt-metal/commit/752990330b96edf4e0ca0c2125592d932f1998cb) — deterministic `lidar2img` in the tests. Replaces `torch.randn(4, 4)` projections with an approximate nuScenes rig in `config/encoder_config/camera_rig.py`. **Changed no model or kernel code** — only the operating point. |
| fixed by | [`d897012f5cc`](https://github.com/tenstorrent/tt-metal/commit/d897012f5cc8b276f6afa6f55ad22818d65764c8) — nests the deformable attention's parameters under their own key |

## The defect

**`TTSpatialCrossAttention` applied its own `output_proj` twice and never applied the nested
deformable attention's at all.**

`preprocess_spatial_cross_attention_parameters` (`tt/model_preprocessing.py:296-318`) built a **flat**
namespace: `parameters["output_proj"]` ← the SCA's own; `value_proj` / `sampling_offsets` /
`attention_weights` ← lifted out of the nested attention; and the nested attention's own
`output_proj` **skipped**, because line 315 guarded it with
`and not hasattr(torch_model, "output_proj")`. `TTSpatialCrossAttention.__init__` then handed that
whole flat namespace to the inner attention, which found `self.params.output_proj` — the SCA's — and
applied it. SCA applied the same matrix again.

| | computes |
|---|---|
| reference | `SCA.output_proj( MSDA.output_proj(attn) / count ) + residual` |
| TT | `SCA.output_proj( SCA.output_proj(attn) / count ) + residual` |

Verified by substitution: replacing the *reference* inner attention's `output_proj` with the SCA's
reproduces the TT module to **0.999646**, against **0.978711** for the correct reference — and
0.999646 matches standalone `TTMSDeformableAttention` accuracy (0.999650) to 4e-6. Nothing else
differed.

**Not a precision problem.** Not bfloat16, not math fidelity, not bilinear interpolation, not the
accumulate/divide/project tail — all four were measured clean to 0.99998 or better (see the ladder).

## Why the geometry exposed it

`lidar2img` enters only through `point_sampling_3d_to_2d`, which decides where `grid_sample` reads.
`padding_mode="zeros"` means an out-of-range sample returns zero on both paths — free agreement that
exercises no interpolation. Random matrices put nearly every valid sample at the extreme border:

| config | geometry | valid_frac | max_len | valid coords in border band | locations inside [0,1] |
|---|---|---|---|---|---|
| nuscenes_tiny | random | 0.1505 | 2847 | **0.9361** | 0.346 |
| nuscenes_tiny | rig | 0.1875 | 2472 | **0.0177** | 0.811 |
| carla_tiny | random | 0.1506 | 2850 | **0.9478** | — |
| carla_tiny | rig | 0.2052 | 2146 | **0.0148** | — |
| nuscenes_base | random | 0.1507 | 2852 | **0.9752** | 0.296 |
| nuscenes_base | rig | 0.1875 | 2472 | **0.0177** | 0.806 |

Per-camera coverage also went from degenerate to balanced — `nuscenes_base` random
`[248, 2603, 780, 1975, 2852, 2102]` versus rig `[1579, 1913, 1919, 1812, 1799, 2472]`. So the rig
raised interpolated samples from ~30–35% to ~81%, close to the 98% the standing
`test_ms_deformable_attention.py` already uses.

## The ladder that found it

Each level held weights and feature inputs fixed and varied only the rig. The
`test_geometry_divergence.py` harness was diagnostic — every level logged PCC and asserted nothing —
and was removed once the cause was found.

> This is a summary. The full ladder — per-level, per-config, per-geometry PCC tables, ~40 measured
> values — was condensed here once the bug was fixed and the numbers stopped being actionable. It is
> in git at `69e140abe2e:models/experimental/bevformer/docs/pcc_drop_after_deterministic_lidar2img.md`
> if a future regression needs to be compared against it cell by cell.

| level | what | verdict |
|---|---|---|
| L1 `point_sampling_3d_to_2d` | `reference_points_cam` PCC 0.9989–0.9999; `bev_mask` never bit-exact (111–404 mismatches of 240k); `max_len` differs host vs device by up to 12 | **Diverges, but geometry-independent** — same magnitude before and after. See [the standing discrepancy](#secondary-and-still-open). |
| L1b | with the rig, a flipped mask query lands at an image edge where the feature is nearly identical: PCC over disagreeing rows **0.9965–0.9990**. With random matrices it lands far out of range: **0.9465–0.9496** | The rig made the mask discrepancy *less* damaging, not more |
| L2 `MSDeformableAttention` alone | rig 0.999728–0.999861, `uniform` 0.999727–0.999858 — agree to the fifth decimal | **Clean.** 0.9997 cannot produce an encoder at 0.9946 |
| L3 `SpatialCrossAttention` | rig 0.995474 (tiny) / 0.998743 (base); row rel-err **p50 ≈ p90 ≈ p99** (0.095 / 0.051) | **The gap appears here**, and it is diffuse, not a tail. Mean rel-err 0.095 on attended rows vs 0.0024 on unattended. |
| L3b wrapper vs inner | TT inner + **host fp32 wrapper** → 0.999916; TT wrapper → 0.995474 — a **53× larger error from the wrapper alone**. bfloat16-rounding the hybrid wrapper moves it 1e-6 | **The isolation.** bfloat16 storage explains none of it |
| L3c each tail op vs fp32 | bf16 round-trip 0.999999 · `div` 0.999999 · `linear` default 0.999980 · LoFi 0.999881 · HiFi2/HiFi4 0.999962 · HiFi4+fp32 acc 0.999980 · residual `add` 0.999993 | **The tail is not the loss.** Raising math fidelity is not the fix — the whole default-to-HiFi4 span is 1.5e-5 |
| L3d capture the module's own call | inputs correct (`query` 0.999999, `reference_points` 0.999993, `value` 0.999999); inner output vs reference **0.978711**; vs reference with SCA `output_proj` substituted **0.999646** | **The bug.** Right inputs, wrong output, and the wrong output is exactly the double projection |
| L4 `TemporalSelfAttention` | driven from the BEV grid, not `lidar2img`; signposts show `10000 - 10000` regardless of geometry | Cannot be affected; no device run spent |
| L5/L6 layer and encoder | rig, per layer: tiny 0.996338 → 0.994639 by layer 3; base 0.998647 → 0.997940 by layer 6 | **Flat after layer 1** — the whole gap is in the first layer and does not accumulate. `num_levels` is the variable, not depth: 1 level gives ~9.5% row error, 4 levels ~5.0% at identical geometry. |

A storage-precision bound (same reference graph, inputs and weights rounded to bfloat16, float32
arithmetic) gives 0.999997–0.999998 at every layer of every config — three orders of magnitude below
the observed gap.

`nuscenes_base` / random is absent from L6: with `return_intermediate=True` the larger random-geometry
`max_len` (2852 vs 2472) pushes the 6-layer 4-level case over DRAM. A limit of the diagnostic harness,
not of the product path.

## The fix, and what it moved

Two changes, both in the parameter path, so one fix covers the standalone SCA and the layer/encoder
path (`model_preprocessing.py:529-531`):

1. The deformable attention's parameters, `output_proj` included, nest under
   `parameters["deformable_attention"]` instead of being flattened next to the SCA's. The line-315
   guard goes away with the collision it was working around. A two-level
   `_convert_sca_parameters_to_object` keeps the nesting through both the fresh and the cached path.
2. `TTSpatialCrossAttention.__init__` passes `params.deformable_attention` to
   `TTMSDeformableAttention` rather than the whole SCA namespace.

| measurement | before | after |
|---|---|---|
| L3d module inner vs reference inner (tiny/rig) | 0.978711 | **0.999650** |
| L3d same, with SCA `output_proj` substituted | 0.999646 | 0.978729 |
| L3 SCA tiny/random · tiny/rig | 0.996641 · 0.995474 | **0.999983 · 0.999909** |
| L3 SCA base/random · base/rig | 0.999327 · 0.998743 | **0.999988 · 0.999952** |
| L3 row rel-err p50 (tiny/rig) | 0.09507 | **0.01274** |
| L5 layer tiny/rig · base/rig | 0.996338 · 0.998647 | **0.999548 · 0.999719** |
| L6 encoder tiny/rig · carla_tiny/rig, last layer | 0.994639 · 0.994968 | **0.999347 · 0.999500** |

The substitution check inverted exactly as predicted: what used to reproduce the module now
reproduces nothing, and the correct reference now matches it.

**Standing suite after the fix:** `test_encoder.py` 5 passed, all at 0.9993 or better against
0.995–0.997 thresholds (base 0.999498, tiny 0.999298, carla_base 0.999631, carla_tiny 0.999444,
base_fast 0.999687). The other five suites: 25 passed, 1 failed — `test_spatial_cross_attention.py`
at bev 200×200, an OOM on a 2963668992 B DRAM buffer that reproduced byte-for-byte with the fix
stashed. Pre-existing and unrelated; since cleared by
[stage 07](perf_reports/07-sampling-grid-in-row-major.md).

## Why the standing tests missed it

`tests/pcc/test_spatial_cross_attention.py` passes at 0.997–0.999 either way: it uses uniform-random
reference points with 95% of `bev_mask` cleared, an operating point where the wrong projection costs
less than the threshold allows. Only the deterministic rig moved the operating point far enough.

`preprocess_temporal_self_attention_parameters` was checked for the same flatten-and-collide shape and
is clean: neither the reference nor the TT temporal self-attention owns an `output_proj`, so its
nested attention's projection is the only one in the namespace.

## Do not re-baseline

Recorded because it was the standing recommendation before the cause was known: the two failing
thresholds were 1.4e-3 and 3e-5 under the line, and the cause was **a wrong matmul**. Re-baselining
would have made the bug the specification. All five encoder parametrizations now sit 3e-3 above their
thresholds; raising them is defensible, nothing forces it.

## Secondary, and still open

Make TT `point_sampling_3d_to_2d` agree with the reference on `bev_mask`. Device computes `max_len`
2484 where host computes 2472 — a boundary-comparison effect (`depth > eps` and the `0 < x,y < 1`
tests in reduced precision), TT consistently the more permissive side. It makes the spatial-path
tensor shapes **device-dependent**, which affects what the profile harness measures and which any
host-derived `max_len` bound must cover ([DEAD_ENDS 3](perf_reports/DEAD_ENDS.md#3-a-static-bound-on-max_len)).

## Note found in passing

`model_config.num_points_in_pillar` (2 for `tiny`, 4 for `base`) is **dead**: the pillar depth
actually used is `dataset_config.z_cfg["num_points"]`, which is 4 for every preset.
`get_encoder_kwargs()` passes both and `z_cfg` wins. Any reasoning that treats `tiny` as a
2-point-pillar configuration is wrong.

## Reproducing

```bash
pytest models/experimental/bevformer/tests/pcc/test_encoder.py -v
pytest models/experimental/bevformer/tests/pcc/test_layer.py -v
pytest models/experimental/bevformer/tests/pcc/test_spatial_cross_attention.py -v
```

Rebuilding a ladder level needs `lidar2img` from `camera_rig.py:lidar2img_for_dataset` (or a
`torch.randn(num_cams, 4, 4)` draw for `random`), and for the sub-op levels the rebatch that
`reference/spatial_cross_attention.py:150-172` performs. Run each level as its own pytest
invocation — several TT modules in one process exhaust DRAM, which is what the two OOMs in the
original run log were.
