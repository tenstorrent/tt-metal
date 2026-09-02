---
title: "[BEVFormer] Fix SCA output projection and replace random lidar2img test geometry"
issue_id: "00"
status: todo
state: open
issue: "55187"
url: "https://github.com/tenstorrent/tt-metal/issues/55187"
parent: "https://github.com/tenstorrent/tt-metal/issues/55048"
blocks:
  - "01 | [BEVFormer] Remove encoder host round-trips and sync points"
  - "02 | [BEVFormer] Fold the offset normalizer into Linear weights"
  - "03 | [BEVFormer] Classify and attribute encoder data-movement cost"
  - "05 | [BEVFormer] Use fused multi-scale deformable attention"
  - "06 | [BEVFormer] Trace-capture the encoder"
  - "07 | [BEVFormer] Evaluate lower-precision weights with explicit math fidelity"
---

## Context

Spatial Cross Attention (SCA) wraps a nested Multi-Scale Deformable Attention (MSDA) module. Both modules own a separate `output_proj` and the reference implementation applies the MSDA projection first, followed by the SCA projection.

Encoder tests currently use random `lidar2img` matrices. Arbitrary 4x4 matrices do not describe a physically valid camera and produce unrealistic sampling patterns: most valid sampling coordinates fall near the image border, where zero padding can hide an incorrect projection. A deterministic camera rig moves most samples into the interpolated image region and exposes encoder PCC failures.

Replacing random `lidar2img` generation is therefore a prerequisite for trusting any encoder PCC result, not a convenience for this fix alone. Every downstream test that depends on projection geometry inherits the same requirement.

## Problem

- [SCA parameter preprocessing stores the outer](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/model_preprocessing.py#L295-L318) `output_proj` [at the top level, flattens nested MSDA parameters into the same namespace, and skips the nested](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/model_preprocessing.py#L295-L318) `output_proj`.
- [SCA passes that flat parameter namespace to the inner MSDA module](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_spatial_cross_attention.py#L83).
- [MSDA applies](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L328-L330) `self.params.output_proj`, which resolves to the SCA projection, and [SCA applies the same projection again](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_spatial_cross_attention.py#L329-L331).
- The TT path therefore applies `SCA.output_proj` twice and never applies `MSDA.output_proj`, unlike the [reference MSDA](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/reference/ms_deformable_attention.py#L258) and [reference SCA](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/reference/spatial_cross_attention.py#L175-L204).



## Proposed direction

- Replace random `lidar2img` [test inputs](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_encoder.py#L46-L69) with deterministic camera-rig geometry that exercises real interpolation. Remove the random-matrix path rather than keeping it as an alternative; leaving it available lets unrealistic sampling patterns hide correctness problems in any test that reuses it.
- Store all nested MSDA parameters, including its `output_proj`, under `deformable_attention`.
- Preserve the nested namespace when converting both freshly preprocessed and cached parameters to objects.
- Pass only `params.deformable_attention` to the inner MSDA module.
- Keep existing PCC thresholds; lowering them would make the incorrect projection part of the expected behavior.



## Acceptance criteria

- [ ] Nested MSDA and outer SCA `output_proj` weights are both loaded and remain independently addressable.
- [ ] The inner MSDA applies its own projection, followed by the outer SCA projection in the same order as the reference.
- [ ] Deterministic camera-rig encoder tests pass without lowering thresholds: `nuscenes_tiny` PCC >= 0.996, `carla_tiny` PCC >= 0.995, and base configurations at their existing thresholds.
- [ ] Standalone SCA, MSDA, point-sampling, temporal-attention, layer, and encoder PCC tests pass at their configured thresholds.
- [ ] Fresh and cached parameter-preprocessing paths produce the same nested parameter structure.
- [ ] No existing PCC threshold is reduced or re-baselined.
- [ ] No encoder test constructs `lidar2img` from arbitrary random 4x4 matrices; every projection matrix comes from a valid camera rig.



## References

- [Deterministic camera-rig prototype](https://github.com/tenstorrent/tt-metal/commit/752990330b96edf4e0ca0c2125592d932f1998cb)
- [Parameter-plumbing fix prototype](https://github.com/tenstorrent/tt-metal/commit/d897012f5cc8b276f6afa6f55ad22818d65764c8)
- [Layer, encoder, and diagnostic test prototype](https://github.com/tenstorrent/tt-metal/commit/a213d3c1135bc66baff97bc57f2712e863c83ef6)
- [Encoder PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_encoder.py)
- [SCA PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_spatial_cross_attention.py)
- [MSDA PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_ms_deformable_attention.py)
