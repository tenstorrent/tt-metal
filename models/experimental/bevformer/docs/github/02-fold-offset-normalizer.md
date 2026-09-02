---
title: "[BEVFormer] Fold the offset normalizer into Linear weights"
issue_id: "02"
status: todo
state: open
parent: "https://github.com/tenstorrent/tt-metal/issues/55048"
enables:
  - "03.02 | [BEVFormer] Build the MSDA sampling grid on unpadded rows"
---

## Context

MSDA predicts sampling offsets with a Linear layer and then divides every output coordinate by the fixed width/height of its feature level. The normalizer is configuration-derived, so applying it on every forward repeats static arithmetic.

The prototype removes the runtime divide and reduces post-fusion layer kernel time by approximately 6.7%. A later sampling-grid prototype reuses the same fold for `2/[W,H]`; neither result describes the current implementation until these changes are applied.

## Problem

- [`spatial_shapes` is uploaded and `offset_normalizer` is rebuilt during `forward`](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L282-L294).
- [A tiled `ttnn.div` applies the static normalizer for every MSDA call](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L289-L294).
- The final coordinate axis is small, so tile padding can make this elementwise operation disproportionately expensive.

## Proposed direction

- Build a per-output-channel scale containing `1/W` and `1/H` for every level, point, head, and axis.
- Apply the scale once to `sampling_offsets` weight and bias during [model preprocessing](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/model_preprocessing.py#L133-L145) or initialization.
- Generalize the [existing one-level VADv2 helper](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/vadv2/tt/tt_utils.py#L9-L41) to SCA's multi-level layout.
- Remove the runtime divide while preserving the sampling-coordinate convention.

## Acceptance criteria

- [ ] The runtime offset-normalizer divide is absent from the profile.
- [ ] TSA and multi-level SCA use the correct scale for every feature level and coordinate axis.
- [ ] Existing MSDA, TSA, SCA, layer, and encoder PCC suites pass at their configured thresholds, including PCC >= 0.997 for the base encoder.
- [ ] Same-configuration profiling reports device time, operation count, and cold-/steady-state behavior.

## References

- [MSDA PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_ms_deformable_attention.py)
- [Prototype implementation](https://github.com/tenstorrent/tt-metal/commit/d86d2f722fbecdd210e96c2927a9b9648211ebf5)
