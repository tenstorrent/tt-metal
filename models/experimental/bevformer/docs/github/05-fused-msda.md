---
title: "[BEVFormer] Use fused multi-scale deformable attention"
issue_id: "05"
status: todo
state: open
parent: "https://github.com/tenstorrent/tt-metal/issues/55048"
depends_on:
  - "02 | [BEVFormer] Fold the offset normalizer into Linear weights"
---

## Context

BEVFormer currently implements multi-scale deformable attention as a sequence of tensor rearrangements, `grid_sample` calls, stacking, weighting, and reduction. TTNN already provides an experimental fused operation for the sample-weight-reduce core.

Prototype integration reduces layer kernel time by 28.1% by deleting the stack, multiply, and reduction tail, even though the fused sampling operation itself is slower than the `grid_sample` work it replaces. The percentage is a reference result from the prototype branch, not the current implementation.

## Problem

- [Each feature level performs multiple layout conversions, permutes, reshapes, and one `grid_sample`](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L36-L108).
- [Outputs are stacked and then multiplied and reduced in separate kernels](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L109-L129).
- Baseline profiling places most layer device time in the SCA MSDA region, with concat, reshape, and grid sampling among the largest operations.

## Proposed direction

- Validate the [fused op contract](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/multi_scale_deformable_attn_device_operation.cpp#L11-L79) at BEVFormer TSA and SCA shapes.
- Use one fused call for single-level TSA and one call per feature level for multi-level SCA.
- Accumulate per-level outputs on device; attention weights are jointly normalized but their weighted sums can be accumulated per level.
- Follow the [existing VADv2 integration](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/vadv2/tt/tt_utils.py#L45-L107) where applicable, while preserving ROW_MAJOR, INTERLEAVED, dtype, alignment, and `align_corners=False` requirements.

## Acceptance criteria

- [ ] The decomposed `grid_sample` plus stack/multiply/reduce core is replaced by the fused operation.
- [ ] TSA and multi-level SCA paths are covered, including shapes near the fused op's useful-size threshold.
- [ ] Existing MSDA, TSA, SCA, layer, and encoder PCC suites pass at their configured thresholds, including PCC >= 0.997 for the base encoder.
- [ ] Same-configuration profiling reports total kernel time, operation count, and per-region changes; regressions in the fused kernel itself are separated from removed tail operations.
- [ ] Follow-up issues separately own fused-op device time, remaining layout preparation, and multi-level support.

## References

- [MSDA PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_ms_deformable_attention.py)
- [Prototype implementation](https://github.com/tenstorrent/tt-metal/commit/2bc69b90e3c1472e29428f40542bac47847cbf36)
