---
title: "[BEVFormer] Classify and attribute encoder data-movement cost"
issue_id: "03"
status: todo
state: open
parent: "https://github.com/tenstorrent/tt-metal/issues/55048"
depends_on:
  - "05 | [BEVFormer] Use fused multi-scale deformable attention"
---

## Context

BEVFormer deformable attention spends substantial device time rearranging tensors between producer and consumer layouts. Optimizing isolated operation names is risky: deleting one permute can simply introduce an untilize or reshape elsewhere.

A baseline capture of the current implementation attributes 67.7% of device time to data movement and 32.3% to compute. Host and pipeline gaps are separate from device data movement and must not be included in that ratio.

Prototype profiling after fused MSDA shows that the child changes can reduce the post-fusion layer kernel time by 38.6%, lower the data-movement-to-compute ratio from 1.00 to 0.51, and preserve PCC. These measurements are reference results from the prototype branch, not the current implementation.

## Problem

- [The MSDA path contains repeated reshape, permute, layout-conversion, stack, multiply, and reduction operations](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L70-L124).
- [SCA permutes both `key` and `value` into camera-major batches](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_spatial_cross_attention.py#L251-L255), although deformable attention consumes only `value`.
- [Sampling-location arithmetic runs on heavily padded tiled shapes before grid sampling](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L282-L307).
- The profile does not state which consumer contract required each transformation.
- Without producer-to-consumer attribution, it is unclear whether storage reordering, L1 placement, fusion, or an op-contract change is the right fix.

## Proposed direction

- Reproduce the baseline and classify every operation as compute, data movement, or mixed using documented rules.
- Map each movement operation to its producer tensor, consuming operation, layout contract, and measured device time.
- Separate required transformations from accidental ones that can be removed by changing storage order or fusion boundaries.
- Validate the independent child changes for dead work, sampling-grid arithmetic, attention preparation, head-axis movement, and layer-invariant camera layout.
- Reserve NoC analysis for the remaining value-path permute and untilize work tracked by ticket `05.02`.

## Acceptance criteria

- [ ] The classification covers all profiled operations and reconciles to total device time within rounding.
- [ ] Host/pipeline gaps are reported separately from device movement and compute.
- [ ] Every movement operation contributing at least 1% of device time has a named producer, consumer, source location, and required/avoidable decision.
- [ ] Each independent optimization is measured against the immediately preceding configuration and reports its percentage change.
- [ ] The final profile reconciles the movement and compute totals and reports the resulting ratio.
- [ ] Any production code retained during the investigation passes existing layer and encoder PCC thresholds, including PCC >= 0.997 for the base encoder.

## References

- [Baseline commit](https://github.com/tenstorrent/tt-metal/commit/d897012f5cc8b276f6afa6f55ad22818d65764c8)
- [Prototype: delete unused SCA key permute](https://github.com/tenstorrent/tt-metal/commit/a933e1059d96ffb98c68f78d6965a6e92d9b35f4)
- [Prototype: build the sampling grid in ROW_MAJOR](https://github.com/tenstorrent/tt-metal/commit/a32ddae6c62363ba9ad45844a8d08e8655d564b4)
- [Prototype: prepare attention once per call](https://github.com/tenstorrent/tt-metal/commit/c649b46fee237ebc5b17e838b80354aa2c337144)
- [Prototype: build a head-major sampling grid](https://github.com/tenstorrent/tt-metal/commit/e6b4ce53fe163df02964a3655d55222a0a3ed5e0)
- [Prototype: split value heads without tile padding](https://github.com/tenstorrent/tt-metal/commit/7820f325bd86cbca8dfcbe27cf460204c8d9773c)
- [Encoder PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_encoder.py)
