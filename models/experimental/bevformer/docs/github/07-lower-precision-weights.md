---
title: "[BEVFormer] Evaluate lower-precision weights with explicit math fidelity"
issue_id: "07"
status: todo
state: open
parent: "https://github.com/tenstorrent/tt-metal/issues/55048"
---

## Context

BEVFormer currently uses bfloat16 weights and activations. Lower-precision weights can reduce bytes moved, but this is the first proposed optimization that may spend numerical accuracy and must be evaluated separately from correctness-preserving layout work.

The expected benefit is limited because matmul is a small share of prototype layer time and fused MSDA requires bfloat16 inputs. No performance result from the prototype branch should be assumed for the current implementation.

## Problem

- [Model preprocessing defaults every weight and bias to bfloat16](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/model_preprocessing.py#L14-L16).
- [MSDA Linears do not specify a compute-kernel configuration](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L234-L337), so math fidelity depends on dtype-sensitive defaults.
- [Encoder FFN Linears also rely on defaults](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_encoder.py#L259-L267).
- A global dtype change can add typecasts or violate a consumer's bfloat16 contract, hiding movement cost behind conversion cost.

## Proposed direction

- First set every Linear's compute configuration explicitly to the current bfloat16 default and verify that performance and PCC do not change.
- Change weights only to bfloat8_b while keeping activations bfloat16.
- Evaluate tensors individually and keep bfloat16 at fused-op boundaries that require it.
- Report each dtype change independently so one PCC result cannot hide which tensor spent the accuracy budget.

## Acceptance criteria

- [ ] Explicitly pinning current math fidelity produces no measurable performance or PCC change.
- [ ] Each lower-precision weight group is profiled independently and reports its layer-time percentage change.
- [ ] Added typecasts and their device-time share are included in every result.
- [ ] Encoder, MSDA, TSA, SCA, and layer PCC plus absolute/relative error metrics are reported for every change.
- [ ] No existing PCC or error threshold is relaxed to accept a lower-precision configuration.
- [ ] Any accepted configuration preserves bfloat16 at consumers whose contracts require it.

## References

- [Encoder PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_encoder.py)
- [MSDA PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_ms_deformable_attention.py)
