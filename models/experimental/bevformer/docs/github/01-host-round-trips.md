---
title: "[BEVFormer] Remove encoder host round-trips and sync points"
issue_id: "01"
status: todo
state: open
parent: "https://github.com/tenstorrent/tt-metal/issues/55048"
---

## Context

The BEVFormer encoder mixes TTNN execution with host-side Torch and Python work.

The largest known block is in Spatial Cross Attention (SCA), but the cleanup spans the encoder, temporal attention, point sampling, and deformable attention.

Baseline host-gap attribution is noisy because queue-entry stalls are charged to the next device operation. The stronger reference result is that the device-side rebatch prototype reduces same-harness layer wall time by 76%. This is a prototype result, not the current implementation.

## Problem

- [SCA reads masks and large activations to host, performs rebatch/scatter logic in Python, and uploads the results again](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_spatial_cross_attention.py#L152-L321).
- [Frame-invariant SCA planning runs inside every encoder layer](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_encoder.py#L464-L497) even though all six layers share the same `bev_mask`.
- [BEV reference points](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_encoder.py#L155-L158) and [MSDA spatial shapes](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_ms_deformable_attention.py#L282-L284) do not change between layers, but they are recreated and uploaded for every layer or MSDA call, causing redundant host work, device allocations, and data transfers.
- The data-dependent `max_len` readback remains a synchronization point even after large tensor transfers are removed.

## Proposed direction

- Move SCA rebatch and scatter-back to device operations.
- Build the frame-level SCA plan before entering the encoder layer loop.
- Cache tensors whose values are fixed for the module or model configuration.
- Stabilize the data-dependent `max_len` shape. Compute camera visibility from `bev_mask` once per frame and reuse it for query selection, `max_len`, and the number of cameras that contributed to each BEV query.

## Acceptance criteria

- [ ] Child issues define and validate each independent optimization.
- [ ] Existing SCA, layer, and full-encoder tests pass at their configured PCC thresholds, including PCC >= 0.997 for the base encoder configuration.
- [ ] Before/after wall, kernel, and host-gap measurements are reported from the same harness and configuration.
- [ ] Gap claims use repeated measured iterations that separate queue-entry cost from steady-state dispatch.

## References

- [Encoder PCC coverage](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_encoder.py)
