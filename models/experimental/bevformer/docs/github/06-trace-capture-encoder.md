---
title: "[BEVFormer] Trace-capture the encoder"
issue_id: "06"
status: todo
state: open
issue: "55210"
url: "https://github.com/tenstorrent/tt-metal/issues/55210"
parent: "https://github.com/tenstorrent/tt-metal/issues/55048"
depends_on:
  - "01 | [BEVFormer] Remove encoder host round-trips and sync points"
  - "01.02.01 | [BEVFormer] Stabilize SCA max_len with a high-water mark"
---

## Context

The encoder executes the same sequence of TTNN operations for consecutive frames. TTNN trace capture can record that sequence once and replay it, reducing Python dispatch work between device operations.

Trace replay requires the same operation sequence, tensor shapes, and device buffers used during capture. The SCA rebatch shape depends on `max_len`, so a trace is reusable only while the cached high-water mark remains unchanged.

After applying the prerequisite optimization prototypes, steady-state host gap is approximately 3% of layer wall time. This is an upper bound from the prototype path, not a measurement of the current implementation, and it makes trace capture a low-value optimization unless an encoder-level profile finds additional per-forward host work.

## Problem

- [The encoder dispatches every layer through Python on each forward](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_encoder.py#L476-L497).
- [`max_len` is calculated from the current frame and used to allocate SCA rebatch tensors](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tt/tt_spatial_cross_attention.py#L154-L199), so their shapes can change between frames.
- A trace captured for one rebatch shape must not be replayed after the high-water mark grows.
- The current encoder harness does not provide a clean profile of host work outside the repeated layer body, so the end-to-end opportunity is still unknown.

## Proposed direction

- Reprofile the encoder after ticket #55191 using repeated measured iterations that separate queue-entry cost from steady-state host dispatch.
- Compare total encoder wall time with the repeated layer body and report the percentage that trace replay can potentially recover before implementing capture.
- Preallocate persistent input, intermediate, and output buffers required by the capture.
- Capture the full encoder device-operation sequence after the `max_len` high-water mark and tensor shapes are established.
- On a frame that increases the high-water mark, discard the old trace, rebuild the affected buffers, run the required uncaptured setup, and capture a new trace before further replay.
- Compare eager and traced execution using the same inputs, shapes, device, and measurement window.

## Acceptance criteria

- [ ] A fixed-shape encoder forward can be captured and replayed repeatedly without reallocating traced buffers.
- [ ] Input buffers are refreshed between replays without changing their addresses or shapes.
- [ ] High-water-mark growth invalidates the old trace before it can be replayed with stale shapes.
- [ ] Eager and traced outputs pass existing SCA, layer, and encoder tests at their configured PCC thresholds, including PCC >= 0.997 for the base encoder.
- [ ] Profiling reports the traced-versus-eager wall-time change as a percentage and separates one-time capture cost from replay cost.
- [ ] The result includes the number of replays required to amortize capture and recapture overhead.
- [ ] If encoder-level profiling does not show a recoverable host-dispatch share above measurement noise, the no-win result is documented instead of adding trace complexity.

## References

- [Encoder PCC tests](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/bevformer/tests/pcc/test_encoder.py)
- [RT-DETR trace-capture example](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/vision/detection/rtdetr/tests/latency.py#L85-L97)
