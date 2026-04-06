---
title: Conv3D Gotchas
type: pitfall
status: active
confidence: high
last_reviewed: 2026-04-06
tags:
  - conv3d
  - pitfalls
  - reduction
  - blocking
source_files:
  - ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d.cpp
  - ttnn/cpp/ttnn/operations/experimental/conv3d/prepare_conv3d_weights.cpp
  - ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp
  - ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp
  - ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp
  - ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp
  - tests/ttnn/unit_tests/operations/conv/test_conv3d.py
  - tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py
related_pages:
  - ../sources/conv3d_op.md
  - ../recipes/conv3d_codegen.md
---

# Conv3D Gotchas

## Weight Preparation Assumptions

- Rank-5 weights are treated as unprepared and must start on host.
- Rank-2 weights are treated as already prepared.
- Grouped conv preparation may force a device tensor back to host before rewriting.
- `C_in_aligned` must be divisible by `C_in_block`.

If you change weight shape conventions, update both wrapper and preparation logic.

## One C_in Block Per Core

Current hard constraint from the program factory:

- each core must process exactly one `C_in` block

Why:

- the writer overwrites the same output address for each `C_in` block instead of accumulating
- bias is re-added for each block if this constraint is violated

Any attempt to batch multiple `C_in` blocks per core requires redesign of reduction/write semantics.

## Non Tile-Aligned Patch Sizes

This op has explicit regression coverage for non tile-aligned patch sizes.

Key risk:

- `patch_size = kD * kH * kW * C_in_block` can fail to land on tile width

The implementation zero-fills tile-alignment padding in the reader. Do not remove or weaken that logic casually.

## Reader Path Split

The reader has a direct path and an L1-prefetch path.

Common mistake:

- editing one path and forgetting the other

If your change affects receptive-field assembly, out-of-bounds handling, or indexing, audit both paths.

## FP32 Partial Reduction Is Not A Local Toggle

`use_fp32_partials` is not just a compute-kernel flag.

It changes:

- CB data formats
- tile sizes
- zero-tile usage
- packer/unpacker reconfiguration
- reduction logic
- untilize reconfiguration

If you touch fp32 partial handling, inspect host setup and compute kernel together.

## Reducer Ownership

Bias addition, final untilize, and final writeback belong to the reducer.

If a worker starts doing any of those, the reduction contract is broken.

## Watcher And Architecture Caveats

Current tests document two practical caveats:

- some conv3d tests are skipped with Watcher enabled due to known failures
- Blackhole skips a `C_in` blocking path because reduction is incorrect there

Do not generalize a Wormhole-passing change to Blackhole without checking architecture behavior explicitly.

## Sharded Layouts Are Part Of The Contract

Nightly tests cover:

- interleaved input
- height-sharded input
- width-sharded input
- block-sharded input
- sharded output
- interleaved vs sharded equivalence

So `conv3d` is not only an interleaved-layout op. Reader and accessor changes need to preserve that.

## Sources

- `ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/prepare_conv3d_weights.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp`
- `tests/ttnn/unit_tests/operations/conv/test_conv3d.py`
- `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py`
