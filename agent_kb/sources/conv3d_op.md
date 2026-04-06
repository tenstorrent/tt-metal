---
title: Conv3D Operation
type: source
status: active
confidence: high
last_reviewed: 2026-04-06
tags:
  - conv3d
  - source
  - ttnn
  - experimental
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
  - ../recipes/conv3d_codegen.md
  - ../pitfalls/conv3d_gotchas.md
---

# Conv3D Operation

## Purpose

This page summarizes how the current `ttnn.experimental.conv3d` op is structured and where agents should look before changing it.

Related pages:

- [Conv3D Codegen](../recipes/conv3d_codegen.md)
- [Conv3D Gotchas](../pitfalls/conv3d_gotchas.md)

## Entry Points

- API wrapper: `ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d.cpp`
- Weight preprocessing: `ttnn/cpp/ttnn/operations/experimental/conv3d/prepare_conv3d_weights.cpp`
- Device/program setup: `ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`
- Kernels:
  - reader: `.../device/kernels/reader_vol2col.cpp`
  - compute: `.../device/kernels/compute.cpp`
  - writer: `.../device/kernels/writer.cpp`
- Functional tests:
  - `tests/ttnn/unit_tests/operations/conv/test_conv3d.py`
  - `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py`

## Current Dataflow Shape

The op is implemented as:

1. reader performs `vol2col` on a `T_out_block x H_out_block x W_out_block` spatial block
2. compute tilizes the row-major patch buffer
3. compute runs matmul against preprocessed weights
4. compute reduces partial sums across `C_in` workers when needed
5. reducer applies bias after reduction
6. compute untilizes the result
7. writer writes row-major output to the destination tensor

The host-side comment in the program factory describes this shape explicitly and is worth treating as the top-level contract.

## Weight Handling

Documented from code:

- rank-5 weights are treated as unprepared and must be on host
- rank-2 weights are treated as already prepared
- non-tile layout weights are converted to tile layout by the wrapper
- grouped conv rewrites weights into a grouped layout before device upload
- if grouped weights are already on device, the op warns and moves them back to host for preparation

Prepared-weight pipeline:

1. optional grouped-layout rewrite
2. upload to device
3. permute to move kernel spatial dims ahead of channels
4. align `C_in`
5. reshape by `C_in_block`
6. flatten to a 2D matrix for matmul

## Important Config Knobs

From the wrapper and program factory:

- `C_in_block`
  - `0` means full `C_in`
  - otherwise controls the inner reduction blocking
- `C_out_block`
  - `0` means use full padded `C_out`
- `T_out_block`, `H_out_block`, `W_out_block`
  - spatial blocking parameters
- `dilation`
  - affects output dims and the reader path
- `compute_with_storage_grid_size`
  - drives how much parallelism the program factory can assign

Default config is conservative:

- output layout row-major
- block sizes `(1, 1, 1)`
- `C_out_block = 32`
- `C_in_block = 0`

## Parallelization Model

The program factory parallelizes outermost across `C_in`, then across output blocks.

Important current rule:

- each core must handle exactly one `C_in` block

This is enforced in the program factory because the current writer overwrites output for each `C_in` block and bias is added per block. The code explicitly rejects `c_in_per_core > 1`.

Reduction model:

- cores processing the same output block but different `C_in` blocks form a reduction group
- the `c_in_idx == 0` core is the reducer
- workers publish partials through CBs and semaphore handshakes
- reducer collects worker partials, reduces them into its own partial buffer, adds bias, untilizes, and writes output

## Reader Behavior

The reader has two modes:

- direct DRAM-to-vol2col assembly
- two-phase L1 prefetch for spatially reusable kernels with no dilation

The L1 prefetch path is enabled only when:

- kernel has spatial reuse (`kT > 1 || kH > 1 || kW > 1`)
- dilation is all ones
- the computed shard fits within the L1 budget left after CB allocation

The reader also explicitly zero-fills tile-alignment padding so non-tile-aligned patch widths do not contaminate matmul input.

## Compute Behavior

The compute kernel does four distinct things:

1. tilize the row-major patch buffer
2. run blocked matmul
3. optionally reduce partials across workers
4. optionally add bias and then untilize

When `fp32_dest_acc_en` is enabled and there are multiple `C_in` blocks, the program factory switches partial accumulation buffers to fp32. The compute kernel then reconfigures packer/unpacker formats around tilize, matmul, reduction, and untilize to avoid losing precision.

Bias is applied only on the reducer and only after reduction.

## Writer Behavior

The writer:

- reads weights and bias tiles for the assigned `C_out` block
- acts as the synchronization point for worker partial reductions
- writes the final reducer-owned row-major output block to the output tensor

For worker cores, the writer is mostly a reduction handshake engine rather than an output writer.

## Test-Derived Constraints

The current tests encode several important facts:

- watcher-related failures exist for some conv3d tests
- Blackhole currently skips a `C_in` blocking path because the reduction path is incorrect there
- non tile-aligned patch sizes have a regression test
- dilation has explicit test coverage
- sharded input and sharded output layouts are tested in nightly coverage
- interleaved vs sharded numerical equivalence is tested
- real model-shaped cases such as Qwen and Mochi drive non-trivial blocking choices

## What To Read Before Editing

Minimum read set for any `conv3d` change:

1. `conv3d.cpp`
2. `prepare_conv3d_weights.cpp`
3. `conv3d_program_factory.cpp`
4. the kernel file you plan to change
5. `tests/ttnn/unit_tests/operations/conv/test_conv3d.py`

If the change touches memory layout, dilation, or unusual blocking, also read the nightly test file.

## Sources

- `ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/prepare_conv3d_weights.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp`
- `tests/ttnn/unit_tests/operations/conv/test_conv3d.py`
- `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py`
