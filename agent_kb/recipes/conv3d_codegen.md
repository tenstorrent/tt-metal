---
title: Conv3D Codegen
type: recipe
status: active
confidence: high
last_reviewed: 2026-04-06
tags:
  - conv3d
  - recipe
  - codegen
source_files:
  - ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d.cpp
  - ttnn/cpp/ttnn/operations/experimental/conv3d/prepare_conv3d_weights.cpp
  - ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp
  - tests/ttnn/unit_tests/operations/conv/test_conv3d.py
  - tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py
related_pages:
  - ../sources/conv3d_op.md
  - ../pitfalls/conv3d_gotchas.md
---

# Conv3D Codegen

## When To Use This Page

Use this page when modifying the existing `conv3d` op or when using it as a template for another patch-extraction or blocked-convolution style op.

Related pages:

- [Conv3D Operation](../sources/conv3d_op.md)
- [Conv3D Gotchas](../pitfalls/conv3d_gotchas.md)

## Safe Edit Order

Read and reason in this order:

1. API wrapper in `conv3d.cpp`
2. weight preparation in `prepare_conv3d_weights.cpp`
3. host orchestration in `conv3d_program_factory.cpp`
4. one or more kernels
5. unit and nightly tests

Do not start from the kernel files alone. Too many constraints are encoded host-side.

## If You Change Weight Semantics

Inspect all of:

- accepted weight ranks
- host vs device ownership expectations
- grouped-conv preparation path
- alignment and `C_in_block` divisibility
- flattening into the matmul-ready 2D form

A weight-shape change is not local to `prepare_conv3d_weights.cpp`.

## If You Change Blocking

Audit together:

- `C_in_block`
- `C_out_block`
- `T_out_block`, `H_out_block`, `W_out_block`
- patch size and padded patch size
- matmul tile counts `M_t`, `K_t`, `N_t`
- CB page sizes and page counts
- L1 prefetch budget logic
- reduction-group formation

The current implementation assumes one `C_in` block per core. If your change violates that, the reduction/write model must change too.

## If You Change Numerical Behavior

Check:

- `fp32_dest_acc_en`
- the `use_fp32_partials` path in the program factory
- format reconfiguration around tilize, matmul, reduction, and untilize
- bias placement after reduction

Do not move bias addition earlier unless you have reworked the reduction logic carefully.

## If You Change Reader Behavior

Decide whether the change affects:

- direct DRAM read path
- L1 prefetch path
- padding semantics for out-of-bounds receptive fields
- tile-alignment padding of patch rows
- TensorAccessor behavior for sharded layouts

For sharded-input changes, verify against the nightly sharded-layout tests, not only the base unit tests.

## If You Change Reduction Or Multi-Core Scheduling

Read the reduction-group logic in the program factory and the worker/reducer handshake in both compute and writer kernels.

Current contract:

- workers compute partials only
- reducer owns final reduction, bias, untilize, and writeback
- semaphores and CB handshakes coordinate reduction

If you violate that ownership split, expect subtle races or silent numerical errors.

## Minimum Validation Matrix

After a non-trivial change, run at least:

1. base unit test for a normal shape
2. non-aligned patch-size regression
3. dilation case if dilation-related code changed
4. grouped-conv case if weight grouping changed
5. sharded-input test if reader/accessor code changed
6. model-shaped case if blocking logic changed materially

## What Durable Knowledge To Add Back

If the edit teaches a durable lesson, update:

- [Conv3D Operation](../sources/conv3d_op.md) for structural facts
- [Conv3D Gotchas](../pitfalls/conv3d_gotchas.md) for failure modes
- a generic KB page only if the lesson is not `conv3d`-specific

## Sources

- `ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/prepare_conv3d_weights.cpp`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`
- `tests/ttnn/unit_tests/operations/conv/test_conv3d.py`
- `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py`
