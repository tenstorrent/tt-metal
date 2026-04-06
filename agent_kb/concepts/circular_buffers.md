---
title: Circular Buffers
type: concept
status: seed
confidence: high
last_reviewed: 2026-04-06
tags:
  - circular-buffer
  - synchronization
  - kernel
source_files:
  - docs/source/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.rst
  - METALIUM_GUIDE.md
  - docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst
---

# Circular Buffers

## Core Model

Circular buffers are single-producer, single-consumer queues used for communication between Tensix threads and kernels.

For code generation, treat them as protocol objects, not just storage.

Related pages:

- [Kernel Pipeline](../concepts/kernel_pipeline.md)
- [Kernel Codegen Gotchas](../pitfalls/kernel_codegen_gotchas.md)

## Producer Protocol

The producer side should follow this shape:

1. `cb_reserve_back(cb, n)`
2. obtain write pointer if needed
3. fill the reserved space
4. `cb_push_back(cb, n)`

Do not push before the data is actually ready.

## Consumer Protocol

The consumer side should follow this shape:

1. `cb_wait_front(cb, n)`
2. obtain read pointer or access via compute API
3. consume the data
4. `cb_pop_front(cb, n)`

Do not pop before all dependent work is complete.

## Interaction With Compute Kernels

- `cb_wait_front` must happen before unpacking or math that needs the tile.
- `cb_reserve_back` must happen before packing into an output CB.
- In the common optimized ordering, input pops and output reserves are placed between `tile_regs_commit()` and `tile_regs_wait()` to overlap communication with the packer.

## Naming Convention (MANDATORY)

**All circular buffers must use semantic names that describe their purpose.** Never use raw numeric names like `cb_0`, `cb_1`, `cb_in0`, `cb_out0`, or `cb_intermed0`.

The numeric CB index is an implementation detail — it goes in the assignment, not the variable name. The variable/constant name must convey what data the CB holds and why it exists.

**Good names** — describe the data or role:
```
cb_input_tiles, cb_output_tiles, cb_scaler, cb_tilized, cb_reduced,
cb_partial_sum, cb_mean, cb_variance, cb_gamma, cb_rm_in, cb_rm_out,
CB_INPUT, CB_SCALER, CB_OUTPUT, CB_TILIZED
```

**Bad names** — describe the index:
```
cb_0, cb_1, cb_in0, cb_out0, cb_intermed0, c_in0, c_out0,
CB_IN, CB_OUT (too generic — what input? what output?)
```

This rule applies everywhere:
- **Design documents** (architecture.md, op_design.md): CB tables and all CB references
- **Program descriptors** (Python): variable names for CB indices
- **Kernel code** (C++): `constexpr uint32_t` CB variable names
- **Agent handoffs**: any text mentioning a CB must use its semantic name
- **Journal entries**: `cb_refined` entries must use semantic names

The index range convention (0-7 inputs, 16-23 outputs, 24-31 intermediates) still applies for choosing which numeric index to assign, but the variable holding that index must be semantically named.

## Agent-Specific Guidance

- Preserve page counts exactly unless you have checked the kernel's tile accounting.
- Be suspicious of changing CB IDs, because they are frequently coupled to host-side setup and compute-kernel init calls.
- When adapting an example, update host-side CB configuration and device-side CB usage together.

## Sources

- `docs/source/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.rst`
- `METALIUM_GUIDE.md`
- `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst`
