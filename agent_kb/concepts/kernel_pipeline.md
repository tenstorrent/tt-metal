---
title: Kernel Pipeline
type: concept
status: seed
confidence: high
last_reviewed: 2026-04-06
tags:
  - kernel
  - pipeline
  - reader
  - compute
  - writer
source_files:
  - METALIUM_GUIDE.md
  - docs/source/tt-metalium/get_started/get_started.rst
  - docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst
---

# Kernel Pipeline

## Core Model

For single-core operator development, the default mental model is a three-stage pipeline:

1. Reader kernel moves tiles into circular buffers.
2. Compute kernel waits on those buffers, runs unpack/math/pack work, and writes results to output buffers.
3. Writer kernel drains output buffers to memory.

This is the baseline pattern agents should start from unless there is a clear reason to deviate.

Related pages:

- [Circular Buffers](../concepts/circular_buffers.md)
- [Kernel Codegen Gotchas](../pitfalls/kernel_codegen_gotchas.md)
- [New Kernel Codegen Checklist](../recipes/new_kernel_codegen_checklist.md)

## Invariants

- Data movement and compute are separate concerns. Do not assume a single kernel should do everything.
- Compute-kernel source code is compiled into separate binaries for the unpack, math, and pack RISC-V cores.
- Synchronization is explicit. A working-looking sequence can still be wrong if ownership or ordering is violated.
- Preserve overlap when possible: wait for input, acquire Dst, compute, commit, pop/reserve, wait, pack, release, push.

## Minimal Compute-Loop Shape

This is the canonical loop shape to preserve unless an existing example shows a different safe ordering:

1. `cb_wait_front(...)` on required inputs.
2. `tile_regs_acquire()`.
3. Unpack and/or compute API calls.
4. `tile_regs_commit()`.
5. `cb_pop_front(...)` on consumed inputs.
6. `cb_reserve_back(...)` on produced outputs.
7. `tile_regs_wait()`.
8. `pack_tile(...)` or equivalent output movement.
9. `tile_regs_release()`.
10. `cb_push_back(...)`.

## Why Agents Fail Here

Common bad generations:

- skipping reader/writer separation and inventing ad hoc data movement
- using compute APIs without respecting Dst register ownership
- reserving or pushing circular buffers in the wrong phase
- changing a known-good ordering without understanding why it existed

## Guidance For Codegen

- Start from the closest programming example, not from a blank file.
- Reuse the shape of a working compute loop before changing APIs or tile counts.
- If the operation is numerically sensitive, read the accuracy notes before editing kernels.

## Sources

- `METALIUM_GUIDE.md`
- `docs/source/tt-metalium/get_started/get_started.rst`
- `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst`
