---
name: tt-lang
description: "Write, port, debug, and validate TT-Lang DSL kernels for Tenstorrent hardware. Use when Codex needs TT-Lang programming model, APIs, dataflow buffers, grid/node logic, pipes, tensor slicing, hardware constraints, or translations from CUDA, Triton, PyTorch, or TTNN kernels."
---

# TT-Lang

Use this skill when implementing TT-Lang kernels or translating existing CUDA, Triton, PyTorch, or TTNN logic into TT-Lang. Treat TT-Lang as a low-level DSL over Tensix nodes: reason about tile movement, dataflow buffers, L1 limits, synchronization, and output layout before writing compute expressions.

## Reference Loading

Load only the files needed for the current task:

- [guide.md](guide.md): practical kernel workflow, translation guidance, common patterns, testing workflow, and debugging checklist.
- [TTLangSpecification.md](TTLangSpecification.md): source of truth for language APIs, including kernels, grids, `ttl.node`, dataflow buffers, blocks, pipes, tensor slices, copy, semaphores, signposts, debug printing, and math functions.
- [examples.md](examples.md): longer worked kernels for pipes, attention, streaming, reductions, and multicore patterns.
- [extern_references.md](extern_references.md): external TT-Lang model projects to inspect when a task needs real model-level examples.

Prefer the bundled specification over memory when API details matter. Use official online TT-Lang docs only when the user asks for current external documentation or when local references are insufficient.

## Working Pattern

1. Identify the target tensor shapes, layouts, dtypes, memory placement, and correctness reference.
2. Decompose the operation into data movement plus tile-level compute. Missing PyTorch or TTNN operations are usually a data movement problem plus primitive math, not a reason to stop.
3. Design the dataflow graph before coding: one producer and one consumer per dataflow buffer, matching loop counts across producer and consumer threads, and explicit output writes.
4. Start with the smallest kernel that proves the operation, usually single-node. Scale to `grid="auto"` or fixed multicore grids after correctness is clear.
5. Keep tensors tile-aligned and use TTNN for setup, padding, layout conversion, and postprocessing when that keeps the TT-Lang kernel focused.
6. Test every kernel. Read logs and tensor outputs; do not trust exit code alone.
7. If the compiler fails with MLIR verifier/assert/segfault behavior, try a simpler rewrite or split. Do not spend time debugging compiler internals unless the user explicitly asks.

## Kernel Checklist

- Define a `@ttl.kernel` function that takes TTNN tensors and mutates output tensors in place.
- Use `@ttl.compute()` for math on blocks in L1.
- Use `@ttl.datamovement()` functions to copy tensor slices into and out of dataflow buffers.
- Use context managers around `dfb.reserve()` and `dfb.wait()` so block push/pop is handled correctly.
- Use `ttl.node(dims=...)` and `ttl.grid_size(dims=...)` for logical coordinates and grid dimensions.
- Prefer `grid="auto"` with bounds checks for production streaming kernels. Fixed grids are appropriate for pipe topologies or early isolation.
- Keep dataflow buffer block shapes small enough for L1 and stream large tensors through loops.
- Verify every `wait()` has a matching producer path and every `reserve()` has a matching consumer path.

## Validation

For local scripts, include a normal Python `if __name__ == "__main__":` path that opens a device or simulator-compatible device, prepares TTNN tensors, runs the kernel, converts results back with `ttnn.to_torch`, and compares against a PyTorch reference.

Default iteration:

```bash
python /path/to/kernel.py
```

If the environment provides a wrapper such as `run-test.sh`, use it, but still inspect the produced log. For hangs, first check dataflow buffer balance and loop counts. For numerical issues, isolate one operation at a time and print input, output, and expected tensors outside the kernel.
