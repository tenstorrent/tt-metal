---
description: 'PR review rules for TTNN operation implementations'
applyTo: 'ttnn/cpp/ttnn/operations/**'
excludeAgent: "cloud-agent"
---

# TTNN Operations Review

## 🔴 CRITICAL

- **Program cache correctness**: any op using optional outputs, aliased buffers, or dynamic runtime args must correctly handle program-cache hits. Stale buffer pointers from a previous invocation will silently produce garbage. Verify `override_runtime_arguments` re-patches all non-Buffer runtime args (use `DynamicRuntimeArg` for values that change between calls).
- **Tracing compatibility**: program factories must only create programs — no host-side tensor allocations, vector fills, or buffer operations inside `create_program`. The host should pass tensors through; any host work breaks metal tracing.
- **Don't regress Device 2.0 API migration**: do not revert device-side kernel code back to deprecated APIs (e.g., `acquire_dst`/`release_dst` → use `tile_regs_*` equivalents). Do not reverse completed migrations.
- **Deprecated API usage**: `dst` register calls are deprecated. Use the `tile_regs` API (`tile_regs_acquire`, `tile_regs_commit`, `tile_regs_wait`, `tile_regs_release`). See the device 2.0 migration tracker.

## 🟡 IMPORTANT

- **Reshape volume invariant**: before and after a reshape, `logical_shape.volume()` must equal `input_tensor.logical_volume()`. If a dimension is inferred (-1), verify the arithmetic produces exact division. Do not confuse `padded_shape()` with `logical_shape()` — padding is layout-specific and must be applied after the logical reshape.
- **Layout alignment constraints**: for `Layout::ROW_MAJOR`, the last dimension must be divisible by 8 (ROW_MAJOR_WIDTH). For `Layout::TILE`, height and width must be divisible by 32 (TILE_HEIGHT/TILE_WIDTH), and physical volume must be divisible by 1024 (TILE_HW).
- **CCL ring buffer consistency**: in CCL operations, `num_buffers_per_channel` must match between host-side `EriscDatamoverBuilder` construction and device kernel compile-time args. Buffer address and semaphore address vectors must have identical sizes. `eth_buffer_size_bytes` must be recomputed if `num_edm_channels`, `num_buffers_per_channel`, or `page_size` change after initial computation.
- **Composite op overhead**: composite ops that introduce intermediate operations (unsqueeze, reshape, transpose) must include a device-time performance measurement in the PR for representative shapes. Break out per-op overhead to show the composition isn't slower than alternatives (e.g., matmul path).
- **Input validation**: ops must validate dtype, layout, storage type, device affinity, and shape constraints before launching kernels. Don't rely on kernel-level crashes for validation. Use `TT_FATAL` with clear messages.
- **`override_runtime_arguments` completeness**: if an op has runtime args that change between invocations (buffer addresses, dynamic shapes, scalar params), `override_runtime_arguments` must update ALL of them. A missing update means the cached program runs with stale values.
- **Init calls placement**: `_init` calls in compute kernels must be at the top of the kernel, not scattered in the middle of compute loops. Misplaced inits cause subtle data format corruption.
- **Experimental includes isolation**: experimental or internal headers must not be included in common/shared headers. Include them directly in the kernel file that needs them.
- **No framework references in API docs**: do not reference PyTorch, TensorFlow, or other framework semantics in TTNN API documentation or nanobind docstrings. Document TTNN behavior independently.
- **Descriptor-style migration**: new ops should follow the descriptor pattern (separate `DeviceOperation`, `ProgramFactory`, types header). When migrating existing ops, preserve functional equivalence and add cache-hit tests.

## 🟢 SUGGESTION

- Remove tests from the legacy eager test suite where a ttnn unit test already covers the same functionality.
- For ops that claim program-cache reuse, add an explicit cache-hit test that calls the op twice with different parameters and verifies `override_runtime_arguments` correctness.
- When `override_runtime_arguments` has nothing to update, it should be a no-op (early return), not omitted entirely.
- Prefer kernel-level tile operations over host-side data manipulation. If the host is generating data into buffers (e.g., twiddle factors for FFT), document why this is necessary and whether it breaks tracing.
- Identical code paths (e.g., duplicated if/else branches) should be collapsed with a comment explaining the shared logic.

## Review Checklist

- [ ] Program cache: `override_runtime_arguments` patches all dynamic values (buffers, scalars, shapes)
- [ ] No host-side allocations or buffer operations inside `create_program`
- [ ] No regression to deprecated APIs (`acquire_dst`/`release_dst`, legacy noc calls)
- [ ] Input validation before kernel launch (dtype, layout, device, shape)
- [ ] Reshape ops: logical volume preserved, alignment constraints met for target layout
- [ ] CCL ops: buffer/semaphore counts consistent between host setup and kernel args
- [ ] Compute kernel `_init` calls at top, not mid-loop
- [ ] Composite ops include device-time perf measurement for representative shapes
- [ ] Experimental/internal headers not leaked into common includes
- [ ] New ops follow descriptor-style pattern
- [ ] Cache-hit test present for ops with dynamic runtime args
- [ ] No external framework references in API documentation
