---
description: 'PR review rules for TTNN operation implementations'
applyTo: 'ttnn/cpp/ttnn/operations/**'
excludeAgent: "cloud-agent"
---

# TTNN Operations Review

## 🔴 CRITICAL

- **Program cache correctness**: any op using optional outputs, aliased buffers, or dynamic runtime args must correctly handle program-cache hits. Stale buffer pointers from a previous invocation will silently produce garbage. Verify `override_runtime_arguments` re-patches all non-Buffer runtime args (use `DynamicRuntimeArg` for values that change between calls).
- **Tracing compatibility**: program factories must only create programs — no host-side tensor allocations, vector fills, or buffer operations inside `create_program`. The host should pass tensors through; any host work breaks metal tracing.
- **Device 2.0 API migration**: do not revert device-side kernel code back to deprecated APIs, and do not reverse completed migrations. The migration moves toward the new `tile_regs` API and the new CircularBuffer/DataFlowBuffer, semaphore, and NOC objects (instead of the free functions `cb_wait_front`, `noc_async_read`, etc.). New and migrated kernels should use `tile_regs` (`tile_regs_acquire`, `tile_regs_commit`, `tile_regs_wait`, `tile_regs_release`) and these objects rather than the deprecated `dst` register calls (`acquire_dst`/`release_dst`) or the free-function NOC/semaphore/CB APIs. See the device 2.0 migration tracker.
- **Kernel API call ordering**: on the compute side, `tile_regs` calls, the CircularBuffer/DataFlowBuffer calls, inits, and reconfigs must be ordered correctly. On the data-movement side, the NOC, semaphore, and CircularBuffer/DataFlowBuffer calls must be ordered correctly. The ordering should also be clear and easy to follow.

## 🟡 IMPORTANT

- **Reshape volume invariant**: before and after a reshape, `logical_shape.volume()` must equal `input_tensor.logical_volume()`. If a dimension is inferred (-1), verify the arithmetic produces exact division. Do not confuse `padded_shape()` with `logical_shape()` — padding is layout-specific and must be applied after the logical reshape.
- **Layout alignment constraints**: for `Layout::ROW_MAJOR`, the last dimension must be divisible by 8 (ROW_MAJOR_WIDTH). For `Layout::TILE`, height and width must be divisible by 32 (TILE_HEIGHT/TILE_WIDTH), and physical volume must be divisible by 1024 (TILE_HW).
- **CCL ring buffer consistency**: in CCL operations, `num_buffers_per_channel` must match between host-side `EriscDatamoverBuilder` construction and device kernel compile-time args. Buffer address and semaphore address vectors must have identical sizes. `eth_buffer_size_bytes` must be recomputed if `num_edm_channels`, `num_buffers_per_channel`, or `page_size` change after initial computation.
- **Composite op overhead**: be wary of composite ops that introduce intermediate operations (unsqueeze, reshape, transpose), since the per-op overhead can make the composition slower than alternatives (e.g., a matmul path). When perf is a concern, the trade-off should be backed by device-time data for representative shapes. Dedicated perf coverage for such ops should be driven by model requirements and owned by the relevant model/perf pipeline — not added ad hoc to a feature PR or to sanity.
- **Input validation**: ops must validate dtype, layout, storage type, device affinity, and shape constraints before launching kernels. Don't rely on kernel-level crashes for validation. Use `TT_FATAL` with clear messages.
- **Init/startup placement**: `compute_kernel_hw_startup` must be at the very start of the kernel, and it must be in `main()` rather than inside a helper function. Short `_init` calls, by contrast, belong in the middle of the kernel next to the ops they configure — not hoisted to the top. Verify that every init's placement is correct for what it configures; misplaced inits cause subtle data format corruption.
- **Experimental includes isolation**: experimental or internal headers must not be included in common/shared headers. Include them directly in the kernel file that needs them.
- **No framework references in API docs**: do not reference PyTorch, TensorFlow, or other framework semantics in TTNN API documentation or nanobind docstrings. Document TTNN behavior independently.
- **Descriptor-style pattern**: new ops should follow the descriptor pattern (separate `DeviceOperation`, `ProgramFactory`, types header). Do not require migrating pre-existing ops as part of an unrelated PR — that breaks the "a PR should do one thing" principle. Instead, file an issue for the owning team to drive the migration; if a migration is the explicit purpose of the PR, preserve functional equivalence and add cache-hit tests.

## 🟢 SUGGESTION

- Where a ttnn unit test already covers the same functionality as a legacy eager test, note the redundancy. Prefer filing an issue for the owning team to remove the legacy test rather than bundling unrelated test cleanup into a feature PR.
- For ops that claim program-cache reuse, add an explicit cache-hit test that calls the op twice with different parameters and verifies `override_runtime_arguments` correctness.
- When `override_runtime_arguments` has nothing to update, it should be a no-op (early return), not omitted entirely.
- Prefer kernel-level tile operations over host-side data manipulation. If the host is generating data into buffers (e.g., twiddle factors for FFT), document why this is necessary and whether it breaks tracing.
- Identical code paths (e.g., duplicated if/else branches) should be collapsed with a comment explaining the shared logic.

## Review Checklist

- [ ] Program cache: `override_runtime_arguments` patches all dynamic values (buffers, scalars, shapes)
- [ ] No host-side allocations or buffer operations inside `create_program`
- [ ] No regression to deprecated APIs (`acquire_dst`/`release_dst`, free-function noc/semaphore/CB calls)
- [ ] Kernel API calls (tile_regs, CB/DFB, NOC, semaphore, inits, reconfigs) ordered correctly and clearly
- [ ] Input validation before kernel launch (dtype, layout, device, shape)
- [ ] Reshape ops: logical volume preserved, alignment constraints met for target layout
- [ ] CCL ops: buffer/semaphore counts consistent between host setup and kernel args
- [ ] `compute_kernel_hw_startup` at the very start and in `main()`; short `_init` calls placed correctly next to the ops they configure
- [ ] Composite ops: intermediate-op overhead considered; perf trade-offs backed by device-time data when perf is a concern
- [ ] Experimental/internal headers not leaked into common includes
- [ ] New ops follow descriptor-style pattern
- [ ] Cache-hit test present for ops with dynamic runtime args
- [ ] No external framework references in API documentation
