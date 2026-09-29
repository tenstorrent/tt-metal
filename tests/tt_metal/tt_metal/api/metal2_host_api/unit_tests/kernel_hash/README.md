# Kernel hash tests

A kernel's JIT cache key is its `compute_hash()`. Two kernels that hash equal share one cached binary. So a spec
change that changes the generated kernel code or compile-time arguments must change the hash; otherwise a stale
binary runs with the wrong layout. Conversely, a change that is only supplied at run time must leave the hash
unchanged, so the binary is reused.

Each test builds Programs from specs that differ in one field and compares the resulting kernels' `compute_hash()`.
There is one file per struct whose field is varied.

Fixtures: `ProgramSpecTestQuasar` (Q) and `ProgramSpecTestGen1` (WH).

| File | Struct | Tests | What is pinned |
|---|---|---|---|
| `kernel_spec.cpp` | `KernelSpec` | 1 | Binding a scratchpad changes the hash |
| `dataflow_buffer_spec.cpp` | `DataflowBufferSpec` | 1 | `tile_format_metadata` changes the bound compute kernel's hash |
| `scratchpad_spec.cpp` | `ScratchpadSpec` | 2 | `size_per_node` and `data_format_metadata` change the binding kernel's hash |
| `kernel_advanced_options.cpp` | `KernelAdvancedOptions` | 4 | How tensor bindings are split into binding sequences changes the hash; compile-time vararg values and count change it; identical varargs hash equal |
| `tensor_parameter.cpp` | `TensorParameter` | 2 | A different `TensorSpec` changes the hash; an identical one does not |
| `tensor_spec_relaxations.cpp` | `TensorSpecRelaxations` | 3 | With `dynamic_tensor_shape`, the hash stays stable across shapes (interleaved tile, interleaved row-major, sharded) |
