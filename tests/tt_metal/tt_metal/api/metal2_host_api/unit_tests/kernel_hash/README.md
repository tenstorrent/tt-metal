# Kernel hash tests

Kernels whose `compute_hash()` is equal share one JIT-cached binary. A spec change that alters the generated code or
compile-time arguments must change the hash, or a stale binary runs; a change supplied only at run time must not.
Each test builds Programs from specs that differ in one field and compares the kernels' hashes, with one file per
struct whose field is varied.

Fixtures: `ProgramSpecTestQuasar` (Q) and `ProgramSpecTestGen1` (WH).

| File | Struct | Tests | What is pinned |
|---|---|---|---|
| `kernel_spec.cpp` | `KernelSpec` | 1 | Binding a scratchpad changes the hash |
| `dataflow_buffer_spec.cpp` | `DataflowBufferSpec` | 1 | `tile_format_metadata` changes the bound compute kernel's hash |
| `scratchpad_spec.cpp` | `ScratchpadSpec` | 2 | `size_per_node` and `data_format_metadata` change the binding kernel's hash |
| `kernel_advanced_options.cpp` | `KernelAdvancedOptions` | 4 | How tensor bindings split into binding sequences, and compile-time vararg values and count, change the hash; identical varargs hash equal |
| `tensor_parameter.cpp` | `TensorParameter` | 2 | A different `TensorSpec` changes the hash; an identical one does not |
| `tensor_spec_relaxations.cpp` | `TensorSpecRelaxations` | 3 | With `dynamic_tensor_shape`, the hash is stable across shapes (interleaved tile, interleaved row-major, sharded) |
