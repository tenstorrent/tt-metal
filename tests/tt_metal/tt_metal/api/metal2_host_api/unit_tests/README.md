# Metal 2.0 Host API unit tests

Host-only tests against a mock device. A test in this folder must:

- run without silicon or an emulator, and have a name starting with `CPU_`;
- be fast;
- use only the default kernel source. A test that needs its own kernel source goes in `kernel_compilation_tests/`,
  which keeps JIT compilation out of this suite.

| Directory | What it tests |
|---|---|
| `invariant_tests/` | Which specs `MakeProgramFromSpec` accepts or rejects, one directory per public header |
| `kernel_hash/` | Which spec fields feed a kernel's JIT cache key (`compute_hash`) |
| `program/` | What `MakeProgramFromSpec` produces from a valid spec |
| `program_run_args/` | `SetProgramRunArgs`, `UpdateProgramRunArgs`, `UpdateTensorArgs`, `MergeProgramRunArgs` |
| `tensor_spec_relaxations/` | The `TensorSpecRelaxations` match and hash relation |
| `utility/` | `Table<K, V>` |
| `spec_type_properties.cpp` | Every spec struct stays an aggregate (designated initializers work) and ttsl-hashable |
