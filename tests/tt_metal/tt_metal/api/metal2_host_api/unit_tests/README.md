# Metal 2.0 Host API unit tests

This folder holds Metal 2.0 unit tests, test included in this category must:

- Be self-contained.
  - e.g. Host only.
- Being able to run without silicon (and preferably without emulator when possible).
  - This means all tests should be prefixed with `CPU_` on it's name to allow CI system to pick up these.
- Be trivially fast to execute.
- Not use any non-default kernel source string.
  - If tests needs to have custom kernel source string, consider adding them to the `kernel_compilation_tests` folder or `integration_tests` folder.
  - This allows harness to skip compilation step, which allows the whole test suite to be quick to execute.

## Layout


| Directory                                              | What it tests                                                                          |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `[invariant_tests/](invariant_tests/)`                 | Which specs `MakeProgramFromSpec` accepts or rejects, one directory per public header  |
| `[kernel_hash/](kernel_hash/)`                         | Which spec fields feed a kernel's JIT cache key (`compute_hash`)                       |
| `[program/](program/)`                                 | What `MakeProgramFromSpec` produces from a valid spec                                  |
| `[program_run_args/](program_run_args/)`               | `SetProgramRunArgs`, `UpdateProgramRunArgs`, `UpdateTensorArgs`, `MergeProgramRunArgs` |
| `[tensor_spec_relaxations/](tensor_spec_relaxations/)` | The `TensorSpecRelaxations` match and hash relation                                    |
| `[utility/](utility/)`                                 | `Table<K, V>`                                                                          |
| `spec_type_properties.cpp`                             | Every spec struct stays an aggregate (designated initializers work) and ttsl-hashable  |
