# Metal 2.0 Host API unit tests

Unit tests are self-contained, fast to run, and need no physical device or emulator: they run on the host against a
mock device (thus every test name starts with `CPU_`). They use only the default kernel source, so they never need the
JIT system to compile custom kernel code.

Put a test elsewhere when:

- it needs its own kernel source: use [kernel_compilation_tests](../kernel_compilation_tests/), which keeps JIT
  compilation out of this suite;
- it has to run kernels on a device to check their results: use [integration_tests](../integration_tests/).

| Directory | Test Category |
|---|---|
| `invariant_tests/` | Which specs `MakeProgramFromSpec` accepts or rejects |
| `kernel_hash/` | Which spec fields feed a kernel's JIT cache key |
| `program/` | What `MakeProgramFromSpec` produces from a valid spec |
| `program_run_args/` | How a program behaves when runtime arguments are supplied or patched |
| `tensor_spec_relaxations/` | The `TensorSpecRelaxations` match and hash relation |
| `utility/` | Utilities used by Metal 2.0 |
| `spec_type_properties.cpp` | Every spec struct stays an aggregate (designated initializers work) and ttsl-hashable |
