# Metal 2.0 Host API integration tests

Integration tests are the heaviest tests in the Metal 2.0 suite. They need physical hardware or full emulation to run,
and custom kernel source code to compile. They run the kernels and generally assert on the end-to-end result of the
device output, including Metal 2.0's integration with the rest of the low-level stack (for example LLK).

Put a test here only if a mock device cannot cover it. Put it elsewhere when:

- it only needs its kernel source to compile, checked with compile-time assertions: use
  [kernel_compilation_tests](../kernel_compilation_tests/);
- it needs no custom kernel source: use [unit_tests](../unit_tests/).

| File | Test Category |
|---|---|
| `binding_loopbacks.cpp` | Resource bindings: DFB, semaphore and tensor accessor names |
| `kernel_args_loopbacks.cpp` | Kernel arguments: named args, varargs, TT_KERNEL kernels and CRTA sections |
| `scratchpad.cpp` | Scratchpads under slow dispatch |
| `scratchpad_fast_dispatch.cpp` | Scratchpads under fast dispatch |
| `llk_operand_mul.cpp` | LLK operands from a DFB, a LocalTensorAccessor and a Scratchpad (Blackhole) |
| `mesh_workload_factories.cpp` | `MakeMeshWorkloadFromSpec(s)` |
