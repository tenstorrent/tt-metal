# Kernel argument compilation tests

Compile-only tests of how kernels read their arguments. Mock Wormhole (`ProgramSpecTestGen1`).

| File | Tests | What is compiled |
|---|---|---|
| `compile_time_varargs.cpp` | 8 | `get_compile_time_vararg*` in DM and compute kernels, including empty and 1024-entry lists; varargs as a prefix ahead of tensor binding CTAs; an out-of-range index fails to compile, with and without kernel asserts enabled |
| `tt_kernel_shim.cpp` | 1 | A TT_KERNEL compute entry: the generated `kernel_main()` shim is emitted and compiles on the TRISC path |

On-device checks of the same argument paths are in
[`../../integration_tests/kernel_args_loopbacks.cpp`](../../integration_tests/kernel_args_loopbacks.cpp).
