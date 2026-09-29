# Kernel compilation tests

This folder holds Metal 2.0 unit tests with custom kernel source, test included in this category must:

Similar to [README for unit_tests directory](../unit_tests/README.md),
this folder should hold **unit tests** that are:

- Self-contained.
- Being able to run without silicon.
 - This means pleaes include `CPU_` in the test's naming prefix.
- Be reasonable fast to execute.

With the sole exception that this folder contains tests with non-default kernel source.
For example: test kernel source generation of bindings.
Different to the unit tests directory,
tests in this folder may take more time to execute due to it requiring the JIT compilation system to kick-in.

Note that tests within this folder should not expect the respect kernel source code to be executable, if that's the desired test environment, please put your tests under [the integration test folder](../integration_tests/).
This means resource binding should be tested using `static_asserts`.

| Directory | What it contains |
|---|---|
| [`bindings/`](bindings/) | Kernels that use resource binding tokens: tensor accessors, tensor binding sequences, scratchpads, `get_token_if_present`, and the LLK operand metadata (`LLKOperandFrom`) of DFB, scratchpad and tensor bindings |
| [`kernel_args/`](kernel_args/) | Kernels that read compile-time varargs, and the TT_KERNEL compute shim |
