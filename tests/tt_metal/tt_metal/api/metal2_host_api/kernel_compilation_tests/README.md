# Kernel compilation tests

Host-only tests that JIT-compile custom kernel source against a mock device. The `unit_tests/` rules apply (no
silicon, `CPU_` prefix), except that custom kernel source is allowed; these tests are slower because they compile.

The kernels are compiled, never run. Check generated binding code with `static_assert` in the kernel source. A test
that must run a kernel and check its output belongs in `integration_tests/`.

| Directory | What it contains |
|---|---|
| `bindings/` | Resource binding tokens: tensor accessors and binding sequences, scratchpads, `get_token_if_present`, and the LLK operand metadata (`LLKOperandFrom`) of DFB, scratchpad and tensor bindings |
| `kernel_args/` | Compile-time varargs, and the TT_KERNEL compute shim |
