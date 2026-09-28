# Kernel compilation tests

These tests build a Program from a ProgramSpec and JIT-compile its kernels against a mock Wormhole device
(`program.impl().compile(...)` or `detail::CompileProgram`). Nothing runs on a device. They catch errors that only
show up when device code is compiled: the generated `kernel_bindings_generated.h`, the device-side accessor APIs,
and the argument shims.

They use the `ProgramSpecTestGen1` fixture (mock Wormhole). Mock-Quasar JIT compilation is not wired up in this
checkout, because the Quasar TRISC firmware objects aren't built and the link step fails. Test names start with
`CPU_`, so CI runs them on host-only runners. Tests whose names contain `FailsToCompile` expect the kernel build to
fail.

| Directory | What it compiles |
|---|---|
| [`bindings/`](bindings/) | Kernels that use resource binding tokens: tensor accessors, tensor binding sequences, scratchpads, `get_token_if_present` |
| [`kernel_args/`](kernel_args/) | Kernels that read compile-time varargs, and the TT_KERNEL compute shim |

Whether a spec with these bindings is accepted is tested in
[`../unit_tests/invariant_tests/`](../unit_tests/invariant_tests/). A test that needs the kernel to actually run
belongs in [`../integration_tests/`](../integration_tests/).

## Running

JIT compilation needs `TT_METAL_HOME` to point at the repository root:

```sh
TT_METAL_HOME=$PWD ./build/test/tt_metal/unit_tests_api --gtest_filter='ProgramSpecTestGen1.*JITSmoke*'
```

`ProgramSpecTestGen1` also contains the Wormhole invariant tests. Add `--gtest_list_tests` to see what a filter
selects.
