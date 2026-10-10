# Kernel compilation tests

Like unit tests, kernel compilation tests are self-contained, relatively fast to run, and need no physical device or
emulator. Unlike unit tests, they may contain custom kernel source code, which the JIT system compiles as part of the
test. The kernels are compiled but never run: a test asserts only on building the program, not on its runtime
behavior. To check the kernel-side setup, use compile-time assertions such as `static_assert` in the kernel source.

Put a test elsewhere when:

- it has to dispatch to a device to check device-side behavior: use [integration_tests](../integration_tests/);
- it needs no custom kernel source, for example to validate a ProgramSpec: use [unit_tests](../unit_tests/).

| Directory | Test Category |
|---|---|
| `resource_bindings/` | Resource binding emissions |
| `kernel_args/` | Compile-time varargs, and 1st world kernel arguments |
