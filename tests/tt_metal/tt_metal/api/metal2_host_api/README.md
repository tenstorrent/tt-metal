# Metal 2.0 Host API tests

This folder holds the tests for the Metal 2.0 Host API. Please categorize your test by the folder
structure below. For more information, please check out the README in each nested folder.

Note that nested folder often have a defined organization,
please read the folder READMEs and respect their organization.

## Test categories

The folder is mostly organized by these 3 test categories:

1. Unit tests.

   These are tests that are self-contained and fast to run;
   do not require a physical device or emulator to be present;
   nor custom kernel source code that requires the JIT system to compile it.

2. Kernel Compilation Tests.

   Like unit tests, these tests should be self-contained and relatively fast to run;
   do not require a physical device or emulator to be present;
   but can contain custom kernel source code that requires the JIT system to perform compilation as part of its validity checks.
   Note that these kernels are not run, they're merely compiled.
   These tests usually involve compile-time assertions or sanity checks on the kernel-side setup.

3. Integration Tests.

   These are the most heavyweight tests in the Metal 2.0 test suite.
   They require physical hardware or full emulation to run, and custom kernel source code to compile.
   They generally assert on the end-to-end result of device output.

## Layout

| Directory | What it holds |
|---|---|
| `unit_tests/` | Host-only tests against a mock device, using only the default kernel source |
| `kernel_compilation_tests/` | Host-only tests that JIT-compile custom kernel source without running it |
| `integration_tests/` | Tests that need silicon or an emulator |
| `test_helpers/` | Shared test utilities: ProgramSpec builders, test fixtures |
