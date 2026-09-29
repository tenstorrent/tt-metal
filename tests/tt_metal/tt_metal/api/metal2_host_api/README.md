# Metal 2.0 Host API tests

This folder holds the tests for the Metal 2.0 Host API. Please categorize your test by the folder
structure below. For more information, please check out the README in each nested folder.

## Layout

| Directory | What it holds |
|---|---|
| `unit_tests/` | Host-only tests against a mock device, using only the default kernel source |
| `kernel_compilation_tests/` | Host-only tests that JIT-compile custom kernel source without running it |
| `integration_tests/` | Tests that need silicon or an emulator |
