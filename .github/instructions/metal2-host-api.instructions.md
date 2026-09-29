---
description: 'Metal 2.0 Host API: test placement, and keeping header invariants, validation and tests in step'
applyTo: 'tests/tt_metal/tt_metal/api/metal2_host_api/**,tt_metal/api/tt-metalium/experimental/metal2_host_api/**,tt_metal/impl/metal2_host_api/**'
excludeAgent: "cloud-agent"
---

# Metal 2.0 Host API Review

Use `tests/tt_metal/tt_metal/api/metal2_host_api/README.md` and the README in each nested
folder to decide where a test goes. Those READMEs are the source of truth for placement;
the rules below are what a review must enforce.

## 🔴 CRITICAL

- **Wrong test category**: a test that needs silicon (enqueue, device readback) must live in
  `integration_tests/`. A host-only test with custom kernel source must live in
  `kernel_compilation_tests/`. `unit_tests/` must not contain custom kernel source strings.
- **Missing from `sources.cmake`**: every new test `.cpp` under
  `tests/tt_metal/tt_metal/api/metal2_host_api/` must be listed in
  `tests/tt_metal/tt_metal/api/metal2_host_api/sources.cmake`. A file that is not listed is
  never compiled into `unit_tests_api`.
- **Invariant without a test**: a new or changed invariant comment in a public Metal 2.0 header,
  or a new or changed validation check in `tt_metal/impl/metal2_host_api/program_spec.cpp`, must
  have a matching test in `unit_tests/invariant_tests/<header>/` (local invariants) or
  `unit_tests/invariant_tests/program_spec/` (structural invariants that read more than one
  `ProgramSpec` field).

## 🟡 IMPORTANT

- **`CPU_` prefix**: tests in `unit_tests/` and `kernel_compilation_tests/` must use the `CPU_`
  name prefix. Integration tests must not.
- **Architecture fixture**: use the shared fixture for the architecture the rule depends on
  (`ProgramSpecTestQuasar` / `ProgramRunArgsTestQuasar` for Gen2, `ProgramSpecTestGen1` /
  `ProgramRunArgsTestGen1` for Wormhole Gen1, `ProgramSpecTestBlackhole` for Blackhole-only
  rules, `PrefetcherPipeSpecTest*` for PrefetcherPipe). Do not invent a one-off fixture when a
  shared one already covers the architecture.
- **Directory README tables**: when adding or removing tests, update the directory README's
  Files table and test counts, and for invariant tests the coverage table.
- **Kernel-compilation bindings**: tests in `kernel_compilation_tests/` check generated binding
  code with `static_assert`. They must not enqueue or otherwise run the compiled kernel; that
  belongs in `integration_tests/`.
- **Shared helper consumers**: changes to `test_helpers.hpp`, `mock_device_fixtures.hpp`,
  `prefetcher_pipe_test_helpers.hpp`, or `run_args_test_helpers.hpp` must keep compiling the
  tests outside this folder that include them (for example under
  `tests/tt_metal/tt_metal/api/dataflow_buffer/` and
  `tests/ttnn/unit_tests/gtests/accessor/`).

## Review Checklist

- [ ] Test is in the correct category (`unit_tests/` / `kernel_compilation_tests/` / `integration_tests/`)
- [ ] New test `.cpp` is listed in `metal2_host_api/sources.cmake`
- [ ] New or changed header invariants / `program_spec.cpp` validation have matching invariant tests
- [ ] Host-only tests use the `CPU_` prefix; integration tests do not
- [ ] Tests use the shared fixture for the architecture the rule depends on
- [ ] Directory README Files / coverage tables are updated
- [ ] Kernel-compilation tests use `static_assert` and do not run kernels
- [ ] Shared helper-header changes keep external includers compiling
