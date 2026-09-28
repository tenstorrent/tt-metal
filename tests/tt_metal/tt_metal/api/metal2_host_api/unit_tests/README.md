# Metal 2.0 Host API unit tests

Tests that run without silicon. Most of them build a Program against a mock device
(`experimental::configure_mock_mode`): a `MeshDevice` on a simulated Quasar, Wormhole or Blackhole
cluster. A few are plain host code with no device at all. Every test name starts with `CPU_`, which
is how CI selects host-only tests (`--gtest_filter='*CPU_*'`).

## Layout

| Directory | What it tests |
|---|---|
| [`invariant_tests/`](invariant_tests/) | Which specs `MakeProgramFromSpec` accepts or rejects, one directory per public header |
| [`kernel_hash/`](kernel_hash/) | Which spec fields feed a kernel's JIT cache key (`compute_hash`) |
| [`program/`](program/) | What `MakeProgramFromSpec` produces from a valid spec |
| [`program_run_args/`](program_run_args/) | `SetProgramRunArgs`, `UpdateProgramRunArgs`, `UpdateTensorArgs`, `MergeProgramRunArgs` |
| [`tensor_spec_relaxations/`](tensor_spec_relaxations/) | The `TensorSpecRelaxations` match and hash relation |
| [`utility/`](utility/) | `Table<K, V>` |
| `spec_type_properties.cpp` | Every spec struct stays an aggregate (designated initializers work) and ttsl-hashable |

## Where a new test goes

- It checks whether a spec is accepted or rejected: `invariant_tests/<header>/`. See that directory's README.
- It checks what a valid spec lowers to: `program/`, or `kernel_hash/` if it is about the kernel hash.
- It checks run arguments: `program_run_args/`.
- It JIT-compiles a kernel: [`../kernel_compilation_tests/`](../kernel_compilation_tests/).
- It needs silicon: [`../integration_tests/`](../integration_tests/).

List every new file in [`../sources.cmake`](../sources.cmake).

## Shared helpers and fixtures

These live in the parent `metal2_host_api/` directory:

- `test_helpers.hpp`: minimal spec builders (`MakeMinimalValidProgramSpec`, `MakeMinimalGen2DMKernel`, ...).
- `mock_device_fixtures.hpp`: `ProgramSpecTestQuasar`, `ProgramSpecTestGen1` (Wormhole), `ProgramSpecTestBlackhole`,
  `ProgramRunArgsTestQuasar`, `ProgramRunArgsTestGen1`.
- `prefetcher_pipe_test_helpers.hpp`: PrefetcherPipe test geometry, spec builders, and the
  `PrefetcherPipeSpecTestQuasar` / `PrefetcherPipeSpecTestGen1` fixtures.
- `run_args_test_helpers.hpp`: ProgramRunArgs builders shared by `program_run_args/` and `program/`.

Two build constraints shape these headers:

- A gtest suite such as `ProgramSpecTestQuasar` spans many files, and gtest requires every test in a suite to use
  the same fixture type. Use the fixtures from `mock_device_fixtures.hpp`; don't redefine them in a test file.
- Unity builds compile up to eight source files as one translation unit. A helper local to one file needs a name no
  other file in `metal2_host_api/` uses; a helper that two files need belongs in a shared header.

## Running

All of these tests are in the `unit_tests_api` binary:

```sh
cmake --build build --target unit_tests_api
TT_METAL_HOME=$PWD ./build/test/tt_metal/unit_tests_api --gtest_filter='ProgramSpecTestQuasar.*'
```

Suites used here: `ProgramSpecTestQuasar`, `ProgramSpecTestGen1`, `ProgramSpecTestBlackhole`,
`ProgramRunArgsTestQuasar`, `ProgramRunArgsTestGen1`, `PrefetcherPipeSpecTestQuasar`, `PrefetcherPipeSpecTestGen1`,
`PrefetcherPipeSpecTestGen1TwoChips`, `AggregateSpecTypes`, `ProgramSpecReflectionTest`, `MergeProgramRunArgs`,
`TableTest`, `TableMiscTest`, `TableHashTest`, `TensorSpecRelaxations`.

A suite spans several directories, so a suite filter does not select one directory. Add `--gtest_list_tests` to
see what a filter selects.
