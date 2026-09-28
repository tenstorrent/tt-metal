# Invariant tests

Each test builds a ProgramSpec, calls `MakeProgramFromSpec` on a mock device, and checks that the spec is either
rejected with a specific error message or accepted. Together they pin the invariants written as comments in the
Metal 2.0 public headers under `tt_metal/api/tt-metalium/experimental/metal2_host_api/`.

## Local and structural invariants

An invariant is **local** to the smallest struct that holds every field it reads. Hardware generation may be an
extra input. A local invariant is tested in the directory named after that struct's header. For example, "a DFB
accessor name is a valid C++ identifier" reads only a `DFBBinding`, so it is tested in `kernel_spec/`.

A rule that reads more than one field of `ProgramSpec` is a **structural** invariant of `ProgramSpec`. An example
is "every `DFBBinding::dfb_spec_name` names a declared DataflowBufferSpec", which reads `kernels` and
`dataflow_buffers`. The structural invariants are listed at the top of `ProgramSpec` in `program_spec.hpp` and
tested in `program_spec/`.

Each header directory's README maps the header's invariant comments to the tests that cover them, and lists the
invariants that no test covers yet.

## Directories

| Directory | Header | Covers |
|---|---|---|
| [`kernel_spec/`](kernel_spec/) | `kernel_spec.hpp` | KernelSpec: threads, bindings, argument schema, hw_config against its bindings |
| [`data_movement_hardware_config/`](data_movement_hardware_config/) | `data_movement_hardware_config.hpp` | DataMovementHardwareConfig |
| [`dataflow_buffer_spec/`](dataflow_buffer_spec/) | `dataflow_buffer_spec.hpp` | DataflowBufferSpec |
| [`scratchpad_spec/`](scratchpad_spec/) | `scratchpad_spec.hpp` | ScratchpadSpec |
| [`prefetcher_pipe_parameter/`](prefetcher_pipe_parameter/) | `prefetcher_pipe_parameter.hpp` | PrefetcherPipeParameter geometry |
| [`advanced_options/`](advanced_options/) | `advanced_options.hpp` | KernelAdvancedOptions, DFBAdvancedOptions, SemaphoreAdvancedOptions |
| [`program_spec/`](program_spec/) | `program_spec.hpp` | WorkUnitSpec, per-field ProgramSpec rules, all structural invariants |

Headers without a directory:

- `semaphore_spec.hpp`: its one local invariant (`target_nodes` is non-empty) has no test yet. Semaphore binding
  rules are in `kernel_spec/semaphore_binding.cpp`; semaphore option rules are in `advanced_options/` and
  `program_spec/semaphores.cpp`.
- `compute_hardware_config.hpp`: states no invariant of its own. The `unpack_modes` rules need the kernel's DFB
  bindings, so they are in `kernel_spec/hardware_config.cpp` and `program_spec/unpack_modes.cpp`.
- `tensor_parameter.hpp`: states no invariant.
- `program_run_args.hpp`: tested in [`../program_run_args/`](../program_run_args/).

## Writing an invariant test

- Start from a minimal valid spec (`test_helpers.hpp`), break exactly one rule, and assert on the error text with
  `::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(...))`.
- When the boundary is not obvious, pair the rejection with an acceptance test at the boundary, as
  `MaxComputeThreadsSucceeds` does for `ComputeKernelExceedingMaxThreadsFails`.
- Use the fixture for the architecture the rule depends on: `ProgramSpecTestQuasar` (Gen2), `ProgramSpecTestGen1`
  (Wormhole, Gen1) or `ProgramSpecTestBlackhole`.
