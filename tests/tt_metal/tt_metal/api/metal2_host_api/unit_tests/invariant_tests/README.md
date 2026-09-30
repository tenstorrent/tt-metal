# Invariant tests

Each test builds a ProgramSpec, calls `MakeProgramFromSpec` on a mock device, and checks that the spec is rejected
with a specific error message or accepted. Together they pin the invariants written as comments in the public
headers under `tt_metal/api/tt-metalium/experimental/metal2_host_api/`.

## Where an invariant is tested

An invariant is **local** to the smallest struct that holds every field it reads (hardware generation may be an extra
input), and is tested in the directory named after that struct's header. For example, "a DFB accessor name is a
valid C++ identifier" reads only a `DFBBinding`, so it is tested in `kernel_spec/`.

A name field is a pointer: a binding's `*_spec_name` or `tensor_parameter_name`, `WorkUnitSpec::kernels`, and
`DataflowBufferSpec::borrowed_from`. A struct may follow the names it holds, so a rule that reads the object a name
points to is still local. For example, "every DFB a compute kernel binds sets `data_format_metadata`" follows
`dfb_spec_name`, so it is local to `KernelSpec` and tested in `kernel_spec/`. Rules on `*AdvancedOptions` fields are an
exception and do not follow names.

A rule that no struct below `ProgramSpec` can state is **structural**. It either checks that a name resolves, for
example "every `DFBBinding::dfb_spec_name` names a declared DataflowBufferSpec", or needs every struct that names an
object, for example "every DataflowBufferSpec is bound by some kernel". Structural invariants are listed at the top of
`ProgramSpec` in `program_spec.hpp` and tested in `program_spec/`.

| Directory | Header | Covers |
|---|---|---|
| `kernel_spec/` | `kernel_spec.hpp` | KernelSpec: threads, bindings and the DFBs they name, argument schema, hw_config |
| `data_movement_hardware_config/` | `data_movement_hardware_config.hpp` | DataMovementHardwareConfig |
| `dataflow_buffer_spec/` | `dataflow_buffer_spec.hpp` | DataflowBufferSpec, including the TensorParameter `borrowed_from` names |
| `scratchpad_spec/` | `scratchpad_spec.hpp` | ScratchpadSpec |
| `prefetcher_pipe_parameter/` | `prefetcher_pipe_parameter.hpp` | PrefetcherPipeParameter geometry |
| `advanced_options/` | `advanced_options.hpp` | KernelAdvancedOptions, DFBAdvancedOptions, SemaphoreAdvancedOptions |
| `program_spec/` | `program_spec.hpp` | WorkUnitSpec and the kernels it names, per-field ProgramSpec rules, all structural invariants |

Headers without a directory:

- `semaphore_spec.hpp`: its one local invariant (non-empty `target_nodes`) is untested. Semaphore binding and option
  rules are in `kernel_spec/`, `advanced_options/` and `program_spec/`.
- `compute_hardware_config.hpp`: no invariant of its own. The `unpack_modes` rules need the kernel's DFB bindings, so
  they are in `kernel_spec/hardware_config.cpp`.
- `tensor_parameter.hpp`: no invariant.
- `program_run_args.hpp`: tested in `../program_run_args/`.

## Writing an invariant test

- Start from a minimal valid spec (`test_helpers.hpp`), break exactly one rule, and assert on the error text with
  `::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(...))`.
- When the boundary is not obvious, pair the rejection with an acceptance test at the boundary, as
  `MaxComputeThreadsSucceeds` does for `ComputeKernelExceedingMaxThreadsFails`.
- Use the fixture for the architecture the rule depends on, and add the test to the directory's Files and coverage
  tables. The tables prefix each test with its fixture:

| Prefix | Fixture | Architecture |
|---|---|---|
| Q | `ProgramSpecTestQuasar` | Quasar (Gen2) |
| WH | `ProgramSpecTestGen1` | Wormhole B0 (Gen1) |
| BH | `ProgramSpecTestBlackhole` | Blackhole (Gen1) |
| PQ | `PrefetcherPipeSpecTestQuasar` | Quasar, with PrefetcherPipe helpers |
| PW | `PrefetcherPipeSpecTestGen1` | Wormhole B0, with PrefetcherPipe helpers |
