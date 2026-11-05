# Invariant tests

The tests in this folder check that `MakeProgramFromSpec` accepts valid `ProgramSpec`s and rejects invalid ones. An
invalid `ProgramSpec` is one that violates one of its invariants. Each subfolder's README lists the invariants it
tests.

## Folder organization

Each subfolder corresponds to one Metal 2.0 Host API header. Its tests check the local invariants of the structs in
that header, and its README lists those invariants. Structural invariants are tested in `program_spec/`. Keeping each
check next to the struct it constrains makes the checks easier to find and verify.

## Kinds of invariants

An invariant is **local** if it can be checked using the given struct (and the structs it contains) alone.
For example, "a DFB must have a non-zero entry size" can be checked by soley looking at the `DataflowBufferSpec` struct.
A field that names another object behaves like a pointer, invariants dependening on these indirections are still considered local.
For example, "the Tensor referred by `DataflowBufferSpec::borrowed_from` by name must be in L1" is a local invariant of
`DataflowBufferSpec`.

A **structural** invariant can only be checked at the top level: `ProgramSpec`.
For example,
"every `DFBBinding::dfb_spec_name` names a declared `DataflowBufferSpec`"
can only be checked when the full view of memory resource and kernel specs are available.
It is inevitable that these invariants exist but we should keep them to a minimum.

| Directory | Header |
|---|---|
| `kernel_spec/` | `kernel_spec.hpp` |
| `data_movement_hardware_config/` | `data_movement_hardware_config.hpp` |
| `compute_hardware_config/` | `compute_hardware_config.hpp` |
| `dataflow_buffer_spec/` | `dataflow_buffer_spec.hpp` |
| `scratchpad_spec/` | `scratchpad_spec.hpp` |
| `semaphore_spec/` | `semaphore_spec.hpp` |
| `tensor_parameter/` | `tensor_parameter.hpp` |
| `prefetcher_pipe_parameter/` | `prefetcher_pipe_parameter.hpp` |
| `advanced_options/` | `advanced_options.hpp` |
| `program_spec/` | `program_spec.hpp` |

Note that `program_run_args.hpp` does not describe any constructs within the ProgramSpec,
thus it is not included in this directory.

## Listing invariants

Each sub-directory README has a "Listed invariants" section that reflects the invariant of it's header.
The section includes: The definition of constructs within the respective headers with their invariant marked as comments.

Keep a listing in step with its header: when a struct gains, loses or renames a field, update the listing; when a new
check is added to `tt_metal/impl/metal2_host_api/program_spec/validation/`, list the rule on the struct that owns it and
add a test. The `*AdvancedOptions` structs have no listing yet.

### Example: `SemaphoreSpec` in `semaphore_spec.hpp`

```cpp
struct SemaphoreSpec {
    SemaphoreSpecName unique_id;

    // Invariant:
    // - Must be non-empty
    Nodes target_nodes;

    SemaphoreAdvancedOptions advanced_options;
};
```
