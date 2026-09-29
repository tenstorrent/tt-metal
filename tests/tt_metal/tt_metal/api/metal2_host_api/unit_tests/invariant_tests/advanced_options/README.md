# advanced_options invariant tests

Local invariants of the `*AdvancedOptions` structs in `advanced_options.hpp`. The header does not state them in
comments yet, so these tests follow the checks in the implementation. Rules that span several structs (DFB alias
groups, compute-bound semaphore options, PrefetcherPipe roles, lanes and relays) are structural and live in
`../program_spec/`.

## Files

| File | Tests | Covers |
|---|---|---|
| `kernel_advanced_options.cpp` | 13 | `tensor_binding_sequences`, `num_runtime_varargs_per_node`, `prefetcher_pipe_bindings` |
| `dfb_advanced_options.cpp` | 3 | `allow_instance_multi_binding`, `prefetcher_pipe_relays` |
| `semaphore_advanced_options.cpp` | 2 | `initial_value` |

## Coverage

### KernelAdvancedOptions

| Rule | Tests |
|---|---|
| A `TensorBindingSequence::sequence_name` is a C++ identifier | WH `TensorBindingSequenceInvalidIdentifierFails` |
| Sequence names are unique within a kernel | WH `TensorBindingSequenceDuplicateSequenceNamesFails` |
| A sequence name differs from every tensor accessor name, and from every generated `<accessor>_t` alias | WH `TensorBindingSequenceNameCollidesWithBindingFails`, WH `TensorBindingSequenceNameCollidesWithGeneratedTypeAliasFails` |
| Every sequence member is one of this kernel's tensor accessor names, with no repeats | WH `TensorBindingSequenceUnknownMemberFails`, WH `TensorBindingSequenceDuplicateMembersFails` |
| `num_runtime_varargs_per_node` entries cover disjoint node sets | Q `VarargPerNodeOverlapFails` |
| A `PrefetcherPipeBinding::accessor_name` is a C++ identifier | PQ `InvalidAccessorNameFails` (the length limit is untested) |
| Pipe accessor names are unique within a kernel | PQ `DuplicateAccessorNameFails` |
| `pipe_parameter_names` is non-empty and has no repeats | PQ `EmptyAccessorGroupFails`, PQ `SamePipeTwiceInOneAccessorFails` |
| A pipe appears in at most one of a kernel's pipe bindings | PQ `SamePipeBoundTwiceInOneKernelFails` |
| Only data-movement kernels bind pipes | PQ `ComputeKernelBindingPipeFails` |

Accepted tensor binding sequences are covered by the JIT tests in `kernel_compilation_tests/bindings/tensor_bindings.cpp`.

### DFBAdvancedOptions

| Rule | Tests |
|---|---|
| Gen2: `allow_instance_multi_binding` is false, even when no instance is multi-bound | Q `MultiBindingFlagOnGen2Fails` |
| `prefetcher_pipe_relays` has no repeats | PQ `RelayListsSamePipeTwiceFails` |
| A relay DFB (non-empty `prefetcher_pipe_relays`) does not set `borrowed_from` | PQ `RelayWithBorrowedFromFails` |

What the flag allows on Gen1 is covered by the `*WithFlag` tests in `../program_spec/work_unit_bindings.cpp` and
`../program_spec/dfb_endpoints.cpp`.

### SemaphoreAdvancedOptions

| Rule | Tests |
|---|---|
| Gen2: `initial_value == 0` | Q `SemaphoreNonZeroInitialValueFailsOnQuasar`. Accepted on Gen1: WH `SemaphoresWithNonZeroInitialValueSucceedOnGen1` |
