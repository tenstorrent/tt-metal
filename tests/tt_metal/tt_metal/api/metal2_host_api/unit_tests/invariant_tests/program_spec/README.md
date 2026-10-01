# program_spec invariant tests

Invariants of `program_spec.hpp`: the local invariants of `WorkUnitSpec`, the per-field rules of `ProgramSpec`, and
the structural invariants listed at the top of `ProgramSpec` below.

## Listed invariants

`WorkUnitSpec` and `ProgramSpec` as declared in `program_spec.hpp`, with every field and only its invariants.

```cpp
struct WorkUnitSpec {
    std::string name;

    // Invariant:
    // - Must be non-empty.
    // - Must not have repeated kernel names.
    // - At most one compute kernel.
    // - Summed num_threads:
    //   - Gen1: at most 1 across compute kernels, at most 2 across data-movement kernels.
    //   - Gen2: at most 4 across compute kernels, at most 6 across data-movement kernels.
    // - Gen1: the data-movement kernels use distinct config_1xx->processor values and the same
    //   config_1xx->noc_mode. Those in DM_DEDICATED_NOC mode also use distinct config_1xx->noc values.
    // - For each DataflowBufferSpec these kernels bind, exactly one of them binds it as PRODUCER and
    //   exactly one binds it as CONSUMER. (The producer and consumer may be the same kernel.)
    // - For each ScratchpadSpec, at most one of these kernels binds it.
    // - These kernels bind at most 32 distinct DataflowBufferSpecs on Wormhole, 64 on Blackhole, or 32 on
    //   Quasar. Subject to additional constraints during build.
    Group<KernelSpecName> kernels;

    // Invariant:
    // - Must be non-empty.
    Nodes target_nodes;
};

struct ProgramSpec {
    // Object invariants (excluding advanced options):
    //
    // Every name resolves:
    // - Each KernelSpec::DFBBinding::dfb_spec_name names a spec in dataflow_buffers.
    // - Each KernelSpec::SemaphoreBinding::semaphore_spec_name names a spec in semaphores.
    // - Each KernelSpec::ScratchpadBinding::scratchpad_spec_name names a spec in scratchpads.
    // - Each KernelSpec::TensorBinding::tensor_parameter_name names a parameter in tensor_parameters.
    // - Each name in WorkUnitSpec::kernels names a spec in kernels.
    // - Each DataflowBufferSpec::borrowed_from, when set, names a parameter in tensor_parameters.
    //
    // Every declaration is used:
    // - Each kernel is listed by at least one WorkUnitSpec.
    // - Each dataflow buffer is bound by at least one kernel.
    // - Each scratchpad is bound by at least one kernel.
    // - Each tensor parameter is bound by a kernel, or by a dataflow buffer's borrowed_from.
    // - (Semaphores are exempt here)
    //
    // Data-movement core assignment:
    // - Gen2: each data-movement kernel can be given num_threads DM cores, the same cores on every node
    //   it runs on, such that kernels in the same WorkUnitSpec get disjoint cores, kernels on the same
    //   side of a DataflowBufferSpec get the same cores, and DM0 and DM1 are never used.
    //
    // DataflowBufferSpec endpoints (across all kernels that bind a given DataflowBufferSpec):
    // - All PRODUCER bindings share access_pattern, and their kernels share num_threads and kind
    //   (compute or data movement). The same holds for all CONSUMER bindings.
    // - Gen1: all data-movement PRODUCER kernels share config_1xx->processor. The same holds for
    //   data-movement CONSUMER kernels.
    // - If any kernel binds it as both PRODUCER and CONSUMER, its PRODUCER kernels and its CONSUMER
    //   kernels are the same set of kernels.
    // - Gen2: on each side, the data-movement kernels agree on whether implicit sync is disabled for it
    //   (via disable_dfb_implicit_sync_for or disable_dfb_implicit_sync_for_all).
    // - With P the PRODUCER kernels' num_threads and C the CONSUMER kernels' num_threads:
    //   - STRIDED consumers: num_entries % max(P, C) == 0 and num_entries / max(P, C) <= 65535.
    //   - ALL consumers: num_entries % P == 0 and num_entries / P <= 65535.
    //
    // Semaphores:
    // - A SemaphoreSpec bound by a compute kernel is not also bound by a data-movement kernel.
    // - At most one SemaphoreSpec is bound by compute kernels.
    //
    // All subobjects uphold their own invariants.

    std::string name;

    // Invariant:
    // - Must not have repeated KernelSpec::unique_id.
    // - Must not be empty.
    Group<KernelSpec> kernels;

    // Invariant:
    // - Must not have repeated DataflowBufferSpec::unique_id.
    Group<DataflowBufferSpec> dataflow_buffers;

    // Invariant:
    // - Must be empty (Not yet implemented).
    Group<CrossNodeDataflowBufferSpec> cross_node_dataflow_buffers;

    // Invariant:
    // - Must not have repeated SemaphoreSpec::unique_id.
    // - Each node must have at most 16 SemaphoreSpecs associated with them.
    Group<SemaphoreSpec> semaphores;

    // Invariant:
    // - Must not have repeated ScratchpadSpec::unique_id.
    Group<ScratchpadSpec> scratchpads;

    // Invariant:
    // - Must not have repeated TensorParameter::unique_id.
    Group<TensorParameter> tensor_parameters;

    // Invariant:
    // - Must have at least one WorkUnitSpec.
    // - All work_units must be on distinct nodes.
    Group<WorkUnitSpec> work_units;

    ProgramAdvancedOptions advanced_options;
};
```

## Known gaps

- `WorkUnitSpecWithNoKernelsFails` and `EmptyWorkUnitSpecsFails` fail earlier, on "Kernel ... is not referenced by
any WorkUnitSpec", so the checks their names describe are never reached.
- `DMKernelsDifferentNocModesOnDistinctNodesSucceeds` sets both kernels to `DM_DEDICATED_NOC`, so it does not test
what its name says.
- `SemaphoreSharedByComputeAndDMIsRejected` and `ComputeBoundSemaphoreWithNonzeroInitialValueIsRejected` use
`EXPECT_ANY_THROW` and do not pin the error message.
- Untested: a kernel repeated within one `WorkUnitSpec::kernels`; an empty `WorkUnitSpec::target_nodes`; more than
16 SemaphoreSpecs on one node; the Gen1 per-WorkUnitSpec thread budgets; the STRIDED / ALL `num_entries` rules.
