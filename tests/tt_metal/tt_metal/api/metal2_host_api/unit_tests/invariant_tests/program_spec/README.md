# program_spec invariant tests

Invariants of `program_spec.hpp`: the local invariants of `WorkUnitSpec`, the per-field rules of `ProgramSpec`, and
the structural invariants listed at the top of `ProgramSpec`. The PrefetcherPipe and alias rules come from
`advanced_options.hpp` and `prefetcher_pipe_parameter.hpp`, but are structural because they relate several
ProgramSpec fields.

## Files

| Section of program_spec.hpp | File | Tests |
|---|---|---|
| `WorkUnitSpec` fields: non-empty kernels, target nodes on the device grid | `work_unit_spec.cpp` | 5 |
| `ProgramSpec` fields: unique ids in each group, non-empty kernels and work units, disjoint work units, no cross-node DFBs yet | `program_spec_fields.cpp` | 11 |
| Every binding refers to a declared resource; every WorkUnitSpec refers to declared kernels | `references.cpp` | 5 |
| Every declaration is used (semaphores exempt) | `declarations_used.cpp` | 5 |
| Borrowed memory: `borrowed_from` names an L1 TensorParameter large enough for the DFB | `borrowed_memory.cpp` | 8 |
| Placement per WorkUnitSpec: one producer and one consumer instance per DFB per node; one binding kernel per scratchpad per node | `work_unit_bindings.cpp` | 14 |
| Placement per WorkUnitSpec: at most one compute kernel; summed `num_threads` budgets | `work_unit_capacity.cpp` | 4 |
| Placement per WorkUnitSpec, Gen1: distinct processors, shared `noc_mode`, distinct dedicated NOCs | `gen1_dm_placement.cpp` | 11 |
| Placement per WorkUnitSpec, Gen2: DM core assignment | `gen2_dm_core_assignment.cpp` | 3 |
| DataflowBufferSpec endpoints: same-role uniformity (including the Gen1 processor), self-loop sides, data format for compute, Gen2 implicit-sync agreement | `dfb_endpoints.cpp` | 12 |
| DataflowBufferSpecs per node | `dfbs_per_node.cpp` | 4 |
| Unpack modes | `unpack_modes.cpp` | 10 |
| Semaphores bound by compute kernels (Blackhole) | `semaphores.cpp` | 6 |
| DFB alias groups (`DFBAdvancedOptions::alias_with`): symmetric, same total size, node set and `borrowed_from` | `dfb_aliasing.cpp` | 5 |
| PrefetcherPipe: accepted geometries, references, usage, sender/receiver roles, accessor groups | `prefetcher_pipe_roles.cpp` | 24 |
| PrefetcherPipe: credit lanes (the receiver's `num_threads`) | `prefetcher_pipe_lanes.cpp` | 9 |
| PrefetcherPipe: relay DFBs | `prefetcher_pipe_relays.cpp` | 9 |
| Minimal specs that satisfy every invariant | `valid_program_specs.cpp` | 3 |

## Known gaps

- `WorkUnitSpecWithNoKernelsFails` and `EmptyWorkUnitSpecsFails` fail earlier, on "Kernel ... is not referenced by
  any WorkUnitSpec", so the checks their names describe are never reached.
- `DMKernelsDifferentNocModesOnDistinctNodesSucceeds` sets both kernels to `DM_DEDICATED_NOC`, so it does not test
  what its name says.
- `SemaphoreSharedByComputeAndDMIsRejected` and `ComputeBoundSemaphoreWithNonzeroInitialValueIsRejected` use
  `EXPECT_ANY_THROW` and do not pin the error message.
- Untested: a kernel repeated within one `WorkUnitSpec::kernels`; an empty `WorkUnitSpec::target_nodes`; more than
  16 SemaphoreSpecs on one node; the Gen1 per-WorkUnitSpec thread budgets; the STRIDED / ALL `num_entries` rules.
- The per-node DFB limit in the `ProgramSpec` comment is marked "TODO: THIS IS LYING" in the header.
