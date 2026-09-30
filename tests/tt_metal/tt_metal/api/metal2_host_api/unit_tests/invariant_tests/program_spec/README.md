# program_spec invariant tests

Invariants of `program_spec.hpp`: the local invariants of `WorkUnitSpec`, the per-field rules of `ProgramSpec`, and
the structural invariants listed at the top of `ProgramSpec`. `WorkUnitSpec::kernels` is a list of names, so the
rules that read the kernels it names, and what those kernels bind, are local to `WorkUnitSpec`. They are equal to
per-node rules because `work_units` are disjoint. The PrefetcherPipe, alias and compute-bound semaphore rules come
from `advanced_options.hpp` and `prefetcher_pipe_parameter.hpp`. As advanced options they keep their structural
classification.

## Files

| Section of program_spec.hpp | File | Tests |
|---|---|---|
| `WorkUnitSpec` fields: non-empty kernels, target nodes on the device grid | `work_unit_spec.cpp` | 5 |
| `WorkUnitSpec::kernels`: one producer and one consumer instance per DFB its kernels bind; one binding kernel per scratchpad | `work_unit_bindings.cpp` | 12 |
| `WorkUnitSpec::kernels`: at most one compute kernel; summed `num_threads` budgets | `work_unit_capacity.cpp` | 4 |
| `WorkUnitSpec::kernels`, Gen1: distinct processors, shared `noc_mode`, distinct dedicated NOCs | `gen1_dm_placement.cpp` | 11 |
| `WorkUnitSpec::kernels`: distinct DataflowBufferSpecs bound, which bounds the DFBs on each node | `dfbs_per_node.cpp` | 4 |
| `ProgramSpec` fields: unique ids in each group, non-empty kernels and work units, disjoint work units, no cross-node DFBs yet | `program_spec_fields.cpp` | 11 |
| Every name resolves: binding names, `WorkUnitSpec::kernels`, `DataflowBufferSpec::borrowed_from` | `references.cpp` | 6 |
| Every declaration is used (semaphores exempt; `borrowed_from` counts as a use of a TensorParameter) | `declarations_used.cpp` | 6 |
| Gen2 DM core assignment | `gen2_dm_core_assignment.cpp` | 3 |
| DataflowBufferSpec endpoints: every bound DFB has a producer and a consumer, same-role uniformity (including the Gen1 processor), self-loop sides, Gen2 implicit-sync agreement | `dfb_endpoints.cpp` | 12 |
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
- The DFB limit on `WorkUnitSpec::kernels` is marked "TODO: THIS IS LYING" in the header.
