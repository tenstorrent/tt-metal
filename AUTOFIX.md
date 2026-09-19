# Terminal MLA+MoE range-lockstep allocation repair

## Failing contract

TT-Lang runtime storage must assign one address to every core of a remote-uniform dataflow buffer while permitting disjoint core sets to reuse addresses. Local singleton arenas and remote-uniform storage must not fragment one another's required common address interval.

Job 109879 showed that reserving one 528,768-byte union arena after model argument allocation does not satisfy this contract. Each bank already contained 189,504 bytes of model lockstep arguments, so only a 313,600-byte common interval remained. Earlier runs failed either a 151,616-byte local arena or a 14,336-byte remote-uniform member after independent allocations fragmented the common interval.

## Diagnosis and correction

The original range-lockstep implementation selected a common address by scanning the requested cores, but reserved allocator 0 and mirrored that reservation globally. This prevented disjoint core sets from reusing an address and made selected-core placement dependent on unrelated banks.

The correction reserves the chosen range in the exact per-core allocator domains selected by the tensor distribution or shard specification. MeshBuffer mirrors and gathers ownership only for those cores. TT-Lang then orders the exact remote-uniform and singleton-local requests deterministically by decreasing size and allocates each with range-lockstep semantics. Intersecting core sets cannot overlap; disjoint sets may reuse an address.

## Changed files

- `tt_metal/impl/allocator/bank_manager.hpp`
- `tt_metal/impl/allocator/bank_manager.cpp`
- `tt_metal/impl/allocator/allocator.hpp`
- `tt_metal/impl/allocator/allocator.cpp`
- `tt_metal/distributed/mesh_buffer.cpp`
- `tests/tt_metal/tt_metal/api/allocator/test_per_core_bank_manager.cpp`
- `tests/tt_metal/tt_metal/api/allocator/test_range_lockstep_allocation.cpp`

The corresponding TT-Lang allocation ordering is in a separate candidate worktree. This report covers the TT-Metal repair only.

## Validation

Source revision: `60e6701fa4e72b7168b0c07e01ff3fa864f47b2b`.

Focused job 109973 built the two allocator test targets and passed:

- `PerCoreAllocation.CPU_RangeLockstep*`: 5/5
- `RangeLockstepAllocationTest.*`: 6/6

The job used Clang 18, `ENABLE_DISTRIBUTED=OFF`, Python 3.12, ccache disabled, and an exclusive qb2 allocation. Its install/import stage and Terminal MLA+MoE device qualification are still in progress. Hardware correctness and performance are not yet verified.
