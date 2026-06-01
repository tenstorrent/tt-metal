# Ring Joint SDPA Kernel Scheduling Plan

## Goal

Avoid deploying Tensix kernels on cores that do not participate in the SDPA work partition, except where multicast protocol correctness requires dummy participant cores inside a rectangular mcast region.

## Implementation Status

- Branch `pjosipovic/ring-joint-sdpa-core-scheduling` implements the main `ring_joint_sdpa_program_factory.cpp` scheduling fix.
- Main ring-joint reader kernels now use the reader/data-movement participant set, including K-mcast dummy readers when K mcast requires row-wide protocol participants.
- Main ring-joint writer and compute kernels now use only cores with real Q work.
- Fused all-gather receiver signaling is now initialized from the reader/data-movement participant set instead of the full configured grid.
- Main ring-joint performance accounting now uses the actual compute core count instead of the configured grid area.
- Current main-factory CB allocation is narrowed to the reader/data-movement participant union. Splitting every CB to the exact per-kernel use set remains a follow-up refinement.
- `exp_ring_joint_sdpa_program_factory.cpp` remains a follow-up. Its fabric MUX writer termination protocol currently assumes a full fixed client group via `num_workers_per_link`; removing zero-work MUX client kernels without changing that protocol risks teardown deadlock.

## Original And Remaining Issues

- Original issue: `ring_joint_sdpa_program_factory.cpp` computed active work count, but still scheduled reader, writer, and compute kernels over the full configured SDPA grid.
- `exp_ring_joint_sdpa_program_factory.cpp` also schedules reader, writer, fabric-writer, and compute kernels over full grid-derived ranges even when `B * NH * num_q_chunks` underfills the grid.
- `SDPAProgramConfig::sub_core_grids` exists but is ignored by the ring-joint factories.
- Original issue: fused all-gather signaling targeted the full SDPA kernel range, so simply dropping reader kernels on inactive cores would deadlock unless the signal target list was also narrowed.
- Mcast handling is mixed with work scheduling. K mcast can force row-wide participants, while head/V mcast rejects gaps instead of representing them as dummy protocol participants.

## Desired Model

Build explicit core sets after work partitioning:

- `work_cores`: cores with `global_q_count > 0`.
- `mcast_dummy_cores`: inactive cores needed only because an enabled mcast rectangle includes them.
- `reader_cores`: `work_cores + mcast_dummy_cores`, because dummy cores may need to run protocol receive/ack behavior.
- `compute_cores`: `work_cores` only.
- `writer_cores`: `work_cores` only, plus fabric writer cores that actually forward data.
- `data_movement_cores`: the union of reader and writer cores. This may intentionally differ from `compute_cores`.
- `signaled_reader_cores`: exactly the cores with reader kernels that wait on fused all-gather semaphores.

## Data Movement vs Compute Grids

The data-movement Tensix grid must be modeled independently from the compute grid.

- Reader kernels may run on a strict superset of compute cores when mcast dummy participants are required.
- Writer kernels may run on a different subset from readers, especially in `exp_ring_joint` where fabric MUX writers are data-movement-only participants.
- Compute kernels must only run on cores with real SDPA Q work.
- CB descriptors must be allocated on the union of cores that use each CB, not blindly on either the compute grid or the full configured grid.
- Semaphores must be allocated on the protocol participant set for the protocol they serve:
  - fused all-gather semaphores on `signaled_reader_cores`;
  - mcast chain semaphores on `reader_cores` involved in that chain;
  - fabric/MUX semaphores only on valid fabric writer/client cores.
- Runtime args must be emitted only for cores in the corresponding kernel descriptor's `core_ranges`.

This means the implementation should avoid a single `core_grid_set` variable for all kernels. Use explicit names such as `reader_core_set`, `writer_core_set`, `compute_core_set`, `mcast_protocol_core_set`, and `fabric_writer_core_set`.

## Implementation Steps

1. Add small host-side helpers in the SDPA factory layer:
   - Convert ordered `CoreCoord` vectors to `CoreRangeSet`.
   - Build a row-major ordered core list from `compute_with_storage_grid_size` or `sub_core_grids`.
   - Build `work_cores` from `core_work[i].global_q_count > 0`.
   - Build rectangular mcast coverage sets and subtract `work_cores` to find `mcast_dummy_cores`.

2. Update `ring_joint_sdpa_program_factory.cpp`:
   - Partition work across the configured candidate cores as today.
   - Build `work_core_set` after `core_work` is populated.
   - For head/V mcast, stop treating an inactive core inside the mcast rectangle as automatic ineligibility. Instead, include it in `mcast_dummy_cores` if protocol participation is required.
   - For K mcast, keep row-rectangle behavior, but make zero-work row cores reader-only dummy participants.
   - Set reader kernel `core_ranges` to `reader_core_set`.
   - Set writer and compute kernel `core_ranges` to `work_core_set`.
   - Allocate each CB on the exact data-movement/compute union that uses it. For example, K/V input CBs may be needed on reader dummy cores, while output and compute intermediate CBs should remain on `work_core_set`.
   - Emit runtime args only for cores in each kernel's range.
   - Initialize fused-op semaphores and all-gather signaler with `reader_core_set`, not the full configured grid.

3. Update `exp_ring_joint_sdpa_program_factory.cpp`:
   - Build `work_core_set` from single-Q-chunk work assignment.
   - For mcast chains, include only required rectangle dummy participants in reader/protocol scheduling.
   - Restrict compute to `work_core_set`.
   - Restrict non-fabric writer to non-MUX `work_cores`.
   - Restrict fabric writer to MUX writer cores that have real `head_work` and a valid link/direction.
   - Do not default empty MUX writer rows to injector `(0,0)`.
   - Restrict semaphores and CB descriptors to the smallest valid reader, writer, fabric-writer, or compute core set.

4. Respect `SDPAProgramConfig::sub_core_grids`:
   - If provided, use it as the candidate SDPA worker set instead of a full rectangle.
   - Preserve existing rectangular behavior when it is absent.
   - Validate that mcast-enabled paths can form required rectangles from the selected cores, or intentionally add dummy protocol cores within the provided grid contract.

5. Add validation:
   - `work_cores` must be non-empty.
   - Every core in `compute_cores` and `writer_cores` must be in `work_cores`.
   - Every mcast receiver/injector must have a reader kernel scheduled.
   - Every fused-op signaled core must have a reader kernel scheduled.
   - No reader-only dummy core may receive compute or non-required writer kernels.
   - No CB required by a reader-only or fabric-writer-only kernel may be allocated solely on `compute_cores`.
   - No compute-only CB may be allocated on dummy data-movement cores unless a kernel on those cores actually accesses it.

6. Add tests:
   - A ring-joint case where total Q chunks are fewer than configured SDPA cores.
   - A case where K mcast requires dummy row participants.
   - A case where head/V mcast rectangle contains inactive cores and should still run via dummy readers.
   - An `exp_ring_joint` underfilled-grid case, including underfilled MUX writer rows.
   - Program descriptor inspection tests, if available, that assert kernel `core_ranges` match active/dummy expectations.

7. Run verification:
   - Existing ring-joint SDPA unit/nightly target on Blackhole.
   - `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` must pass.
   - Existing T3000 ring-joint attention target.
   - Existing exp-ring-joint SDPA target.
   - Program cache tests to ensure descriptor hash and runtime-arg patching still behave correctly.

## Suggested Change Order

1. Implement helper core-set construction and descriptor range conversion.
2. Fix `ring_joint_sdpa_program_factory.cpp` without enabling new mcast gap behavior yet.
3. Narrow fused-op signaling to reader cores and run non-mcast/regular tests.
4. Add dummy reader support for mcast rectangles.
5. Apply the same scheduling model to `exp_ring_joint_sdpa_program_factory.cpp`.
6. Add tests and update performance accounting to report actual compute cores, not configured grid size.

## Risks

- Fused all-gather signaling and reader scheduling must stay in lockstep, or readers can wait forever.
- Dummy mcast participants must not consume or produce Q work, output writes, or compute CB traffic.
- Program cache hash behavior may change when core ranges are narrowed; cache-hit runtime patching should be checked.
- Fabric writer narrowing in `exp_ring_joint` must preserve MUX teardown and termination-master semantics.

## Acceptance Criteria

- Inactive non-mcast cores have no reader, writer, or compute kernels.
- Inactive mcast rectangle cores have only the minimal dummy reader/protocol participation needed.
- Fused-op signal targets equal the reader kernel participant set.
- No correctness regressions in ring-joint and exp-ring-joint SDPA tests.
- `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` passes.
- Performance reports use actual compute core count.
