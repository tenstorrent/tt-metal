# Runtime early-exit implementation plan

> Original approved design/implementation record. The current branch follows
> the newer parent PR APIs and construction-time geometry; see
> [rebase integration](KDA_PADDING_REBASE.md) for current names and validation.

## Goal and constraints

Implement the approved [design](KDA_EARLY_EXIT_DESIGN.md) on PR #56632 head
`84ae832f178`. Fixed physical shapes and placement, device start/end metadata,
32-token alignment, immutable input carries, and one capture are required.
Projection/convolution arithmetic reduction and arbitrary-token padding are
outside scope. Performance parity remains a measurement question.

## Existing boundaries

Existing: `tt/kda/kda.py` owns layer orchestration; `tt/kda/recurrence.py` owns
grouping and distributed prefix; `tt/kda/device_chronology.py` owns semantic
device selections. These paths are relative to `models/demos/deepseek_v3_d_p/`.

Existing: native `chronological_topology/chronology.hpp` and
`factory/chronology_binding.hpp` derive and distribute runtime controls.
Preparation has separate reader/compute/writer work counts. Summary and scan
share the recurrent-scan operation. Reduction and exclusive scan communicate
through per-head barriers. Native paths are relative to
`ttnn/cpp/ttnn/operations/experimental/kda/`.

## Proposed changes and data flow

1. Extend shared topology with local valid rows, bounded head groups, valid
   chunks per group, and final-owner selection. Cover the arithmetic with
   hand-verifiable boundary cases before wiring kernels.
2. Thread optional `actual_end` through native operation interfaces, bindings,
   tensor inputs, and chronology binding. Validate metadata representation at
   public boundaries. Derive controls on device on every execution. Preserve
   the omitted-end path.
3. Preparation uses identical valid-work filtering in all three kernels while
   retaining physical tensor addressing and physical work distribution.
4. Summary and scan use valid chunk bounds independently of physical strides.
   Exited groups consume their control record but never wait for chunk inputs.
   Clip segmented reset logic when the valid interval ends before the reset.
5. Reduction/exclusive scan use active-only participant counts and stage
   bounds. Empty local partitions publish only the identity/carry required by
   the fixed SP graph, without chunk computation.
6. Extend chronology selections to choose convolution history at the local
   valid end and the correct final device. Preserve the existing global prefix
   as final state when no valid separated tail remains. Publish the last valid
   group state through the fixed final-state layout when a tail does remain.
7. Layer and recurrence orchestration pass one canonical runtime interval;
   preserve fixed output shapes and existing behavior without an end bound.

## Validation ownership

- Shared topology arithmetic: small explicit expected rank bounds, including
  SP1, split boundary, empty ranks, exact boundaries, and a partial final group.
- Native operation tests: prepared terms, summaries, reduction, group entries,
  and final scan against independent expected results; multiple heads and
  poisoned padding expose stale/incorrect physical indexing.
- Layer tests: valid output and both states versus independent CPU recurrence,
  padded-value invariance, nonzero carries, cache-hit rebinding, and repeated
  changing-start/end trace replay on SP1 and both SP axes.
- Existing full-length offset tests establish compatibility.
- Native build precedes safe-wrapper device tests. Preserve complete logs,
  command, revision, return status, and skipped/unvalidated coverage.
- Rotating paired timing compares full physical, early exit, and trimmed
  execution where feasible, with separate recurrence and layer results.

## Risks and unknowns

Required: readers, compute, writers, and barriers agree on their bounds before
any kernel can exit. Physical group stride cannot be replaced by active count.
Unknown: overhead of runtime controls/identity publication and available build
capacity. Numerical tolerances must account for differing grouping in trimmed
oracles; tail invariance and repeated execution remain exact contracts.
