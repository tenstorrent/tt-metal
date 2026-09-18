# Runtime aligned-tail early exit for KDA

> Original approved design/implementation record. The current branch follows
> the newer parent PR APIs and construction-time geometry; see
> [rebase integration](KDA_PADDING_REBASE.md) for current names and validation.

## Problem and base

Build the fourth padding prototype on PR #56632, published head
`84ae832f1787fc1b8495e9a1b40f8090601a658d`. Preserve the PR's runtime offsets,
physical placement, grouping, and one-capture execution while skipping padded
recurrence chunks and groups. This supersedes the main-based draft.

The approved algorithm is early exit, not masked recurrence arithmetic.
Padding and valid boundaries are aligned to 32 tokens; no partial chunks.

## Existing contracts (observed)

- `tt/kda/kda.py::ttKDA.forward` accepts a caller-owned replicated UINT32
  row-major `actual_start` scalar. Its address stays alive through capture and
  replay; its contents may change. No host readback validates scalar values.
- `tests/kda/layer/test_dynamic_trace.py` exercises changing offsets and
  nonzero continuation states in one capture on SP1 and both axes of SP2/SP4.
- `tt/tt_prefill_runtime.py::prefill_chunk` defines absolute exclusive bounds
  `[actual_start, actual_end)`; their difference is the number of real tokens.
- `tt/kda/device_chronology.py` owns semantic selections for convolution
  history, affine transforms, and entry/final recurrent state.
- `tt/kda/recurrence.py::_scan_sp_grouped_chunks` performs one summary pass,
  partition reduction/SP prefix, group exclusive scan, and recurrent scan.
- `tests/kda/chronology_oracle.py` documents the boundary device's physical
  head-then-tail row order and chronological separation by the other ranks.

These Python paths are relative to `models/demos/deepseek_v3_d_p/`.
Native code references below are relative to
`ttnn/cpp/ttnn/operations/experimental/kda/`.

- `chronological_topology/chronology.hpp` derives rank-local topology from
  runtime start and static mesh geometry.
- `factory/chronology_binding.hpp` binds runtime metadata and transports derived
  controls to cooperating kernels.
- `reduce_affine_transforms/device/kernels/dataflow/reader_writer_reduce_affine_transforms.cpp`
  already avoids inactive head-summary arithmetic, but every physical group
  still participates in each stage barrier. It is not yet full group early exit.

## Proposed interface and guarantees

Extend the layer with an optional device `actual_end` scalar, with the same
representation, replication, ownership, and captured-address lifetime as
`actual_start`. Omission retains the PR's full physical chunk behavior.
For padding-enabled calls, both scalars remain device-resident and can change
between replays of one capture. Cache identity includes metadata structure,
not its runtime contents; rebinding must refresh input addresses.

Caller preconditions follow the existing no-readback contract:

- Both bounds are nonnegative, UINT32-representable, and 32-aligned.
- `0 < actual_end - actual_start <= SP * local_rows`, without integer overflow.
- Every rank observes the same metadata for a launch/replay.

The global empty interval is excluded from this prototype. Empty local
partitions are required. Input carries remain immutable. Returned convolution
and recurrent states represent exactly the last valid token. Output retains
the PR's fixed physical shape and placement; only valid rows are observable.
Padded output values are unspecified. Neither valid rows nor carries depend on
padded input values.

## Proposed validity derivation

Let `L = actual_end - actual_start`, `C = local_rows`, `P = SP`, and let
`h = C - actual_start % C` be the first chronological segment length for SP>1.
Let `j` be the rank's chronological distance from the boundary device.

- SP1: local valid rows are `L`; no separated head/tail chronology.
- Boundary rank (`j=0`): valid head rows are `min(L,h)`; valid tail rows are
  `clamp(L - (h + (P-1)*C), 0, C-h)`.
- Other ranks: valid rows are `clamp(L - (h + (j-1)*C), 0, C)`.

The boundary rank stores its head followed by its tail, so valid local rows
remain a prefix. A nonempty tail implies a completely valid head and all
intervening ranks. Keep these semantic bounds in the native chronology owner;
do not expose independent per-stage chunk counts to callers.

## Proposed stage behavior and synchronization

| Stage | Behavior |
| --- | --- |
| Preparation | Read, compute, and write only valid local chunks; retain physical strides. All cooperating kernels agree on the exact count. |
| Group summaries | Summarize only valid chunks in each group, preserving existing head/tail segmentation; empty groups perform no recurrence arithmetic. |
| Reduction | Compose active head summaries only. No sends, waits, or barrier counts may require exited groups. |
| SP prefix | Keep collective participation fixed. Empty local partitions contribute an identity transition and preserve the incoming carry. |
| Exclusive scan | Process active groups only, retaining the tail reset where a valid tail exists. Physical worker addressing remains unchanged. |
| Final scan | Process valid chunks only, with the PR's boundary seed switch if reached. Publish the last valid state. |

Skipped internal slots are unspecified and must not enter valid arithmetic.
Where a fixed collective requires a tensor contribution from an empty rank,
publish the small identity/carry result required by that boundary; do not run
the padded recurrence to manufacture it. Inter-chip collectives remain uniform.

Convolution state selection must move from the physical end to the logical
valid end, including ends inside a group or on another rank. Convolution and
projection compute optimization is outside this recurrence prototype.

## Invariants and acceptance

- Full-length calls preserve the existing offset semantics and topology.
- Exactly one summary pass, SP prefix, group prefix, and final scan; no host
  scalar readback, input rotation, tensor cropping, or identity-mask arithmetic
  over padded chunks.
- Test nonzero input carries, multiple heads, one valid chunk, partial groups,
  exact group/rank boundaries, entirely padded ranks, and a valid boundary tail.
- Test SP1 plus both SP axes; changing starts and ends in one trace, including
  transitions between empty/nonempty ranks and split/non-split boundaries.
- Check valid output and both carries against the independent CPU reference;
  check zero/random padding invariance, repeated replay determinism, cache-hit
  input rebinding, and immutable input state.
- Directly test preparation, summaries, reduction, exclusive scan, and final
  scan so a layer path cannot hide a stage that was not exercised.
- Build host API/binding changes, then use serialized safe pytest. Preserve
  exact revision, commands, logs, outcomes, and unsupported/unvalidated cases.

## Performance evaluation

Measure recurrence and whole-layer latency separately. Compare full physical
execution, the early-exit implementation, and physically trimmed execution
where supported. Rotate case order and retain paired raw samples.

Inferred: recurrence arithmetic should scale with valid chunks, but fixed
placement, metadata delivery, collective participation, and launch overhead
remain. Whole-layer projection/convolution/norm work still scales with physical
length. No claim of zero regression or trimmed-run parity is established.

## Decisions and remaining risks

- Proposed: reuse absolute `actual_end` rather than introduce a separate length
  convention. The existing runtime defines the interval, but PR #56632 itself
  does not implement padding.
- Proposed: retain fixed output shape and leave invalid rows unspecified,
  consistent with one-capture execution.
- Proposed: support empty local partitions but reject a globally empty interval
  as a caller precondition for this prototype.
- Unknown: minimum control/identity-publication overhead and whether generic
  conversion of skipped summary slots should be eliminated or initialized.
- Inferred: fixed physical grouping can differ numerically from trimmed-shape
  grouping. Require reference accuracy and padding invariance; do not assume
  bit identity across different grouping geometries.
- Base is the published PR SHA. Separate in-progress local PR-review changes
  are not part of this checkout and must not be overwritten or implicitly used.

## Approval status

All contracts were approved by the user on 2026-09-17. Implementation and
validation follow `KDA_EARLY_EXIT_DEV.md`.
