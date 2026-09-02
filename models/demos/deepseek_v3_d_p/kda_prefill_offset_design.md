# KDA prefill over MLA offset-rotated tokens

## Status and approval gate

**Status:** Proposed design; not approved for implementation.

This document specifies KDA behavior for a full 5120-token prefill interval
whose input is already in MLA's offset-rotated, block-cyclic SP layout. It does
not constitute an implementation plan. Implementation planning must wait until
the decisions in [Decisions requiring approval](#decisions-requiring-approval)
are approved.

The detailed MLA layout investigation, including the producer integration gap,
is in [MLA prefill at a non-chunk-aligned offset](mla_prefill_offset_report.md).

## Problem

**Observed:** On Galaxy SP8 x TP4, a full 5120-token activation has 640 token
rows per SP rank. At `actual_start=960`, MLA assigns the rows as follows:

| Physical SP rank | Local rows | Absolute token positions |
| ---: | ---: | --- |
| 0 | `0..639` | `5120..5759` |
| 1 | `0..319` | `960..1279` (head) |
| 1 | `320..639` | `5760..6079` (tail) |
| 2 | `0..639` | `1280..1919` |
| 3 | `0..639` | `1920..2559` |
| 4 | `0..639` | `2560..3199` |
| 5 | `0..639` | `3200..3839` |
| 6 | `0..639` | `3840..4479` |
| 7 | `0..639` | `4480..5119` |

The chronological order is therefore:

```text
960                                                                  6080
 | SP1 head | SP2 | SP3 | SP4 | SP5 | SP6 | SP7 | SP0 | SP1 tail |
 |   320     | 640 | 640 | 640 | 640 | 640 | 640 | 640 |   320    |
```

Every row is present on four TP devices with a different hidden-dimension
shard. TP does not change token order.

**Observed:** Current KDA assumes that chronological SP partitions are ordered
by physical rank. Its convolution exchange gives rank 0 the input carry and
rank `r>0` the tail from rank `r-1`
([`tt/kda/convolution.py`](tt/kda/convolution.py#L16-L22)), while its
distributed recurrent prefix composes summaries in `range(sp_size)`
([`tt/kda/recurrence.py`](tt/kda/recurrence.py#L217-L296)).

**Inferred:** Feeding MLA's offset-rotated tensor directly to current KDA is
incorrect whenever `actual_start mod 5120 != 0`. A rank rotation is required
when the start is a multiple of 640; otherwise one physical rank also has to be
split into the first and last causal segments.

### Goals

- Make a full 5120-token KDA prefill mathematically equivalent to processing
  `[actual_start, actual_start + 5120)` in natural chronological order.
- Support the wrap on any of the eight SP ranks and at any tile-aligned row
  within that rank.
- Preserve MLA's physical input and output row placement; KDA must not impose a
  different activation layout on adjacent layers.
- Preserve the existing caller-owned `KdaState` continuation contract.
- Avoid exchanging the full hidden activation solely to restore chronological
  SP partitions.

### Non-goals

- Fixing the producer path that currently does not construct MLA's required
  offset-rotated input. That is a separate integration concern documented in
  the supporting report.
- Supporting non-tile-aligned offsets. MLA already requires 32-token alignment
  ([`tt/mla/mla.py`](tt/mla/mla.py#L887-L896)), and KDA recurrence chunks are
  one TT tile ([`tt/kda/config.py`](tt/kda/config.py#L12-L19)).
- Defining arbitrary partial final-prefill behavior in the first change. This
  design's required interval contains 5120 real tokens; padding semantics need
  a separate extension.
- Changing TP sharding, model mathematics, weights, or decode behavior.

## Current system

### Layout

For global chunk size `G`, SP size `P`, local rows `C=G/P`, and absolute start
`S`, define:

```text
boundary rank b = floor(S / C) mod P
split row     o = S mod C
head length   h = C - o
tail length   t = o
```

If `o=0`, there is no split and chronological SP order is
`b, b+1, ..., P-1, 0, ..., b-1`. If `o>0`, there are `P+1` logical segments:

```text
(b, rows 0:h), (b+1, all rows), ..., (b-1, all rows), (b, rows h:C)
```

Ranks are interpreted modulo `P`. For `S=960`, `P=8`, and `C=640`, this gives
`b=1`, `o=320`, `h=320`, and `t=320`. The same equations are independently
implemented by MLA's Python position oracle and cache writer
([`tt/mla/utils.py`](tt/mla/utils.py#L83-L111),
[`writer_update_padded_kv_cache.cpp`](../../../ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp#L100-L122)).

### KDA execution and state

**Observed:** `ttKDA.forward` currently accepts only `(hidden_states, state)`.
It projects the complete local sequence, performs causal QKV convolution,
computes gates, runs the recurrent scan, applies the gated RMSNorm, and projects
the result ([`tt/kda/kda.py`](tt/kda/kda.py#L379-L408)).

**Observed:** The convolution carry and recurrent state are explicit,
caller-owned replacement values. The recurrent state is FP32 tile-layout DRAM;
the convolution state is the last `kernel_size-1` projected-QKV rows in BF16
row-major DRAM ([`tt/kda/kda.py`](tt/kda/kda.py#L168-L187)). Both are replicated
across SP and retain their current TP distribution
([`tt/kda/convolution.py`](tt/kda/convolution.py#L16-L22),
[`tests/kda/layer/test_distributed.py`](tests/kda/layer/test_distributed.py#L112-L141)).

**Observed:** Distributed recurrence already represents each partition as an
affine summary, gathers all summaries, derives each partition's entry state,
and obtains the global final state before local chunk scans finish
([`tt/kda/recurrence.py`](tt/kda/recurrence.py#L217-L296),
[`tt/kda/recurrence.py`](tt/kda/recurrence.py#L317-L375)).

**Observed:** The production K3 configuration uses 32-token recurrence chunks,
20 local chunks for each 640-token SP partition, and grouped scanning for
`T=5120` ([`tt/kda/config.py`](tt/kda/config.py#L27-L36),
[`tt/kda/config.py`](tt/kda/config.py#L67-L77)).

## Proposed design

### Central abstraction: ordered logical segments

**Proposed:** Derive an ephemeral logical-segment topology from
`actual_start`, `P`, and `C`. Do not create a new persistent layout or state
object. Each segment contains only:

- its physical SP owner;
- its half-open local-row range; and
- its position in chronological order.

The topology has `P` segments when `o=0` and `P+1` when `o>0`. The boundary
rank owns both the first `head` and last `tail` segments. All other ranks own
one full segment. Because `actual_start` and `C` are tile-aligned, every
non-empty segment contains an integral number of KDA chunks.

```text
physical storage                         causal topology

SP0 [-------------- full --------------]       +-------------------+
SP1 [---- head ----|---- tail ----------]  ---> head -> SP2 -> ...  |
SP2 [-------------- full --------------]       ^                   v
...                                             +-- tail <- SP0 <---+

Output rows return to the same physical slices; only causal carry order changes.
```

This topology is the single canonical source for both convolution predecessor
routing and recurrent-summary composition. The two stages must not derive
independent notions of order.

### Data and control flow

1. **Validate and derive topology.** `ttKDA.forward` receives
   `actual_start`. All SP ranks use the same scalar. The layer derives `b`,
   `o`, and the ordered segments; callers do not pass a precomputed rank list.
2. **Run token-local projection once.** Input projection and auxiliary gate
   computation operate on all 640 physical rows. These operations are
   token-local and do not need chronological reshaping.
3. **Route convolution halos by logical predecessor.** Collect the last
   `kernel_size-1` projected-QKV rows needed by each logical segment. The head
   receives the caller's input convolution state. Every subsequent segment
   receives its chronological predecessor's raw projected-QKV tail. The final
   replacement convolution state comes from the logical tail segment.
4. **Convolve logical segments.** Full ranks convolve one 640-row segment. The
   boundary rank convolves head and tail separately using their distinct entry
   carries, then writes both results back to their original local-row slices.
   Head and tail must never be convolved as adjacent local rows.
5. **Summarize recurrence per logical segment.** Produce separate affine
   summaries for boundary head and tail and one for every full segment. An
   inactive or padded execution lane, if required by uniform mesh kernels,
   contributes the affine identity; it must not execute a synthetic zero token.
6. **Compose in logical order.** Gather the segment summaries and compose them
   in topology order. Produce an entry recurrent state for every segment and
   the final replacement recurrent state after the last segment.
7. **Scan and restore physical placement.** Scan each segment from its computed
   entry state. The boundary rank scans head and tail independently. Place the
   outputs back into their original physical row ranges, then run the existing
   token-local RMSNorm and output projection over the full local buffer.
8. **Return canonical state.** Return the unchanged output layout and a normal
   `KdaState` whose two carries describe the stream through
   `actual_start+5120-1`.

The recurrent tail scan depends on the prefix-composed entry state, not on the
completion of every preceding local scan. This refines the initial two-phase
idea: the tail remains causally last, but affine summaries avoid an unnecessary
full-scan synchronization. Convolution has an analogous optimization because
its entry halo depends on predecessor projected QKV, which exists immediately
after token-local projection.

### Galaxy SP8 x TP4, offset 960

The topology and carry edges are:

```text
input state
    |
    v
SP1 rows 0:320       tokens  960:1280   (head)
    |
    v
SP2 rows 0:640       tokens 1280:1920
    |
   ...
    |
    v
SP7 rows 0:640       tokens 4480:5120
    |
    v
SP0 rows 0:640       tokens 5120:5760
    |
    v
SP1 rows 320:640     tokens 5760:6080   (tail)
    |
    +--> replacement recurrent state and convolution state
```

The same topology is repeated independently in each TP column. Collective
payloads remain sharded by TP exactly as today.

## Contracts

### KDA forward boundary

**Proposed input:** `actual_start: int` is added to the KDA prefill boundary.
Its exact Python signature and default are an implementation-plan decision;
there must be no silent assumption of zero once the offset-aware path is used.

Preconditions:

- `hidden_states` is the MLA-compatible offset-rotated, block-cyclic tensor for
  `[actual_start, actual_start+5120)`; KDA does not reorder natural host input.
- Batch size is one, every SP rank has 640 real rows, and all existing dtype,
  memory, TP, and state-shape requirements continue to hold.
- `actual_start >= 0` and `actual_start mod 32 == 0`.
- Every SP rank observes the same `actual_start`; disagreement is caller error.
- Input `state` is the canonical KDA state immediately before `actual_start`.

Postconditions:

- Each output local row corresponds to the same absolute token as its input
  local row.
- Output values and replacement state are equivalent, within the accepted
  numerical threshold, to natural-order KDA over the same interval.
- Input state remains read-only, matching the current replacement-state
  contract ([`tt/kda/kda.py`](tt/kda/kda.py#L384-L389)).
- Replacement recurrent and convolution states are replicated across SP and
  preserve the current TP distribution.

Failure semantics:

- Invalid alignment, negative offsets, unsupported local length, or inconsistent
  static mesh geometry must fail before launching order-sensitive work.
- The layer must not guess whether an input is natural or block-cyclic. Layout
  is a caller contract; debug-only validation may check metadata but cannot
  infer row identity from tensor values.

### Segment-order contract

- Exactly one segment contains every absolute position in the requested
  interval; none is duplicated or omitted.
- The first segment consumes caller state, every other segment consumes its
  immediate logical predecessor's carry, and the last segment produces
  replacement state.
- When `o=0`, the boundary rank is not split and no zero-length tail exists.
- Head and tail may belong to any physical SP rank.
- Convolution and recurrence use the identical ordered segment list.

### Non-functional contract

- The production path must not introduce a full-hidden-width gather or
  all-to-all solely for offset correction.
- Offset support must remain traceable and program-cache safe. Supported
  offsets may use a bounded set of tile-aligned segment shapes, but must not
  trigger unbounded compilation keyed by absolute position.
- No performance improvement or acceptable regression is claimed until Galaxy
  SP8 x TP4 measurements exist.

## Invariants and acceptance criteria

### Invariants

1. Physical row identity is stable across KDA: MLA and KDA name the same token
   in every row.
2. Causal state flows by chronological segment order, never raw SP-rank order.
3. Boundary head and tail are causally non-adjacent when `o>0`.
4. Each real token updates convolution and recurrence exactly once.
5. Returned state represents the chronological end of the interval and is
   suitable for the next prefill call.
6. Offset zero preserves current behavior and numerical results.

### Required correctness tests

- A host topology test exhausts all 160 tile-aligned starts modulo 5120 and
  asserts coverage, uniqueness, row ranges, boundary ownership, and logical
  order for all eight possible boundary ranks.
- Distributed device tests compare inverse-rotated output, recurrent state,
  and convolution state with the natural-order PyTorch reference for:
  `o=0`; smallest split (`o=32`); midpoint (`o=320`); largest split
  (`o=608`); and at least one split on every SP rank.
- The production Galaxy SP8 x TP4, `T=5120`, `actual_start=960` case matches
  the real-weight reference with PCC at least `0.9995`, the existing production
  threshold ([`tests/kda/perf/test_layer_perf.py`](tests/kda/perf/test_layer_perf.py#L44-L48)).
- A segmented-continuation test proves that replacement state can feed the next
  call and matches one-shot natural execution. Existing zero-offset segmented
  coverage is the baseline
  ([`tests/kda/layer/test_distributed.py`](tests/kda/layer/test_distributed.py#L144-L253)).
- Tests verify convolution carry and recurrent state independently, not only
  final projected output.
- Trace replay covers at least two different boundary ranks and two different
  split rows without stale offset capture or program-cache corruption.

### Required performance evidence

- Measure warm trace wall time and device-program breakdown on Galaxy SP8 x
  TP4 for offsets `0`, `32`, `320`, `608`, and `960`.
- Report the added collective bytes and program count relative to offset zero.
- Compare the proposed segment-aware design with the simple sequential-tail
  baseline before accepting its added control complexity.
- Set a regression budget only after the first measurements; the current Galaxy
  reference is intentionally unset
  ([`tests/kda/perf/test_layer_perf.py`](tests/kda/perf/test_layer_perf.py#L51-L60)).

## Decisions and trade-offs

### Proposed: segment-aware causal execution

Use the logical-segment topology described above, preserve the physical MLA
layout, exchange only causal halos and affine summaries, and scan boundary head
and tail separately.

Why preferred:

- it makes chronological order explicit at the two stateful boundaries;
- it avoids two full-activation layout conversions per KDA use;
- it generalizes uniformly to any boundary rank and tile-aligned split; and
- it reuses KDA's existing affine-summary property instead of serializing the
  tail behind completion of all preceding local scans.

Cost: recurrence orchestration and convolution carry exchange must represent
two segments on one rank, while current code represents exactly one partition
per rank. Kernel shape and trace behavior need prototyping and Galaxy evidence.

### Alternative A: sequential tail after the existing workflow

Run the boundary head plus all intervening ranks first, obtain final state, then
run the boundary tail.

This is mathematically valid only if the first phase also rotates physical rank
order and both convolution and recurrence exclude the boundary tail. It is a
useful correctness baseline and may require less initial kernel work. However,
it serializes up to 608 tail tokens after the main scan and makes the boundary
rank execute twice on the critical path. The proposed affine-prefix approach
retains its causal decomposition without the avoidable scan-completion
dependency.

### Alternative B: reshard into non-wrapping temporal partitions

Exchange the incoming hidden rows so physical ranks hold consecutive natural
640-token partitions, run current KDA unchanged, then exchange output rows back
to MLA layout.

This has the simplest KDA reasoning and is valuable as a bring-up oracle. It
also moves the full hidden activation twice and adds synchronization at every
KDA/MLA layout boundary. Whether that is actually too expensive is
**Unknown** until measured, but it conflicts with the goal of limiting offset
handling to small state/summary payloads.

### Alternative C: logical rank rotation only

Rotate the rank order used by existing carry collectives. This completely
solves offsets with `o=0`, including `actual_start=640`, but cannot represent
the same physical rank at both ends when `o>0`. It is therefore a useful fast
path, not a complete design.

### Alternative D: gather, run locally, scatter

Gather all 5120 tokens to one SP rank, execute natural-order KDA, and scatter
the result. This is a straightforward diagnostic oracle but sacrifices SP
compute parallelism and moves the largest payload. It is rejected for
production.

### Decisions requiring approval

| Decision | Proposal | Material alternative |
| --- | --- | --- |
| Physical layout | Preserve MLA row placement through KDA | Full reshard before and after KDA |
| Causal model | One ordered list of `P` or `P+1` logical segments | Special-case sequential second pass |
| Tail scheduling | Derive tail entry from affine prefix; do not wait for all local scans | Simpler serialized tail baseline |
| API input | Add absolute global `actual_start` at KDA prefill boundary | Pass precomputed boundary/split metadata |
| Initial scope | Full 5120 real-token intervals, tile-aligned starts | Include partial/padded final chunks now |
| Performance gate | Measure first, then approve a budget | Choose a speculative regression threshold |

## Blast radius

Expected affected areas:

- `tt/kda/kda.py`: forward boundary, topology derivation, split orchestration,
  and output placement;
- `tt/kda/convolution.py`: logical predecessor halo routing and final carry;
- `tt/kda/recurrence.py`: multiple logical summaries/entry states on the
  boundary rank and logical-order prefix composition;
- KDA layer/component/contract tests and Galaxy performance coverage;
- the model call site that already owns `actual_start`.

Expected unchanged areas:

- MLA cache mapping and attention;
- checkpoint weights and reference KDA equations;
- `KdaState` public representation and ownership;
- TP collectives and output hidden-dimension sharding;
- decode.

## Risks, assumptions, and unknowns

- **Assumed:** Every KDA invocation receives the same MLA-compatible physical
  row layout as the surrounding transformer. The known producer mismatch must
  be resolved before end-to-end production validation.
- **Assumed:** A full production call always has 640 real rows on every SP rank.
- **Risk:** Current TTNN mesh operations generally present uniform local shapes.
  Head and tail have complementary variable lengths, so an implementation may
  need bounded shape variants or identity-masked lanes. Padding with ordinary
  zero tokens is not semantically safe because they can still update KDA state.
- **Risk:** Splitting boundary convolution and recurrence can add launches and
  reduce occupancy even without a large collective.
- **Risk:** BF16 transport of additional affine summaries can change numerical
  error relative to the current one-summary-per-rank prefix.
- **Unknown:** Whether one segment-aware collective or two phase-specific
  collectives gives the best Galaxy latency and program-cache behavior.
- **Unknown:** Whether offset should be a runtime tensor/scalar or a trace-fixed
  argument. Absolute position must not create an unbounded program-cache key;
  only `actual_start mod 5120` affects topology.
- **Unknown:** The required behavior for partial/padded final chunks. Adding it
  may introduce empty segments and per-rank real-token counts and should be
  specified before extending this contract.

## Evidence summary

- MLA position equations and the offset-960 visualization:
  [supporting report](mla_prefill_offset_report.md).
- Current forward pipeline and state contract:
  [`tt/kda/kda.py`](tt/kda/kda.py#L168-L216),
  [`tt/kda/kda.py`](tt/kda/kda.py#L379-L408).
- Physical-rank convolution carry ordering:
  [`tt/kda/convolution.py`](tt/kda/convolution.py#L10-L88).
- Physical-rank affine-prefix ordering and local scan separation:
  [`tt/kda/recurrence.py`](tt/kda/recurrence.py#L217-L296),
  [`tt/kda/recurrence.py`](tt/kda/recurrence.py#L317-L375).
- Existing distributed output/state and continuation coverage:
  [`tests/kda/layer/test_distributed.py`](tests/kda/layer/test_distributed.py#L37-L141),
  [`tests/kda/layer/test_distributed.py`](tests/kda/layer/test_distributed.py#L144-L253).
- Existing production dimensions and acceptance threshold:
  [`tests/kda/perf/test_layer_perf.py`](tests/kda/perf/test_layer_perf.py#L44-L60).
