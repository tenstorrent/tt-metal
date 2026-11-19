# Decode trace retention risk (source review)

No full48 decode capacity limit has been measured yet; the reduced retention probe below has now passed. `batch32_hf_v1.json` failed
during prefill, before any decode record. Its post-prepare DRAM view had
4,169,844,224 free bytes/rank and a 517,866,944-byte largest free block per bank.
The eventual failed **prefill** trace requested 143,278,080 bytes/rank, or
17,915,904 bytes per bank; only 17,688,000 contiguous bytes/bank remained.
These numbers establish the old prefill-retention failure, not a decode limit.

The current `_capture` key is the sorted active-slot tuple. Sorting removes
caller-order permutations but retains separate programs for every distinct
active subset. `_sample` similarly includes the active tuple in its key because
inactive token restoration is compiled for those slots. No trace is evicted.

For the B32 HF test, baseline retirement creates all prefix widths 1..32. After
the reversed migration, retirement creates suffix widths 1..31; the width32
subset is shared. Therefore the planned test retains **63 model plus63 sampling
signatures**, alongside up to six canonical prefill signatures. Its sum of
logical decode widths is 1,024. The general fixed-slot API has up to
2^32−1 nonempty subsets, so sorting alone cannot bound long-lived serving trace
memory. This is a source-level scalability risk, not a measured maximum.

For scale only, the separate full48 B1 `full_accuracy_v2.json` grew by
1,085,865,984 bytes/rank from prepared to final with nine captured traces. That
delta includes every retained allocation effect and different prefill shapes;
it must not be multiplied by63 as an estimate of decode trace bytes.

`tests/probe_decode_trace_memory.py` ran successfully as `decode_trace_memory_reduced_v1.json` in56.599s, using the prior native LM head. This reduced real0/1/3
probe It keeps width1/2/4/8/16/32 model and common-sampling traces alive, records
allocator deltas separately, then verifies reversed row order reuses each trace
without additional retained growth. It uses shared BF16 K/V, capacity128,
tracking enabled, and normal generator-wide warmup; it does not execute
prefill. The module docstring contains the bounded serialized command. CPU
compile and formatting checks passed. Every reversed-order replay reused the
trace with zero retained growth. This is not a capacity bracketing test.

| Width | Model retained bytes/rank | Sampling retained bytes/rank |
| ---: | ---: | ---: |
| 1 | 6,750,208 | 327,680 |
| 2 | 12,451,840 | 327,680 |
| 4 | 23,658,496 | 327,680 |
| 8 | 46,071,808 | 327,680 |
| 16 | 90,898,432 | 327,680 |
| 32 | 180,420,608 | 327,680 |

Six widths retain362,217,472 bytes/rank. A least-squares planning fit is
`M(w)=1,235,076+5,600,650*w` bytes, maximum residual85,519 bytes across measured
widths. It estimates5,833,518,985 bytes for63 model+sampling signatures and
3,007,151,266 bytes for32 canonical exact-width signatures **even at reduced
depth**. These are interpolated sums, not full48 measurements. Exact-width
compaction alone is therefore not a credible complete memory solution.

If measured width costs approximately follow `M(w)=a+b*w`, the current63-key
test's model cost estimate is `63*a+1024*b`; sampling contributes its separately
measured63-signature cost. A canonical exact-width design would retain
`32*a+528*b`. A six-width bucket design would retain approximately
`6*a+63*b`. These are planning formulas only: fixed LM-head/sampling costs,
layer-kind ratios, trace alignment, temporary high-water overlap, and allocator
fragmentation prevent treating a reduced-layer fit or simple16× scaling as
full48 physical-fit proof.

If actual retention fails, the smallest defensible first design is canonical
**exact width**, not dummy execution:

1. At scheduler membership changes, maintain an external-slot to resident-slot
   bijection and use the existing device cycle-copy machinery to compact active
   requests to physical slots0..N−1 while retaining dormant requests elsewhere.
   Move all private recurrence, conv, PLE history, QSA index/position/page tables,
   token feedback, seeds/counters and supported sampler state. Shared K/V pages
   stay in their global pool; their page tables move with the request.
2. Key model and sampler traces by exact active count and semantic sampler mode.
   Translate public input/output/cache ownership through the persistent mapping.
   Preserve n1 FP32 reduction and n>1 native BF16 reduction exactly. Keep logits
   snapshots and delayed outputs tied to submission-time logical slots.
3. Reuse the mapping unchanged for steady-state replay. Membership transitions
   may perform explicit device state copies; they must not copy cache data to
   host, rebuild per-token positions, or reset dormant seeds/history. A host
   metadata bijection is not host cache ownership. Existing public `slot_remap`
   semantics must compose with the internal map rather than accidentally apply
   two migrations.

This bounds subset-dependent traces to32 widths but may still exceed memory;
it is a proposal, not implemented or validated. Avoid adding32 full private
scratch caches: existing cycle scratch is sufficient for permutations.

Width buckets need substantially more work and are not a harmless extension of
the above. Padding only the MoE path would not bound captured attention/cache
addresses or varying attention command counts. Padding the whole decode path
would execute dummy recurrent and QSA updates; setting a dummy host position to
−1 does not mask these low-level writes. Safe buckets require traced active
masks on every cache/history/position update, dedicated nonaliasing dummy pages,
inactive logits and feedback preservation, seed no-advance, and no position
overflow after arbitrarily many replays. Physical dummy rows also change expert
grouping and require numerical controls. In particular width1 must stay its own
FP32 path. No bucket or dummy-cache implementation should be selected merely
from this source review.


## Native rebinding API review

Legacy TTNN trace exposes begin/end/execute/release only. Metal
`experimental/command_list.hpp` does provide persistent CQ-ordered
`CommandList::update_args(CmdListArgPatch)`, but the TTNN command-list wrapper
(`ttnn/api/ttnn/command_list.hpp` and nanobind implementation) exposes neither
parameters nor update_args. It accepts only Metal2 ProgramSpec operations and
prohibits coexistence with legacy traces. Current generic ProgramDescriptor and
native model operations make this a substantial runtime port, not a small
model-side trace-rebinding shortcut. No unsupported trace-buffer patching is
proposed.

## Factored trace candidate

A source-feasible alternative retains each layer/physical-slot attention segment
once, and each layer/width grouped MoE segment once. Fold the previous layer's
post-MoE residual update into the next layer's attention segment. Persistent
per-slot residual/mixed/injection/block buffers are allocated before all captures
and reused across layers. An ordered same-CQ sequence preserves arithmetic and
private cache bindings, without per-token cache migration. Prefix-width proof
can use fixed input ordering; arbitrary active sets subsequently need either
transition-only canonical mapping or persistent device gather/scatter indices.

This costs approximately `48*(N+1)+N` trace submissions per full48 token, versus
one model trace:1,616 atB32 including32 terminal segments (plus sampling). A
vector-enqueue native wrapper could remove Python crossings but not the CQ
commands; no existing TTNN list-of-trace-IDs API was found. Keep the original
whole-model B1 path. The new reduced proof must measure exact logits/cache
state, command-storage growth, and clean end-to-end submission cost before this
can be selected. It is more localized than masking every attention/recurrence
write in dummy-width buckets, but is not yet a proven performance-preserving
solution.

## Measured factored v1 and indexed v2 candidate

`factored_decode_tracked_v1.json` passes all nine width/step comparisons at
widths1/2/32: logits, private/logical-KV state, and existing common-sampler tokens
are bitwise equal. It retains134 factored segments and186,777,600 bytes/rank.
The unchanged v1 source is `tests/probe_factored_decode_traces.py`; its artifact
contains the model/generator/decoder source hashes. Capture deltas include bank
alignment; they are allocator observations, not raw command-payload lengths.

| Segment | Count | Bytes each/rank | Total bytes/rank |
| --- | ---: | ---: | ---: |
| First GDN attention, actual layer0 | 32 | 1,179,648 | 37,748,736 |
| PLE GDN attention, actual layer1 | 32 | 2,490,368 | 79,691,776 |
| QSA attention, actual layer3 | 32 | 1,703,936 | 54,525,952 |
| MoE width2, each of three layers | 3 | 458,752 | 1,376,256 |
| MoE width32, each of three layers | 3 | 983,040 | 2,949,120 |
| Terminal | 32 | 327,680 | 10,485,760 |

The clean v1 parent run reports original/factored token-out ms of9.5197/9.5175
atB1 (identical dispatch),18.2804/18.4484 atB2 (+0.92%), and
248.3538/251.4865 atB32 (+1.26%). These are reduced-model measurements, not
full-model acceptance or a claim that the candidate is faster.

The pinned HF config at snapshot812d96e10a0dd82b69d749e68693698cf6c36daa has
35 non-PLE GDN layers, one PLE GDN layer, and12 QSA layers. A planning scenario
substituting the measured first-GDN segment for all35 non-PLE layers gives
2,055,208,960 bytes/rank for fixed-slot attention. Linear interpolation of the
measured MoE width2/32 endpoints over widths2..32 gives1,072,693,248 bytes/rank
across48 layers. Adding terminal gives3,138,387,968 bytes/rank (2.92285GiB).
This excludes baseline/other B1 traces, sampling, prefill traces, persistent
workspaces, weights/cache and all other allocations. It is **not a bound or fit
proof**: later ordinary GDN folds a previous residual update instead of the
first embedding, unmeasured widths may cross command-size boundaries, and the
v2 copy schedule differs. Even monotonicity of the endpoint interpolation has
not been established. No context or batch capability changes follow from it.

`tests/probe_factored_decode_slots.py` is the independent v2 candidate;
`tests/factored_slot_copy.py` contains its dataflow-only indexed copy. Native
`ttnn::scatter` supports integer indices but transforms tiled inputs through
row-major storage and allocates its result; it cannot directly update32
independent fixed-address row buffers. The candidate reads one persistent
UINT32[32] index payload, with32 static row-buffer addresses in kernel runtime
arguments. It gathers each selected BF16 tile row into an exact-width packed
activation and scatters the unchanged MoE output to the same fixed slots.
32-byte face writes preserve inactive rows and use no floating-point arithmetic.
Source reads complete before use; gather writes finish before scratch reuse;
scatter writes finish before reading the next tile. One core owns each feature
tile, so no two cores write the same destination tile. Same-CQ segment ordering
preserves layer dependencies. The kernel does not touch request caches.

The sorted unique slot payload changes only when membership changes. All
attention/terminal segments remain tied to their original cache addresses.
V2 captures widths2..32 (222 segments total), checks nonprefix and irregular
slots plus membership transitions without reset, and asserts zero index uploads
inside clean token windows. Redundant token/word work on non-PLE intermediate
layers is removed. Primary slot0 B1 retains the original whole-model trace.
CPU checks cover face offsets, untouched bytes, invalid indices, arbitrary-slot
dispatch, and transition-only metadata; actual kernel compilation, trace
tracking, exact numerical behavior and performance require the serialized TT
run. V2 hashes its own source and helper as well as production sources.

Remaining production issues are explicit: existing sampler capture keys still
contain active subsets (a separate dynamic feedback-copy diagnostic is being
prepared); single-active nonzero slots still use whole-model captures. Keeping
32 full48 B1 captures may consume significant additional memory. A factored
width1 route for nonprimary slots could preserve the original n1 FP32 MoE
arithmetic without cache migration, but is not implemented or accepted here.
No dtype, fidelity, residual layout, expert ownership, or sampler algorithm is
changed by the indexed-copy proposal.
