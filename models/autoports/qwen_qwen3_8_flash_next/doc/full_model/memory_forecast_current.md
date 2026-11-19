# Memory forecast: units, measured baselines and remaining constraints

This is source arithmetic and an extrapolation of existing artifacts. It is not
a new allocation measurement, context reduction, B32 acceptance or max-context
execution result. No device was accessed for this audit.

All totals below are **bytes per rank/device**, unless a column explicitly says
per bank. Each rank has eight DRAM banks. Multiply allocator `*_per_bank`
numbers by8 to compare aggregate allocation against a rank's tensor ledger;
do not multiply them by4 mesh ranks. The inherited allocatable bound is
4,272,341,376 bytes/bank ×8 = **34,178,731,008 bytes/rank**.
An aggregate free count does not guarantee one allocation fits: its required
per-bank extent must also fit each bank's largest contiguous free block.

| Measured baseline | Free per bank | Free per rank |
| --- | ---: | ---: |
| `full_accuracy_v2.json`, B1/C262144 prepared | 260,930,112 | 2,087,440,896 |
| Same run, final | 125,196,864 | 1,001,574,912 |
| `batch32_hf_v1.json`, B32/C512 prepared | 521,230,528 | 4,169,844,224 |

These baselines predate the selected DRAM head and compact checkpoints. New
persistent charges are655,360 bytes/rank of head padding and34,576,896 bytes
per configured request per rank. The latter includes independent eight-bank
rounding of each QSA raw-history buffer. The initial34,478,592 estimate was
98,304 bytes too small. Checkpoint migration needs no extra snapshot scratch.

## B1 at context262144

Historical prepared free minus the new head/checkpoint charges leaves
**2,052,208,640** bytes/rank. Six canonical prefill capture/replay deltas measured
at C384 in `prefill_signatures_full48_v2.json` total **765,526,016**:

| Physical count | Retained allocator growth per rank |
| --- | ---: |
| 1 | 81,330,176 |
| 2 | 130,744,320 |
| 4 | 132,710,400 |
| 8 | 135,790,592 |
| 16 | 141,688,832 |
| 32 | 143,261,696 |

Subtracting that total and an illustrative81,330,176 decode allowance leaves
**1,205,352,448** bytes/rank. This is not a max-context trace-size estimate:
`qsa.py::indexer_scores` unrolls eight8192-key blocks at C262144 versus one at
C384, increasing dispatch/trace storage and intermediates. The count1 figure
above is measured prefill growth, merely a rough decode allowance here.

Stronger historical evidence is the actual C262144 prepared-to-final growth:
**1,085,865,984** bytes/rank across nine captures in `full_accuracy_v2.json`.
Its final free minus new head/checkpoint charges remains **966,342,656**.
New wrapper/program/trace sizes and peak temporaries still need measurement.
The historical largest contiguous extent after preparation was257,336,640
bytes/bank, distinct from aggregate free260,930,112 bytes/bank.

Current conservative tensor planning in `capacity_plan.json` gives B1/C262144
resident31,831,409,664 and headroom **2,347,321,344** before runtime; after its
separate2GiB runtime envelope,199,837,696 remains. The envelope includes unknown
trace/program/temporary costs: do not subtract measured traces a second time
without identifying which costs the envelope already covers. Historical32
prefill-input bucket allowance is still conservatively retained in this plan.
The later single-workspace edit gives no B1 output-storage saving.

## Current selected planning estimator

`cache_budget.py::runtime_forecast` now feeds default pool admission instead of
knowingly reserving only2GiB for a larger retained trace set. It exposes each
component and provenance; explicit runtime-reserve overrides remain visible in
the returned plan. Configured B1 retains2GiB and no factored-owner allocation.
For configured B>1 the default is the larger of2GiB and:

| B32/C512 runtime component | Bytes/rank |
| --- | ---: |
| Per-slot attention captures, all48 layers | 2,055,208,960 |
| Six MoE bucket captures, all48 layers | 128,974,848 |
| Per-slot terminal plus clear captures | 10,493,952 |
| Six canonical prefill captures | 765,526,016 |
| Two sampling trace allowances | 655,360 |
| Non-trace allowance | 268,435,456 |
| Selected runtime reserve | **3,229,294,592** |

The non-trace allowance rounds upward from historical B32/C512 prepared minus
its same-revision listed tensor plan:30,008,886,784−29,776,167,424 =232,719,360.
This covers unlisted prepared allocations in that measured workload; it does
not certify all future fragmentation, kernels or peak temporary shapes.

The production factored owner allocates only actual configured slots and no
inactive-logit template. Its source-exact persistent reservation is
`B*(3*163840+16384) + (ceil(log2(B))+1)*163840 +1024` for B>1:17,236,992 at B32,
1,344,512 at B2, zero at B1. Metadata is UINT32 row-major[2,32], banked1024B.
These persistent buffers are charged in resident bytes, not again in runtime.

At B32/C512, current shared-pool planning therefore gives30,575,773,184 resident
before global KV and373,663,232 available pool bytes after the selected runtime
reserve:950 logical full-stack blocks. The pinned API can still return16416
budgeted tokens, after which the worker adds32 output pages:545 blocks total,
including one null page, within950. This is a conservative allocation budget,
not a measured maximum or a reduction in advertised per-request context.

For long-context B>1, QSA attention and the whole prefill allowance scale by
`ceil(context/32768)`, following the source's8192-key indexer-block dispatch
unrolling. Charging the whole prefill cost is intentionally conservative;
this is an unverified command-count extrapolation, not a proven physical
bound. Configured B1 is exempt and retains its actual-max-context-backed prior
allowance. Fresh full48 B32 allocation will validate or replace the projection.

## Earlier B32 context512 forecast and measured-baseline comparison

Historical prepared free minus head padding655,360,32 checkpoints1,106,460,672
and the measured six-prefill total765,526,016 leaves **2,297,202,176** bytes/rank
for factored decode, its workspaces, sampling captures and other growth.

The reduced `factored_decode_slots_tracked_v3.json` measured per-slot attention
captures1,179,648 for ordinary GDN,2,490,368 for GDN+PLE and1,703,936 for QSA;
each exact-width MoE trace is458,752. Extrapolating by the full model's35 ordinary
GDN +1 PLE +12 QSA layers, with32 fixed slots:

| Prospective retained capture component | Bytes/rank |
| --- | ---: |
| All per-slot attention traces | 2,055,208,960 |
| MoE, all31 widths2..32 across48 layers | 682,622,976 |
| Terminal traces across32 slots plus clear | 10,493,952 |
| Total with31 MoE widths | **2,748,325,888** |
| MoE, six measured buckets across48 layers | 128,974,848 |
| Total with six MoE buckets | **2,194,677,760** |

All31 widths exceed the preceding residual budget by451,123,712. Six buckets
leave102,524,416 before extra working storage; the integrated owner's source
reservation17,236,992 (no inactive canvas) leaves **85,287,424**. These totals extrapolate
C128 reduced traces to C512/full48; exact programs, bucket geometry, temporary
lifetimes and clean latency remain unverified. They are not a selected fallback.

The newer production prefill workspace edit changes32 persistent per-slot
outputs into one shared output, independently of the sampler canvas. Source
saving at B32 is31×3,981,312 = **123,420,672** bytes/rank. If hardware validates
that edit, the preceding six-bucket remainder becomes225,945,088 before factored
working storage, or208,708,096 after the source-exact17,236,992 charge. The
current capacity ledger now charges exactly one shared output at every batch
size, with this source-only saving explicitly recorded. Historical allocator
measurements remain unchanged; the saving is not yet an isolated measured delta.

Retaining whole-model singleton traces for arbitrary B32 slots is an additional
hazard: approximately80MB per singleton can approach2.6GB across32 slots.
A whole-model B1 path for a generator configured with one slot is separate from
arbitrary singleton subsets in a B32 generator. The selected factored owner
includes bucket1 with accepted FP32 MoE reduction for all singleton subsets;
configured B1 keeps its original whole-model path. Bounded sampling traces
alone do not solve model-trace cardinality. Full48 integrated retained-trace
allocation and lifecycle validation remain required.

For comparison, current source private-cache planning at B32/C512 (including
one full migration cache, checkpoints and one shared prefill output) gives
resident30,783,391,232 including factored buffers and headroom3,395,339,776
before runtime, or166,045,184 after the selected3,229,294,592 reserve.
Shared-pool planning gives30,575,773,184 before global KV; the12-layer512-page
pool adds201,326,592, with the pinned worker's null/output reservations separate.
These tensor sums exclude measured runtime and cannot be substituted for the
historical prepared free value in the same subtraction.

## Original B32 OOM was a contiguous-allocation failure

`batch32_hf_v1.json` failed allocating143,278,080 bytes/rank, requiring17,915,904
bytes in each bank. At failure, aggregate free was21,051,584×8 =168,412,672,
but the largest compatible extent was17,688,000×8 =141,504,000:1,774,080 short.
Prepared-to-failure growth was **4,001,431,552** bytes/rank. The old `(count,slot)`
prefill keys retained many complete full-model traces; the six canonical keys
address that cardinality without proving every later decode configuration fits.

All figures preserve the advertised262144 context and selected dtype/fidelity,
KV and mesh policy. Fresh integrated allocator evidence, per-bank contiguous
headroom and the single required max-context execution remain acceptance work.
