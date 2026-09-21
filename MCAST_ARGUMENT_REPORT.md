# Multicast argument and measurement report

Compact implementation: `5745f4b6d71`. Three-baseline counts cover matched
matmul workloads and 22 additional configurations in the companion tables.
Validation and measurements are recorded below. No general performance
improvement is claimed; residual increases are explicit.

## Source states

Immediate pre-revamp implementation: `1674fc3cf08` (feedback fixes complete).
All operation factories, helper serializers, kernels, and model consumers were
captured before compact-layout edits in `/tmp/mcast-compact-current-sources.tar`.
These are complete source definitions, including placements, conditional tails,
and padding; snapshots are not themselves evaluated emitted-word counts.
The immutable Git commits are the durable source of the same snapshots.

| Operation | Pre-helper source commit | Migration commit | Comparability note |
| --- | --- | --- | --- |
| Matmul fixed/rotating 1D/2D, sparse, group attention | `0c9a8d8ed7169fffb117e4389afe15a29bd2876d` | `c38e9ccd484c6405773567044fa51a1f04dc6aa1` | Dedicated DRAM-sharded factory still has no helper; report as unaffected. |
| Conv2D | `b4f4e642099a15926cc863ace2175904d5855a26` | `966c4aaec5ba11f5ef7659502fa2ca7ac61aaa23` | Includes preceding matmul/API changes, not a main-branch comparison. |
| TopK | `966c4aaec5ba11f5ef7659502fa2ca7ac61aaa23` | `adc78c21e05102bef4dc50139faf6357c9d7be82` | Inspect the changed reduction path, not unrelated router implementations. |
| GroupNorm | `7b34ebc1e0df3da2ae9bbaf4c1ae87b66bea895e` | `acbdefe60db41f2d1a55c097e8e6fab43cd51f0f` | Includes wrapped-family and distributed consumer migration. |
| Conv3D | `a870e083f63349fc4d9d5cd61df440c5b434a123` | `f949d2207f89a9c37a8ecd51a3373ffc6cc31420` | Preserve the unrelated compact core-work placement change immediately before migration. |
| LayerNorm ordinary/pre/post-allgather | `f949d2207f89a9c37a8ecd51a3373ffc6cc31420` | `7e5149ba1e161ffc330c2b5e94320d23f8a51402` | Native ProgramSpec already existed; do not compare against an older positional interface. |

The corresponding full operation-source snapshots are in
`/tmp/mcast-compact-pre-{matmul,conv2d,topk,groupnorm,conv3d,layernorm}.tar`.
Quasar and CCL multicast implementations not using this helper remain unaffected.

## Count convention

One logical word means one emitted uint32 scalar, not aligned dispatch bytes.
Count positional CT and named CT once each; do not count named values again when
the build generates their representation. Resource bindings and compiler defines
are separate. CT is counted once per kernel variant, not multiplied by placement.
RT totals include operation prefixes, named values, helper blocks, padding, and
fusion tails on every placed core. Zero-RT compute kernels remain variants.

Before revamp, a present positional helper has 11 CT words (12 for chain):
tag, remote-presence, data-ready ID, consumer-ready ID, ACK, flags, rotating span,
sender mode, uniform remote count, uniform loopback count, rectangle capacity;
chain appends signal-source ID. Descriptor attachment adds two named CT offsets.
Absent positional helpers have one tag, zero RT, and descriptor offset names.
Native spec metadata/resources must be counted separately from this positional
description: semantic correspondence is not an extra emitted positional block.

Current RT block: rectangle count, ACK, `2*S` ordered sender coordinates,
`7*C` rectangle words (bounds x4, remote, loopback, mode), optional five-word
chain neighbors, then two role/phase words. Thus multicast RT is `4 + 2*S + 7*C`,
with `S=max(1, rotating_span)`. Chain RT is 11 words. Every placed core receives
the full family block, including single-role and inactive cores. Prefixes are
padded to the maximum per placed kernel before appending the block.

## Matched measurement method

Explicitly collect `tests/ttnn/unit_tests/kernel_lib/mcast_argument_audit.py`
through `scripts/run_safe_pytest.sh`. Its four fixed/rotating 1D/2D workloads
record full emitted descriptor CT/RT and placement, 20 warmed descriptor
construction samples, and 20 warmed operation calls including synchronization.
The initial operation call includes any compilation/cache lookup and dispatch;
it is not an isolated compile-time measurement. Use `--profile` separately for
kernel durations; profiled wrapper success alone does not prove pytest passed.
Record architecture, profiler overhead, cache state, and measurement limitations
alongside before/after results. Binary counts, sizes, and stack/storage require
separate artifact/compiler inspection, not inference from argument counts.

Current-helper Blackhole runs passed all four workloads (3.48 s unprofiled,
5.00 s profiled). Full snapshots/samples: `/tmp/mcast-compact-baseline-measurements.log`
and `/tmp/mcast-compact-baseline-profile.log`. Profile CSV:
`generated/profiler/reports/2026_09_20_17_22_22/ops_perf_results_2026_09_20_17_22_22.csv`.
All workloads use BF16 tiled input/output, interleaved DRAM output, no fusion,
LoFi math, and block width 4 tiles. Rotating inputs are width-sharded (1D) or
block-sharded (2D), row-major. Current helper-only CT is included in full counts.

| Current-helper case | M/K/N | Grid | Descriptor kernel groups | Sum CT positional + named | Aggregate RT | Warm construction median ns | Warm call + sync median ns |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 1D | 256/1024/1024 | 8x1 | 4 | 134 | 244 | 6010 | 63494.5 |
| Rotating 1D | 256/1024/1024 | 8x1 | 3 | 100 | 360 | 6600 | 63255 |
| Fixed 2D | 256/1024/256 | 4x4 | 7 | 234 | 656 | 16240 | 37530 |
| Rotating 2D | 256/1024/256 | 4x4 | 4 | 143 | 752 | 21319.5 | 35770 |

These CT sums count each distinct kernel variant for these workloads. Fixed 2D
has seven variants: its two receiver sources each compile for both NoCs.
The profiler CSV's source-level kernel listing omits those extra variants; the
compiler commands and ELF artifacts establish the complete count.
The fixed-1D in0 sender has 40 positional + 6 named CT, 17 RT on one core
(4 operation + 13 helper). Its receiver has 17 positional + 3 named CT,
13 helper-only RT on seven cores. The unaffected in1 reader/writer has
36 positional + 7 named CT and 17 RT on eight cores. The compute kernel has
18 positional + 7 named CT and zero RT.

The complete non-matmul three-baseline tables are in
[MCAST_OPERATION_COUNTS.md](MCAST_OPERATION_COUNTS.md).

Profiled current-helper medians over the 20 warmed calls are 39602 ns fixed 1D,
38335 ns rotating 1D, 11712 ns fixed 2D, and 10345.5 ns rotating 2D. Profiler
compiler artifacts show respectively 4, 3, 7, and 4 distinct kernel variants
(including compute). Maximum DM0/DM1 binary sizes
reported in the CSV are respectively 1848/1976, 1848/2256, 2256/1784, and
2256/1736 bytes. These are profiled binaries; compare only against the same
profiling mode, not uninstrumented ELF sizes.

## Matmul: complete three-source-state counts

The shapes/configurations are the matched cases above. Pre-helper counts below
are source-calculated from `0c9a8d8ed71`; v1/v2 counts are emitted descriptor
snapshots. `P+N` is positional plus named CT, including helper offset names.
No fusion tail is enabled. Resource bindings and compiler defines are not scalar
CT arguments. All listed RT counts include operation fields and padding.

| Case / kernel role | Pre CT P+N | v1 CT P+N | v2 CT P+N | Δ vs pre | Δ vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 1D or 2D: in0 sender | 33+4 | 40+6 | 47+6 | +16 | +7 |
| Fixed 1D or 2D: in0 receiver | 8+1 | 17+3 | 24+3 | +18 | +7 |
| Either 1D: in1 reader/writer, absent channel | 39+5 | 36+7 | 36+7 | −1 | 0 |
| Rotating 1D: mixed in0 | 37+5 | 25+7 | 32+7 | −3 | +7 |
| Either 2D: in1 sender/writer | 39+5 | 46+7 | 53+7 | +16 | +7 |
| Either 2D: in1 receiver/writer | 21+3 | 30+5 | 37+5 | +18 | +7 |
| Rotating 2D: mixed in0 | 41+3 | 25+5 | 32+5 | −7 | +7 |
| All four: compute | 18+7 | 18+7 | 18+7 | 0 | 0 |

The pre-helper rotating factories emitted unused TensorAccessor tails after
their 22-word operation/protocol prefix: input accessor (12 words in 1D, 16 in
2D), empty sparsity accessor (2), and num-batch placeholder (1). Those tails were
removed during migration, not this revamp. Accessor sizes are independently
captured in `input_accessor_ct` on the matched tensors; their serializer is
unchanged across the compared source states. Counting only fields read by the
old rotating kernel would undercount these emitted values.

| Case / kernel role | Cores | Pre RT | v1 RT | v2 RT | Δ vs pre | Δ vs v1 | v2 operation + helper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 1D: in0 sender | 1 | 8 | 17 | 8 | 0 | −9 | 4+4 |
| Fixed 1D: in0 receiver | 7 | 2 | 13 | 2 | 0 | −11 | 0+2 |
| Either 1D: in1 reader/writer | 8 | 21 | 17 | 17 | −4 | 0 | 17+0 |
| Rotating 1D: mixed in0 | 8 | 14 | 28 | 12 | −2 | −16 | 1+11 |
| Fixed 2D: in0 sender | 4 | 8 | 17 | 8 | 0 | −9 | 4+4 |
| Fixed 2D: in0 receiver | 9+3 | 2 | 13 | 2 | 0 | −11 | 0+2 |
| Either 2D: in1 sender/writer | 4 | 21 | 30 | 21 | 0 | −9 | 17+4 |
| Fixed 2D: in1 receiver/writer | 9+3 | 15 | 26 | 15 | 0 | −11 | 13+2 |
| Rotating 2D: in1 receiver/writer | 12 | 15 | 26 | 15 | 0 | −11 | 13+2 |
| Rotating 2D: mixed in0 | 16 | 10 | 20 | 10 | 0 | −10 | 1+9 |
| Compute | 8 or 16 | 0 | 0 | 0 | 0 | 0 | 0+0 |

The `9+3` receiver classes use different NoCs and therefore distinct binaries,
even though their scalar counts match. The rotating 1D mapped axis has one gap:
its coordinate payload is six words, not the four-word contiguous-line target.
The rotating 2D per-row schedules use four coordinate words. The operation's
one-word rotating prefix remains separate from the helper's phase. In1's fixed
17-word prefix retains the two bias placeholders even without fusion; those
are operation-owned values, not multicast savings.

| Matched case | Variants pre/v1/v2 | CT pre/v1/v2 | CT Δ pre / v1 | Aggregate RT pre/v1/v2 | RT Δ pre / v1 |
| --- | --- | --- | --- | --- | --- |
| Fixed 1D | 4/4/4 | 115/134/148 | +33 / +14 | 190/244/158 | −32 / −86 |
| Rotating 1D | 3/3/3 | 111/100/107 | −4 / +7 | 280/360/232 | −48 / −128 |
| Fixed 2D | 7/7/7 | 172/234/276 | +104 / +42 | 320/656/320 | 0 / −336 |
| Rotating 2D | 4/4/4 | 137/143/164 | +27 / +21 | 424/752/424 | 0 / −328 |

These are logical counts, not measured command-queue bytes or aligned dispatch
traffic. No dispatch-byte saving is inferred from them.

## Matched v1/v2 performance and artifact evidence

Final unprofiled v2 samples: `/tmp/mcast-compact-final-measurements.log` (four
passing workloads). Profiled v2: `/tmp/mcast-compact-candidate-profile.log`
(four passed, 5.70 s), with CSV under
`generated/profiler/reports/2026_09_20_18_14_49/`.
Each median uses 20 warmed samples, as above. These are separate batches on the
same Blackhole; no confidence interval or statistical speedup is asserted.

| Case | Construction ns v1 → v2 | Warm call + sync ns v1 → v2 | Kernel ns v1 → v2 | Kernel change | Max DM0/DM1 bytes v1 → v2 |
| --- | --- | --- | --- | ---: | --- |
| Fixed 1D | 6010 → 6240 | 63494.5 → 67104.5 | 39602 → 39626 | +0.06% | 1848/1976 → 1848/1520 |
| Rotating 1D | 6600 → 7085 | 63255 → 63570 | 38335 → 38417 | +0.21% | 1848/2256 → 1848/1908 |
| Fixed 2D | 16240 → 17455 | 37530 → 41164.5 | 11712 → 11517.5 | −1.66% | 2256/1784 → 2188/1372 |
| Rotating 2D | 21319.5 → 18244.5 | 35770 → 34924.5 | 10345.5 → 10274.5 | −0.69% | 2256/1736 → 2188/1572 |

Host medians are not uniformly better: fixed 2D construction rose by 1.215 µs
and warm call+sync by 3.635 µs in the final batch. Placement analysis adds a
host scan; the generic metadata cache and allocation-free encoding merge avoid
unnecessary repeated family analysis. Remaining host variation/overhead is
reported, not hidden by lower logical RT counts.

SFPI object compilation was replayed without ccache, three times per data-movement
variant, from saved v1 sources and frozen v2 sources with the original profiled
flags/generated headers. All 84 object compiles passed. Artifacts and timing logs:
`/tmp/mcast-compile-audit.NoorT4/`. These are compiler-to-LTO-object timings, not
whole JIT/link times, and exclude unchanged compute kernels. Per-variant medians
range 0.247–0.328 s v1 and 0.250–0.329 s v2. The regular rotating in0 object
medians are 0.306 → 0.311 s (1D), 0.309 → 0.317 s (2D).

ELF symbol/disassembly inspection gives these profiled `kernel_main` frames and
symbol sizes. They are frame allocations, not a claim about maximum stack depth
including firmware/callees. No memcpy/memmove call occurs in these functions.

| In0 kernel | Frame bytes v1 → v2 | kernel_main bytes v1 → v2 |
| --- | --- | --- |
| Fixed 1D sender | 192 → 112 | 1596 → 1140 |
| Fixed 2D sender | 208 → 128 | 1404 → 992 |
| Rotating 1D, eight senders | 128 → 144 | 1876 → 1528 |
| Rotating 2D, four senders per row | 112 → 128 | 1352 → 1188 |

Range expansion is performed once in pipe construction, directly into owned
storage. The optional receiver uses in-place construction; direct returns use
C++17 prvalue elision. The 64-sender range table occupies 512 bytes even though
only six coordinate words arrive in RT.

For the same 8x8 receiver geometry, Counter protocol, 67 rounds, and NoC 0, the
large lifetime fixture passes with both a regular row-major schedule and a
first-two-senders-swapped schedule that deliberately selects explicit fallback.
The comparison isolates storage/encoding costs; the changed sender ordering
means it is not a matched latency benchmark. Logs:
`/tmp/mcast-compact-64-artifact.log` and
`/tmp/mcast-compact-64-explicit-artifact.log` (both passed).

| 64-sender v2 encoding | Coordinate RT words | Main-channel RT words | kernel_main frame | kernel_main symbol bytes |
| --- | ---: | ---: | ---: | ---: |
| Explicit pairs | 128 | 134 | 192 B | 2888 B |
| Two X ranges, one Y range | 6 | 12 | 720 B | 3084 B |

The range variant reserves 528 additional frame bytes while saving 488 bytes
of logical coordinate RT. Active receivers initialize 128 table entries in one
64-iteration setup loop, directly at the final stack address. Neither artifact
has a memcpy/memmove call. Runtime-inactive cores take a separate one-time
528-byte memset when constructing the empty `std::optional`; active receivers
bypass that clear and populate the table. This is residual initialization cost,
not repeated expansion per receive. The frame includes optional/protocol state
and spills beyond the 512-byte table, and is not total call-chain stack usage.

## Required Claude review: disposition

The explicitly requested CLI review completed after initial validation; full
output is `/tmp/mcast-compact-claude-review.log`. The following dispositions
refer to its numbered findings, not unverified acceptance of every claim.

| Finding | Disposition |
| --- | --- |
| 1: Mixed-role sender coordinates zeroed | Fixed. Fill every participating core's reserved coordinate slot; include sender-only groups in compatibility selection. Literal, cross-group host, and Watcher device accessor regressions passed. |
| 2: Latch generic versus placement-specific API use | Declined. Placement-specific CT and RT are only exposed together. A family may legitimately emit different kernels through different complete APIs; a global latch would prohibit that supported use without preventing arbitrary caller vector splicing. |
| 3: Recheck coordinate copy length | No redundant runtime check added. Equal schedule length is guaranteed when groups are added; encoding dimensions/range counts are checked before selection. New cross-group fallback coverage pins that boundary. |
| 4: Ternaries might read omitted offsets | The correctness claim is declined: C++ evaluates only the selected conditional operand, independent of optimization. Added compile-time nonempty compressed-axis checks and early positional version rejection. |
| 5: New placement-grid restriction | Removed from shared inference; existing frontend validation remains at its owning boundary. |
| 6: Assert in malformed range fallback | Deferred. Shared constexpr host/device code consumes host-validated ranges; host replay checks every phase. Adding device-only assertion dependencies to the common wire definitions is unnecessary for the supported emitter contract. |
| 7: Cache every placed core's group/phase | Deferred larger refactor. Some repeated placement/role lookup remains; measured host overhead is disclosed. Do not add a second per-core preparation representation without evidence it pays for its complexity. |
| 8–9: Repeated generic analysis and pointer set | Fixed: cache generic metadata during preparation and merge placement encodings inline without a pointer-ordered allocation. |
| 10: Remove staged CT copy | Retained the small staged value for the explicit two-destination transaction contract and readability. |
| 11: Sender rectangle value copies | No speculative API change. Optimized inspected sender/rotating functions contain no memcpy/memmove call; symbol/frame sizes are reported above. |
| 12: Stale v1 layout helpers | Removed; chain has named fixed offsets and remains 11 RT words. |
| 13: Prepared LOOPBACK word | Retained the full prepared rectangle representation shared with the pipe structure; the compact wire omits it and derives remote+1. |
| 14: Defaulted metadata equality | Retained explicit five-field comparison, compatible with the SFPI C++17 frontend. |
| 15: External schedule fallback unclear | Documented as the plan's deliberate scope choice; arbitrary/external schedules retain pairs. |
| 16: Constrain receiver coordinate constructor | Deferred C++20 requires syntax in shared SFPI C++17 code; the actual owning/pointer/native-view constructions all compile and run. |
| 17: Use returned direct CT offset | Fixed in the matmul utility. |

Additional coverage now includes absent-coordinate and short-old-positional-tag
rejection, native ProgramSpec old-tag rejection, mixed-role sender access, and
compressed schedules with multiple rectangles and unknown per-rectangle mode.
The expanded launcher passed 19 cases, including all 48 native host contracts
and native device suites (`/tmp/mcast-compact-review-contracts-fixed.log`). The
final combined-geometry host rerun passed too. Standalone Watcher passed its
mixed-role smoke and 12-case coordinate-lifetime/matmul matrix; triage was off.

## Other consumers: complete counts and source derivations

The [per-kernel tables](MCAST_OPERATION_COUNTS.md) cover 22 configurations and
96 kernel variants across those configurations, with all three CT/RT counts,
signed per-core deltas, multiplicities, helper payloads, and exact aggregates.
The [captured CT/count data](MCAST_OPERATION_COUNTS.json) also retains defines
and v2 compile hashes. Hashes were distinct within each captured program; the
unchanged pre/v1 variant counts were checked against their factory construction
and specialization inputs, not inferred from profiler source-name listings.

Unlike the four matmul cases, pre-helper and v1 non-matmul counts are calculated
from the immutable source states, not claimed as before/after execution captures.
V2 was measured by temporarily lowering descriptors, inspecting direct Programs,
and lowering/invoking native ProgramSpecs. Native tensor-layout CT words and bound
common-RT addresses are included. The temporary hook code and its build include
path were removed completely; no operation or tt_metal instrumentation is retained.
The captures add host construction work and are not performance measurements.

Capture logs: `/tmp/mcast-all-operation-hashes.log` (171 tests passed),
`/tmp/mcast-dit-count-hash.log`, and `/tmp/mcast-dram-count-hash.log` (one each).
They include placements and full per-core arguments. The corresponding source
snapshots, final CT values, and source-derived formulas make each table auditable.
All cases ran on Blackhole; the available worker grid is 11x10. Individual program
placements are listed below and in the appendix; they are not all the whole grid.

| Configuration | Variants pre/v1/v2 | Aggregate CT pre/v1/v2 | CT Δ vs pre / v1 | Aggregate RT pre/v1/v2 | RT Δ vs pre / v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conv3D compact chain | 3/3/3 | 118/129/136 | +18 / +7 | 3136/3136/3136 | 0 / 0 |
| Conv3D compact mixed/idle | 3/3/3 | 118/129/136 | +18 / +7 | 3376/3392/3392 | +16 / 0 |
| Conv3D rectangular/passive | 3/3/3 | 118/128/135 | +17 / +7 | 3376/3520/3136 | -240 / -384 |
| GroupNorm wrapped, legacy | 5/5/5 | 98/122/136 | +38 / +14 | 957/2457/1001 | +44 / -1456 |
| GroupNorm rectangular, legacy | 5/5/5 | 172/196/210 | +38 / +14 | 660/992/648 | -12 / -344 |
| GroupNorm local, legacy | 5/5/5 | 173/195/209 | +36 / +14 | 800/992/576 | -224 / -416 |
| Attention Q10, interleaved | 3/3/3 | 16/29/36 | +20 / +7 | 8580/13860/13640 | +5060 / -220 |
| Attention Q50, interleaved | 3/3/3 | 16/29/36 | +20 / +7 | 8580/13860/6710 | -1870 / -7150 |
| Attention Q10, sharded | 3/3/3 | 54/67/74 | +20 / +7 | 8580/13860/13640 | +5060 / -220 |
| Attention Q50, sharded | 3/3/3 | 94/107/114 | +20 / +7 | 8580/13860/6710 | -1870 / -7150 |
| LayerNorm ordinary, two-stage | 7/7/7 | 64/123/165 | +101 / +42 | 185/441/225 | +40 / -216 |
| LayerNorm ordinary, per-line | 7/7/7 | 64/123/165 | +101 / +42 | 178/430/218 | +40 / -212 |
| LayerNorm pre-allgather | 7/7/7 | 60/89/110 | +50 / +21 | 373/785/435 | +62 / -350 |
| LayerNorm post-allgather | 7/7/7 | 68/97/118 | +50 / +21 | 233/437/263 | +30 / -174 |
| Sparse matmul compact output | 4/4/4 | 116/135/149 | +33 / +14 | 174/214/178 | +4 / -36 |
| Sparse matmul wide subblock | 4/4/4 | 116/135/149 | +33 / +14 | 454/564/468 | +14 / -96 |
| TopK local/final | 6/6/6 | 69/88/102 | +33 / +14 | 226/655/294 | +68 / -361 |
| Conv2D width-sharded | 3/3/3 | 83/90/97 | +14 / +7 | 576/1324/356 | -220 / -968 |
| Conv2D block-sharded | 4/4/4 | 154/180/194 | +40 / +14 | 2236/2780/2032 | -204 / -748 |
| Conv2D height-sharded | 4/4/4 | 165/191/205 | +40 / +14 | 767/1863/655 | -112 / -1208 |
| DiT GroupNorm local | 4/4/4 | 110/138/152 | +42 / +14 | 1236/1920/1536 | +300 / -384 |
| Dedicated DRAM-sharded matmul | 3/3/3 | 62/62/62 | 0 / 0 | 132/132/132 | 0 / 0 |

### Workloads and accounting checks

- Conv3D: 8x8 program grid, BF16, HiFi2, FP32 destination accumulation, bias,
  no packer accumulation. N/C/D/H/W is (1 or 2)/64/5/5/5, kernel 3x3x3,
  stride 1, padding (0,1,1), output channels 160/320/64 and input-channel block
  64/32/32 for chain/mixed/rectangular respectively. Reader 53 CT/11 RT and
  compute 31 CT/12 RT are unchanged. The pre-helper writer had three more CT
  protocol scalars and 11 raw RT protocol fields. Its 15-word operation prefix
  gains four reduction-coordinate words on 60 active cores in mixed/rectangular
  cases, but not on four idle cores. Attachment pads those four prefixes to 19;
  that accepted 16-word aggregate overhead remains for chain. Chain RT stays 11.
  The first chain case has 45 active and 19 idle cores. The rectangular case has
  60 active and four inactive landing cores; its mixed-role helper shrinks 13→7.
- Wrapped GroupNorm: legacy BF16 tiled height-sharded input, seven batches,
  128 channels, spatial length 288, 16 groups, 7x9 grid; each reduction spans
  nine ordered cores. Source replay of the old first/middle/last splitting gives
  six senders with two rectangles and one with three. Their old RT is 18 gather
  coordinates plus 12/17 protocol words: 30×6 and 35×1. V1 padded all seven to
  18+27=45; v2 uses 18+19=37. Receiver RT returns 27→2 (56 cores), but the fixed
  three-record sender capacity leaves +44 total versus pre-helper. CT moved two
  positional semaphore IDs into the helper plus one independent named gather ID
  and two attachment offsets. Column-major/transposed and Welford forms were also
  validated; their count-equivalent geometry is not duplicated in the table.
- Rectangular GroupNorm: DRAM interleaved BF16 tiled, N=1, C=128, spatial=512,
  16 groups, grid 8x4. Four senders have 5 operation words + 16 gather coordinates;
  28 receivers keep five operation words. Old single-rectangle protocol was
  seven sender words/two receiver words; v1 13/13 becomes v2 4/2. The local
  configuration uses N=8, spatial=64 on the same grid, 32 singleton groups. Its
  present LocalCopy channel has zero helper RT; the seven operation/gather words
  remain. Two empty reader/compute placements are still compiled variants and
  count toward CT, but contribute zero RT. Legacy and Welford kernels passed.
- Attention: Q length 1, batch 32; (K, sequence, Q heads, KV heads) is
  (96,576,10,2) or (64,256,50,5), BF16, both DRAM interleaved and column-major
  height-sharded L1 variants. All three kernels are placed on all 110 cores,
  including idle cores. The reader's unchanged prefix is 20 words. Pre-helper
  added 13 protocol scalars plus four X and ten Y sender-axis coordinates: 47
  total. V1 uses a 75-word helper for the 32 phases. Q10 has external senders
  and nonuniform ACK/mode, so exact explicit fallback remains 73 helper words:
  93 total, +46/core versus pre-helper. Q50 compresses the exact partial-column
  schedule to four coordinate words, giving ten helper / 30 total words. The
  idle cores retain the same kernel layout. Writer 17 RT and compute 14 RT are
  unchanged; the sharded TensorAccessor CT sizes differ by shape and are counted.
- LayerNorm ordinary: 256x320, BF16 input/residual, 5x2 grid, Welford off, both
  two-stage and per-line reduction. The two-stage layout uses one sender, seven
  all-to-all receivers, two other receivers; per-line uses 2/6/2. Pre-allgather
  uses a 32x2048 local shard on 8x4; post-allgather uses 32x2048 on 8x2, BF8 input,
  weights and output. These selected distributed cases are RMSNorm, four logical
  tensor partitions tested sequentially on the device, not a multi-device
  dispatch measurement. LayerNorm/RMSNorm, residual, dtype and Welford variations
  were covered by the full consumer run. Native readers keep operation-owned
  gather coordinates/named fields. Ordinary uses two independent channels;
  pre/post each use one. Each adds 10 named metadata CT words in v1, 17 in v2;
  the old sender-only fanout CT scalar is removed. Each sender channel uses four
  RT words and each receiver channel two; only the original four sender bounds
  disappear during migration. Thus ordinary retains +4 RT/core, pre/post +2
  per receiver. No channel/gather geometry deduplication was attempted. Native
  post-allgather writer tensor bindings add two positional CT and one common RT
  address per placed core; these are included, not mistaken for zero-cost handles.
- Sparse matmul: compact optional-output M/K/N=32/128/192, four batch blocks and
  eight experts on six cores; wide-subblock uses the test's 4x4 grid, N=1024 and
  two-tile output subblock, M/K=32/128, four batch blocks and eight experts.
  The compact case is BF16; the wide case has BF16 in0/output and BF8 in1.
  Both DRAM cases use the generic paired emitter.
  Its fixed-channel payload is 13→7 on both sender and receiver kernels, not
  placement-specialized 4/2. Complete sender RT is 8/17/11, receiver 2/13/7;
  the absent in1 writer retains 22 RT versus the old 26 including unused bounds.
  Remaining aggregate increases are +4/+14 versus pre-helper. CT accounts for
  the actual sparsity/output TensorAccessors and the one-word absent channel.
- TopK: BF16 (1,1,32,8192), k=32, sorted, both largest/smallest validated. There
  are 32 local workers plus one final worker, six variants. Old readiness sender
  CT contained two semaphore IDs, four bounds and fanout; the new operation
  prefix retains one independent arrival counter and four operation values.
  The helper moves geometry to four sender RT words and adds two receiver RT
  words to each local writer, leaving +68 aggregate RT versus pre-helper. Local
  reader, final writer, and both compute argument lists are unchanged.
- Conv2D: all three use BF16, tiled output, batch 2, 3x3 filter, stride 2,
  padding (1,2,2,3), HiFi4, FP32/packer accumulation, bias, no activation fusion. Width-sharded
  case has Cin/Cout=384/353, input 8x8: 22 activation-reader cores (12 active
  operation workers), 12 weight/compute cores. Its old activation CT had six
  protocol values; old RT had three operation words plus the full 11+10 device
  coordinate axes. V1 is 3+55; v2 is 3+11, with six coordinate range words for the
  22-phase grid. Block-sharded case is Cin=Cout=128, input 32x32, 80 total cores:
  44 sender-kernel placements include 40 input-only/role-none cores; four active
  senders serve 36 receivers. Its dynamic-role sender helper is 5 words, receiver
  2, after operation prefixes 6/1. The raw activation multicast remains unmigrated
  and unchanged (36 CT, 17 RT); it is included in the full counts. Height-sharded
  case is Cin=Cout=16, input 256x256, act_block_h=32, with 105 active and five
  noop landing cores on 11x10. Sender/receiver operation prefixes are 5/2;
  they include the unconditional zero remaining-tile field added at migration
  (old prefixes were 4/1 when activation reuse was off). Both fixed weights
  variants add 11/18 helper CT plus two offset names; no old CT values disappear.
- DiT local GroupNorm: local 1x1 mesh, NCHW (1,128,64,64), 32 groups, BF16,
  no activation, 8x8 reduction grid, four 16-core groups, Welford. Generic paired
  emission retains seven helper words on senders and receivers. The sender's
  37-word prefix includes the 32 gather coordinates. Old/current/revamped RT is
  44/50/44 on four senders, 7/18/12 on 60 receivers, plus unchanged writer 10.
  Local receiver migration also added the two-word output TensorAccessor, absent
  from its old CT, and both readers add two named helper offsets. Legacy unused
  named protocol values remain here and are included. Distributed mux/fabric
  kernels are not migrated to this helper; no savings or cross-device validation
  is claimed for them.
- Dedicated DRAM-sharded matmul is unchanged across all three source states,
  verified by the factory diff: float32 32x96 × 96x32, no padding, sharded input
  and L1 output. Three kernels each span 30 cores; complete RT classes are
  in0 9×4 + 1×26, in1 11×1 + 1×29, compute 1×30. CT=62 and RT=132 in every
  state. Its raw multicast is deliberately marked unaffected, not compacted.

The listed residual RT increases are explained by fixed-capacity padding,
retained gather/channel duplication, generic role unions, exact external-sender
fallback, or moving formerly CT geometry into RT. New CT increases of seven per
present positional channel (also seven per native metadata block) pay for role
and encoding specialization. No measured dispatch alignment or packet-byte
reduction is claimed. No unresolved correctness issue remains from validation;
performance/host regressions and deliberately deferred review simplifications
are documented rather than treated as gains.

## Final clean validation

After removing all temporary instrumentation, the host rebuild passed
(`/tmp/mcast-compact-clean-build.log`). The final 176-case consumer/audit batch
passed in 21.44 s; all six local DiT cases passed in 3.70 s; the four strict
matmul count checks and eight compressed-coordinate lifetime cases passed in
6.56 s. All three dedicated DRAM-sharded padding cases passed too. Logs are
`/tmp/mcast-compact-clean-{validation,dit,measurements,dram}.log`.
No device hang/reset occurred. These supplement the native/compiler, chain,
Watcher, and profiled artifact evidence above; no multi-device DiT claim is made.
