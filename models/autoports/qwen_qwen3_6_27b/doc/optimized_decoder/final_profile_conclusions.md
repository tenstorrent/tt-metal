# Final194 profiler conclusions

All eight final profiles reference delivered source SHA256
`315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.
This is a read-only synthesis of their rendered tt-perf-report CSV/text,
metrics and accounting, not an independent stage-review verdict. No additional
optimization, device run, implementation or test change was performed.

For every run, independently recomputing CSV kernel/gap sums and operation
counts agrees with `metrics.json`. Decompressing `ops.csv.gz` reproduces the
recorded raw CSV SHA256. Summing physical weight-read bytes and necessary cache
traffic agrees with `accounting.json`; division by one-chip512GB/s reproduces
its bandwidth floor and floor/same-run-host ratio. BFP exponent bytes are
included; unused setup copies and transient activation traffic are excluded.
The bound is bandwidth-only, not an assertion that SFPU computation is free.

## Final measurements and exact artifacts

Times below are milliseconds. P-kernel/P-gap are the profiled warmed prefill
window; P-host is the same run's warmed wall measurement. D-kernel/D-gap are
one traced decode window; D-host is the same run's trace median. These are
**profiler-enabled** observations, not substitutes for profiler-off candidate
ranking. Each run link opens its provenance/metrics; P/D open the complete
prefill/decode CSV, and A opens same-run accounting. Corresponding advice-enabled
`.txt` reports and compressed raw operations are in the same directories.

| Kind, batch, logical prefix | P-kernel | P-gap | P-host | D-kernel | D-gap | D-host | Floor / D-host | Artifacts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| [GDN B1 S512](perf/final194_profile_profile_l0_b1_s512/metrics.json) | 5.496 | 0.316 | 6.561 | 1.135 | 0.019 | 1.186 | 78.082% | [P](perf/final194_profile_profile_l0_b1_s512/prefill_perf_report.csv), [D](perf/final194_profile_profile_l0_b1_s512/decode_perf_report.csv), [A](perf/final194_profile_profile_l0_b1_s512/accounting.json) |
| [GDN B1 S513](perf/final194_profile_profile_l0_b1_s513/metrics.json) | 7.408 | 0.414 | 8.590 | 1.139 | 0.019 | 1.189 | 77.887% | [P](perf/final194_profile_profile_l0_b1_s513/prefill_perf_report.csv), [D](perf/final194_profile_profile_l0_b1_s513/decode_perf_report.csv), [A](perf/final194_profile_profile_l0_b1_s513/accounting.json) |
| [GDN B1 S1025](perf/final194_profile_profile_l0_b1_s1025/metrics.json) | 12.989 | 0.495 | 14.577 | 1.139 | 0.020 | 1.195 | 77.478% | [P](perf/final194_profile_profile_l0_b1_s1025/prefill_perf_report.csv), [D](perf/final194_profile_profile_l0_b1_s1025/decode_perf_report.csv), [A](perf/final194_profile_profile_l0_b1_s1025/accounting.json) |
| [GDN B32 S33](perf/final194_profile_profile_l0_b32_s33/metrics.json) | 20.953 | 2.651 | 24.753 | 1.546 | 0.020 | 1.602 | **70.120%** | [P](perf/final194_profile_profile_l0_b32_s33/prefill_perf_report.csv), [D](perf/final194_profile_profile_l0_b32_s33/decode_perf_report.csv), [A](perf/final194_profile_profile_l0_b32_s33/accounting.json) |
| [Full attention B1 S512](perf/final194_profile_profile_l3_b1_s512/metrics.json) | 3.116 | 0.084 | 3.934 | 1.002 | 0.029 | 1.060 | 73.104% | [P](perf/final194_profile_profile_l3_b1_s512/prefill_perf_report.csv), [D](perf/final194_profile_profile_l3_b1_s512/decode_perf_report.csv), [A](perf/final194_profile_profile_l3_b1_s512/accounting.json) |
| [Full attention B1 S513](perf/final194_profile_profile_l3_b1_s513/metrics.json) | 4.928 | 0.205 | 5.903 | 1.002 | 0.030 | 1.061 | 73.040% | [P](perf/final194_profile_profile_l3_b1_s513/prefill_perf_report.csv), [D](perf/final194_profile_profile_l3_b1_s513/decode_perf_report.csv), [A](perf/final194_profile_profile_l3_b1_s513/accounting.json) |
| [Full attention B1 S1025](perf/final194_profile_profile_l3_b1_s1025/metrics.json) | 8.539 | 0.220 | 9.663 | 1.010 | 0.030 | 1.066 | 72.867% | [P](perf/final194_profile_profile_l3_b1_s1025/prefill_perf_report.csv), [D](perf/final194_profile_profile_l3_b1_s1025/decode_perf_report.csv), [A](perf/final194_profile_profile_l3_b1_s1025/accounting.json) |
| [Full attention B32 S33](perf/final194_profile_profile_l3_b32_s33/metrics.json) | 12.123 | 1.183 | 14.219 | 1.038 | 0.035 | 1.102 | **70.555%** | [P](perf/final194_profile_profile_l3_b32_s33/prefill_perf_report.csv), [D](perf/final194_profile_profile_l3_b32_s33/decode_perf_report.csv), [A](perf/final194_profile_profile_l3_b32_s33/accounting.json) |

GDN B32 necessary traffic is575,307,776B, giving1.123648ms. Its1.546358ms
kernels +0.019922ms gaps +0.036188ms host remainder reconcile to1.602468ms.
The residual uses its actual11520physical columns; counting obsolete16640
columns would inflate the roofline. Full-attention B32's bound is0.777240ms
against1.101611ms host. The owner-selected stop rule remains the separately
recorded [selection](stop_rule_selection.json): ≥70% and the last two kept
improvements0.384237%/0.393176%, both <3%. These final-source profiles corroborate
it. The8-hour device-time alternative was not used.

## Dominant decode costs

For GDN B32, the six matmuls total967.857us (62.6% of device kernel time):
MLP gate/up/down586.459us, packed input primary183.888us, narrow residual
129.194us, and output68.316us. Their reported bandwidth is approximately
456–463GB/s, or89–90% of the nominal one-chip512GB/s. The exact recurrence
is383.494us (24.8%), convolution59.967us, affine/layout/gate31.655us, and
beta/decay6.105us. The two three-phase hidden norms total27.560us; fused
correction reconstruction is9.609us. These identify actual GenericOp roles
from their delivered call order, rather than treating all11 GenericOps as one
opaque bottleneck. GDN decode has33 device operations and no reported host ops.

GDN B1/S512 is dominated by the same matrix reads:966.043us (85.1% of kernel
time), including MLP584.246us. Recurrence falls to55.113us, convolution5.719us,
affine8.611us, beta3.467us. Across S512/513/1025 the kernel sum remains
1.135–1.139ms; fixed recurrent-state size explains the near context-independent
decode cost. This is not the same as full-attention cache traffic.

Full-attention B32 spends820.761us (79.1%) in six matmuls: MLP588.351us,
QKV92.542us, gate70.056us and output69.812us. Remaining leaders are four
norms32.349us, paged decode SDPA30.425us, batched QKV split26.363us and
paged fused KV update13.016us. Its51-op path has35.216us gaps. B1/S512 has
44ops and818.507us matmuls (81.7%); SDPA36.862us, norms30.771us and four
permutations19.193us follow. B1 uses the native single-user head-creation path,
not the B32 staging kernel. At S1025 B1 SDPA rises to43.983us and necessary
KV traffic increases, while matrix costs remain nearly fixed.

All eight decode reports mark their matmuls optimized and have no actionable
CSV advice entries. The tool's Blackhole argument uses its nominal P150 model
while the device is P300: some displayed FLOPs percentages exceed100%.
Those per-op compute percentages and the pipeline's reported8core count are
not used to claim impossible efficiency or justify blindly increasing grids.
Explicit byte accounting and same-run host time determine the reported bound.

## Prefill, internal chunks and non-aligned tails

B1/S512 GDN's main costs are packed input primary1073us, residual712us,
fused SwiGLU1045us, down465us, output394us, elementwise binary521us,
slices355us and two native ChunkGDN calls239us. Full attention at the same
shape is led by SwiGLU1039us, down485us, packed projection419us, SDPA330us,
four norms221us and output186us. The different head/cache operations are
visible rather than conflated as interchangeable attention implementations.

B32/S33 is1056logical matrix rows, not32 independent full decoder calls.
GDN has236device operations:64 ChunkGDN calls total2491us, five reshape
operations2081us, SwiGLU2071us, binary1862us,78slices1547us and typecasts1258us.
Its input primary partitions640/416 take1096/1055us;32 native causal convolutions
cost991us. Full attention has442ops, dominated by six reshapes3495us,
SwiGLU2063us,270slices1281us, two full down chunks979us, two full packed
projection chunks840us, norms655us and concatenations478us. The measured
reshapes are not assumed free metadata views merely because their names
contain `View`. These movement and per-user native-composite costs help explain
prefill's larger gap totals; decode fusion gains do not erase prefill costs.

| B1 logical sequence transition | Extra prefill kernel time | Extra device ops | What the CSV actually shows |
|---|---:|---:|---|
| GDN512→513 | 1.912ms | 43 | One full512chunk plus a one-token tail; second fused SwiGLU still has physical M512 and costs954.868us. Native ChunkGDN count stays2; tail uses the one-token path. |
| Full attention512→513 | 1.812ms | 53 | Second fused SwiGLU M512 costs970.272us; tail SDPA80.108us, packed projection M32 155.010us and down M32 193.044us, plus tail movement/normalization. |
| GDN513→1025 | 5.581ms | 46 | Two full512chunks plus the tail: four ChunkGDN calls477.420us total; SwiGLU1044.576 +1044.504 +955.323us. |
| Full attention513→1025 | 3.611ms | 33 | Two full512chunks plus the tail: SDPA330.386 +627.199 +113.501us as accessible cache grows; SwiGLU1038.822 +1039.239 +969.814us. |

These are cross-run profiled kernel differences, not isolated per-token
latencies. Public S513/1025 remain valid; internal padding/chunking explains
the discontinuity and does not impose sequence divisibility. These eight
profiles contain B1 long-boundary shapes and B32/S33; B32 long-prefill claims
must use the separate `final194_longprefill_*` artifacts, not extrapolation.

## Advice disposition and deferred work

The actionable suggestions occur in prefill, and must be interpreted with
actual legality, precision and whole-decoder evidence from [work_log](work_log.md)
and [checklist](checklist.md):

| Report advice / observed opportunity | Evidence and disposition |
|---|---|
| Increase88/110 or77/110 active prefill cores | Source audit showed row-grid underfill for512rows. Internal640-row partitioning was kept and helps larger flattened batches. Larger768/1024 K8 configs first failed physical L1; [legal K4 retry](prefill127_cap768.json) and row-cap147 experiments were measured and slower. Do not treat the first API/L1 error as rejection. Filling a B1/S512 grid by further padding is not established as a win and is deferred. |
| Place input0 in L1 | Regular role-specific input-L1 was legal but slower (B1 7.543/B32 29.519ms in prefill94). Direct fused-SwiGLU [swiglu137 B1](swiglu137_b1.json)/[B32](swiglu137_b32.json) also lost:6.755/26.573ms versus6.749/26.462ms DRAM controls. The measured added transfer is why this advice is rejected for that topology, not a blanket claim that L1 is ineffective. Norm-to-projection L1 handoff was separately retained after rounding/lifetime repair. |
| Set program configs / enlarge small output subblocks | Prefill86/94 measured subblocks and role K-blocks. Residual-only [K10/SW3](prefill171_res_k10_sw3_b1_s512.json) was kept, improving B32/S512176.516→174.323ms. Primary K10 initially exceeded L1; [repaired block-height1](primary181_b32_s512.json) ran correctly but174.670ms lost, so it was rejected. The MinimalMatmul report's generic “no program_config” does not prove a different composite configuration wins. Further full-attention subblock/tail-specific composite changes are untried after the stop rule. |
| HiFi2 may replace GDN attention HiFi4 | Residual-only HiFi2 failed long PCC0.992758; all-attention HiFi2 failed0.985621; [prefill-only attention HiFi2](phase139_attention_sequence.json) failed0.993217. Keep required two-term HiFi4 input attention. Full-attention prefill's generic “use HiFi2/HiFi4 for accuracy” applies to selected LoFi rows; raising fidelity is not a demonstrated performance gain and quality is judged by unchanged decoder gates. |
| Weight-bandwidth dominated MLP / lower dtype | Real-weight BFP4/LoFi and profiler80 evidence exist; decode MLP LoFi on GDN failed long0.990742. BFP8 remains default, with stage8 datatype-sweep/full-model precision probe responsible for a final lower-precision choice. No decoder-only acceptance waiver. |
| More decode readers or different packing | [geometry119](geometry119.json) C32/R8–13 was slower0.448–0.669ms versus0.424ms corrected-pair control; lower-padding C8/C16 high-reader sweeps also lost. C64's K-divisibility error was repaired using legal C40/C80 families; [geometry123](geometry123.json) C40/R5 won0.404ms versus0.425ms and was retained. Packed primary/residual and legal separate same-input alternatives were actually measured; selected packing/layout is not justified solely by fewer ops. |
| Remaining GDN recurrence/conv/movement | Exact update/readout, packed Q/K reuse, balanced owners, state rounding at the persistence boundary, affine and beta fusions are retained with exact probes. Grouped reader126, depth3conv141 and departed writer172 did not improve whole latency consistently and were rejected. Historical persistent-L1-state budget exceeded physical L1; any new partition must recompute final194 budgets. See [kernel contracts](kernel_contracts.md) and [attention contracts](kernel_contracts_attention.md). |

Explicit handoff, **not additional work or claimed rejection**: a cheaper
one-token prefill SwiGLU/tail program; additional B32 prefill movement fusion;
independent narrow-residual geometry reducing physical padding; further exact
norm fusion or recurrence NoC scheduling. None has final194 whole-path evidence.
Changing any of them would require exact own-input checks, unchanged PCC/context/
trace/determinism gates and warmed before/after timing. The satisfied stop rule
ends optimization here; it does not waive the remaining stage quality review.
