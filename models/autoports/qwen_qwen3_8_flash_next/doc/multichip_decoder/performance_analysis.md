# Multichip decoder performance analysis

CPU analysis of existing device captures, 2026-11-13. The default now uses
ownership-masked matmul reduction for logical n=1 and native post-combine
reduction for n>1. Final-source short contracts and long smoke pass; final
operator attribution is available for layers0/1/3. All final captures and
tt-perf-report commands passed. Final and historical attribution are separated
below; independent stage acceptance remains separate.
This document is not a whole-model or final-stage acceptance claim.

## Adopted change and final-source latency

The historical layer0 decode profile identified a231.835μs one-core
post-combine reduction. The selected replacement distributes the n=1 weighted
sum across2560 output features using matmul. Rank ownership is applied with
`where(..., values, 0)` **before** arithmetic so stale nonowned slots, including
NaNs, cannot enter the sum. It still executes only gate-selected experts.
Multirow inputs, including grouped B32 decode, retain native reduction; there
is no B32 reduction change or attributed B32 speedup from this optimization.

The historical tracked/audited layer0 experiment improved B1 decode from
2.590861ms to2.399465ms, a7.387% diagnostic latency reduction, and was kept
after correctness validation. These instrumented candidate numbers are not
the final clean timing comparison. See
[original contract](mesh_l0_contract.json),
[candidate result](mesh_l0_matmul_reduce_v2.json), and their provenance.
Two immutable ownership bounds are shared in`MeshWorkspace`:8192 bytes of
payload and65536 bytes of eight-bank reservation, charged inside the existing
static allowance; see[memory_capacity_plan.json](memory_capacity_plan.json).

[performance_summary.json](performance_summary.json) pins separate clean
timing and tracked correctness reports, source hashes, PCC and trace evidence.
The table uses `clean_timing_{baseline,mesh}_l{0,1,3}.json`: single-chip
OptimizedDecoder versus four-chip MultichipDecoder with allocation tracking,
runtime audit, watcher and profiler disabled. Prefill is the median of five
warmed eager33-token calls; decode is the recorded lower median of30 timed
trace replays after excluding two of32 replays. Restore/transfers are outside
timing. Speedup=baseline/mesh; efficiency=speedup/4. The tracked final contract
reports separately retain tails, stack, runtime-audit and allocation checks.

| Layer | Clean B1 mesh ms | B1 speedup / efficiency | Clean B32 mesh ms | B32 speedup / efficiency |
|---|---:|---:|---:|---:|
| 0 GDN |2.264450|1.358851x /33.971%|57.873066|1.671165x /41.779%|
| 1 PLE/GDN |3.431540|1.237314x /30.933%|95.144455|1.424109x /35.603%|
| 3 QSA |4.747707|1.043726x /26.093%|131.113815|1.163694x /29.092%|

All three[final short reports](final_mesh_l0.json),
[layer1](final_mesh_l1.json), and[layer3](final_mesh_l3.json) pass prefill,
decode, every1..32 tail, whole-batch B32, stack-boundary and runtime-fallback
checks; B1/B32 trace output and state are exact against eager execution.
Minimum whole-output PCCs are0.99961370 for prefill/decode,0.99954444 for
tails,0.99889278 for B32, and0.99630183 for stacked layers. Final32769-token
prefill plus decode passes for[PLE/GDN](final_mesh_l1_long32769.json) and
[QSA](final_mesh_l3_long32769.json), with minimum PCC0.99984349/0.99917488
and full-stack planning reservation. These are representative layer tests;
reservation does not instantiate the full model, and B32 capacity128 does
not imply B32 maximum-context capacity.

## Final layer0 operator attribution

The final[report provenance](tracy/final_mesh_l0/provenance.json) links the
valid raw operation CSV, exact commands, and prefill/decode human-readable
tables and CSVs. [final_operator_summary.json](final_operator_summary.json)
records sums, categories, families, source CSV hashes and the reduction row
for all three layer kinds. Values sum merged worst-rank device
rows, not wall time or a critical path. Segmented profiler-drain gaps and
`Total %` cannot establish throughput.

| Final layer0 phase | Merged rows | Summed device time, ms | AllGather rows / μs | Tool CCL rows / μs | Native post-combine rows / μs |
|---|---:|---:|---:|---:|---:|
| Prefill33 |744|21.243162|20 /436.381|24 /654.423|1 /192.228|
| Decode1 |283|2.14309175|10 /133.53075|12 /166.98675|0 /0|

Native post-combine reduction is absent from final decode. Prefill retains
one native row for the32-token chunk; its one-token tail uses the new path.
Decode's replacement matmul is merged row19438, global call4975617,
selected device1: logical M/K/N=`1/10/2560`, physical`32/32/2560`,
**3.593μs,88 reported cores, HiFi4 BF16×BF16→BF16**, FP32 destination
accumulation enabled, packer L1 accumulation disabled. Inputs/output are
TILE DRAM-interleaved. The raw attributes confirm1D multicast, grid11×10,
one-tile K block and1×1 subblock. The88 count describes program participation;
it does not mean88 useful output tiles. The matmul row excludes ownership
masking and layout preparation, so its ratio to the old231.835μs row is
not an end-to-end speedup. The historical tracked whole-layer diagnostic benefit and final clean
baseline/mesh comparison are distinguished above.

| Final layer0 data-movement evidence | Prefill rows / μs | Decode rows / μs |
|---|---:|---:|
| Tool TM category |340 /1521.852|149 /451.171|
| Tool DM category |16 /52.527|11 /36.710|
| Typecast |132 /279.151|67 /130.143|
| Slice |98 /224.687|25 /34.985|
| Concat |12 /135.673|3 /62.239|
| ReshapeCodegen |70 /1214.702|7 /24.182|
| UntilizeCodegen |2 /275.249|1 /10.171|

Family rows overlap tool categories and must not be added to category totals.
Codegen/generic operations remain partly unclassified. Tool CCL still includes
dispatch/combine with`dispatch_group_size=1`, so12 CCL rows do not mean12
network collectives. Actual decode AllGather attribution is10 rows/133.53075μs.

The remaining largest individual decode rows are GenericOp132.779μs,
fused MoE116.139μs, BinaryNg89.813μs, and two one-core GR injection matmuls
84.213/84.133μs. Family totals are BinaryNg414.659μs, Reduce310.327μs,
GenericOp242.795μs and Typecast130.143μs. Injection remains168.346μs in
total; removing the old reduction does not resolve its one-output-tile limit.
The new matmul's modeled91.770GB/s/17.924% DRAM and1.199% FLOP utilization
describe a very small operation, not a demonstrated bandwidth ceiling.
Its K is one tile, so generic advice to increase`in0_block_w` beyond1
requires padding or changing the geometry; retain numerical validation.

## Final PLE/GDN and QSA attribution

Final[layer1 report provenance](tracy/final_mesh_l1/provenance.json) and
[layer3 report provenance](tracy/final_mesh_l3/provenance.json) link both
phase CSVs and human-readable tables. All four layer3 report-generation
commands exit0. Its prefill stdout is split at line boundaries; the
original`.txt` is an index and provenance records ordered parts plus the
reconstructed SHA256. Repository hooks subsequently normalized trailing
whitespace and final newlines, as recorded in provenance. Numeric report
content and CSV data are unchanged.

| Final phase | Merged rows | Summed device time, ms | AllGather rows / μs | Tool CCL rows / μs | Native post-combine rows / μs |
|---|---:|---:|---:|---:|---:|
| Layer1 prefill33 |1144|26.31009775|22 /407.21475|26 /670.05175|1 /190.307|
| Layer1 decode1 |487|3.191712|11 /154.145|13 /187.648|0 /0|
| Layer3 prefill33 |1719|79.273100|18 /424.563|22 /704.006|1 /191.553|
| Layer3 decode1 |397|4.57938475|9 /129.46975|11 /163.60675|0 /0|

Every final decode has zero native post-combine rows. The n=1 replacement
measures3.551μs on layer1 and3.548μs on layer3, each with88 reported cores
and HiFi4 BF16×BF16→BF16. Prefill retains one native32-row reduction and
one matmul for its tail. These timings exclude the associated masking and
layout preparation; they do not establish a standalone end-to-end speedup.

| Final phase | Compute category μs | TM rows / μs | DM rows / μs | Unclassified/Other μs |
|---|---:|---:|---:|---:|
| Layer1 prefill |13321.359|498 /2926.121|25 /71.989|9320.577|
| Layer1 decode |1498.305|230 /724.690|18 /53.702|727.367|
| Layer3 prefill |8443.335|1009 /3616.719|214 /390.942|66118.098|
| Layer3 decode |1224.249|217 /580.022|20 /53.243|2558.264|

The large Other category includes real compute and movement, not idle time.
Layer1 decode has BinaryNg610.642μs, GenericOp428.431μs and
Reduce379.249μs. Its one-core UINT32 GenericOp row28969 costs133.887μs;
raw shapes identify the striped code-table lookup with table width50000256
and16×20 selected words (`PackedEmbedding._lookup_striped`,
`tt/striped_gather.py`). This is a concrete lookup-parallelism candidate,
but the report supplies no reliable DRAM utilization for this generic op.
Layer1 decode Typecast totals154.712μs and UntilizeWithUnpadding129.108μs;
changes to lookup or representation must include those surrounding costs
and preserve row/scale addressing and full PLE output parity.

QSA's replicated ordered index projection dominates: row33012 is60194.890μs
for the32-token prefill chunk; tail row38120 is2099.663μs; decode row39715
is2099.874μs. Raw kernel paths identify
`tt/kernels/ordered_projection_compute.cpp`; BF16 weights are2560×640,
FP32 temporaries are32×640, and20 cores execute the pinned pair-order
algorithm in[ordered_projection.py](../../tt/ordered_projection.py).
The output is subsequently converted to BF16. This behavior preserves
sensitive index selection, so replacing it with an ordinary matmul or
lowering fidelity is not an accepted optimization. Output-column parallelism
across ranks while retaining each column's K order, followed by reconstruction
of the640-wide indexer result, is a concrete future experiment requiring
exact projection/selection and whole-decoder checks.

QSA decode selected-KV materialization is another GenericOp, row40216,
236.470μs on65 cores; raw inputs include local paged K/V, page table and
2080 selected IDs, matching`local_selected_paged_kv`. SDPA costs165.384μs
and routed FFN135.125μs. GenericOp totals2388.248μs, versus129.46975μs
AllGather; CCL-only changes cannot remove the dominant replicated indexer.
Prefill also has99 paged-cache-update rows/276.543μs and99
interleaved-to-sharded rows/70.224μs; these counts come from the CSV, not
an estimated public graph.

Modeled DRAM rates remain partial evidence: layer1's packed GDN projection
uses55 cores at202.896GB/s/39.628%; its two2560-wide PLE matmuls use88 cores
near135.7GB/s/26.5%. QSA's packed projection uses66 cores at215.705GB/s/42.130%,
and its output projection88 cores at132.083GB/s/25.798%. The ordered indexer,
lookup, selected-cache and fused FFN rows lack trustworthy tool utilization.
Do not call them bandwidth-saturated or estimate dense512-expert FLOPs.
AllGather is only one part of movement; TM/DM labels exclude these custom
kernels. Fidelity and DRAM-grid advice must therefore be tested against the
exact source arithmetic, memory ledger, and producer/consumer timing scope.

## Historical valid operator attribution and its limits

The generated `tt-perf-report` tables, CSV exports, and exact command/raw-CSV
provenance are under [tracy/mesh_l0_v2](tracy/mesh_l0_v2/provenance.json) and
[tracy/baseline_l0_segmented](tracy/baseline_l0_segmented/provenance.json).
Both have prefill/decode `.txt` and `.csv` reports. These valid captures
predate the adopted n=1 matmul reduction and cannot be presented as the final
operator graph or its final bottleneck proportions. Prefill is logical33,
executed as32+1; decode is one token. The earlier unsegmented baseline and
first mesh capture lost markers and are excluded; see
[profiler_audit.md](profiler_audit.md).

Values here sum the report's `Device Time [μs]` column. The mesh report merges
four devices using per-operation device attribution; the selected worst-rank
device can change between operations. These sums are not a critical path,
wall-clock latency, or measured scaling efficiency. Baseline segmented
capture has intentional profiler-drain gaps; its `Total %`, op-to-op gaps,
and automatic tracing-savings advice must not be interpreted as throughput.
Use separate warmed trace runs for latency/speedup/efficiency.

| Layer0 report | Merged op rows | Summed device time, ms | AllGather rows / time, ms |
|---|---:|---:|---:|
| Baseline prefill | 1212 | 61.132905 | 0 / 0 |
| Mesh prefill | 741 | 21.575215 | 20 / 0.473900 |
| Baseline decode | 215 | 2.955560 | 0 / 0 |
| Mesh decode | 280 | 2.378958 | 10 / 0.134801 |

These collective counts come from CSV rows, not an estimated source graph.
The tool's broader CCL category contains24 prefill rows/0.692121ms and12
decode rows/0.168322ms: AllGather plus native dispatch/combine. Dispatch and
combine use `dispatch_group_size=1` in this implementation; their CCL label
does not establish inter-device traffic. Mesh routing inputs are replicated,
experts are EP4, and rank contributions are reduced by gather plus ordered
FP32 summation.

## What dominates

| Operator family | Baseline prefill, μs | Mesh prefill, μs | Baseline decode, μs | Mesh decode, μs |
|---|---:|---:|---:|---:|
| BinaryNg | 27605.825 | 7606.866 | 945.978 | 410.182 |
| Reduce | 13445.458 | 3882.989 | 532.852 | 308.818 |
| GenericOp | 11291.645 | 4339.858 | 515.852 | 242.907 |
| Typecast | 270.744 | 293.983 | 122.279 | 128.793 |
| Native ordinary Matmul | 529.072 | 603.160 | 264.034 | 301.199 |
| Routed expert matmul/fused FFN | 2201.484 | 751.828 | 152.801 | 115.988 |

The last row compares baseline SparseMatmul with mesh MoeFusedSwiGlu; it is
not identical operator scope or datatype. The mesh fused op includes
gate/up, activation, and down, while baseline activation/cleanup appears
elsewhere. Neither sparse active count nor fused FFN FLOP/DRAM utilization
is reliably populated by these tables. Do not fill missing utilization
using512 dense experts: execution remains gate-selected.

Historical mesh decode's largest individual row is post-combine reduction:
231.835μs on one core (ID19317). The adopted n=1 replacement, measured
whole-layer benefit and final layer0 per-op attribution are recorded above.
The next
large rows include GenericOp132.790μs (ID18535), fused MoE115.988μs
(ID19270), and BinaryNg102.401μs (ID18890). Two GR injection matmuls consume
84.220 and84.241μs (IDs18481/19120),168.461μs combined. Their physical shape
is32x10240x32, with one output tile on one core. Decode M and N each occupy
one tile, so ordinary M/N partitioning or a larger core grid cannot create
additional work. Parallelism requires a K reduction or a different output
packing, both with numerical validation; blindly increasing the grid is
not an actionable fix. The report models about16GB/s and3% DRAM utilization
for these rows, consistent with poor parallelism rather than a demonstrated
full-chip DRAM-bandwidth ceiling.

The wider GDN packed projection32x2560x1568 uses55 cores,41μs, and modeled
202GB/s/39.5% DRAM utilization. Shared-expert160-wide gate/up matmuls each
use five cores and about31μs; their five output tiles explain limited
column parallelism. Source precision and full-layer PCC must constrain
any lower-fidelity experiment. The perf tool's generic suggestion of HiFi2
is not numerical acceptance for sensitive projections or routing inputs.

Data movement is material: mesh decode has65 Typecast rows/128.793μs,
29 Slice rows/49.185μs, and4 Concat rows/63.162μs. Prefill has69
ReshapeCodegen rows/1210.916μs. The tool leaves several codegen and generic
ops unclassified, so its TM/DM totals are not a complete physical data
movement census. The explicit TILE-to-ROW_MAJOR combine input conversion
is a correctness repair and must remain until an alternative passes the
same slot-level and whole-output checks. Communication is visible, but
AllGather's134.801μs decode total is smaller than BinaryNg, Reduce,
post-combine reduction, and combined GR injection; optimizing CCL alone
does not address the largest measured contributors.

## Precision actually executed

`MeshMoe` in [multichip_decoder.py](../../tt/multichip_decoder.py) stores
routed experts as BFP4, calls the native fused FFN with LoFi,
`fp32_dest_acc_en=False`, and `packer_l1_acc=False`; its public activation
input/output is BF16. Its shared expert uses BFP8 weights and the loader's
HiFi4/FP32-destination compute policy. GR exact projections use FP32 weight
storage and explicitly preserved BF16 boundaries.

The fused native FFN is not exclusively BF16 activation arithmetic despite
the top-level CSV displaying `BF16, BFP4 => BF16`. In
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_geometry.cpp:328-347`,
CB_X_TILES, CB_H, gate/up gather/accumulator, and hidden-slice/local CBs use
`FormatKey::Bfp8`. `moe_fused_swiglu_program_factory.cpp:97-104` maps that
to `DataFormat::Bfp8_b`; weight format derives from actual input tensor
dtype at386. Thus BFP4 weights, LoFi compute, and BFP8 internal staging are
all part of the tested mesh policy. The single-chip table's sparse rows
instead show LoFi BF16xBFP8=>BF16. Cross-policy accuracy is established by
decoder comparisons, not by assuming identical arithmetic.

## Residual-layout and collective experiments

[residual_contract.json](residual_contract.json) and its
[provenance](residual_contract_provenance.json) compare a common native
row-parallel real layer0 WO, collective, residual write, and complete next
MLP GR read. Producer features are seeded synthetic; residual inputs and
weights are real baseline operands. This is warmed eager producer/consumer
scope, not shipping exact-GDN or whole-layer timing.

| Candidate | T1, ms | T2, ms | T31, ms | T32, ms |
|---|---:|---:|---:|---:|
| FP32 gather then sum | 0.990671 | 1.632232 | 3.404884 | 3.507201 |
| BF16 gather then widen/sum | 1.101083 | 1.633655 | 3.429110 | 3.466622 |
| Native FP32 all-reduce | 0.948009 | 1.648282 | 3.250868 | 3.369436 |
| Hidden-sharded FP32 RS | 1.418001 | 1.810404 | 3.519524 | 3.670263 |

All16 cases pass residual/mixed/injection whole-output PCC; the minimum
across reported metrics is0.9999911403. BF16 transport is bit-exact with
the FP32-gather control for all tested outputs and halves transported
operand width before widening. Its five-sample eager medians are not a
uniform latency win. Hidden RS stays sharded through residual addition and
distributed per-stream norm, then gathers normalized values for exact GR
projections. It does not immediately undo RS with AG. This measured
implementation is slower at every tested length; that result does not
prejudge fused matmul-RS or AG-matmul candidates, whose target-shape recipes
are in [indexed_decode_audit.md](indexed_decode_audit.md).

The integrated persistent native-AR candidate completes all tails and16
exact B1/B32 trace replays, but two per-request B32 diagnostic PCCs are
0.990708/0.990973, missing the historical probe's stricter per-request gate.
That report did not record whole-B32 aggregate PCC; it does not prove failure
of the user's whole-output gate. B1 decode2.547588ms and B32
decode56.850929ms offer only about2% gain in the owner's comparison.
The candidate is conservatively rejected for less accurate per-request
results and small measured benefit, and ordered
gather/FP32 accumulation remains selected. See
[mesh_l0_persistent_ar.json](mesh_l0_persistent_ar.json) and
[AUTOFIX.md](AUTOFIX.md); this is an optimization failure, not failure of
the passing selected path.

Indexed decode remains a design comparison, not a blanket rejection.
Native sparse_matmul requires packed tensors rather than the current
list-of128 expert allocations; no safe existing Python zero-copy bridge
was found. Keeping both formats for48 layers adds16.9869312GB/device and
does not fit. Packed-only prefill/decode is still a viable design to test.
EP indexed decode also needs a static trace contract for rank-local active
counts0..10; the native API derives its count from index shape. TP4 avoids
that count mismatch with ten selected fragments per rank but has a
five-tile160-wide intermediate. Padding to192 exceeds the current
maximum-context headroom without another storage saving. These are exact
storage/API constraints and measurement targets, not unmeasured claims
that indexed or TP execution is slower.

## Advice, decisions and remaining experiments

| Area | Evidence / constraint | Decision and next useful measurement |
|---|---|---|
| Post-combine n=1 | Historical231.835μs single-core row replaced by3.548–3.593μs matmul plus preparation; historical tracked layer0 B1 latency improves7.387% (diagnostic only); final three-layer and long contracts pass | Keep ownership-masked matmul for n=1. Retain native n>1/B32 reduction. All final profiles confirm no native post-combine row in decode. |
| CCL transport | BF16 gather then widen is bit-exact in all residual probe outputs; native AR misses two per-request diagnostic thresholds, with no historical whole-B32 aggregate | Keep ordered accumulation conservatively. Do not infer latency improvement solely from narrower payload or source-level collective counts. |
| Residual ownership | Shape-faithful hidden RS through residual write and next GR consumer is slower at all four tested lengths | Keep stream ownership for this implementation. This does not reject an unmeasured fused RS path. |
| Combine layout | Native TILE combine corrupted selected slots; explicit ROW_MAJOR conversion passed slot-level and layer contracts | Keep the conversion. Remove it only after an alternative reproduces ownership, finite values, tails and whole outputs. |
| GR injection | One physical M tile and one N tile,32x10240x32; two historical rows total168.461μs | Enlarging the ordinary M/N grid cannot distribute one output tile. Evaluate explicit K partition/reduction or combined output packing, with exact-GR and whole-layer PCC. |
| QSA indexer | Replicated20-core ordered projection costs60.195ms for32 tokens and2.100ms for decode; pair order affects index selection | Measure output-column mesh partition preserving K order, reconstruct indexer output, and validate exact selection. Current evidence does not establish an ordinary matmul replacement. |
| PLE lookup | One-core striped UINT32 code lookup costs133.887μs in decode; generic utilization absent | Evaluate parallel selected-word lookup with the same packed code/scale mapping; include conversion costs and PLE history/output checks. |
| DRAM geometry | Injection modeled16GB/s/3%; wider projection202GB/s/39.5%; shared160-wide projections have five output tiles | These rows do not prove a DRAM bandwidth ceiling. DRAM-sharded weights, bank placement or alternate multicast grids require actual target-shape timing and capacity checks. |
| Fidelity | Routed BFP4 weights, LoFi and internal BFP8 are already executed; shared weights BFP8; sensitive exact projections retain FP32/BF16 policy | Perf-tool HiFi2 advice is not an accuracy result. Further fidelity/storage changes need separate scope, full-layer PCC and memory accounting. |
| Indexed active experts | Native sparse op needs packed weights; current fused grouped op uses128 tensor allocations/rank; dual full-stack formats exceed memory | Packed-only remains viable. Resolve static trace active-count0..10 and prefill storage compatibility; do not label it slower without measurement. |

Unmeasured future performance work includes fused matmul-RS/AG-matmul using
the exact shape/API recipes in[indexed_decode_audit.md](indexed_decode_audit.md),
packed-only indexed EP or TP4 experts, injection K reduction, DRAM geometry,
and precision/fidelity changes. TP4's160-wide intermediate has five tiles;
padding to192 needs a compensating storage saving to preserve context.
These are open engineering experiments, not evidence that the current path
is optimal or that a performance stop rule has been satisfied.

Final layer0/layer1/layer3 attribution above supersedes historical profiles
for current source. All final report-generation commands passed, and the
human-readable tables, CSV sums and source-specific advice were reviewed.
Trace latency, watcher acceptance and independent stage review remain
separate evidence and completion gates.
