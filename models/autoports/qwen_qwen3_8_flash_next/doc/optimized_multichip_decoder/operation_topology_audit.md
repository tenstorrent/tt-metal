# Stage 5 operation-topology audit

CPU-only starting audit, 2026-11-13, checkout `79582226b281a19aba1cc5ee3dd4fb80aa87317c`.
No device experiment or implementation change was performed for this audit. This
is a candidate plan, not an optimization acceptance claim. References to source
line numbers describe the starting source; named classes remain the stable keys.
Installed optimize method, matmul and advisory references and repository
`tech_reports/LLMs/llms.md` section 3.3 were consulted.

## Starting contract and measured priority

The target is all four p300c Blackhole devices, logical 1x4 ring. Each device
owns one entire 2560-wide hyperconnection residual stream, local attention heads,
128 complete experts, and one quarter of packed PLE rows. Decoder input and
output are BF16 TILE DRAM-interleaved `[1,1,T,2560]` per rank. There is **no
inter-layer collective or reshard**. Preserve this contract across layer kinds
unless an adapted whole producer/consumer family is measured faster. A gathered
comparison tensor belongs outside the measured layer.

The final stage-4 CSVs were read directly, rather than relying only on the prose
report. These are sums of merged worst-rank per-operation times, not critical
paths or wall latency. Their segmented profiler-drain gaps are unsuitable for
throughput claims.

| Layer kind | Decode rows / summed device microseconds | Actual AllGather rows / microseconds | Leading opportunities |
|---|---:|---:|---|
| 0 GDN | 283 / 2143.09175 | 10 / 133.53075 | Exact projection GenericOp 132.779; routed fused FFN 116.139; two one-core injection matmuls 168.346 combined; BinaryNg and Reduce families |
| 1 PLE/GDN | 487 / 3191.712 | 11 / 154.145 | Striped lookup 133.887 on one core; exact projection 132.879; injection; PLE transforms |
| 3 QSA | 397 / 4579.38475 | 9 / 129.46975 | Ordered index projection 2099.874 on 20 cores; selected-cache materialization 236.470; SDPA 165.384 |

Sources: `../multichip_decoder/tracy/final_mesh_l{0,1,3}/decode_perf_report.csv`,
the corresponding `.txt` tables and `provenance.json`. Those provenance files
name persistent raw CSVs under `/home/ttuser/qwen38-flash/multichip_profiles/`.
Stage-4 clean timing and correctness are separate artifacts, described in
`../multichip_decoder/performance_summary.json`. Recollect stage-5 final-default
timings; these starting attribution rows cannot stand in for them.

## Operation audit before tuning

All implementation references below are relative to the model root unless they
start with `ttnn/`. CCL payload quantities are specified separately below.

| Producer and current sequence | Repeated work / movement | Candidate and arithmetic constraint | Action / prior evidence |
|---|---|---|---|
| `tt/multichip_decoder.py:MeshResidual.read` (137–148): local FP32 RMSNorm → normalized stream AG → exact down → small AG/slice → exact up → weighted-stream AG/FP32 sum; ordinary injection reads the same gathered normalized input | Two GR calls per decoder. Down and injection share input but differ dtype/arithmetic: FP32-weight accurate down versus ordinary BF16 injection. Full normalized gather is already shared between them | Pack compatible output work only if the precise reduction and BF16 boundaries survive; otherwise parallelize the one-column injection over K, or pack the four injection outputs and distribute after one larger projection. Preserve both injection and next GR outputs | Injection is 32×10240×32 with one useful output column/rank, about 84 microseconds twice. More M/N grid cores alone cannot fix it. K-split/packed variants are untried |
| `MeshGDN.__call__` (222–227): same x → ordinary packed z/a/b and separate exact packed QKV | Already packs z/a/b, and QKV separately; naively combining both would replace QKV's pinned summation | Compare a packed family retaining exact QKV arithmetic, or keep separate exact QKV and tune z/a/b. Include slices and exact-kernel layout costs | Exact QKV uses reordered K and a dedicated kernel (`tt/fused_decoder.py:FusedExactQKV`), so generic native packed-QKV fusion is not an equivalent drop-in. No stage-5 target experiment yet |
| `MeshGDN.out` (205–207,303): local1536 → AG6144 → exact output-column projection640 → AG2560 | Two collectives, custom precise reduction; a conventional row-parallel projection is a different numerical algorithm | Compare AG+native output-column MM and fused AG-MM against exact projection, carrying hidden640 output into residual write/next GR; compare local full-output MM+RS and fused MM-RS with the same consumer | Stage-4 hidden residual probe is useful but uses ordinary matmul, not the shipping exact GDN projection. Fused families remain untried |
| `MeshQSA.build/forward_chunk` (981–994,1014,1084): packed Q/gate,K,V; local heads and local KV; native output MM1536→2560; ordered gather/sum | Packed QKV already adopted. Row-parallel output and subsequent reduction can be fused or moved | Best initial native fusion target because QSA WO is already a native BFP8 matmul. RS2560→640 must feed a hidden-sharded residual consumer; AG local1536→6144 plus output-column640 is another decomposition | Compare all native/fused families with the same activation/weight precision and producer/consumer scope; no target fused evidence yet |
| QSA ordered indexer (`tt/ordered_projection.py`, `tt/kernels/ordered_projection_*.cpp`) | Identical 2560→640 projection redundantly runs on four devices; 60.195 ms for the32-token prefill chunk and 2.100 ms decode in stage4 | Output-column mesh shard to160/rank, retain K pair order and RNE BF16 conversion, AG640 after projection. Also distribute token/column work over more local cores while retaining each output's ordered dot product | Highest payoff for QSA. Current wrapper fixes640/20cores; adapting reader/writer stride, validation and weight packing is needed. Do not infer impossibility from this wrapper assertion |
| `MeshMoe` (565–695): exact router+AG512 → token-selected grouped dispatch → fused SwiGLU → TILE-to-ROW_MAJOR combine → selected weighted reduction → shared expert → ordered AG/sum | Native dispatch/combine are tagged CCL but use dispatch_group_size1; those tags do not prove Ethernet traffic. Expert execution remains selected only. Combine conversion is large and correctness-sensitive | Keep selected execution. Tune grouped geometry/precision; compare packed-only indexed EP B1 where trace-active-count contract is solved. Do not introduce a union-masked all-expert B32 path | Earlier TILE combine corrupted selected slots; explicit ROW_MAJOR repaired them. n1 reduction already uses ownership-masked matmul; n>1 retains native reduction. Removal needs slot-level plus whole-output evidence |
| `MeshMoe.shared` (615–622,691–694): same x → gate160, up160, scalar route; shared down160→2560 | Two narrow native projections, five output tiles each; scalar route is another same-input matmul | Pack gate/up to320 and compare split with activation fusion; include split/slice/multiply/down and CCL. Consider adding scalar route to the pack only with inert padding and exact sigmoid boundary | Untried; lower fidelity and DRAM geometry must be crossed with selected packing |
| `MeshPLE` (1432–1451), inherited `tt/ple.py` | Key and value consume embeddings, key is stream-specific output2560, value2560 is replicated. Per-rank ordinary projections plus norm/conv/elementwise movement | Pack key/value if their actual inherited call inputs match; otherwise tune separately. Parallelize stripe selected-word lookup and include untilization/scale decode | Lookup ownership is row%4, then ordered sum. Preserve packed code and scale addressing and history state. No target parallel-lookup measurement yet |
| `Collectives.gather/sum` (61–75) | Synchronous AG, DRAM result; sum AG along dim1, BF16 transport then FP32 ordered local reduction. Every invocation allocates framework-managed output | Async AG with explicit ring, links, semaphores and preallocated shape-specific output first; preserve local reduction arithmetic. Then native AR family separately | Existing persistent async AR passed allocation/tail/replay but changed numeric output. It does **not** reject exact-order async AG persistence |

`tt/decoder_components.py:linear` selects custom precise projections by type;
ordinary tensors use `ttnn.linear` with DRAM outputs and the loader's HiFi4,
FP32 destination accumulation, no packer L1 accumulation. Ordinary weights are
DRAM-interleaved; no selected DRAM-sharded decode config is present. Therefore
phase-specific L1 activation shards, DRAM-sharded weights and one/two/three
reader tests are open candidates for the ordinary projection roles, not evidence
that the custom exact kernels can be replaced without validation.

## Payloads and persistence plan

For a four-rank ring all-gather, logical transport per rank is approximately
three times its local input payload (excluding packet overhead). For a reduction
scatter it is three quarters of the full partial payload. Decode rows are
physically padded to32. The table shows BF16 physical B1 payloads, not network
counter measurements; multiply by physical_rows/32 for larger tiled operands.

| Collective | Local input bytes | Approx. ring bytes/rank | Output bytes/rank |
|---|---:|---:|---:|
| Normalized stream AG2560→10240, twice | 163840 | 491520 | 655360 |
| GR down AG96→384, then remove rank padding to320, twice | 6144 | 18432 | 24576 |
| Weighted stream or MoE partial AG(dim1)2560 | 163840 | 491520 | 655360 |
| GDN head AG1536→6144 | 98304 | 294912 | 393216 |
| GDN output AG640→2560 | 40960 | 122880 | 163840 |
| Router AG128→512 | 8192 | 24576 | 32768 |
| QSA WO RS2560→640 candidate | 163840 | 122880 | 40960 |
| Sharded ordered-indexer AG160→640 candidate | 10240 | 30720 | 40960 |

FP32 CCL doubles these payloads. BF16 transport followed by widening preserves
the original BF16 operands; BFP8 transport is a distinct accuracy experiment.
Native reduction algorithms may change summation and rounding even at FP32.
The current gather does not explicitly pass `num_links`; physical two-link
connectivity is not proof that its API default uses two links.

Persistent buffers should belong to a stage/model workspace, indexed by physical
shape, dtype, dim and lifetime, with semaphores initialized before capture.
Do not reuse one output arena across two simultaneously live gathered tensors
(for example GR full input and down output), nor overwrite a retained trace
output before its consumer. Serial CQ0 alone does not prove alias safety.
Record intended trace allocations and repeated changed-input replay with
allocation tracking. Buffer additions belong in the context ledger.

## Coherent producer/consumer comparison matrix

| Family | Residual before → after / next consumer | Projection / CCL precision | Movement and persistence | Status |
|---|---|---|---|---|
| S0 current stream ownership | stream2560 → stream2560; local full-stream RMSNorm | Sensitive exact GDN/GR; ordinary QSA BFP8/HiFi4; BF16 AG then local FP32 sum | Framework output allocation; GR gathers normalized streams internally; no inter-layer conversion | Accepted stage4 reference |
| S1 exact-order async | Same as S0 | Identical math, BF16 AG; separately test BFP8 communication only if numerical gate permits | Preallocated AG outputs, fixed semaphores, explicit ring/link/workers; avoid changing reduction arithmetic concurrently | Untried, strong low-risk CCL experiment |
| S2 native AR | Same as S0 | BF16/FP32 AR alternatives at same projection policy | Persistent RS/AG intermediates and semaphores; removes explicit gathered partial plus local sum | Stage4 async AR evidence exists; whole-B32 aggregate missing, so old per-row miss is not a valid whole-output rejection |
| H0 native MM+RS | hidden `[4,T,640]` → same; residual add and distributed per-stream norm, then normalized gather for exact GR | BF16 operands, FP32-output projection/RS control then BF16 candidate; separate QSA and GDN sensitivity | Carry640 block through four residual streams. Gather four injection scalars, not full block. Persistent partial/output and norm-stat buffers | Stage4 shape-faithful producer/next-consumer slower at T1/2/31/32; not fused or full-layer acceptance |
| H1 fused MM-RS | Same as H0 | Match H0 policy and geometry; BF16/BFP8 candidates crossed with same family | Full-output weight shards along K; persistent fused intermediate and640 output; no immediate restoration to replicated block | Untried |
| H2 AG-MM | Same as H0 | Real output-column weights6144×640; native and fused AG-MM under same dtype/fidelity | Gather local1536 input, produce640 directly, feed H0 consumer. Persistent6144 gather buffer; output does not gather at this boundary | Untried; exact GDN/native conversion requires independent check; QSA is a better native first target |
| S3 projection packing + L1 working shards | S0 stream boundary, or H families if compatible | Compare shared packed320 vs split160 at identical dtype/fidelity, then LoFi/HiFi2; QSA packed3584 vs split control | One shared conversion into the work shard per same-input group; include restoring required residual/norm contract only at its true boundary; DRAM reader1/2/3 with same precision | Untried |

Stage4 `residual_contract.json` measured H0 through residual addition and the
next full GR read; it did not immediately undo RS. Its T1 latency was1.418001ms
versus0.990671ms for FP32-gather control, and all four tested lengths lost.
This is legitimate rejection of that measured implementation, not rejection
of H1/H2 or lower-precision compatible families. The stage4 probe uses seeded
producer features and real weights/residual operands, so it is screening
evidence; real model whole-layer numerical tests decide integrated acceptance.

## Exact fused/async API entry points and retries

* `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/all_gather_async_nanobind.cpp`
  exposes the persistent-buffer overload, semaphores, optional cluster axis,
  barrier, chunks/workers/buffers. Preserve rank ordering and dim1 for sums.
  Test dim3 and dim1 shapes separately, changed inputs, cached variants and
  trace replay before integrating all collectives.
* `.../all_gather_matmul_async/all_gather_matmul_async_nanobind.cpp` accepts
  persistent output, global semaphore, core-grid offset, separate AG/MM memory
  configs, explicit program config, barrier and workers/buffers. Its device
  validation requires rank4 input/weights, input leading dimensions1,1 and
  dim3. There is no public cluster_axis argument in this fused API. Interleaved
  AG storage is the simplest first legal shape; sharded AG has an additional
  ring-size/core-shard restriction.
* `.../matmul_reduce_scatter_async/` requires dim3 and a 2D multicast matmul
  program config. Start local `[1,1,T,1536]` × `[1,1,1536,2560]`, output640,
  explicit CCL worker placement disjoint from MM workers and FP32 intermediate.
  `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/watcher/test_matmul_reduce_scatter_fp32_bh.py`
  is a target-platform construction reference. Start smaller legal subblocks
  if L1 fails; do not dismiss fusion at the first default-config error.

The complete starting shape recipes and exact storage limitations are in
`../multichip_decoder/indexed_decode_audit.md`. Existing source/API coverage
does not prove target-model numerical equivalence or latency. When a native
fused projection fails the exact GDN gate, try output-column ordered/custom
projection plus async communication or retain GDN exact while measuring the
native family on QSA. GR exact SFPU products and ordered-indexer pair order are
not native matmul program-config choices.

## Prioritized experiment order and acceptance scope

1. Record same-harness final-stage4 multichip baseline B1/B32 and prefill short,
   long and32-boundary lengths. Retain real diverse routing and all three
   meaningful kinds. Final optimized numbers must come from final defaults.
2. Shard/order-preserving parallelize QSA indexer output, the largest measured
   cost. Require exact indexer BF16 and selected IDs for changed inputs, then
   whole-output PCC, cache-consuming trace and awkward tails. Measure its
   added640-wide AG together with the projection.
3. Async persistent AG, retaining ordered sum; then shared gate/up packing and
   injection K/packing families. Measure whole layer, not just launches saved.
4. QSA WO H0/H1/H2 producer/next-consumer family; repeat GDN only with its
   sensitive exact control. Sweep compatible activation/CCL dtype and persistent
   buffers together, and record shape/layout adaptations after failures.
5. Ordinary projection DRAM sharding, L1 working layouts, legal K blocks and
   Blackhole readers1/2/3. Cross fidelity LoFi/HiFi2 with selected geometry;
   capture measured rows proving dtype/fidelity actually reached the ops.
6. PLE lookup parallelism and grouped MoE geometry/batch32 attribution. Packed
   indexed EP is an open candidate, but dual expert formats cost16.99GB/device
   extra over48 layers and are physically excluded. Packed-only remains open.

Expert BFP4/LoFi with internal BFP8 staging is inherited from accepted stage4
because all-BFP8 expert weights plus PLE exceed32GiB before context state.
Document that inherited physical constraint separately from any new stage5
precision choice; the optimize default prohibition is not permission to make
new groups BFP4 without the later datatype decision. Precision reductions in
new groups remain recorded candidates, not silently selected defaults.

Every kept candidate must preserve the advertised context ledger, nonaligned
logical lengths, gate-selected experts, whole-output baseline PCC, B1 and
diverse B32, no runtime host fallback, changed-input allocation-tracked replay
and a separate watcher run. A first TTNN/API error is a retry/AutoFix trigger.
If the optimize stop rule fires, explicitly move untried rows to the work-log
handoff section; this audit alone neither satisfies those experiments nor
waives correctness gates.


## Stage5 evidence update: coherent families and initial defaults

This is an intermediate readback of completed artifacts, not a stage pass.
The earlier matrix records the starting audit; its “untried” labels are historical
and are superseded below where a measured experiment exists. All latencies below
are milliseconds on the 1×4 target mesh. Clean timing runs are distinct from
allocation-tracked, runtime-audit and watcher acceptance runs.

### Lower-movement producer, residual and next-consumer family

`fused_family_HiFi4.json` and `fused_family_LoFi.json` exercise real QSA WO weights
at logical Q1,2,31,32, with internally padded32-row producer inputs. Native
MM+FP32 RS, fused MM-RS, AG+MM and fused AG-MM all feed the hidden-sliced residual
update and entire next exact GR directly. There is no immediate gather/restore
to stream ownership. The component has two eager input variants; its optional
trace repeats one unchanged input, with at most one live trace. It is screening,
not changed-input trace acceptance. Existing JSON predates the added own-eager
bit-exact trace check; do not retroactively claim that field was measured.

At Q1/HiFi4, traced ordered BF16 control was0.7177, native RS0.5836, fused
RS0.6016, AG-MM0.5948 and fused AG-MM0.5835. Eager ranking differed. That control
predates improved injection10, so component gains alone cannot select a default.
The following whole-decoder comparison uses the improved stream control:

| Whole QSA path | Warm prefill33 | Traced B1 | Traced B32 | B32 whole-output PCC |
|---|---:|---:|---:|---:|
| Stream control, injection10 + parallel valid-row indexer | 41.991554 | 2.999988 | 75.122368 | 0.999922335 |
| Hidden residual, native FP32 RS | 41.900307 | 3.008545 | 76.548355 | 0.999916553 |
| Hidden residual, fused AG-MM | 42.017933 | 2.999908 | 76.450859 | 0.999918461 |

Sources: `hidden_stream_control.json`, `hidden_qsa_rs.json`,
`hidden_qsa_agmm.json`, and corresponding logs/provenance (successful candidate
runs use `_v2_provenance.json`). All pass existing numerical gates and exact
own-eager output/state replay. These are clean timing runs with tracking off.
The first candidate attempt failed only the host adapter's TTNN Shape slicing;
converting to tuple fixed the adapter and both reruns passed. This is an adapted
and successfully measured family, not rejection at a first API error.

The candidate module `tests/multichip_hidden_candidate.py` retains
`[1,4,T,640]` hidden ownership logically across attention, MoE and the next GR;
its public storage is a local reshape/permute into `[1,1,T,2560]`. Host pack/unpack
is outside timing, with no device boundary collective. QSA WO produces640
columns directly, MoE uses FP32 RS to640, and exact GR consumes the distributed
stream norm. Thus the measured cost includes the changed next consumer and
collective placement, not an artificial restore penalty. Active experts remain
gate-selected. The fused AG-MM candidate owns gathered6144-element BF16 buffers
for32 logical-count variants (12,582,912 bytes/rank); only the selected weight
representation is retained. Native RS uses framework outputs.

Reject these two measured whole-QSA hidden paths for the default: native RS
regresses B32 by1.898%, and fused AG-MM regresses B32 by1.768% while B1 is equal
within0.003%. The result does not reject unseen implementations or claim measured
PLE/GDN equivalence. Extension to their sensitive exact output paths remains a
handoff option, rather than a prerequisite for rejecting the already coherent
QSA alternatives. Existing stream ownership already has no inter-layer gather,
reshard or all-reduce, so rejecting hidden ownership introduces no boundary tax.

### Material movement, packing, precision and kernel experiments

* Async ordered BF16 AG retained the same residual and sum arithmetic but cost
  B1/B32=2.30024/58.7469 versus2.26639/57.8812 control. Preallocated-output async
  AG was retried and measured2.27989/58.0905. Global BFP8 transport measured
  2.33470/59.6262, whole-output B32 PCC0.997227; three per-row misses are diagnostic.
  These measured implementations are slower. Subgroup-only payload precision
  remains distinct from global conversion and is not silently rejected.
* Shared gate/up packing and HiFi4/HiFi2/LoFi were tried end-to-end: roughly0.8%
  B1 gain with0.2% B32 regression; separate LoFi gave no material gain. DRAM-sharded
  packing versus separate projections was subsequently measured in the role sweep.
* Injection10 performs ten1024-K FP32 partials before ordered sum/BF16 conversion.
  Initial L0 B1/B32 improves2.26639/57.8812 to2.17083/54.8231 (4.22%/5.28%).
  Other5/20/40-way splits were measured. This is now the default and the relevant
  comparison control for any residual family.
* Parallel valid-row ordered QSA indexer preserves product/partial order.
  Actual multichip binding reduces initial L3 prefill/B1/B32 from
  80.1750/4.75097/131.1169 to42.36697/3.09050/78.06350; combined injection10
  measured42.30587/2.99593/75.12868. The first nominal experiment missed the fused
  constructor and is explicitly excluded. `parallel_indexer_all_counts_v3`
  verifies Q1..32, six actual HF windows, exact BF16 and pre-cast FP32, cross-shape
  changed-input replay with allocation tracking. Component and full-mesh evidence
  are distinct; the final binding enables valid-row compute/copy-once.
* Pipelined PLE lookup batches independent64-byte NoC reads into scratch slots,
  then barriers/scatters. Batches16/32/64 with injection10 measure B1
  3.25050/3.24242/3.24216 and B32 89.03596/88.86462/88.75695. Default32 stays within
  the fixed16KiB CB (scratch starts12288); batch64 exactly fills its remaining4KiB.
  `pipelined_lookup_component.json` passes UINT32/BF16/FP32, all ranks,
  logical16×20,17×33,33×65,512×20, changed-input traces, exact values and zero tails
  with program-cache allocation tracking. Whole-default watcher remains separate.
* QSA precision artifacts `qsa_hifi2.json` and `qsa_lofi.json` pass existing whole
  gates, respectively prefill/B1/B32=42.28685/2.97583/74.23139 and
  42.26735/2.96394/74.00025. Their B32 PCC0.99981117/0.99922478 differs from control;
  final choice must preserve the accepted baseline requirement, not merely cross
  an absolute threshold. `qsa_bfp4.json` fails with B32 whole PCC0.98170942;
  its74.01091ms B32 is not an acceptable optimization. No new BFP4 default implied.

### DRAM and activation-sharding retries

`dram_seed_{gdn,qsa,wo,shared,gr}.json` first proved legal real-weight shapes.
`dram_geometry_v3_*` then sweeps8/16 storage shards,1/2/3 readers, compatible
K blocks1/2/5/10/20, and packed/separate projection families. These trace times
include input reshard, output conversion, and logical unpack/slices. Inputs are
seeded screening activations; model selection requires whole-decoder follow-up.
A prior split-QSA block40 case requested1,998,784 bytes against1,572,864 L1 bytes;
this hard physical failure triggered reduced-block retry, not family dismissal.

| Role | Native component | Best legal component | Storage cores/readers/K block | Precision, packing |
|---|---:|---:|---|---|
| GDN input projections | 0.070074 | 0.038815 | 8/2/2 | BF16 HiFi4, packed |
| QSA QKV | 0.069724 | 0.042912 | 8/3/2 | BFP8 HiFi4, packed |
| QSA WO | 0.052611 | 0.027122 | 8/3/2 | BFP8 HiFi4, separate |
| Shared gate/up | 0.075720 | 0.026586 | 8/3/5 | BFP8 HiFi4, separate |
| Injection | 0.096977 | 0.028420 | 8/3/5 | BF16 HiFi4, separate |

These GR numbers compare against the probe's ordinary native injection, not
injection10; integrated ablation must decide whether replacing that improved
control helps. `dram_geometry_selected_v1.json` records configurations.
`dram_weight_memory_plan.json` accounts per-device steady weight deltas: QSA/WO0,
GDN+1,146,880 bytes/layer, shared+522,240, injection+4,587,520 per projection
(two per decoder). Original and prepared tensors overlap during construction,
then originals are released; the ledger is arithmetic, not allocation high-water.

At this audit readback all `dram_integrated_v1_l{0,1,3}.json` reports pass, with
prefill/B1/B32 respectively21.67613/2.20889/56.72865,
26.24606/3.27384/91.22306, and41.69348/2.94332/72.97012. These are candidate results,
not final defaults. Against the improved QSA stream control, L3 gains1.89% B1 and
2.87% B32. L0/L1 selection needs role ablation against their improved controls;
aggregate component gains do not prove a whole-layer gain. Parent is conducting
that ablation; no default or context change is asserted here.

### Material work still required at this readback

1. Finish DRAM integrated ablations and choose the actual default. Update the
   context memory ledger if prepared layouts persist; preserve the advertised
   maximum unless hard physical evidence demonstrates the largest feasible value.
2. Finish the KV BFP8 branch decision. `bfp8_kv_component.json` proves selected-row
   unpacking, page permutations and changed IDs, but not cache-consuming whole
   decoder correctness or context. Any selected KV dtype/layout change needs
   updated `doc/context_contract.json`,32K–64K smoke and maximum-context evidence.
   Retaining the current KV policy avoids inventing an unmeasured capacity claim.
3. Produce final-default whole-output PCC, warm prefill and traced B1/diverseB32
   for all three kinds, plus MoE share/routing attribution. Real fixture rows are
   actual first-layer embeddings; later-layer inputs are explicitly shape-faithful
   stress, not claimed full-stack activations. Per-row PCC stays diagnostic.
4. Execute final-default allocation-tracked changed-input/state/cache replay,
   nonaligned tails, long context, risk-matched stress, runtime fallback audit,
   and separate watcher runs. Earlier component passes do not replace these gates.
5. Capture final-default per-kind profiler CSV/provenance, process advice-enabled
   tt-perf-report tables and account for actionable advice. Prepared commands are
   `plans/final_profiles.json` and `final_profile_reports.py`. Reconcile observed
   final op dtype/fidelity and movement with the selected configuration.
6. Record the stop-rule evidence and any remaining untried topology options in the
   work log: subgroup CCL dtype, further lower-movement PLE/GDN exact variants,
   packed-only active-expert geometry, and untried fusion suggested by final
   profiles. Do not describe these as measured rejections. Already tested coherent
   families need not be rediscovered merely because the default remains stream-owned.
7. Reconcile README/work_log/context contract with final numbers, obtain independent
   stage-review, fix/rereview findings, and locally commit stage-owned changes with
   logged SHAs. This audit is not stage-review and does not mark the stage complete.

The residual contract to preserve in downstream work remains one complete BF16
hyperconnection stream per rank, local `[1,1,T,2560]`, globally four streams.
Decoder return and the next decoder entry use this same layout. Internal exact
GR collectives operate only where their consumer requires them; no inter-layer
collective or restoration is part of the default contract.


## Subsequent selection evidence: real precision, CCL and KV

This appendix supersedes earlier “pending real-input trial” and “subgroup CCL
untried” labels for the exact cases below. Final combined selection and acceptance
remain pending. All referenced JSON has adjacent command/environment/provenance
and logs; this audit adds no individual-artifact hashing. Times are milliseconds,
on the same real-embedding fixture/1×4 mesh used by the improved controls. These
clean timing runs do not claim watcher, allocation-tracked or long-context gates.

### Whole-output gating correction and real projection fidelity

`real_dram_gdn-compact_LoFi_native.json` preserves its historical `passed:false`
and provenance exit1. That result comes solely from the inherited B32 per-row
PCC threshold: one row is0.994474590 versus the legacy0.995 requirement. Its
whole B32 PCC is0.999070942, short whole-output minimum is0.999327660 and state
parity passes. Under the user's whole-output0.995 contract this is **not a
numerical rejection**. The parent corrected batch acceptance to whole-output
PCC while retaining row diagnostics; historical JSON is not rewritten. LoFi
still offers only0.294% B1 improvement and0.006% B32 improvement over compact
HiFi4, so retaining HiFi4 can be a measured precision/performance decision.

The role-wise fidelity screens completed on actual decoder inputs:

| Artifact | Warm prefill33 | Traced B1 | Traced B32 | B32 whole PCC |
|---|---:|---:|---:|---:|
| `real_dram_gdn-compact_HiFi2_native` | 21.561970 | 2.167059 | 53.741016 | 0.999234557 |
| `real_dram_gdn-compact_LoFi_native` | 21.493917 | 2.163312 | 53.768890 | 0.999070942 |
| `real_dram_qsa_HiFi2_native` | 42.054405 | 2.991395 | 74.790337 | 0.999945164 |
| `real_dram_qsa_LoFi_native` | 42.065720 | 2.986344 | 74.682564 | 0.999806643 |
| `real_dram_shared_HiFi2_native` | 21.517874 | 2.129607 | 54.449799 | 0.999309778 |
| `real_dram_shared_LoFi_native` | 21.522894 | 2.132383 | 54.451962 | 0.999264002 |

These are DRAM-projection subgroup policies: compact GDN BF16 weights, QSA
BFP8 weights, and shared BFP8 weights with their recorded geometry; activation
remains BF16. Shared HiFi2/LoFi B32 times54.449799/54.451962 do not beat the
HiFi4 phase-specific packed-row candidate54.281044. QSA DRAM HiFi2/LoFi are
measured alternatives, not automatically cumulative wins with other roles.

Separate native attention fidelity/weight tests are:

| Artifact | Warm prefill33 | Traced B1 | Traced B32 | B32 whole PCC |
|---|---:|---:|---:|---:|
| `real_qsa_hifi2` | 42.003052 | 2.972940 | 74.188126 | 0.999928594 |
| `real_qsa_lofi` | 42.042500 | 2.968432 | 73.859505 | 0.999367952 |
| `real_qsa_bfp4` | 41.833923 | 2.957922 | 73.886467 | 0.982301474 |

HiFi2/LoFi whole-output checks pass. LoFi's minimum short PCC is0.997453392,
versus0.998797059 for HiFi2; timing gains must be assessed together with the
accepted quality baseline. **Real QSA BFP4 fails whole-output quality**:
B32 PCC0.982301474 and short minimum0.964503348. This is an actual whole-output
rejection, unlike the legacy GDN LoFi diagnostic-row failure. Do not select the
faster BFP4 row or conflate inherited expert precision with this attention trial.
Final combined candidate runs decide fidelity selection; this audit does not.

### Activation and collective payload dtype on real inputs

BFP8 activation was tested independently in the DRAM QKV, WO and shared groups:

| Artifact | Warm prefill33 | Traced B1 | Traced B32 | B32 whole PCC |
|---|---:|---:|---:|---:|
| `real_activation8_qsa_l3` | 42.027973 | 2.998147 | 74.855629 | 0.999894917 |
| `real_activation8_wo_l3` | 41.947800 | 2.984542 | 74.592117 | 0.999879062 |
| `real_activation8_shared_l0` | 21.512840 | 2.130665 | 54.459503 | 0.999324977 |

All pass whole-output gates. Against matched BF16-activation DRAM ablations,
QKV B1/B32 is0.259%/0.199% slower; WO B1 is0.112% faster but B32 is0.013%
slower; shared B1/B32 is0.060%/0.026% slower. Thus these concrete conversions
have no material performance reason to replace BF16 activations. This is
real-weight/input whole-decoder evidence, not synthetic precision vetoing.

Targeted BFP8 CCL payloads were also tried after the global conversion result:

| Artifact | Warm prefill33 | Traced B1 | Traced B32 | B32 whole PCC |
|---|---:|---:|---:|---:|
| `real_ccl8_moe` | 21.599504 | 2.173458 | 54.269424 | 0.999310076 |
| `real_ccl8_qsa` | 42.035389 | 3.000693 | 74.995354 | 0.999893963 |

`real_ccl8_moe` changes the MoE group on top of injection10 and compares against
`dram_ablation_control_l0`; B1/B32 improves only0.031%/0.023%.
`real_ccl8_qsa` changes QSA communication on top of parallel-indexer/injection10
and compares against `hidden_stream_control`; B1 is0.024% slower while B32
improves0.169%. Both pass whole-output gates, but neither supplies a material
performance gain. These tested subgroup policies now have disposition evidence;
further untested subgroup combinations remain distinct optional opportunities.

### Reduced KV dtype integrated trial

| Artifact | Warm prefill33 | Traced B1 | Traced B32 | B32 whole PCC |
|---|---:|---:|---:|---:|
| `real_bfp8_kv` | 42.817631 | 3.362617 | 75.486511 | 0.999900401 |

The real QSA decoder uses selected-row BFP8 cache unpacking, not conversion of
the entire cache to BF16 on each token. It passes the short whole-output checks
(minimum0.998448372), B32 whole PCC0.999900401 and exact own-eager replay.
Against `hidden_stream_control`, prefill/B1/B32 regresses1.967%/12.088%/0.485%.
Reject this measured short-workload implementation for the optimized default
on performance. This is no claim about long-context bandwidth break-even.
BF16 KV remains the proposed default; the trial does not justify changing the
advertised context contract or claiming BFP8 maximum-context acceptance.
If selected later, the changed cache format still requires maximum-context
validation and updated memory/context records. For retained BF16, perform the
required final32K–64K nonaligned smoke.

### Best-control fused family rerun and full-shard GR block

`fused_family_bestcontrol_HiFi4.json` and `_LoFi.json` now use injection10 in
the stream control. All24 rows per file pass Q1,2,31,32 and compare capture replay
bit-exactly against that candidate's own eager residual/mixed/injection outputs.
Trace input remains unchanged and at most one trace is live; these are component
screens, not changed-input decoder acceptance. Eager inputs still include two
feature variants. The corrected best-control traced measurements are:

| Fidelity / Q | Ordered BF16 | Native RS | Fused MM-RS | AG-MM | Fused AG-MM |
|---|---:|---:|---:|---:|---:|
| HiFi4 / 1 | 0.666309 | 0.583294 | 0.601044 | 0.594661 | 0.584743 |
| HiFi4 / 2 | 1.082152 | 0.996813 | 1.014302 | 1.008836 | 0.997539 |
| HiFi4 / 31 | 3.317994 | 3.194437 | 3.194542 | 3.192282 | 3.191326 |
| HiFi4 / 32 | 3.432744 | 3.286002 | 3.301622 | 3.297610 | 3.282947 |
| LoFi / 1 | 0.654862 | 0.571707 | 0.589732 | 0.594456 | 0.583510 |
| LoFi / 2 | 1.070099 | 0.988187 | 1.001562 | 1.008246 | 0.996824 |
| LoFi / 31 | 3.305008 | 3.167365 | 3.186180 | 3.191461 | 3.176828 |
| LoFi / 32 | 3.423702 | 3.272907 | 3.291243 | 3.298211 | 3.283248 |

The component still favors lower-movement families; this does not override
whole-QSA B32 regressions recorded above. Both experiments are useful: the
component isolates the producer/next-GR gain, while the whole path includes all
layout, normalization, attention and MoE consequences with no immediate boundary
restoration. Retaining stream ownership is therefore based on measured complete
work, not on the misleading eager ranking or obsolete component control.

`dram_gr_largeblock40.json` retries a full-shard K block for the injection role,
where the earlier QSA block40 L1 overflow is not a general blocker. All six
8/16-core ×1/2/3-reader candidates pass component whole PCC1.0. Times for8 cores
are0.037111/0.032768/0.030434;16 cores0.040653/0.036169/0.031139, versus native
0.096796. Best block40 is still slower than the earlier legal block5 result
0.028420. These different-run component measurements support retaining block5
for integrated evaluation; they do not substitute for comparison with the
already improved injection10 whole-decoder control.

### Cumulative advice disposition at this point

| Applicable opportunity | Evidence now available | Current disposition |
|---|---|---|
| DRAM matmul geometry, reader count, larger K block | Five roles,8/16 storage shards,1/2/3 readers; reduced-block retry plus full-shard GR40 | Measured; retain winning legal geometry for combined evaluation |
| Packed versus separate projections | Native/shared and DRAM QKV/GDN/shared; compact logical GDN output and phase-specific shared rows | Measured/adapted; final combination pending |
| Precision crossed with geometry | Ten B1/B32 component sweeps plus real GDN/QSA/shared fidelity, activation and native attention trials | Measured; do not treat diagnostic-row legacy failure as whole-output rejection |
| Material async/persistent/CCL payload changes | Global and subgroup real-input dtype; exact-order async/persistent buffers | Measured; subgroup gains below0.2%, global/async slower |
| Fused MM-CCL and residual consumer placement | Best-control Q1,2,31,32 components plus complete hidden QSA RS/AG-MM | Component wins do not survive complete B32 path; keep stream contract provisionally |
| Reduced KV-cache dtype | Exact selected-row component and cache-consuming real decoder | Short-workload performance rejection; no context change selected |
| Final op dtype/fidelity and advice confirmation | Final profiler plans prepared | Pending actual selected-default profiles/tables |
| Cumulative integrated correctness/performance | Last combined selection jobs | Pending; no final optimized number asserted |

Previously listed optional hidden PLE/GDN and packed-only expert ideas remain
untried; subgroup CCL dtype and whole-decoder BFP8 KV are no longer wholly
untried. The owner must use the actual optimize Stop Rule and log remaining
handoff items if it fires. Final-default PCC, nonaligned/tail and context checks,
changed-input tracked replay, fallback audit, watcher, independent stage-review
and stage-owned local commit evidence remain mandatory and are not closed by
these clean candidate timings.

## Integrated selected policy and cumulative disposition

Selection is now integrated in `tt/multichip_decoder.py:MESH_DRAM_POLICIES`;
final-default measurements and acceptance gates are still pending. This section
supersedes earlier selection-pending labels, without promoting historical
candidate-wrapper results into final-default performance.

| Family / role | Selected default or measured disposition | Evidence |
|---|---|---|
| GR injection | DRAM BF16 HiFi4,8 storage cores,3 readers,block5; supersedes injection10 | `dram_ablation_gr_l{0,3}`, combined final-candidate rows; full-shard block40 screen slower than block5 |
| GDN Z/A/B | Compact packed BF16 HiFi4,8 cores,2 readers,block2 | Initial padded integration regressed; compact-output retry and real fidelity trials followed |
| Shared gate/up | Packed BFP8 HiFi4,8 cores,3 readers,block5 for both phases | Separate/native/phase-specific packing measured; `dram_packed_shared_l0_v2` and combined rows |
| QSA QKV | Compact packed BFP8 HiFi4,8 cores,3 readers,block2 | Packed/split searches and `dram_compact_qsa_l3` |
| QSA WO | BFP8 HiFi4,8 cores,3 readers,block2 | Row projection ablation; selected DRAM output role |
| Remaining QSA attention | HiFi2, including SDPA; independent QKV/WO remain HiFi4 | `dram_final_candidate_l3` → `dram_final_candidate_hifi2_l3` |
| Sensitive exact indexer | Retain parallel ordered valid-row implementation | Exact BF16/precastFP32 all-count and whole-mesh evidence |
| PLE lookup | Retain pipelined NoC read batch32 |16/32/64 whole decoder plus all-rank/dtype/tail tracked component |
| Residual / CCL placement | Retain full stream/rank; no boundary gather/reshard/reduction | Hidden QSA nativeRS/fusedAGMM pass quality but regress whole B32; coherent next GR/MoE included |
| Activation / CCL payload | Retain BF16 | Real activation8 no material gain; subgroup CCL8 below0.2%; global/async/persistent paths slower |
| KV format | Retain BF16 page32 | Integrated real BFP8 KV passes short quality but B1 regresses12.09%; no new capacity claim |
| Other fidelity/weight reductions | Do not select new BFP4 or LoFi groups | Real QSA BFP4 whole failure; other precision trials do not beat selected complete policy materially; diagnostic-row GDN LoFi legacy failure is not a whole-output rejection |

The successful combined selected-policy candidates are:

| Artifact | Prefill33 ms | B1 ms | B32 ms | B32 whole PCC |
|---|---:|---:|---:|---:|
| `dram_final_candidate_l0.json` |21.480122|2.082665|52.601185|0.999270976|
| `dram_final_candidate_l1.json` |26.086248|3.143290|87.032532|0.999606848|
| `dram_final_candidate_hifi2_l3.json` |41.627953|2.858654|71.619102|0.999913871|

These are evidence for selection; final numbers must come from the actual
integrated default. `candidate_summary.csv` indexes all existing mesh timing
artifacts, preserving reported status separately from whole-output0.995 PCC
checks, diagnostic row minima, matching fixture and provenance/disposition.
Neither a CSV PCC boolean nor a historical clean `passed` substitutes for the
complete unchanged quality/state/context gates.

The owner records Stop Rule2: last retained QSA compact-output refinement
74.706688→74.268517ms B32 (0.5865%), followed by remaining-attention HiFi2
72.503128→71.619102ms (1.2193%), both below3%. The integrated roles compose
already measured choices. `stop_rule_handoff.md` enumerates untried extensions
and separates them from mandatory closure. No final stage acceptance is claimed.
