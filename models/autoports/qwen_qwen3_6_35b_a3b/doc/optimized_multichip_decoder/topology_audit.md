# Operation-topology audit before stage-5 optimization

CPU-only inspection, 2026-11-13. Source baseline: `5ebe699604445e7b30ea0813e15aa913efbd3188`.
Target is the existing 1x4 Blackhole TP4/EP4 decoder. No device was opened and
no implementation changed for this audit. Unless stated otherwise, artifact
paths below are relative to `../multichip_decoder/`; source paths are relative
to this autoport. Inherited measurements establish the starting point, not
stage-5 improvements or a rejection of unmeasured combinations.

## Starting measurements and bottlenecks

The `passed` fields and mesh timing fields in `final_l{0,3}_b{1,32}.json`
confirm these clean, serialized S32 results. Prefill is whole batch; decode
is one blocking warmed trace replay of the complete batch.

| Kind | B | Prefill ms | Traced decode ms |
|---|---:|---:|---:|
| linear | 1 | 63.692517 | 1.015573 |
| linear | 32 | 1340.355774 | 21.726680 |
| full | 1 | 2.030614 | 0.393623 |
| full | 32 | 44.025494 | 6.453329 |

`perf_b32/perf_summary.json`, `profile_findings.md` and raw
`perf_b32/tracy/layer_{0,3}/ops.csv` show the workload to prioritize:

| B32 work, maximum-work rank | Linear | Full |
|---|---:|---:|
| Sum of device kernels per decode replay | 19.730907 ms | 5.802994 ms |
| CCL | 0.610642 ms | 0.338154 ms |
| Movement/conversion | 4.481995 ms | 1.925423 ms |
| Source-boundary MoE | 0.921866 ms (4.67%) | 0.838284 ms (14.45%) |
| Final row split | 0.200190 ms | 0.200174 ms |

These are kernel-work sums, not elapsed-time bounds or latency shares. The
profile prefill is only one representative request, unlike clean whole-batch
timing. Profiler-drain gaps invalidate using report `Total [%]` as a clean
latency denominator. All four ranks have actual work; never sum their durations
to claim mesh latency.

## Operation topology and actions to take

Source anchors: `tt/multichip_decoder.py::{linear,_normalized_input,_reduce,
_full_attention,_linear_attention,_moe,_grouped_moe,decode_batch}`.

| Boundary / present sequence | Material repeated work or movement | Candidate and constraints | Audit action / evidence status |
|---|---|---|---|
| B32 attention: 32 repetitions of norm → attention → local output → direct AR → residual | Full: 32 H2048 norms cost 787.689 us; 32 QKVG 512.673 us; 32 O projections 240.259 us; 32 AR 291.591 us. Linear AR 436.522 us. | Batch independent norms/projections and output reductions while retaining per-request state/cache. Physical singleton tensors already occupy 32 rows, so batching can amortize both launches and padded communication. Preserve partial-RoPE and heterogeneous page/position ownership. | Highest-priority coherent B32 family. Historical B1 CCL winners do not reject this. |
| Linear same-input packed projections: `attention_packed` K/V/a/b and `output_only_packed` Q/Z | 64 ordered input helpers cost 3633.828 us at B32; 32 ordered output helpers 1712.669 us. | Pack the two input groups into one ordered scan or batch rows while preserving ordered accumulation and separate state semantics. Ordinary GEMM changes arithmetic. | Material packing opportunity, not covered by full QKVG packing. Any candidate needs real-weight whole-output and recurrent-state checks, then trace. |
| Full attention packed Q/K/V/gate → slices → head creation → Q/K norm → partial RoPE → cache/SDPA → gate → concatenate → O | Packed QKVG already avoids four separate input reads; Q/K norm changes HEIGHT↔WIDTH shards; dense projection helper returns DRAM after L1 output. | Preserve L1 shard compatibility through producer and consumer; compare packed against tuned split projections under each legal precision policy. Batched full attention needs native batch-aware head/cache layout or separate cache updates. | Existing local DRAM geometry screen applies to singleton projections; packed/split local TP4 comparison remains open. |
| Row output → direct AR for singleton; row output → RS → AG for multirow | B1 AR uses L1 WIDTH shards. B32 attention repeats singleton AR; grouped MoE uses one M32 RS+AG. | Batched direct AR on a single M32 carrier; hidden-sharded RS with next norm consuming the shard; fused MM-RS; AG-MM after distributed norm. | Existing helpers provide coherent controls; qualify any changed batch topology together with activation/CCL dtype and next consumer. |
| Post-attention norm → router and shared packed GU/scalar → active experts → routed + shared local sum → one reduction | Router and shared consume the same normalized input but need different precision. Grouped dispatch requires DRAM; singleton sparse GU selected DRAM. | Reuse one input conversion. Keep router FP32; compare shared packed/split independently. Fusing shared-down RS alone adds another collective unless routed contribution is reorganized. | Do not pack router with BFP8 shared weights without preserving FP32 router arithmetic. Shared B32 ordinary matmuls still use K-block2/subblock1×1, unqualified by B1 DRAM geometry. |
| Grouped EP dispatch → fused SwiGLU → combine → weighted reduction | Selected capacity C2240 at B32; combine still materializes BF16 `[1,1,32,8,2048]` (1 MiB/chip) and zeroes remote slots. | Retain per-token gate-selected execution; examine grouped compute geometry/combined weighted reduction, not union-masked sparse B32. | Local-only dispatch already measured; further combined-output movement remains. Kernel work is only 4.67%/14.45% of linear/full B32, so prioritize attention first. |
| Batched MoE residual → 32 singleton outputs → next layer input list | One Slice plus 31 Untilize/Slice/Tilize triplets: 94 ops, about 200 us. | Carry batched residual to next layer; gather/split only at public test boundary, or implement a direct tiled row extraction. | Meaningful lower-movement family. A faster output contract must be consumed by the next layer and validated mixed-kind; moving split into the next layer is no win. |

## Coherent collective families

Let `M=32*ceil(logical_rows/32)` and `P=M*2048*element_bytes` for a full
partial on each rank. Nominal bandwidth-optimal four-rank traffic is `.75P`
for RS or AG and `1.5P` for AR, excluding forwarding, headers and alignment.
For BF16 M32 these are 96 KiB and 192 KiB per rank. FP32 doubles them.
Thirty-two singleton attention calls therefore have nominal BF16 AR traffic
6 MiB/rank (12 MiB for FP32 linear partials); a compatible M32 batched
reduction could reduce it to 192 KiB (384 KiB FP32). These are estimates,
not fabric counters or measured speedups.

| Family | Residual, next consumer, buffers | Evidence inherited and limitation |
|---|---|---|
| Replicated, local MM + direct AR | Replicated H2048; residual add and local norm. Shared persistent BF16/FP32 scratch and semaphore, 1.5 MiB/chip total. | Full B1 `collective_l3_direct_ar.json`: 0.405460 ms; linear FP32 `collective_l0_direct_fp32.json`: 1.029183 ms. Applies to singleton carrier only in current `_reduce`. |
| Replicated, MM + RS + AG | Replicated H2048; local norm; `.75P + .75P`; generic ping-pong semaphores. | Full B1 `collective_l3_replicated.json`: 0.465064 ms. Current grouped-MoE and multirow control. Direct B32 AR remains unmeasured here. |
| Hidden-sharded, MM + RS | H512 across ranks; add local residual; ordered norm gathers before arithmetic, or distributed norm gathers stats then normalized input as needed. No inter-layer boundary gather. | Linear B1 `collective_l0_sharded.json`: 1.057356 ms. Sharded residual is functional; no B32 ranking is established. |
| Hidden-sharded, fused O MM-RS | Reduced H512 feeds residual add and next norm directly; no restore solely for the interface. | `collective_l3_fused_ordered.json`: 0.470544 ms, passing B1. Its norm is ordered; it does not qualify fused RS + distributed norm + fused AG-MM together. |
| Hidden-sharded, distributed norm + AG-MM | Local normalized H512 feeds packed QKVG; persistent gathered buffer, dedicated semaphores. `.75P` AG plus small norm stats. | Adapted Linear-supporting minimal API passes: `collective_l3_ag_minimal_{32,64}.json`: 0.516678/0.514373 ms. Both record `attention_collective=separate`, so combined fused-input/fused-output family remains open. |
| Hidden-sharded shared AG-MM + reused gathered input | Packed GU consumes fused gather; returned H2048 reused by router/active experts. Down+routed share final reduction. | Source-compatible proposal in `collective_candidate_audit.md`, no measured result. Fused SwiGLU needs GU256 tile-pair packing and separate scalar; current packed N288 is not a family rejection. |

The listed collective JSONs use real checkpoint weights but seeded random
activations (`activation_file=null`), unlike natural B32 acceptance. They all
record `passed=true`. They are historical screening results, not final default
stage-5 numbers. A loss of isolated AG-MM or MM-RS does not reject their
compatible combination or a changed B32 residual contract.

## Packing, precision and geometry coverage

`geometry/search.json` covers full B1 dense R1/R2/R3, K-block4/8/16,
input8/16 and output2/4 controls at BFP8/HiFi2. R1 was slower
(0.437512 versus 0.406818 ms); R2 nearly tied (0.407104 ms). Global controls
are not an exhaustive per-role sweep. `sparse_geometry/search.json` covers
12 B1 indexed layouts; selected DRAM-input GU block64 is 0.392621 ms versus
block16 0.405796 ms. L1-input and smaller-core/subblock alternatives were
actually measured slower. Neither screen qualifies B32 shared/grouped kernels.

Current runtime policy: BF16 activations; BFP8/HiFi2 full/shared projections;
BFP8/LoFi experts; FP32 router weights/logits; sensitive linear ordered
BF16/FP32 helpers; BF16 KV. Grouped native kernel disables FP32 destination
accumulation. Before selecting changed precision, confirm actual profile
attributes rather than inherited policy dictionaries with single-chip geometry.
Local full-attention LoFi, activation/CCL reductions, reduced KV and BFP4
candidate evidence are not established by the above local screens. BFP4 is
experimental only in this stage. KV changes require updating the context
contract and maximum-context validation, otherwise 32K–64K smoke suffices.

## Persistent-buffer and stack contract

`CollectiveWorkspace` owns direct-AR scratch/semaphore, CCL semaphores and
`ep_outputs[capacity]`; prepare every needed bucket before capture. At B32,
one BF16 EP output is `2240*2048*2 = 8.75 MiB/chip`. Dispatch payload and
metadata, combine output and ordinary `_gather`/RS outputs are not explicitly
preallocated by this model; trace allocations alone do not prove safe reusable
buffers across independently live traces. Measure persistent alternatives
where APIs allow them and retain exact alias/lifetime invariants.

The inherited inter-layer contract is replicated `[1,S,2048]` per device,
BF16 residual, identical on four ranks; intra-device layout may be L1 sharded
for singleton direct AR or DRAM for multirow. Layer output already satisfies
the next layer's input, without inter-layer collectives. Batched public input
and output are tuples of singleton tensors, incurring the row split above.
One serial CQ0 workspace is shared across layers; concurrent forwards using
that workspace are unsupported. Preserve this contract or explicitly replace
it with a tested mixed-layer compatible batched/sharded contract.

## Suggested experiment order and acceptance

1. Batch input norm and full-attention projections; combine singleton output
   partials before one batched reduction. Compare direct AR, RS+AG and sharded
   residual through the next consumer, under the same BF16/BFP8 policy.
2. Pack/batch linear ordered projections without changing accumulation order;
   its input/output helper group is about 5.35 ms of B32 device work.
3. Remove output row conversion via stack-compatible batched residual or a
   direct tiled extraction; retain heterogeneous positions/states/page tables.
4. Qualify coherent fused AG-MM + MM-RS and persistent-buffer families; compare
   phase-specific shards and local DRAM reader/block geometry for newly batched
   projections. Revisit shared B32 geometry only after top attention costs.
5. Cross selected geometries with applicable LoFi/HiFi2 and precision choices;
   preserve FP32 linear/router controls unless real-weight evidence supports
   changing them. Measure packed/split local projections and reduced KV as
   required, or list exact untried items when the optimize stop rule fires.

New paths need unchanged whole-output PCC for both kinds, distinct B32 routing,
changing-input trace replay with allocation tracking, nonaligned and chunk-tail
inputs, mixed-kind stack/state/cache ownership, context smoke and separate
watcher evidence. Rerun clean timing and current-source profiler tables on the
final default. This audit does not claim those stage-5 gates have run.
