# Qwen3.6 TP4/EP4 plan — chosen before implementation

Baseline: OptimizedDecoder at 2901d8a5 (implementation 17810c63). Target: four
Blackhole p300c devices 0–3, 1x4 FABRIC_1D, Linear CCL, two links candidate.
The bounded mesh smoke passed; all devices were unowned. No reset used.

## Initial tensor contract (before measurement)

H=2048, 40 layers (verify config arithmetic in capacity manifest), E=256,
k=8, I=512, shared I=512. TP4 leaves I/4=128=4 tiles, below the eight-tile
heuristic. B32 touches u=1-(1-8/256)^32=0.638 of all experts on average.
Therefore EP4 grouped dispatch is selected for batched decode and prefill;
indexed TP4 is a separate batch 1 candidate, never union-masked batch 32.

| Tensor | Global logical shape | Per chip | Mapping / padding |
|---|---|---|---|
| residual | [1,S,2048] | [1,S,512] | hidden-sharded; replicated control also measured |
| full Q/gate | [2048,4096] each | [2048,1024] each | four contiguous query heads |
| full K,V | [2048,512] each | [2048,256] each | rank r owns global KV head floor(r/2); duplicated on pairs |
| packed QKVG | [2048,9216] | [2048,2560] | local Q1024,K256,V256,gate1024 packing |
| full output | [4096,2048] | [1024,2048] | row TP, reduce-scatter to residual |
| KV K and V | [pages,2,32,256] | [pages,1,32,256] | BF16 TILE; same permuted page table on every rank |
| linear Q,K | [2048,2048] each | [2048,512] each | four key heads |
| linear V,Z | [2048,4096] each | [2048,1024] each | eight value heads; Q/K repeat twice locally |
| linear a,b | [2048,32] each | [2048,32] each | eight logical channels padded individually to 32 |
| linear state projection | [2048,6208] | [2048,1600] | K512,V1024,a32,b32; ordered K scan retained |
| linear output projection input | [2048,6144] | [2048,1536] | Q512,Z1024 |
| convolution | [1,3,8192] | [1,3,2048] | local Q512,K512,V1024, BF16 row-major |
| recurrent state | [1,32,128,128] | [1,8,128,128] | FP32; independent heads |
| linear output weight | [4096,2048] | [1024,2048] | row TP; preserve ordered accumulation locally |
| router | [2048,256] | same | replicated FP32 logits, global top-8 |
| indexed expert GU | [1,256,2048,1024] | [1,256,2048,256] | local gate128/up128, BFP8 LoFi |
| indexed expert down | [1,256,512,2048] | [1,256,128,2048] | indexed top-8 only |
| grouped experts | 256 full experts | 64 full experts | rank r owns [64r,64r+64); BFP8 LoFi |
| shared GU/scalar | [2048,1056] | [2048,288] | gate128/up128/scalar32; scalar replicated, applied once to partial |
| shared down | [512,2048] | [128,2048] | row TP; combine with expert partial before one collective |
| norms | [2048] | [512] | distributed statistics for sharded residual; local norm for replicated |

Initial per-device matmuls use explicit geometry derived from local K/N, 32-element tiles,
BF16 activations, BFP8 weights, FP32 accumulators. Indexed GU uses 8 output
cores (one 32-wide tile/core); down 64 cores, K block 4. Full/shared projections
start from baseline DRAM-sharded configuration rebuilt for local shape; legal
reader/K/core candidates will be measured. No source-runtime weight conversion.

## Collective candidates

Each row-parallel site has local [1,1,M,2048] partials. M is physically rounded
to 32; payload P=32*2048*2=131072 bytes at B1/B32. BF16 CCL starts unchanged.

| Family | Before/after residual | Next consumer | Approx rank traffic | Persistent resources / decision |
|---|---|---|---|---|
| local MM + RS + AG | replicated -> replicated | local RMSNorm | 1.5P | generic ping-pong semaphores; control |
| local MM + RS | sharded -> sharded | distributed norm, then AG normalized input | .75P output plus .75P next input and small stats | retained candidate; no immediate reconstruction |
| AG+MM fused | sharded -> column-sharded projection | local attention/shared GU | .75P gathered input | shape-faithful probe with next consumer required |
| MM+RS fused | local intermediate -> sharded residual | residual add/distributed norm | .75P | probe supported output projection packing |
| direct AR | replicated -> replicated | local RMSNorm | topology dependent ~1.5P | persistent all-reduce buffers candidate |

Reusing TTTv2 attention directly mismatches partial 64-dimensional RoPE,
Q/gate interleaving, and linear gated-delta recurrence. Preserve its local-head,
row-TP, distributed norm and reduce-scatter contracts in model-local code.
GPT-OSS CCLManager supplies semaphore ownership, extended to all 11x10 worker
cores, as in the existing Kolibri port. Kolibri grouped expert dispatch is the
shape-adapted reference; its sigmoid routing is replaced with Qwen top-8 softmax.
No canonical Qwen36 Blackhole demo exists in this checkout.

## Capacity and context

Keep advertised 262144 context. Full KV per rank per full layer is
262144*1*256*2(K,V)*2 bytes=268435456 bytes, half the single-chip layout, with
necessary duplicate KV ownership across query-head pairs. Linear recurrent
payload falls from 2097152 to 524288 bytes per request/layer; convolution from
49152 to 12288. Full-model capacity arithmetic must include all layers, both
EP and indexed-TP weight representations, shared/norm/embedding/head estimates,
trace and activation reserve. These are planning estimates until manifest and
device evidence exist; no capability reduction is authorized by this plan.

The initial smoke plan used logical lengths 32767/32768. Acceptance now uses
32769 tokens at capacity 65536 for both kinds, including a nonaligned tail;
the changed local KV layout also requires a maximum-context case. Keep all
nonaligned public lengths; internal attention chunks of 64/256 remain independent
of logical length. Grouped
MoE will be invoked across the larger prepared activation chunk where possible;
2048 is the intended routed-prefill chunk, without padding 1024 to 2048.

## Rejected by arithmetic, not by measurement

Expert replication on every device multiplies full-model weight memory by 4;
union-masked B32 computes ~163 experts for every row versus 8 actual selections.
Plain TP4 grouped experts has only 4 intermediate tiles per rank. EP/hybrid is
preferred; any departure requires measurement. TP2 attention plus replication
wastes two chips' query/state work; TP4 duplicated KV halves total KV savings
relative to an imaginary fractional head but uses all chips' query compute.
Final topology and shape choices require PCC, trace, perf and review evidence.

## Measured selection update and final tensor contract

The initial hidden-sharded design passed correctness. Whole-layer measurements
then selected a replicated residual `[1,S,2048]` on all ranks, with local RMSNorm
and direct L1-width-sharded all-reduce for singleton decode. Prefill and grouped
MoE rows use RS+AG because the measured direct kernel candidate is a singleton
carrier. Layer outputs already satisfy the next layer's input contract.

The final tensor table differs from the initial plan in three rows: residual
activations are replicated `[1,S,2048]`, norm weights are replicated `[2048]`,
and full/linear/shared row-parallel outputs reduce into the replicated residual.
All local attention, expert, state and KV shapes in the initial table remain
unchanged. Prefill concatenations use a fan-in-64 tree in DRAM; batched decode
uses fan-in 16 in L1. This bounds native runtime arguments and CB use while
preserving arbitrary valid logical lengths and row order.

Screening on real checkpoint weights, S32/B1, tracked warmed decode:

| Layer / path | ms | Evidence |
|---|---:|---|
| full, DRAM projections, sharded ordered norm | 0.48151 | projection_l3_dram_ordered_v2.json |
| full, fused output MM-RS, sharded | 0.47054 | collective_l3_fused_ordered.json |
| full, replicated RS+AG | 0.46506 | collective_l3_replicated.json |
| full, replicated direct AR | 0.40546 | collective_l3_direct_ar.json |
| full, adapted Linear minimal AG-MM, 64-row carrier | 0.51437 | collective_l3_ag_minimal_64.json |
| full, adapted Linear minimal AG-MM, 32-row carrier | 0.51668 | collective_l3_ag_minimal_32.json |
| linear, sharded ordered norm | 1.05736 | collective_l0_sharded.json |
| linear, replicated direct BF16 MoE AR | 1.04074 | collective_l0_direct_ar.json |
| linear, replicated direct FP32 attention/BF16 MoE AR | 1.02918 | collective_l0_direct_fp32.json |

All listed candidates passed whole-output PCC and changed-input trace checks.
These are screening numbers, not final clean latency claims. The initial legacy
AG-MM topology failure was adapted to the separate Linear-supporting strided
minimal API; both carrier sizes passed but were slower. Fused O MM-RS directly
fed residual add and the following norm, so its comparison does not charge an
immediate gather merely to recreate a replicated boundary.

The selected direct-AR scratch uses 32 cores in an 8x4 grid, input/output shards
32x64 and scratch shards 32x256. BF16 and FP32 scratch total 1.5 MiB/chip and
are shared by all serial layers through CollectiveWorkspace, avoiding an L1
allocation per layer. CCL semaphores and EP output buffers share that lifetime.
DRAM-sharded projection weights/configs are rebuilt from per-chip K/N using the
optimized baseline builder; no runtime weight conversion occurs.

Resident constructor census from the mixed-stack run measured 465569792 bytes
for a linear layer and 450185216 bytes for a full layer per device. Extrapolated
40-layer resident weights fit inside the conservative weight reserve. The plan
now reserves 6 GiB/chip for the replicated maximum-context activation working set
and traces; available DRAM from the allocator is 34111622144 bytes/chip, leaving
4135717888 bytes after the conservative full-model plan. This is capacity
arithmetic, not a claim that the full model was built or loaded in this stage.


## Selected expert execution update

The initial dispatch control packed selections for all 256 expert IDs on each
rank, executed only its 64 resident full experts, and masked remote experts in
the reduction. The selected default is now `local_expert_dispatch=True`:
rank r dispatches only IDs `[64r,64r+64)`. The global top-8 gate and global
256-ID namespace remain unchanged; this is local payload compaction, not
cross-rank token dispatch or dense all-expert execution. The native dispatch
uses `dispatch_group_size=1`, `cluster_axis=0` on the 1x4 mesh. Grouped EP
remains the prefill and B>1 path; B1 retains indexed TP4 top-8 execution.

Each rank's INT32 row-major dispatch/reduction table is `[1,256]`, with zero
for its 64 IDs and -1 elsewhere. A static FP32 TILE mask `[1,1,1,256]`
zeros nonlocal counts before the aligned prefix sum. Counts and offsets become
UINT32 row-major `[1,256]`; selected IDs remain UINT16 `[1,T,8]`. The native
reader skips -1 entries before offset access or payload writes. Nonlocal counts
are zero, so grouped compute and combine do not consume their unwritten regions.
`experts_per_chip=256` and `num_routed_experts=256` preserve this global index
space; each rank still has exactly 64 full gate/up/down weights.

For internal grouped token count T, let B=max(32,next_power_of_two(T)). The
per-rank capacity is `C=round_up(8*B + 31*64,32)`, replacing the control's
`round_up(8*B + 31*256,32)`. This bounds worst-case top-8 assignments plus
per-expert tile padding even when every selected expert is local. It does not
assume an average two assignments per token per rank. Nonaligned logical rows
are padded only within MoE and sliced back after computation.

| Token bucket B | Selected C | Control C | One selected BF16 payload MiB | One control BF16 payload MiB |
|---:|---:|---:|---:|---:|
| 32 | 2240 | 8192 | 8.75 | 32.00 |
| 64 | 2496 | 8448 | 9.75 | 33.00 |
| 128 | 3008 | 8960 | 11.75 | 35.00 |
| 256 | 4032 | 9984 | 15.75 | 39.00 |
| 512 | 6080 | 12032 | 23.75 | 47.00 |
| 1024 | 10176 | 16128 | 39.75 | 63.00 |
| 2048 | 18368 | 24320 | 71.75 | 95.00 |

Dispatch produces BF16 row-major `[1,1,C,2048]` plus INT32 row-major metadata
`[1,1,C,3]`. Grouped expert output is a prepared, shared BF16 TILE DRAM buffer
`[C,2048]`, one per capacity bucket. Payload columns in the table describe one
buffer, not total peak memory. Combine still returns BF16 row-major
`[1,1,T,8,2048]`; it explicitly sets `init_zeros=True` so absent remote slots
are finite zeros before the weighted masked reduction. This reduces populated
expert regions and dispatch workspace; it does not eliminate the final eight
slots or their initialization. Weight and KV capacities are unchanged, and the
existing conservative activation reserve is retained.

Real-activation S32/B32 comparisons (whole-batch prefill and one whole-batch
decode step, milliseconds) are recorded in `local_dispatch_l{0,3}_{false,true}.json`:

| Layer kind | Prefill control -> selected | Prefill change | Decode control -> selected | Decode change |
|---|---:|---:|---:|---:|
| Linear layer 0 | 1321.33548 -> 1331.78233 | +0.79% | 22.20906 -> 21.72988 | -2.16% |
| Full layer 3 | 46.46513 -> 43.01756 | -7.42% | 6.84678 -> 6.44336 | -5.89% |

The linear prefill screen regressed slightly; local compaction is not a
universal prefill speedup. Both pairs have identical whole-output prefill/minimum
eight-step decode PCC: linear 0.999966521/0.999902546, full
0.999969649/0.999854728. Determinism, changed-input eager/trace equivalence and
live-trace prefill replay pass. These are serial mesh-only screening pairs;
saved single-chip baselines provide correctness controls, not paired speedups.

For indexed B1, the subsequent local geometry screen selects DRAM input and
GU K-block64 on eight cores, output/subblock one tile per core. GU takes
BF16 `[1,1,1,2048]` and local BFP8 `[1,256,2048,256]`, producing BF16 L1
`[1,8,1,256]`. Its 128-wide gate/up halves feed down K128/N2048 on 64 cores,
K-block4, subblock1x1, with BF16 L1 input/output. The default geometry is
`{"gu":{"block_w":64}}`; explicit `{}` retains the block16 control.
The selected whole-layer full-attention B1 screen is 0.39262 ms versus
0.40580 ms for block16 (3.25% less latency). The twelve measured sparse
layouts, including rejected L1-input and reduced-core variants, and seven
dense controls are retained in [the local geometry audit](local_geometry_audit.md).
This is the best measured sparse candidate in that screen, not an exhaustive
optimality result or a B32 sparse-speedup claim.

Source contract: `tt/multichip_decoder.py`, `dispatch_capacity`,
`from_state_dict`, `_moe`, `_grouped_moe`, and `prepare_moe`; native validation
and -1 skip in
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/dispatch_device_operation.cpp`
and `dispatch/device/kernels/dataflow/reader_worker_dispatch.cpp` (under the
same deepseek-prefill prefix). Combine's dynamic count/offset readers are
`combine/device/kernels/dataflow/reader_untilize.cpp` and `writer_untilize.cpp`.

## Historical acceptance checkpoint (before these selection updates)

The real-activation batch-32 runs pass for linear layer 0 and full layer 3:
`natural_l0_b32.json` and `natural_l3_b32.json`. Their minimum decode PCC values
are 0.999902546 and 0.999854728 respectively over eight teacher-forced steps.
State comparisons, repeated trace replay, all-step eager controls, live-trace
prefill, and runtime fallback audits pass under allocation tracking.

`context_l3_32k.json` passes a 32769-token full-attention prefill with a 65536-token
cache, followed by two eager decode steps. Prefill PCC is 0.999975805 and minimum
decode PCC is 0.999976699; the page-table, local-head replica, written-cache and
untouched-cache checks pass. The final-source linear 32769-token smoke and full 262142+2 maximum check
also pass (`context_l0_32k.json`, `context_fullmax.json`). Fresh paired
performance is in `latency_comparison.csv`; watcher/profiler and independent
review status is tracked by the README and gate manifest.
