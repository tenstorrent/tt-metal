# Multichip mesh plan and implemented candidates

This document retains the original preimplementation plan and records the selected
1x4 Linear replicated-residual hybrid EP-prefill/TP-decode path. Detailed final
precision and geometry are in selected_policy_audit.md; historical measurements
belong to their recorded source hashes. Final-policy sharded-residual/Ring/fused
AGMM and shared-DRAM controls pass correctness but are slower (final_policy_alternatives.json).

## Current ownership and precision

The target is four physical Blackhole P300c ASICs in a 1x4 mesh. The replicated
candidate uses `FABRIC_1D`, Linear CCL, and one link. The measured hidden-sharded
alternative uses `FABRIC_1D_RING`, Ring CCL, and one link. TP attention and KV
ownership are the same under either residual layout. Allreduce/RS occurs after
local expert-score weighting and summation, never separately for each expert.

| Tensor or weight | Per-device shape / ownership | Retained dtype / policy |
| --- | --- | --- |
| Public residual, replicated candidate | `[1,1,S,2816]`, identical on four ranks | BF16 input/output; internal normalization/residual precision follows optimized decoder |
| Public residual, hidden-sharded alternative | `[1,1,S,704]`, contiguous hidden shard per rank | BF16 input/output; distributed RMSNorm and projection-input gathers |
| Sliding Q / KV heads | 4 Q heads and 2 KV heads, head dimension 256 | Existing Gemma4 head norms and RoPE; BFP8 K/V cache |
| Full Q / KV heads | 4 Q heads and 1 KV head, head dimension 512; KV head 0 on ranks 0/1, head 1 on ranks 2/3 | Existing tied-KV and partial-RoPE semantics; BFP8 K/V cache |
| Sliding packed QKV / output weights | `[2816,2048]` / `[1024,2816]` | BFP8, column / row partition respectively |
| Full packed QKV / output weights | `[2816,3072]` / `[2048,2816]`; packed QKV includes pair-duplicated KV | BFP8, column / row partition respectively |
| TP expert gate/up / down weights | `[1,128,2816,384]` / `[1,128,192,2816]`; expert intermediate 704 padded to 768, then split four ways | Gate/up and down BFP4; BFP8 decode input; LoFi |
| EP expert gate/up / down weights | `[1,32,2816,1408]` / `[1,32,704,2816]`; rank `r` owns experts `[32r,32r+32)` | Gate/up BFP8 sliding or BFP4 full; down BFP4; LoFi |
| Shared BF16 gate/up / down weights | `[1,1,2816,1088]` / `[1,1,544,2816]`; global intermediate 2112 padded to 2176; each rank packs its up/gate pair | BF16 prefill weights retained |
| Selected shared decode copies | Same local shapes as shared BF16 weights | GU BFP4 both kinds; down BFP8 sliding/BFP4 full; LoFi, BF16 activations/output; BF16 prefill retained |
| Router, norm data, positions and page tables | Replicated router/top-8 and logical page/position indices; sharded residual mode additionally owns local norm weights | Existing dtype contracts; setup-owned position and routing-index tensors |
| Per-kind RoPE tables | Absolute context 262144; prefill and decode layouts shared across layers of the same kind in the capacity plan | BF16; no per-layer full-context RoPE duplication assumed |

The implemented `hybrid_experts=True` candidate retains **both** expert layouts:
EP4 handles fresh multirow prefill, while indexed TP4 handles each one-row
decode. Prefix continuation uses one-row decode and therefore also uses TP4.
This retains all 128 checkpoint experts without replicating each complete
expert on every rank. The EP prefill route gather uses setup-owned rank-local
expert indices and 32-token sparse unions. Inputs and router scores are already
replicated, so this candidate introduces no token all-to-all dispatch. TP
decode executes its eight selected experts on every rank's intermediate shard;
EP-only decode remains an optional measured control with dynamic 0..8 local
active experts and zero-filled inactive outputs. No dense all-expert decode is
introduced. See [EP diagnosis and experiment](AUTOFIX_ep.md).

`optimized_shared=True` adds the quantized shared decode copies while retaining
BF16 prefill matrices. Its compute policy is LoFi with
`fp32_dest_acc_en=False`, `packer_l1_acc=True`, and BF16 output. This matched the
TP1 shared-decode policy and passed the mixed layer-0 → layer-5 direct device
handoff test in [stack_shared_policy.json](stack_shared_policy.json).
`fused_tail=True` preserves independent shared/routed post norms, then the
existing combined norm and residual. It is exercised by the replicated
candidate; the hidden-sharded path has its own distributed norm tail.

Selected shared geometry is2 for sliding and1 for full; grouped MoE reduction
is enabled. Explicit alternatives remain available for reproducible controls.
Grouping joins local shared/routed BF16 outputs as `[1,2,S,2816]`, uses one
existing RS/AG pair, and splits before the independent norms. It currently
requires replicated residuals. The selected default uses grouped reductions
and the per-kind explicit geometry above. DRAM-sharded shared weights are a separate trial with
additional rank-local packing/padding requirements; they are not a selected
layout. See [geometry audit](shared_geometry_audit.md) and
[grouped-reduction audit](grouped_moe_reduce_audit.md).

## Compatible measured residual and collective alternatives

These historical batch-1 4096-prefill/128-advancing-decode results are medians of warmed
**host-wall** intervals, not device durations. Every listed run passes output
PCC ≥0.995 and local cache comparison. The configurations are separate measured
alternatives; component savings cannot be added to complete-layer timings.

| Configuration | Sliding TP4 prefill / decode, µs | Full TP4 prefill / decode, µs | Evidence |
| --- | ---: | ---: | --- |
| Linear, replicated, hybrid + optimized shared + fused tail | 95012.64 / 817.75 | 80566.95 / 812.31 | [sliding](sliding_shared_policy.json), [full](full_shared_policy.json) |
| Ring, hidden-sharded, hybrid + optimized shared, separate QKV gather/matmul | 89719.09 / 936.42 | 74611.52 / 937.34 | [sliding](agmm_layer0_headline_separate.json), [full](agmm_layer5_headline_separate.json) |
| Ring, hidden-sharded, fused AGMM using matching 1D geometry, DRAM output | 88730.11 / 949.47 | 74517.92 / 938.63 | [sliding](agmm1d_layer0_headline_fused.json), [full](agmm1d_layer5_headline_fused.json) |
| Ring, hidden-sharded, fused AGMM using matching 1D geometry, L1 output | 89654.34 / 943.43 | 74536.10 / 933.93 | [sliding](agmm1d_l1_layer0_headline_fused.json), [full](agmm1d_l1_layer5_headline_fused.json) |

The Ring alternatives use hidden width 704 directly between layers. The
`fused_agmm` option requires both Ring topology and hidden-sharded residuals.
Its current 1D implementation reuses the separate projection's matmul geometry
and L1 QKV output, with setup-owned gathered buffers and semaphores. Prefill
still gathers explicitly and uses the existing prefill projection. The 1D/L1
runs have minimum output PCC 0.999328527 sliding and 0.999337831 full, and minimum
cache PCC 0.999982003 / 0.999965247. They use runtime `d4c8f9d9...`; the
replicated policy rows use `5f75aa5f...`. The earlier fused-2D variants and tuned
MMRS controls remain in [the fused-CCL investigation](AUTOFIX_fused_ccl.md).
The final-policy compatible controls repeat this comparison at810.761us sliding
and877.898us full versus650.445us/724.702us selected. See
final_policy_alternatives.json; the table above is historical, not final evidence.

## Bounded output assembly and maximum context

The logical context remains **262144**, with page size 32 and paged-attention
read extent rounded up to 128. Fresh replicated prefill retains physical,
tile-aligned outputs of at most 1024 rows, merges them in groups of at most 32,
and combines at most eight final groups. It releases child references before
slicing to the requested nonaligned logical length. The fresh hidden-sharded
alternative retains its inherited prefill assembly and has separate short/
4096-token evidence; the replicated maximum-capacity passes are not transferred
to it.

Nonzero-prefix continuation preserves exact per-token paged cache updates,
including partially occupied pages. Its implemented assembly collapses 32
one-token outputs into one tile, then merges 32 tiles into 1024-row chunks and
32 chunks into 32768-row groups. Only the last output block repeats a final row
reference as temporary padding; no extra cache update is issued, and the final
logical slice discards the padding. The largest final concat has eight inputs.
See [continuation capacity audit](prefill_continuation_capacity_audit.md).

| Historical context control, replicated hybrid/shared/fused-tail candidate | Minimum output / local-cache PCC | Resident reservation per device | Evidence |
| --- | ---: | ---: | --- |
| Sliding, 262143 prefill + one decode to absolute position 262143 | 0.999960840 / 0.9999999995 | 21474836480 bytes | [sliding_max_capacity.json](sliding_max_capacity.json) |
| Full, 262143 prefill + one decode to absolute position 262143 | 0.999924724 / 0.999971021 | 21072183296 bytes | [full_max_capacity_verified.json](full_max_capacity_verified.json) |
| Sliding, aligned 262144 prefill, zero decode | 0.999967650 / 0.9999999995 | 21474836480 bytes | [sliding_max_aligned.json](sliding_max_aligned.json) |
| Sliding, prefix 31 + continuation 1025, then decode | Prefill 0.999954979 / decode 0.999951751 | No full-stack reservation in this control | [sliding_long_prefix.json](sliding_long_prefix.json) |
| Full, prefix 31 + continuation 1025, then decode | Prefill 0.999914782 / decode 0.999750100 | No full-stack reservation in this control | [full_long_prefix.json](full_long_prefix.json) |

The long-prefix controls check exact prefix preservation, replica equality,
repeated trace equality, refreshed ownership, and device-only execution. Their
batch-1 other-slot check has no second request to exercise; independent-slot
coverage is recorded separately in [batch32_sliding_hybrid.json](batch32_sliding_hybrid.json),
which predates the optimized-shared policy. The maximum-context controls use repeated fixture
input and anonymous DRAM reservations for the other resident tensors. They
exercise real layer computation and the last legal position on both attention
kinds; they do not instantiate a complete model stack or measure its allocator
fragmentation. Current a12a913c passes both262143+1 and262144+0 for both kinds,
both batch32 controls, and both31+1025 continuation controls; see
final_validation_summary.json. Arbitrarily long prefix continuation uses the
same bounded assembly, but only the recorded continuation lengths were measured.

The [memory plan](memory_capacity_plan.json) budgets25 sliding and5 full layers,
all262144 absolute cache tokens at batch1, tied replicated BF16 embedding/head,
and RoPE shared per kind/layout. BFP8/BFP4 tiles occupy1088/576bytes. The selected
hybrid's TP-decode base is4,337,336,320 weight bytes/device; EP prefill adds
4,797,562,880 and shared decode copies96,701,440, for9,231,600,640 total.
With8,556,380,160 cache bytes, embedding/RoPE,2GiB reserve and4,429,185,024
long-prefill live-buffer bytes, conservative peak is27,451,657,216bytes/device.
Current reservation rounding gives27,513,266,688 sliding and27,455,189,504 full,
both below32GB. Historical capacity artifacts retain their older larger budget.
See memory_audit.md for constructor lifetimes and the raw BFP4 setup allowance.

Final-source runtime gates pass; native profile accounting and independent
stage review are recorded separately in README.md. Historical alternatives do
not by themselves establish current acceptance.

---

## Historical plan before implementation

The following original plan is retained as history. Its statements that EP4
was unmeasured, prefix assembly was inherited, and full-stack capacity work was
still required describe that earlier point, not current implementation status.

Target: four Blackhole P300c ASICs, one 1x4 mesh, FABRIC_1D, Linear CCL,
initial num_links=1. All four open/close successfully. Shape-faithful reduce-scatter
[1,1,32,2816] passes exactly in BF16 and FP32 (collective_smoke.log).
Baseline: OptimizedDecoder at a9259624f2 / documentation 2e3a1779d3.
This is a candidate plan, not an optimality or completion claim.

| Tensor | Global logical shape | Local physical shape / mapping |
| --- | --- | --- |
| residual | [1,1,S,2816] | candidate replicated or hidden-sharded [1,1,S,704] |
| sliding Q | 16x256 | 4x256 |
| sliding KV | 8x256 | 2x256; cache [pages,2,32,256] |
| full Q | 16x512 | 4x512 |
| full KV | 2x512 | 1x512, head0 on ranks0/1, head1 on ranks2/3 |
| packed sliding QKV weight | [2816,8192] | [2816,2048] column shards |
| packed full QKV weight | [2816,10240] | [2816,3072], duplicated KV across rank pairs |
| attention output weight | [4096 or8192,2816] | [1024 or2048,2816] row shards |
| expert gate/up | [128,2816,2x704] | pad I=768; each rank [128,2816,2x192], paired packing |
| expert down | [128,704,2816] | [128,192,2816], zero padded K |
| shared gate/up | [2816,2x2112] | pad J=2176; [2816,2x544] |
| shared down | [2112,2816] | [544,2816] |
| router/norm/RoPE | original | replicated; router top8 identical on all ranks |
| page table/positions | original logical indices | replicated; identical physical-page ownership per rank |

Sparse experts retain all128 checkpoint experts partitioned over intermediate
width; only gate-selected active experts execute decode. Weight dtypes start
from optimized policies (BFP8 sliding gate/up, BFP4 full gate/up and both down).
Do not replicate full experts: it wastes capacity and single-user bandwidth.
EP4 is an alternative but creates active-expert imbalance and dispatch traffic;
it remains unmeasured. DP does not help the batch1/concurrency1 target.

H=2816/4=704 is tile aligned. I and J padding stays inside MLPs and has zero
down weights. Public logical S remains arbitrary; inherit optimized chunk/tail
and prefix orchestration. Every layer must consume its own output layout.
Allreduce occurs after score-weighted local expert reduction, never per expert.

### Collective candidates

Let A = physical_rows *2816*bytes_per_element; each logical width shard is A/4.
Ring-equivalent traffic below is per-rank algorithmic volume, excluding headers;
actual Linear topology timing must be measured.

| Boundary family | Residual before/after | Next consumer | Nominal bytes/rank | Dtype | Persistence / decision |
| --- | --- | --- | --- | --- | --- |
| local matmul + allreduce | replicated/replicated | post norm | 1.5A | FP32 and BF16 controls | common ping-pong semaphores; reference candidate |
| reduce-scatter delayed gather | replicated/sharded | distributed post norm + residual | .75A plus small norm stats | FP32/BF16 | gather only before next column projection |
| fused gather-matmul | sharded/local partial | next row projection | .75A | op-supported | test packed projection contract |
| fused matmul reduce-scatter | replicated/sharded | distributed post norm | .75A | op-supported | persistent output/semaphores; check BH support |
| hidden-sharded residual end to end | sharded/sharded | distributed input norm | stats + gathered projection inputs | FP32/BF16 | adapt both shared/routed post norms; gather only at test boundary |

Measure producing and consuming operations together. Immediate RS->AG alone
does not reject a sharded residual. No performance rejection is earned yet.
Common Attention1D does not express Gemma4 Q/K/V head RMSNorm, tied global KV,
partial RoPE and post-attention normalization directly. Common MLP1D differs
in GeGLU and the independent shared/expert post norms. Reuse baseline attention
math and GPT-OSS sparse experts/CCL families with explicit local dimensions.

### Capacity target

Retain262144 logical tokens. BFP8 tile is1088 bytes. Local K+V payload is
2*262144*2*256*1088/1024=285212672 bytes per sliding layer and
2*262144*1*512*1088/1024=285212672 per full layer. Global KV pair replication
makes full cache reduction2x, sliding4x. Full-stack weight/cache accounting and
reserved trace/activation peak are still required; no capability reduction.
