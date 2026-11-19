# Mesh plan, before implementation

Target: four P300c Blackhole chips, IDs 0–3, 1x4 FABRIC_1D, Linear
collectives on axis1. `topology_probe.log` proves fabric open/close. 110 worker
cores and eight DRAM banks per chip. Baseline: OptimizedDecoder at
0915d931, source SHA256 315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22.
Dense model: no router/experts, EP and active-expert decisions are inapplicable.

## Tensor ownership

| Tensor | Global | Per device / mapping |
|---|---|---|
| residual | [B,S,5120] | [B,S,1280] hidden shard, or replicated control |
| input/post norm | 5120 | 1280 shard with distributed statistics; replicated control5120 |
| full Q and gate | 24x256 each | six heads, 1536 columns each |
| full K,V | 4x256 each | one head, 256 columns each |
| full QKV | [5120,8192] | [5120,2048], head-local Q then K then V |
| full output | [6144,5120] | [1536,5120], partial then reduce-scatter |
| full KV each | [pages,4,32,256] | [pages,1,32,256], BFP8; identical replicated page table |
| GDN Q,K | 16x128 each | four heads,512 channels each |
| GDN V,Z | 48x128 each | twelve heads,1536 channels each |
| GDN B,A | 48 each | twelve values, internal32-column tile padding |
| GDN packed input | [5120,16512] | [5120,4160]: Q512,K512,V1536,Z1536,B32,A32 |
| GDN correction | Q/K/V/B/A residual | BFP8 correction, Z zero, FP32 accumulation before BF16 |
| GDN conv | [B,3,10240] | [B,3,2560], ROW_MAJOR BF16 |
| GDN state | [B,48,128,128] | [B,12,128,128], BF16 persistence; FP32 prefill recurrence |
| GDN output | [6144,5120] | [1536,5120], partial then reduce-scatter |
| MLP gate/up | [5120,17408] each | [5120,4352] column shard (136 tiles) |
| MLP down | [17408,5120] | [4352,5120] row shard |
| positions / rope | per user / token | replicated; never sharded across users |

No model-dimension padding is needed for TP4. GDN A/B have20 zero columns
per shard. DRAM-sharded N is rounded to lcm(banks*readers, worker cores)*32;
outputs are sliced back to logical local N. K-aligned worker choices must
respect local1536 (48tiles) and4352 (136tiles), not baseline K6144/17408.
Start with8 workers,2 readers, K-block2 (K48/136/160 all divisible); sweep
role-specific geometry on measured dominant rows. BF16 activations/CCL,
BFP8 weights/cache; GDN correction and HiFi4 retained. MLP policy follows
baseline kind. Context remains262144. Capacity accounting will include all
64 layers, embeddings/head and phase-specific copies before acceptance.

## Boundary topology candidates

Let A=ceil(B*S/32)*32*5120*2 payload bytes per full BF16 activation.
For B1 decode A=327680. Approximate ring-style lower-bound per-chip traffic
at TP4 is .75A for reduce-scatter or gather; Linear routing differs by rank.
Actual fabric/profile measurements decide, not this estimate.

| Family | Residual before/after | Next consumer | Traffic per boundary | Buffers / decision |
|---|---|---|---|---|
| local matmul + all-reduce | replicated / replicated | local norm then column projection |1.5A| ping-pong semaphores; correctness/timing control |
| reduce-scatter + delayed gather | sharded / sharded | distributed norm, gather normalized input at projection |1.5A + tiny stats| avoids replicated residual/add/norm; primary candidate |
| fused all-gather matmul | sharded / local expanded | next projection consumes normalized shard |.75A gather +.75A previous RS| adapt packed projection, persistent CCL buffers |
| fused matmul RS/AR | local expanded / sharded or replicated | residual add + next norm |.75A or1.5A| test supported local K/N, persistent buffers |
| RS then immediate AG | replicated / replicated | local norm |1.5A| control only; does not reject sharded family |

Rejected analytically: data parallel replicates all model weights and cannot
improve single-user latency; TP2 leaves half the available bandwidth unused;
2D TP2x2 duplicates KV heads or adds cross-axis reductions with no benefit for
this four-chip model's divisible heads. Pipeline parallel is inappropriate
for a decoder-layer baseline and adds single-user stage bubbles.

Reusable TTTv2 attention/MLP/norm contracts and GPT-OSS CCLManager are references.
Their standard attention lacks Qwen's output gate, partial RoPE, GDN recurrent
state and precision-correction branch. Model-local subclass retains baseline
norm/matmul/rope/chunk handling and adds local heads plus explicit CCL; reusable
async semaphore manager is used. This mismatch does not reject fusion families.

## Runtime evidence and candidate adaptations

The actual mesh provides2 links on axis1. CCL semaphore coverage now matches
all110 worker cores. The initial reusable8x8 semaphore grid was insufficient
for native worker selection and caused a first-use correctness risk; see
AUTOFIX.md and debug_b32/AUTODEBUG.md.

The original B32 decode physical residual has1024 padded rows, although only
32 logical rows exist. A candidate packs B*S rows before each collective and
distributed norm, restoring [B,S,H/4] afterward. At B32/S1 this reduces
BF16 full-width collective payload from10,485,760 to327,680bytes (32x).
At B32/S33 it reduces20,971,520 to10,813,440bytes. Public tensors and cache
head ownership stay unchanged; layout conversion cost is included in timings.

Fused MMRS prefill adapts the row-parallel producer and following sharded
residual/norm consumer, with explicit finite M padding and output slicing.
The local interleaved row-weight copy is setup-only and must be counted in
all-layer capacity if this candidate is selected. Native AGMM rejects Linear;
newer minimal AGMM is being evaluated with2links x4workers rather than
rejecting the fusion family on the first API failure.

## Selected implementation (supersedes initial candidate choices)

The public residual is hidden-sharded[B,S,1280]. Decode compacts B*S rows
internally. GDN gathers before hidden normalization in decode; full attention
gathers first only for a single user. Other prefill norms gather statistics,
normalize shards, then gather normalized activations at the projection.

MLP gate/up is one local[5120,8704] BFP8 weight (gate4352 then up4352), replacing
the separate decode weights. Prefill owns another[5120,8704] tile-pair-interleaved
copy for native SwiGLU. The down and attention output retain interleaved copies
for fused prefill matmul/RS. All copies count toward residency. Eight DRAM
banks, two readers/bank and eight activation storage cores remain selected;
K-block2 beats4/8/16 and four activation storage cores under the final LoFi
attention policy. DRAM matrices pad N to512; GDN input/correction4160→4608;
packed MLP8704 needs no N padding. Exact physical shapes, bank shard specs and
allocation sizes are in `capacity_plan.json`.

Attention projections use LoFi, including GDN primary/correction and output;
GDN MLP stays HiFi2, full-attention MLP LoFi. Norm and recurrence retain their
FP32 policies. BFP8 weights/KV and BF16 activations/CCL remain the default.
The candidate BFP4 setup adapter casts through BF16 only to perform finite
N padding, then repacks before DRAM placement; it does not affect BFP8 defaults.

The fused gather family was repaired with a device admission fence and
persistent buffers, then rejected against the best separate8x8 projection:
0.431357ms vs0.339615ms. Fused row matmul/RS is retained through the next
sharded residual/norm consumer. Replicated residual controls retain slower
prefill and do not justify changing the stack contract. See candidate_v2/v3,
geometry_v1, fidelity_v1, geometry_precision_v1 and lofi_geometry_v1 JSON and
command logs. CCL BFP8 adds casts without a useful measured decode gain.

The full-model expectation is serialized execution on one mesh/command queue.
Native fused-MMRS state is shared per mesh and reset by the operation; concurrent
execution on independent queues is not established. The64-instance resource
probe is a decoder test with shared weights, not a full-model implementation.
Full-model logits must use last-token or bounded token chunks; all-token logits
at maximum context alone would require30.3125GiB/device in BF16.

### Selected local narrow GDN correction

The single-chip baseline's narrow-correction optimization also wins after TP4.
The primary remains `[5120,4160]` (physical N4608), while correction retains
only QKV2560 plus padded BA64: `[5120,2624]`, physical N3072. The DRAM bank
shard changes from `[5120,576]` to `[5120,384]`; activation core shard remains
`[32,640]`. QKV and BA tile boundaries preserve BFP8 exponent blocks; the zero
Z correction is never read. The fused combine uses logical tile widths130/82,
QKV boundary80 and BA boundary128; actual padded strides come from tensors.
Decode FP32 primary/correction results remain L1 width-sharded until combine.

`narrow_v1_{control,narrow_l1,narrow_dram,full_l1}_b{1,32}` tests the four
combinations against recorded-activation optimized references. Selected narrow
plus L1 decode: B1 0.57404ms vs0.60360ms, B32 0.73298ms vs0.76418ms.
Narrow DRAM-only:0.58259/0.74344ms. All PCC/state/trace gates pass, with identical
PCC values across controls. Narrowing saves8,355,840 bytes/layer/device,
401,080,320 bytes across48 layers. Final full-stack residency accounting is
regenerated in `capacity_plan.json`; context remains262144.

### Final residual-layout decision

`residual_layout_comparison.json` pairs final-default sharded and replicated
runs with identical weights, inputs, precision and cache contract. Sharding
wins every prefill row. At B32 it also wins decode: GDN0.733ms vs0.790ms;
full attention0.702ms vs0.831ms at S33 and0.749ms vs0.879ms at S513.
Replicated residual is about3% faster at B1 decode (GDN0.556 vs0.574ms;
full0.473 vs0.490ms), while its B1 prefill is slower (GDN2.688 vs2.154ms;
full1.879 vs1.631ms). The selected single stack interface prioritizes the
largest supported decode batch, prefill and maximum-context working memory.
A residual tensor at full context is671,088,640 bytes/device sharded versus
2,684,354,560 replicated. Future phase-specific layouts may trade the small
B1 decode difference against an explicit phase transition; no unmeasured
phase-transition speedup is claimed here.

### Watcher-safe row-projection plan

The original unpartitioned native MMRS is rejected after two watcher failures.
Fresh AutoDebug establishes a dedicated-NoC counter ownership defect when
resident weight reads use NoC1; there is no public descriptor/read-NoC override.
Shared native changes are outside this stage. The accepted candidate design
uses ordinary row matmul plus BF16 reduce-scatter at flattened M<=512, and
native MMRS in <=128-row pieces for larger M. Each piece keeps local K1536
(attention) or4352(MLP), output N5120 with N-scatter to1280/rank; M is tile-padded
only internally. For128 and32 rows, native plans report M-on-Y grids4x10 and1x10,
resident footprints602/659KiB for attention/MLP, and NoC0 weight reads. Piece
outputs concatenate in original row order before logical crop/reshape. No
attention/recurrent sequence is split by this additional linear-only partition.
The serial layer-stack contract is unchanged. Live-L1/resource-pressure watcher
validation is required because native planning depends on available L1.

Measured selection: separate wins B1/S512 by19%(GDN)/10%(full) over chunked;
chunked wins B32/S254 by10% for both kinds and B32/S33 GDN by10%. Unchunked
fusion is numerically correct and faster but fails the hard watcher gate.
For native SwiGLU,33..512 rows use M2/K8/N26,subblock1x2,grid11x8;
larger shapes and<=32-row tails retain M4/K4/N16,subblock1x4,grid11x10.
The88-core small configuration has95.1% useful tile-volume and837632 main
DFB bytes/core at M512, versus76.1% and598016 bytes for the prior110-core
configuration. Exact64-core partition and large/tail alternatives were
measured and rejected; see native_prefill_candidates_summary.csv.

Ordinary interleaved prefill uses grid11x10 and K-block8 in the selected
source. Coupled safe-path measurements favor this for GDN/full B32 and full
B1, accepting a0.094ms GDN B1 cost versus8x8/K4. Native MMRS has its own
planner; these ordinary matmul settings do not override its NoC/grid choices.
Final source before hard gates:
`63d2e53af8490dd63ec081ea202611e21dfb3ddca3e033e851e89837d0debb88`.

### Final large-matrix crossover

Exact-policy B32/S513 controls show separate row matmul/RS wins beyond the
intermediate regime: GDN67.347→59.182ms; full52.531→44.758ms. Final native
dispatch is therefore restricted to512<M<=8192, with128-row pieces. M<=512
and M>8192 use separate matmul/RS. Phase copies remain allocated for later
intermediate prefills and remain counted in capacity. Delivery source
`082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`
is the final plan; previous paragraphs retain the measured decision history.
