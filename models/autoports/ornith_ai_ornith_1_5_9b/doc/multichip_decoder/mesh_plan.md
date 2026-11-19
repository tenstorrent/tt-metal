# Stage 4 mesh plan (recorded before implementation)

Target: four P300c Blackhole chips, IDs 0–3, 1x4 mesh, FABRIC_1D,
Linear CCL (final path uses both available links), 11x10 compute grid and eight DRAM banks/chip. Mesh open/close
passed in mesh_smoke.log. Baseline: optimized_decoder.py, stage 3 clean-pass.
HF revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53. Dense model, no experts:
EP, top-k routing and expert-union heuristics do not apply.

Choose TP4 across the one mesh axis. Each chip owns contiguous head groups and
MLP channels; no head replication. Compare replicated and hidden-sharded residual
streams before selecting the measured default. Both attention kinds share the same
layer interface. Candidate local BF16 residual is [1,1,S,1024], versus replicated
[1,1,S,4096]. Decode's physical token tile is 32; logical token count remains 1.

| Tensor | Global | Per chip | Mapping/padding |
|---|---|---|---|
| residual | S x 4096 | S x 1024 or S x 4096 | hidden shard or replicated |
| Q/gate full | 4096 x (4096+4096) | 4096 x (1024+1024) | 4 adjacent Q heads and their gates |
| K,V full | each 4096 x 1024 | each 4096 x 256 | one KV head |
| full packed input | 4096 x 10240 | 4096 x 2560 | pack local Q,gate,K,V |
| full WO | 4096 x 4096 | 1024 x 4096 | input/row parallel |
| linear Q,K,V | 4096 x (2048+2048+4096) | 4096 x (512+512+1024) | 4 key heads, 8 value heads |
| linear z | 4096 x 4096 | 4096 x 1024 | contiguous value head group |
| linear b,a | each 4096 x 32 | each 4096 x 32 | 8 logical heads, zero pad each projection to 32 |
| linear packed input | 4096 x 12352 | 4096 x 3136 | local QKV,z,b,a tile packing |
| linear WO | 4096 x 4096 | 1024 x 4096 | input/row parallel |
| conv taps/history | 4 x 8192 / 3 x 8192 | 4 x 2048 / 3 x 2048 | local Q,K,V groups; history RM |
| recurrent state | 32 x 128 x 128 FP32 | 8 x 128 x 128 FP32 | no cross-chip recurrence |
| KV cache each | C/32 x 4 x 32 x 256 | C/32 x 1 x 32 x 256 | BFP8, replicated shuffled page table |
| MLP gate/up | each 4096 x 12288 | each 4096 x 3072 | output parallel, packed tile pairs for prefill |
| MLP down | 12288 x 4096 | 3072 x 4096 | row parallel |
| input/post norm weight | 4096 | 1024 or 4096 | distributed stats when residual sharded |
| attention head norm | 128 / 256 | unchanged | replicated norm per local head |
| RoPE and positions | 262144 x 128 / 262144 | replicated | shared tables can be reused by stack |

Decode DRAM weight shards: eight banks, shard [local_K, ceil(local_N/256)*32].
Activation width shard: initially 32 cores, [32,ceil(width/1024)*32]; sweep against
64 cores. Input/output matmul layouts derive from local dimensions, never the old
full-model geometry. Prefill minimal-matmul starts M4/K16/N8, subblock1x2,
11x10, inherited BFP8 HiFi2 attention and BFP8 LoFi MLP with FP32 accumulation.
GDN stays FP32 HiFi4; no precision reduction to recurrent state.

## Collective topology candidates

Let A = max(32,S)*4096*2 BF16 bytes; 4-chip ring-equivalent payload
per chip is 3A/4 for RS or AG, 3A/2 for AR (line edge loads differ).

| Boundary family | Residual before/after | Next consumer | Payload | Buffers/decision |
|---|---|---|---|---|
| local MM + RS + AG (AR) | replicated → replicated | local RMSNorm then packed projection | 1.5A | generic CCL semaphores; correctness/control |
| local MM + RS, delayed AG | hidden shard → hidden shard | residual add, distributed RMSNorm, AG normalized values | .75A + stats + .75A | preserves residual through norm; must measure |
| fused AG + input MM | hidden shard → local head/channel result | conv/SDPA or SwiGLU | .75A plus stats | persistent gathered scratch; adapt local packing |
| fused MM + RS | local head/channel result → hidden shard | residual add/distributed norm | .75A | inspect supported geometry and measure if expressible |
| replicated residual with direct AR | replicated → replicated | norm | 1.5A | test supported fused/one-op reduction |

BF16 activation/CCL baseline. Setup owns semaphores and any persistent CCL outputs;
forward allocates only temporary tensors and returned output. Boundary test gathers
are excluded from layer timings. Sharded residual is consumed by next norm before
gather, including cross-kind stack. No per-layer test-boundary gather is allowed.

Pipeline parallelism rejected as initial strategy: one-token dependency serializes
stages and applies only one chip to a layer. DP improves independent throughput but
not single-user latency. TP2 leaves half the available compute idle for one request.
2D 2x2 offers no clean extra axis for this small dense model and introduces another
collective axis. TP4 aligns every head and MLP dimension without model padding.

Reusable modules: GPT-OSS CCLManager / common TT_CCL provide semaphore ownership.
TTT attention/MLP contracts guide local heads, paged cache, RS and distributed norm.
The baseline Qwen3.5 full attention has per-head output gates and partial RoPE;
linear layers have causal conv and FP32 GDN, which generic softmax attention cannot
express. Preserve those optimized composites with local dimensions. No common MoE
module applies. Keep all valid logical lengths; chunk/page/tile padding is internal.

## Context capacity plan

Advertised and intended supported context: 262144 (unchanged). One chip's two full
KV caches: 2*C*1*256*(1088/1024) = 142606336 bytes per full-attention layer,
8 layers = 1140850688 bytes. Linear recurrence: 24*8*128*128*4 = 12582912 bytes;
conv history 24*3*2048*2 = 294912 logical bytes. Decoder weights approximately
9B/4*1.0625 = 2.39 GB before extra decode/preload copies; budget 6 GiB for all
weights/copies including embeddings and head, 2 GiB for activation/trace/CCL,
and 1 GiB shared tables/allocator margin. Total < 11 GiB per 32 GiB chip.
Exact stored-weight accounting and runtime capacity evidence will refine this
conservative estimate. No physical reason to reduce advertised context.

## Measured topology selection

Initial coherent-family results are in perf_{baseline,sharded,replicated,gather,
scatter,both,direct,hybrid}_{0,3}.json. All candidates use real checkpoint weights,
BFP8 projections, BF16 activation/CCL and the same HiFi2 attention / LoFi MLP policy.
Fused scatter's cross-chip accumulation uses FP32 before BF16 output; no weight
precision fallback. Fused AG+MM retained a hidden-sharded residual and consumed it
through distributed RMSNorm; it was not rejected after an immediate full gather.
Both fused families were combined and measured, including the compatible precision.

Selected final family: replicated mesh residual [1,1,S,4096], locally width-sharded
L1 residual/norm/MLP during decode, two-link direct all_reduce_async for decode, and separate
minimal matmul + two-link reduce-scatter + all-gather for prefill. The original faster
resident fused-prefill family violated dedicated-NoC accounting under watcher.
AutoTriage verified the cause; 64/96-KiB public-API planner constraints avoid its
split-NoC regime but measured slower than the separate path. See
watcher_triage/AUTOTRIAGE.md and AUTOFIX.md. No watcher check is disabled.

The sharded-residual and fused gather/scatter families were adapted and measured
before this decision. The default preserves the optimized baseline's local L1
residual/norm contract and exposes one common replicated mesh interface to both
layer kinds. Prefill matmul inputs use L1 after a measured advice trial; the
explicit minimal-matmul config remains M4/K16/N8, subblock1x2, grid11x10.

Decode AR owns two ping-pong BF16 L1 intermediate buffers per serial stack, each
[1,1,32,16384], width sharded over8x4 cores with [32,512] shards, and two semaphores.
Input/output AR use [32,128] shards over the same8x4 grid. Each layer issues two
ARs, restoring its ping-pong index; static trace capture uses stable buffer bindings.
The shared generic CCL manager owns prefill RS/AG semaphores. Prefill outputs are
gathered to the replicated residual before the next sublayer norm.

Final decode geometry (width shards; one logical row occupies a physical32-row tile):

| Role | Local K,N | Input cores / shard | Requested output cores / shard | K block | Readers/bank |
|---|---|---|---|---|---|
| linear packed input |4096,3136|32 / [32,128]|32 / [32,128]|2|2|
| full packed input |4096,2560|64 / [32,64]|64 / [32,64]|2|2|
| gate, up |4096,3072|64 / [32,64]|64 / [32,64]|2|2|
| down |3072,4096|48 / [32,64]|64 / [32,64]|1|4|
| output projection |1024,4096|32 / [32,32]|64 / [32,64]|1|4|

TTNN compacts the output compute grid when fewer cores cover N (for example25
linear packed-output cores and48 gate/up cores). Consumers use the returned actual
layout; the requested rectangle is not asserted as the physical output grid.
Packed attention results convert to interleaved for head/conv consumers. Gate/up
stay sharded through SwiGLU; down/output reshard directly into the AR layout.
Global RMSNorm uses32 cores, block_h1/block_w4/subblock_w2, and stays in that layout
through residual adds. Layer output returns to replicated interleaved DRAM for the
shared stack boundary; no mesh gather is inserted between layers.

## Stack and state interface

All four devices participate in every layer. Default entry uploads use
ReplicateTensorToMesh; outputs remain replicated and pass directly to the next
layer. Tests gather/compose only when evaluating the sharded residual candidate,
outside measured forward. Every row has an independent DecoderState. Full KV is
local [physical_pages,1,32,256]; physical page tables and int32 current positions
are replicated, but local cache contents correspond to each chip's own KV head.
Linear head ownership is contiguous: rank r owns key heads4r:4r+4 and value heads
8r:8r+8; convolution packs local Q,K,V. No collective touches recurrent state.

The context memory plan budgets one maximum-context request. Batch32 testing uses
independent short contexts; it does not claim32 simultaneous262144-token caches.
Memory demand is additive across resident requests. The advertised per-request
context remains262144. Exact tile/DRAM-bank padding arithmetic is in
memory_capacity_plan.json; physical usable DRAM is8*4278190080 bytes per chip,
as recorded by the Blackhole SoC descriptor (slightly below32 GiB).


## Capacity and geometry refinements before final validation

The exact-K input planner separates input storage from requested output geometry.
For requested output cores16/32/64, K1024 uses input16/32/32, K3072 uses16/32/48,
and K4096 uses16/32/64. Tile count per input shard is K/(32*input_cores); each
in0 block divides that count. This avoids the source-confirmed single-reader
sender-coordinate truncation in the initial64-core down sweep. See AUTOTRIAGE.md
and AUTOFIX.md. The output core grid remains independently configured.

Full-context temporary storage is larger than the initial2 GiB allowance:
replicated BF16 [1,1,262144,4096] input, accumulated chunk outputs, and concatenated
result each occupy2 GiB. Reserve8 GiB for those6 GiB plus temporary/trace/CCL
headroom, and another1 GiB allocator margin. Exact total is16.340 GiB/chip against
31.875 GiB usable DRAM. This is a conservative capacity plan, not a claim of
measured allocator peak. Final source executes all three long lengths; see final_long_{0,3}.json.

Construct one `MultichipCollectives(mesh)` and pass `collectives=workspace` to every
layer in a serial stack. Its two L1 receive buffers consume2 MiB/chip,64 KiB on each
of32 cores. Separate per-layer buffers would consume2 MiB/core at32 layers and
exceed Blackhole's1.5 MiB/core physical L1. The explicit shared workspace avoids
that full-stack failure and shares the generic CCL semaphore manager. Independent
concurrently executing stacks require independent workspaces; this decoder's
validated contract is serialized execution on one command queue.

## Fabric-link refinement

`ttnn.get_num_links(mesh,1)` reports2. Initial family trials deliberately used1;
`perf_available_links_{0,3}.json` measured2-link prefill, and
`perf_two_link_reduce_{0,3}.json` extended2 links to decode. Final source uses both
for RS/AG and direct AR. All final gates were rerun after this change;
`pre_two_links/` preserves the previous complete passing suite/source/profiles.
Final whole-layer timings, speedup and efficiency are in performance.csv.

Direct AR of the full1024-token prefill needs8 MiB/worker at1 link or4 MiB at2
according to `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/
all_reduce_async_program_factory.cpp` output-page scratch allocation, exceeding
physical1.5 MiB L1. Smaller-chunk direct AR is an unmeasured future candidate.
