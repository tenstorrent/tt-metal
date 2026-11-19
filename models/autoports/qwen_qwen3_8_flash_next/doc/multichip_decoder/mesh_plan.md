# Stage 4 mesh plan (before implementation)

Baseline: `OptimizedDecoder` at `fb462731`, checkpoint
`812d96e10a0dd82b69d749e68693698cf6c36daa`. Target: all four Blackhole
p300c chips, logical 1x4 ring, two links per edge. UMD discovery and a native
all-gather passed (`cluster_initial.yaml`, `mesh_smoke_initial.log`). Fabric is
`FABRIC_1D_RING`, descriptor `p300_x2_mesh_graph_descriptor.textproto`.
The zero ETH_LIVE_STATUS field in tt-smi is not treated as a connectivity test:
UMD finds all eight link pairs and the four-chip collective completes.

This is a design and candidate matrix, not acceptance or performance evidence.

## Tensor ownership

H=2560, four hyperconnection streams (residual width 10240), 48 layers;
36 GDN and 12 QSA layers. E=512, k=10, I=640. TP4 expert width would be
160=5 tiles, so choose EP4: 128 complete experts/device, 20 tiles wide.
At B=32, expected union fraction is `1-(1-10/512)^32 = 0.4680` (approximately).
Use per-token grouped dispatch/fused SwiGLU/combine at B>1 and prefill;
evaluate indexed EP batch1 against grouped batch1. Never compute all experts
densely. Router uses FP32 logits before top-k; selected weights may be BF16.

| Tensor / op | Global logical geometry | Proposed per-device geometry / layout |
|---|---|---|
| Residual | [1,1,T,10240] BF16 | [1,1,T,2560], one entire stream, DRAM interleaved |
| GR norm | [4,2560] | [1,2560], local whole-stream RMSNorm |
| GR down | [10240,320] FP32 | Output-parallel 80 columns padded to96; input normalized streams gathered; gather 4x96 then remove per-shard padding |
| GR up | [320,10240] FP32 | [320,2560] output parallel; exact baseline reduction |
| GR mixing | [4,T,2560] | Local stream products, FP32 collective/ordered reduction divided by4, BF16 boundary |
| GR injection | [10240,4] | One local column, padded internally32; input normalized stream gather shared with down |
| GDN Q/K/V | [2560,2048+2048+6144] | [2560,512+512+1536], heads4/4/12, no head padding |
| GDN z/a/b | [2560,6144+48+48] | [2560,1536+12+12]; explicit packed segment padding if matmul requires it; BF16 HiFi4 |
| GDN recurrence | [1,48,128,128] FP32 | [1,12,128,128], wholly local head ownership |
| GDN conv carry | [1,1,3,10240] | [1,1,3,2560], Q/K/V segments packed by head ownership |
| GDN output | [6144,2560] BF16 exact | Compare row-parallel FP32 partial reduction with gathered6144 + output-parallel640 columns to preserve baseline summation |
| QSA gated Q | [2560,24*512] | [2560,6*512], six Q/gate heads |
| QSA K/V | [2560,2*256] each | [2560,256], one KV head; duplicate each KV head on two ranks to serve six Q heads/rank |
| QSA paged KV | [pages,2,32,256] BF16 each | [pages,1,32,256], same page permutation and absolute positions on each rank |
| QSA indexer | four128-wide Q heads, one K head | Initially replicated ordered projection, compressed-key cache and selection; sharding candidate must preserve global selection |
| QSA output | [6144,2560] | [1536,2560] row parallel, collective into consumer contract |
| Experts gate/up | [512,2560,1280] | [128,2560,1280], whole experts in DRAM; selected-only execution |
| Experts down | [512,640,2560] | [128,640,2560], same global-to-local expert map |
| Shared expert | [2560,640], [640,2560] | Compare TP4 160-wide intermediate against replication; shared gate applied once |
| PLE packed rows | 320001536 x160 logical values | Interleaved row ownership id%4, local row id//4; 80000384 rows/device, 96 encoded bytes/row, same group10 scale LUT and quantizer |
| PLE lookup output | 16 selected rows/token x160 | Mask nonlocal rows on device, combine unique ownership, produce2560 features/token |
| PLE key | [2560,10240] | [2560,2560], one residual stream; norm and dilation3 conv entirely local |
| PLE value | [2560,2560] | replicated full value projection; local output-stream modulation |
| PLE conv carry | [1,1,9,10240] | [1,1,9,2560] |

All logical T>0 remain valid. Physical chunk tails and tiles are internal;
no padded row advances recurrence, hashing, current position, or cache.
The decoder input/output contract is identical across layer kinds and requires
no layer-boundary gather. Comparison fixtures gather once outside measurement.

## Row-projection and residual candidates

Bytes below are logical BF16 payload per token; tile padding multiplies decode
payload by up to32. Four-device ring reduction/scatter sends roughly3/4 of the
full input per rank per phase. The selected native synchronous CCL path uses CQ0 and framework-managed
resources. Async/fused alternatives require persistent buffers and explicit
semaphore ownership prepared before capture; they are candidate requirements,
not claims about the current implementation.

| Candidate | Residual before/after | Next consumer | Payload / dtype | Buffer plan / decision |
|---|---|---|---|---|
| Local matmul + all-reduce | replicated10240 | GR full norm/read | 5120B block; BF16 or FP32 partials | replicated control; pays two collective phases |
| Reduce-scatter, delayed gather | hidden-sharded [4,T,640] | distributed per-stream norm then gathered normalized input | block640x4; BF16/FP32 | replicate four injection scalars; consume hidden-sharded residual through next GR read |
| Fused all-gather + matmul | fractured input to output-parallel projection | local stream residual add | attention6144 gathered (12288B) | compare exact GDN output tree; cannot assume native FPU matches sensitive projection |
| Fused matmul + RS/AR | local-head input | sharded block then GR injection | block2560 (5120B) | retry supported padded shapes before rejecting |
| Stream-sharded residual, replicated block | local2560 stream throughout | local PLE/GR norm | block2560 replicated (5120B) | avoids full residual gather except GR's required down/injection input |

TTTv2 attention assumes conventional RMSNorm/residual and full attention, whereas
this model uses four learned hyperconnection streams, recurrent GDN, and a global
ordered sparse indexer. Reuse head ownership, paged caches, CCL workspaces and
row/column parallel families; ordinary Attention1D is not a direct substitute.
GPT-OSS CCL helpers and Kolibri grouped EP are implementation references.

## Memory and context gate

The advertised context remains262144. One local BF16 KV head costs
`2*262144*256*2=268435456` bytes per QSA layer/device; twelve layers cost3GiB
at B=1, before replicated compressed indexer state. EP4 BFP8 expert weights
alone cost `48*128*3*2560*640*1088/1024 = 32086425600` bytes/device.
Packed PLE adds7680036864 bytes/device. Thus the unchanged all-BFP8 stack
cannot fit a32GiB chip even at zero context. Lowering context cannot fix it.
Evaluate a memory-feasible expert storage policy using real-weight accuracy;
BFP4 candidates passed predecessor layer0 probes but are not yet accepted
jointly or on this mesh. No context reduction or full-stack fit claim is made
until the complete arithmetic and candidate validation are recorded.

Reject pure data parallelism (duplicates the memory problem), TP4 narrow
experts (five tiles), replicated PLE (30.72GB/device), and TP8 (not available).
Do not reject any feasible lower-movement topology merely for a first API error.

## Selected implementation after measurements

The preceding tables preserve the pre-code plan. Final selection uses EP4
whole experts and stream-sharded residuals, with all setup weights explicitly
mapped by MeshLoader. Weight/state byte arithmetic is now in
`memory_capacity_plan.json`; short all-kind, nonaligned, B32 and stack evidence
is `final_mesh_l{0,1,3}.json`. The target remains the entire1x4 mesh.

| Boundary | Selected geometry and policy | Decision evidence |
|---|---|---|
| GR mixed block | local[1,1,T,2560] BF16 products; gather across rank axis1, widen locally, ordered FP32 sum, divide4 | BF16 transport preserves values and halves transport width; persistent native AR changes batched output numerics for~2%gain and is not selected |
| GR residual | one complete2560 stream/rank; local whole-stream norm, normalized gather for exact down/injection | Hidden640-sharded residual consumed through distributed norm and next GR read was slower atT1/2/31/32 in residual_contract.json |
| GDN output | gather6144 head features, exact projection640 outputs/rank, then gather2560 block | Retains optimized sensitive summation; row-parallel probe is a separate experiment, not an assertion of identical arithmetic |
| QSA output | local1536 input features times[1536,2560] BFP8 weight; ordered gather/sum | Native BF16 TILE KV pages unchanged; six Q and one local KV head/rank |
| MoE grouped |128 complete experts/rank, BFP4 storage; active-only native fused SwiGLU LoFi with BFP8 internal staging | I/TP=160=5tiles, E512; B32 union0.468 requires per-token dispatch; all-kind whole-output PCC passes |
| MoE reduction n1 | ownership-masked BF16[1,1,1,10]@[1,1,10,2560], HiFi4 FP32 destination, BF16 result | Replaces one-core padded native reducer; final default improves B1 and passes all gates run so far |
| MoE reduction n2..32 | native post_combine_reduce with internal32-row padding and rank ownership table | Matmul replacement brought no larger-group benefit; attention/routing positions stay logical n |
| Shared expert | TP4 I160, BFP8 gate/up/down, HiFi4; scalar gate replicated and applied once per partial | Avoids replicated full shared-expert weight bandwidth; partial joins routed contribution before one collective |
| PLE | interleaved row id%4; code/scales padded16columns each; local norm/conv and key stream, replicated full value | Repack byte exact and complete32769-row public path under full-stack DRAM reservation |

Persistent expert scratch, RoPE, context indices, and the two rank bounds must
be shared through one MeshWorkspace across serial layers. Runtime CCL uses
native synchronous gather and framework-managed resources on CQ0; there is no
production-owned semaphore reset or overlap between layers. Input/output
residual shards remain[1,1,T,2560] BF16 at every stack boundary.

Indexed packed-only experts, fused MM-RS/AG-MM, GR injection K reduction and
further projection geometry/fidelity combinations remain unmeasured performance
work, not correctness limitations or claims of inferiority. The exact API,
storage and candidate-shape analysis is in indexed_decode_audit.md. Persistent
dual expert formats are physically excluded; packed-only designs are not.
