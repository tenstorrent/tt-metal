# Blackhole p300c 1×4 mesh plan

This plan was selected before coding the final multichip path. Logical dimensions remain public; padding below is internal and sliced before semantic boundaries.

## Selected dataflow

- TP axis: all four chips on mesh axis 1, Ring topology, one link.
- Weights: column parallel for Q/K/V/gate, linear-attention input projections, and MLP gate/up; row parallel for attention output, linear-attention output, and MLP down.
- Full-attention cache: one local KV head per chip. Page tables and current positions are replicated; each device updates/reads only its local-head paged cache.
- Linear-attention state: four key heads and twelve value heads per chip. Convolution and recurrent state follow those local head owners.
- Activations: replicated hidden-5120 layer boundaries are final. The coherent width-fractured residual candidate was faster in isolation but slower in an integrated whole layer (results below).
- Collectives: row-parallel projections produce partial hidden-width results. Compare all-reduce replicated residual against reduce-scatter with delayed gather, fused matmul/CCL forms supported on this Blackhole build, and a stack-compatible fractured residual. CCL payload starts at BF16 to preserve the optimized baseline's activation policy; lower precision is an A/B candidate, not assumed safe.
- Constants/norm weights/page tables/current positions are replicated. Tensor conversions, slicing, and packing happen at load time or test/model entry boundaries, not inside traced replay.
- MoE: not applicable; all 64 layers use dense gated MLPs.

## Calculated per-device tensor contract

| Family | Global logical shape/dimension | TP=4 local dimension | Placement and padding |
|---|---:|---:|---|
| Residual | `[..., 5120]` | replicated 5120 control; fractured 1280 candidate | 1280 is tile aligned; no logical padding |
| Full Q | `5120 × 6144` (24×256) | `5120 × 1536` (6 heads) | column shard, exact |
| Full K/V | each `5120 × 1024` (4×256) | each `5120 × 256` (1 head) | column shard, exact; local cache head owner |
| Full gate | `5120 × 6144` | `5120 × 1536` | column shard, exact |
| Packed Q/K/V/gate | `5120 × 14336` | `5120 × 3584` | device-major repack at load time; exact |
| Full O | `6144 × 5120` | `1536 × 5120` | row shard, hidden partial then collective |
| MLP gate/up | each `5120 × 17408` | each `5120 × 4352` | column shard, exact; decode local weight is DRAM width-sharded `[5120,544]` over 8 banks |
| MLP down | `17408 × 5120` | `4352 × 5120` | row shard, decode local weight `[4352,640]` over 8 DRAM banks, then collective |
| Linear key heads | 16×128 = 2048 | 4×128 = 512 | exact head ownership |
| Linear value heads | 48×128 = 6144 | 12×128 = 1536 | exact head ownership |
| Linear Q/K/V | `5120 × 10240` | `5120 × 2560` | column shard, exact |
| Linear Z | `5120 × 6144` | `5120 × 1536` | column shard, exact |
| Linear B/A | each `5120 × 48` | each logical `5120 × 12` | pad each local scalar projection to 32; slice to 12 before head math |
| Packed linear input | logical local output 4120 | physical local output 4160 | internal 40-element padding from two 12→32 scalar groups |
| Linear output | `6144 × 5120` | `1536 × 5120` | row shard, hidden partial then collective |
| Full KV cache/layer | `[blocks,4,page,256]` K and V | `[blocks,1,page,256]` K and V | block/page mapping unchanged and replicated; head dim exact |
| Linear convolution state | 2×2048 + 6144 channels | 2×512 + 1536 = 2560 | local channels, exact |
| Linear recurrent state | `[batch,48,128,128]` | `[batch,12,128,128]` | local value-head ownership, exact |

Physical BFP8 cache tiles are 1088 bytes. B1 C=262144 uses 2,281,701,376 cache bytes/device and B32 uses 73,014,444,032. The auditable loader-object sum is 10,599,141,888 bytes/device after official-weight PCC selected BF16 full projections; B32 linear state is 572,522,496 bytes/device. Final trace allocator snapshots measure a maximum 1,969,152-byte load-to-warmed/captured delta. B1 C=262144 passes; fresh-process bracketing finds B32 C=82432 passes and C=82496 fails.

Decode projection placement after profiling is:

| Projection | Local logical weight | 8-bank DRAM shard | `in0_block_w` |
|---|---:|---:|---:|
| full QKV/gate | `5120×3584` | `5120×448` | 4 |
| full O / linear O | `1536×5120` | `1536×640` | 3 |
| MLP gate/up | `5120×4352` | `5120×544` | 4 |
| MLP down | `4352×5120` | `4352×640` | 17 |
| linear packed input | logical `5120×4160`, physical N=4352 | `5120×544` | 5 |

Decode inputs and outputs use eight-core L1 width sharding around these matmuls. Prefill retains interleaved weights and arbitrary logical sequence lengths.

## Row-parallel topology candidates

For BF16 B32 decode, one full hidden activation is `32×5120×2 = 327,680` bytes.

| Candidate | Residual before → after | Next consumer | Collective payload/placement | Persistent plan | Decision gate |
|---|---|---|---|---|---|
| local matmul + all-reduce | replicated → replicated | local RMSNorm | hidden partials; composite reduction/gather | reuse output buffers where API permits | correctness control; accept only if fastest |
| reduce-scatter + delayed gather | replicated → 1280-wide fractured | distributed RMSNorm, then next projection | 327,680-byte logical result reduced/scattered; gather delayed | persistent stats/semaphore/output buffers | required coherent measurement |
| fused all-gather + next matmul | fractured → consumer-local output | next QKV or gate/up projection | gather fused at consumer boundary | prepacked column weights and persistent gather buffer | test if supported for exact shapes |
| fused matmul + reduce-scatter/all-reduce | local projection input → fractured/replicated | residual add/norm | collective fused with O/down where supported | persistent CCL buffers | test exact op contract; first API error is not rejection |
| immediate reduce-scatter → all-gather | replicated → fractured → replicated | unchanged norm | recreates most communication | none beyond control | diagnostic only, not evidence against fractured family |

The coherent fractured candidate (`reduce_scatter → distributed RMSNorm stats → delayed hidden gather`) measured 0.293329 ms versus 0.304505 ms for the isolated replicated family, with rank PCC 0.999822–0.999828. Integrated into the real full layer, however, it measured 0.855622 ms versus 0.773775 ms for replicated boundaries (10.6% slower). The final path therefore keeps replicated boundaries and ring all-reduce after row projections.

Prefill uses the same semantic placement, but collective bytes scale with logical sequence length. Non-aligned sequence lengths are padded/chunked internally and sliced at decoder/test boundaries; page size, tile size, or CCL alignment never becomes a public divisibility requirement.

## Rejected alternatives before coding

- Data parallelism: improves throughput but does not apply four-chip memory bandwidth to one decoder instance and cannot fit the intended full-model layer stack as efficiently.
- TP=2 plus DP=2: halves per-request bandwidth and leaves each replica with roughly half rather than quarter weight/cache memory; unsuitable for the single-user latency and full-stack baseline goal.
- 2-D 2×2 tensor parallelism: hidden and intermediate shapes divide, but this four-chip PCIe-attached ring has no Galaxy-style orthogonal high-link axes. It adds two-axis reductions/replication without a capacity need; retain only as a future option if 1-D profiling exposes a ring bottleneck larger than saved DRAM time.
- Sequence parallel as the primary prefill strategy: useful only for long-prefill scaling and does not solve decode weight bandwidth. It may complement TP later but is not the layer-stack contract.
- Replicating KV heads across chips: unnecessary at TP=4 because four KV heads map exactly one per device; it quadruples per-device cache memory and SDPA work.
- Dense all-expert execution / expert parallelism: model has no MoE experts.
