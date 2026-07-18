# Full-mesh decoder plan

This plan was fixed before creating `tt/multichip_decoder.py`. The source of
truth is the completed single-chip `OptimizedDecoder`, the compiler provenance
in `doc/functional_decoder/multichip_provenance.json`, and the hardware probes
run on 2026-07-18.

## Target hardware and strategy

- `tt-smi -ls --local` exposes four Blackhole P300c devices, IDs 0 through 3.
- `ttnn.get_num_devices()` reports four and `MeshShape(1, 4)` opens and closes.
- The physical and logical adjacency histograms are `{2: 4}`, so all four
  devices form the ring expected by `FABRIC_1D_RING` / `Topology.Ring`.
- The allocator reports eight DRAM banks of 4,272,341,376 bytes, or
  34,178,731,008 bytes per device.
- The compiler prior is also a `1x4` mesh with TP on mesh dimension 1. The
  selected starting strategy is therefore dense 1-D tensor parallelism with
  TP=4, not data parallelism, sequence parallelism, or a partial device set.
- Llama 3.1 8B is dense. There are no router or expert tensors, so MoE and
  expert-parallel strategies are not applicable.

Weights keep the optimized baseline's BFP4/LoFi projection policy. Norms,
residual activations, RoPE tables, positions, and the CCL payload start in BF16.
BF16 and BFP8 caller-owned cache storage remain supported. The first runnable
path uses rank-local DRAM-sharded projection weights and two persistent,
asynchronous sum reductions per layer. A fractured-residual/fused-AGMM family
is an explicit TP4 candidate below and must be measured through its consuming
norm and projection before the final residual contract is selected.

## Tensor ownership and per-device shapes

The model has hidden size 4096, 32 query heads, eight KV heads, head dimension
128, intermediate size 14,336, and 32 identical dense decoder layers.

| Tensor or activation | Global logical shape | TP=4 local shape | Placement / padding |
| --- | --- | --- | --- |
| input/post norm gamma | `[4096]` | `[4096]` | replicated BF16; no padding |
| Q weight | `[4096,4096]` HF | `[4096,1024]` runtime-transposed | output-column shard; Q heads 8/rank |
| K weight | `[1024,4096]` HF | `[4096,256]` runtime-transposed | output-column shard; KV heads 2/rank |
| V weight | `[1024,4096]` HF | `[4096,256]` runtime-transposed | output-column shard; KV heads 2/rank |
| packed QKV weight/output | `[4096,6144]` / `[...,6144]` | `[4096,1536]` / `[...,1536]` | rank-local Q8/K2/V2 pack; no padding |
| query heads | `[batch,32,seq,128]` | `[batch,8,seq,128]` | head-local |
| key/value heads | `[batch,8,seq,128]` | `[batch,2,seq,128]` | head-local |
| attention context | `[...,4096]` | `[...,1024]` | eight local query heads |
| O weight | `[4096,4096]` HF | `[1024,4096]` runtime-transposed | input-row shard; full-width partial |
| gate weight/output | `[14336,4096]` HF / `[...,14336]` | `[4096,3584]` / `[...,3584]` | output-column shard |
| up weight/output | `[14336,4096]` HF / `[...,14336]` | `[4096,3584]` / `[...,3584]` | output-column shard |
| down weight | `[4096,14336]` HF | `[3584,4096]` runtime-transposed | input-row shard; full-width partial |
| residual boundary, initial | `[1,batch,seq,4096]` | same logical replica | replicated across mesh; decode is 32-core L1 width-sharded within each chip |
| residual boundary, candidate | `[1,1,batch,4096]` | `[1,1,batch,1024]` | mesh-shard dimension 3; only selected if the full consuming boundary wins |
| contiguous K/V cache | `[batch,8,max_context,128]` | `[batch,2,max_context,128]` | mesh-shard KV-head dimension 1 |
| paged K/V cache | `[blocks,8,64,128]` | `[blocks,2,64,128]` | mesh-shard KV-head dimension 1; final page internally padded |
| page table | `[batch,ceil(max_context/64)]` | same | replicated INT32; arbitrary physical block order |
| current positions | `[batch]` | same | replicated INT32 |
| RoPE cosine/sine | `[1,1,max_context,128]` | same | replicated BF16 |

All dimensions divide TP=4 and tile constraints. No load-time feature padding is
needed. Logical sequence lengths do not need to divide 32 or 64: tiling and
last-page padding stay internal, and tests will include sequences 7 and 31 plus
decode positions 63, 64, and 65.

## Initial local program and shard specifications

Decode pads the batch row to 32. Each P300c has an 11x10 worker grid and eight
DRAM banks.

| Role | Local K x N | DRAM weight shard (8 banks) | Initial storage/N partitions | Input L1 shard | Expected `in0_block_w` |
| --- | ---: | ---: | ---: | ---: | ---: |
| packed QKV | `4096x1536` | `[4096,192]` per bank | 16 | `[32,256]` width shards | 8 |
| O | `1024x4096` | `[1024,512]` per bank | 8 first candidate | `[32,128]` width shards | 4 |
| gate | `4096x3584` | `[4096,448]` per bank | 16 | `[32,256]` width shards | 8 |
| up | `4096x3584` | `[4096,448]` per bank | 16 | `[32,256]` width shards | 8 |
| down | `3584x4096` | `[3584,512]` per bank | 16 | `[32,224]` width shards | 7 |
| residual/norm | width 4096 | n/a | 32 | `[32,128]` width shards | n/a |

The initial 16-way QKV/gate/up/down partition divides both tiled K and tiled N;
the dense 32-way optimized configs do not divide the TP4 local N values.  The
legal eight-way projection family and O's legal 8/16/32 choices will be swept. Local
interleaved-weight 1-D programs, packed gate/up, one/two CCL links, and BFP8 CCL
payloads remain measured candidates under the same BFP4/LoFi projection policy.

## Collective and residual topology table

For batch 32 BF16 decode, one full hidden partial is 262,144 bytes/device. Ring
reduce-scatter or all-gather peer traffic is `3/4` of that, 196,608 bytes.

| Candidate | Residual before / after | Next consumer | Per-device communication per layer | Persistent state | Pre-code disposition |
| --- | --- | --- | ---: | --- | --- |
| local row matmul + async all-reduce | replicated / replicated | local RMSNorm | two `(RS+AG)` boundaries = 786,432 B | shared `TT_CCL` barrier/RS/AG semaphores; trace-owned outputs | compiler-prior baseline |
| reduce-scatter + immediate all-gather | replicated / replicated | local RMSNorm | same 786,432 B plus layout work | same | not a meaningful optimization by itself |
| reduce-scatter + local add + distributed norm + delayed all-gather + matmul | fractured / local projection | QKV or gate/up | 786,432 B hidden traffic + 12,288 B stats = 798,720 B | persistent RS, stats AG, and hidden AG buffers | required shape-faithful candidate |
| fused all-gather-matmul after distributed norm | fractured / local projection | QKV; gate gather reused by up | same 798,720 B, but AG overlaps matmul | interleaved BFP4 local weights and four-shard persistent AG output | preferred fractured candidate if TP4 ring/API supports it |
| fused matmul + reduce-scatter | replicated input / fractured output | distributed norm | same ring bytes; persistent fused output/RS buffers | generic API adapted to exact BFP4 O and down shapes; minimal-strided API architecture-audited | measure both row-parallel projections before disposition |
| fully fractured residual stack | fractured / fractured | next layer distributed norm | avoids per-layer replicated boundary; same hidden CCL phases, plus stats | all buffers shared/owned at stack scope | only valid final form for a winning fractured family |

The fractured family is not rejected by the archived TP2 result: TP2 resolved to
a physical line and could not run fused AGMM, while the full TP4 mesh is a real
ring. The final default must measure the TP4 family through add, distributed
RMSNorm, gather/fusion, and the next BFP4 projection. A test-only gather at the
comparison boundary is allowed; an in-layer gather that merely restores the
old contract is not.

## Context and full-stack memory plan

The advertised context remains 131,072 tokens. With 2 local KV heads/device:

- BF16 K/V cache/device: `32*2*131072*128*2(K,V)*2` = 4,294,967,296 bytes.
- BFP8 K/V cache/device: the same 2,147,483,648 values at 1,088 bytes per
  1,024-value tile = 2,281,701,376 bytes.
- BFP4 projection weights/device: `32*218,103,808/4` values at 576 bytes per
  1,024-value tile = 981,467,136 bytes.
- Conservative non-decoder allowance: 4,294,967,296 bytes/device.
- Reserved trace, activations, CCL buffers, and allocator headroom:
  4,294,967,296 bytes/device.
- Conservative BF16-cache total: 13,866,369,024 bytes/device, 40.6% of the
  34,178,731,008-byte allocator.

This leaves more than 20 GB/device free, so there is no hard physical reason to
reduce the Hugging Face context. A 2,048-block pool at 64 tokens/block covers
one 131,072-token request or can be shared across a batch.

## Rejected alternatives before measurement

- TP=2 or a `TT_VISIBLE_DEVICES` subset: rejects half the detected hardware and
  contradicts the 1x4 compiler provenance.
- Data parallelism: improves throughput of independent requests, not the
  requested single decoder latency or per-device full-stack memory.
- Sequence parallelism for decode: decode sequence length is one.
- 2-D TP: the target is a four-device 1-D ring, not Galaxy; all model tensor
  dimensions divide the 1-D TP degree exactly.
- Expert parallelism: the model has no MoE layers.
- Dense all-expert execution: not applicable.
- Fully column-sharded O/down: it introduces gathers of attention context and
  the 3,584-wide local MLP intermediate before the narrowing projections; the
  compiler prior and standard Llama topology use row sharding instead.

Performance-sensitive candidates listed above are not rejected until their
TP4 correctness and warmed measurements exist.

## Final measured selection

The final path keeps the replicated residual family.  The required full
boundary probe measured the fractured sequence (reduce-scatter, local add,
distributed RMSNorm, standalone all-gather, BFP4 QKV) at 0.085773 ms versus
0.084869 ms for the selected async all-reduce boundary, or 1.066% slower.  It
also moves 798,720 rather than 786,432 bytes/layer.  The generic fused
all-gather-matmul factory cannot safely recover that difference on TP4 because
its two-direction transfer ledger is hardcoded for eight gathered slices; the
source-level result is recorded in `AUTOTRIAGE.md` and `AUTOFIX.md`.

The separate fused-MM+RS audit covered both exact TP4-local row projections.
The generic 2-D multicast fusion was correct, but O regressed from 0.052418 to
0.066379 ms (26.6342%) and down from 0.098161 to 0.112457 ms (14.5637%), each
at minimum rank PCC 0.999999940395. The minimal-strided fused API is explicitly
gated off on Blackhole by repository issue `#46181` because of a known
producer/consumer race. `fused_mm_rs_audit.md` records the API-by-API evidence.

The final local geometry is a 16-way storage/N partition for packed QKV, O,
separate gate/up, and down, plus 16 actual residual/RMSNorm cores.  The DRAM
factory uses eight weight-owning bank workers inside an 80-core bounding launch;
`profiler_geometry_audit.md` derives that mapping.  In the initial-program table above this
changes O from eight to 16 partitions and residual/norm from 32 to 16 cores; their
decode input shards are respectively `[32,64]` for local O K=1024 and
`[32,256]` for the full residual.  The complete sweep selected BFP4/LoFi
projection weights, BF16 CCL payloads, two ring links, and separate gate/up.
It rejected O32, one link, and BFP8 CCL for decode regression, and rejected
packed gate/up for prefill regression.  The full eight-way storage family also
regressed decode from 0.320058 to 0.331311 ms.  The material interleaved BFP4/
LoFi 1-D family was adapted to every TP4-local projection and regressed decode
to 0.405666 ms.  `candidate_results.csv` contains every measured value and log
path.

The final decode `in0_block_w` values are QKV 8, O 2, gate/up 8, and down 7
tiles.  Each equals the full K-tile width of one 16-core input shard
(`4096/16/32=8`, `1024/16/32=2`, and `3584/16/32=7`), so no larger value is a
legal divisor of that shard.  These are not inherited small-block guesses.

The public boundary remains unaligned and logical.  Hardware tests cover
prefill lengths 7 and 31, contiguous and 64-token paged caches, arbitrary
physical pages, positions 63/64/65, and a direct second-layer prefill/decode
consumer.  The final context and memory calculations above remain unchanged.
