# Pre-implementation mesh plan

Target: four Blackhole p300c chips, ClusterType.P300_X2, 1x4 TP.
`topology_initial.log` proves all four open/close with FABRIC_1D. Start with
Linear CCL; compare Ring with matching fabric configuration. Each device has
11x10 workers, eight DRAM banks, approximately 32 GiB DRAM. Baseline is
`OptimizedDecoder`, checkpoint 1477a0eecf4f6654cebb4010aa89b852d2ca75fb.

## Tensor ownership (logical local shapes)

| Tensor | Global | Local TP4 | Mapping / padding |
| --- | --- | --- | --- |
| Residual | B,S,5120 | B,S,1280 or B,S,5120 | Hidden sharded candidate / replicated control |
| Full Q/gate | 24 heads x 256 | 6 heads x 256 each | Pack Q,K,V,gate separately within each shard |
| Full K/V | 4 heads x 256 | 1 head x 256 each | Local head ownership |
| Packed full projection | 5120,14336 | 5120,3584 | Output shard; no semantic padding |
| Full output projection | 6144,5120 | 1536,5120 | Input shard, sum partial results |
| Linear Q/K | 16 heads x 128 | 4 heads x 128 each | Independent head groups |
| Linear V/Z | 48 heads x 128 | 12 heads x 128 each | Three value heads per key head |
| Linear B/A gates | 48 each | 12 each, padded to 32 | Zero weights in padding; slice to 12 |
| Packed linear projection | 5120,16512 | 5120,4160 | Q512,K512,V1536,Z1536,B32,A32 |
| Linear output projection | 6144,5120 | 1536,5120 | Input shard, sum partial results |
| Conv taps/history | 10240 / B,3,10240 | 2560 / B,3,2560 | Split Q,K,V independently |
| Recurrence | B,48,128,128 | B,12,128,128 | FP32, independent value heads |
| MLP gate/up | 5120,17408 | 5120,4352 | Output shard |
| MLP down | 17408,5120 | 4352,5120 | Input shard, sum partial results |
| Full KV | P,4,32,256 each | P,1,32,256 each | BFP8, identical replicated page tables |
| Positions/RoPE | B / B,S,64 | Replicated | No host reads during forward |
| Layer norms | 5120 | 1280 or replicated 5120 | Distributed statistics for sharded residual |
| Q/K norm, gated linear norm | 256 / 128 | Replicated | Norm axis stays entirely local |

Decode projections retain BFP4/LoFi with BF16 activations. Initial local
activation storage: K5120 -> 80 cores x64 (block2); K1536 ->24 cores x64
(block2); K4352 ->8 cores x544 (block17). Three readers per DRAM bank;
bank shards are [K,ceil(N/(32*8*3))*96]. These are initial geometry
candidates, not measured winners. Prefill preserves optimized minimal-matmul
and bounded 2048-token chunks, including arbitrary valid tails and prefixes.

## Collective/residual families to compare

For M physical token rows, BF16 full hidden payload F=M*5120*2 bytes.
Ideal per-rank ring-volume bounds: RS=3F/4, AG=3F/4, AR=3F/2.
For decode B1, tile storage makes M=32, F=327680 bytes.

| Family | Before/after residual | Next consumer | Communication per boundary | Buffers |
| --- | --- | --- | --- | --- |
| Local projection + AR | Replicated / replicated | Local norm then column projection | 1.5F | Async CCL semaphores; persistent buffers candidate |
| RS + delayed AG | Sharded / sharded | Residual add, distributed norm, then gather normalized input | 1.5F plus small stats | Shared CCL ownership; keep residual sharded |
| Fused AG-matmul | Sharded / local projected | Local attention or SwiGLU | AG 0.75F plus prior RS | Adapt rank/packing and fused-op config |
| Fused matmul-RS | Local partial / sharded | Add and distributed norm | RS 0.75F | Adapt fused output to consumer, no immediate gather |
| Gather local heads then output matmul | Replicated / replicated | Local norm | Gather attention width vs RS hidden | Requires different output weight placement; measure if supported |

BFP8 CCL halves tile payload versus BF16 except exponent overhead; cross this
with the residual candidates. No bias is repeated in reductions: projections
have no bias. Setup owns weights, constants, persistent state and semaphores.
Trace callers restore state after capture and refresh device inputs/positions.

## Context and alternatives

Retain 262144 logical tokens. Local full-attention K+V cost is 544 bytes/token,
142606336 bytes/layer, and 2281701376 bytes across 16 full-attention layers.
Linear recurrence across 48 layers is 37748736 bytes/user plus 737280 bytes
of convolution history. Full weight accounting must include both prefill and
decode copies, bank padding, embeddings/head reserves, trace/activation reserve.

Dense model: no experts, router, or MoE replication. TP2 leaves half the chips
unused for single-user latency. DP4 replicates weights and cannot reduce
single-user latency. 2x2 TP adds independent reduction axes without improving
head divisibility at four chips. TP4 is the selected ownership scheme; residual
and CCL families remain subject to whole-layer measurements.

Common Attention1D does not implement this model's partial RoPE, per-head
Q/K RMSNorm plus output gate, or gated delta recurrence. MLP1D supplies the
column/row projection pattern but not the selected baseline's exact public
logical-tail/packed-row contract. Reuse optimized native attention/recurrence
methods and the common TT_CCL manager; implement model-local weight packing
and boundary collectives. Distributed norms follow RMSNorm1D's primitive pair.

## Adapted fused families

AG-minimal-matmul repacks WO/down by output: local weights6144x1280 and
17408x1280, gathering local1536/4352 inputs. Output stays1280 through residual
and distributed norm. MM-strided-RS instead keeps1536x5120/4352x5120 and
produces1280 residual. Both start on10x8 cores, M1 decode/M4 prefill, K8,N8,
1x4 subblocks, BFP4/LoFi. AG's in0 sender axis10 requires ten workers/link
with one link (or five workers/link with two links); its returned vector strips
internal gather storage, so element0 is the first matmul output. Initial
validation identified this grouping requirement; adapt it before comparison.

Direct all-reduce uses L1-width-sharded BF16 hidden activations on80 cores,
[32,64] shards, and persistent[1,1,32,20480] scratch with[32,256] shards.
This preserves replicated residual and avoids DRAM CCL intermediate storage.
Its decode0.5259ms passes; compare against the same larger-block/carry policy.

## Final combined-family revision (before selection)

With packed MLP, output block6, and two links, direct L1 all-reduce measures
0.47299ms versus replicated RS/AG default~0.487ms; hidden-sharded packed paths
remain~0.599/0.609ms. Validate direct AR as the decode candidate; prefill keeps
RS/AG. The AR scratch is BF16 `[1,1,32,20480]`,80 L1 cores with `[32,256]`
shards:1,310,720B total,16,384B/core. Allocate it once on a shared TT_CCL
context across the layer stack. Independent copies for64 layers would occupy
1MiB/core before dynamic matmul buffers, so sharing is part of the contract.
The stack test must pass the same collective context and verify alias safety.
Weight/head ownership and DRAM/context planning remain unchanged.

## Batched stack layout refinement

The selected public logical contract remains replicated BF16 `[B,1,5120]`.
B1 preserves compact L1 carry. For B>1, the inherited `_public_rows` must
materialize B*32 TILE rows; B32 therefore occupies10MiB per public tensor.
The B32 stacked control exposed L1 overlap at second-layer packed MLP even
though each layer alone fit. `public_dram_batch=2` places these expanded
public tensors in DRAM before repacking, preserving compact L1 internal
residuals, norms, projections, and all-reduce. It does not alter logical
shapes, values, cache ownership, or supported batch sizes. Threshold2 treats
all expanded public rows consistently rather than relying on small-batch
allocator headroom. See AUTOFIX_stacked_batch_layout.md and
stress_stack_batch32_dram.json for source arithmetic and100-step control.

## TP2 capacity comparison under the selected representation

Doubling TP4's sharded persistent terms for TP2 gives40,417,132,544B/device,
including the same dual projection copies, full-context KV, and18GiB
activation/trace reserve. Repacking TP2's linear B/A gates (24 logical gates
padded to32 per field) reduces this estimate by44,236,800B across48 layers,
yielding40,372,895,744B/device versus measured usable DRAM
34,138,688,512B. Two TP2 data-parallel replicas therefore exceed
capacity under this representation and use only two chips for one request.
This is a comparison of the selected storage/activation contract, not proof
that every conceivable TP2 implementation is impossible. TP4 retains the
advertised context within29,939,351,552B/device. Pipeline
parallelism could distribute layers but would leave a single decoder executing
on one chip, preserving its single-chip latency rather than the measured
TP4 per-layer speedup.
