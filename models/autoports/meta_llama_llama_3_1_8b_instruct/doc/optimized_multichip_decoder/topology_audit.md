# Operation-topology audit

This audit was written before optimized-stage knob tuning.  It combines the
live source with the inherited current-source Tracy capture under
`../multichip_decoder/tracy/`.  Layer 16 is representative because all 32
Llama layers have the same dense attention/MLP kind and tensor geometry.

## Current measured operation topology

| Phase / boundary | Current material sequence and layout | Repeated input / movement | Candidate fused or lower-movement form | Dtype / fidelity constraints | Initial action |
| --- | --- | --- | --- | --- | --- |
| decode input + norm | public replicated BF16 `[1,batch,1,4096]`; reshape; one I2S to 16-core width-sharded L1; sharded RMSNorm | public-boundary I2S only; 7 us norm | retain the inter-layer 16-core replicated-L1 contract; compare advisor layout seed, not a boundary gather | residual/norm BF16 | retain as topology baseline; advisor candidate required this pass |
| QKV projection | one packed `32x4096x1536` local matmul; BFP4 DRAM-sharded weight; LoFi; S2I; dedicated decode head creation | Q/K/V already share one input and one packed matmul | dedicated packed QKV is already the graph rewrite; advisor 1-D seed and DRAM geometry controls | head op consumes BF16; local Q8/K2/V2 ownership | keep packed; compare this-pass advisor seed with strongest DRAM path |
| RoPE/cache/attention | dedicated RoPE Q/K; two paged updates; explicit SDPA decode; dedicated concat heads | head-layout conversions and cache ops are required API boundaries | keep composite ops; test explicit SDPA/cache policies only if profiler makes them material | BF16 query/update; BFP8 or BF16 caller-owned cache | retained composite graph; no primitive attention rewrite exists |
| O + attention residual | local `32x1024x4096` row-parallel BFP4/LoFi DRAM matmul; reshard/S2I; async all-reduce decomposed as RS+AG; I2S; residual add | one material RS (about 24 us), AG (about 14 us), and three movement rows | true minimal all-reduce directly from width-sharded L1 with persistent 4x buffer; fused MM+RS; fractured RS carry-forward; fused AG+matmul consumer | compare BF16 and BFP8 payload in the compatible topology; preserve BF16 residual output | implement/measure true persistent minimal all-reduce first; retain inherited fused/fractured evidence as controls and retry material contracts if source permits |
| post-attention norm | 16-core width-sharded L1 RMSNorm | no cross-device movement | distributed norm only belongs to fractured family | BF16 norm/statistics | keep replicated norm unless fractured family wins end-to-end |
| gate/up | two same-input local `32x4096x3584` BFP4/LoFi DRAM matmuls; fused SiLU on binary multiply | repeated same-input matmuls; separate family uses two launches | packed gate/up then on-device split versus best separate fused-elementwise family | BFP4/LoFi is real-weight accepted; count split/layout and prefill cost | inherited TP4 adapted packed candidate lost prefill; re-rank only if the selected CCL family changes whole-layer ordering materially |
| down + layer output | local `32x3584x4096` BFP4/LoFi DRAM matmul; reshard/S2I; second composite RS+AG; I2S; residual add | same material CCL/movement family as O | persistent minimal all-reduce; fused MM+RS; carry fractured residual directly into next layer | final public output must be replicated BF16 width-sharded L1 unless a fully compatible fractured stack wins | implement minimal all-reduce; final output must feed next layer with no added collective/reshard |
| prefill projections | packed QKV, separate gate/up, O/down with explicit 2-D matmul configs; DRAM-interleaved activations | two row-parallel BF16 composite all-reduces | true minimal op is L1-width-sharded-only and is not directly applicable to variable-M DRAM prefill; packed gate/up remains a whole-prefill comparison | BFP4/LoFi weights; BF16 residual/CCL | retain current async composite for prefill unless a shape-owned candidate wins |
| prefill norms | two DRAM-interleaved RMSNorms, profiler reports one-core 50 us rows and 23.51% norm class | no collective, but material one-core compute | phase-specific sharded multi-core norm plus boundary conversions | BF16 norm; public logical length remains unaligned | mandatory measured candidate because the class exceeds 15% in the inherited report |

## Coherent multidevice family matrix

| Family | Residual before -> after | Collective / bytes per layer | Persistent state | Next-layer compatibility | Evidence plan |
| --- | --- | --- | --- | --- | --- |
| current replicated composite | L1 width-sharded partial -> replicated L1 width-sharded BF16 | two RS+AG all-reduces; 786,432 BF16 fabric bytes/layer in prior ledger; conversions around each composite | shared cyclic RS/AG/barrier semaphores; no persistent data buffers | direct, no boundary collective | fresh unchanged baseline plus inherited profiler |
| persistent minimal all-reduce | same L1 contract -> same replicated L1 contract | one dedicated all-reduce row per boundary; same algebraic minimum bytes, fewer launches/conversions | persistent width-sharded intermediate buffer sized `ring_size * output_shard`; dedicated cyclic semaphore | direct, no boundary collective | implement exact TP4 shapes; sweep standard/llama worker placement, NOC policy, links, and BF16/BFP8 payload |
| fractured carry-forward | replicated input -> reduce-scattered/fractured residual; distributed norm; delayed or fused gather only at consumer | inherited ledger 798,720 bytes/layer including stats | RS/AG semaphores/buffers | must be consumed by adapted next norm/projection, with harness-only gather outside measured layer | inherited full shape-compatible probe was 1.066% slower; fused consumer has a source-proven TP4 transfer-ledger blocker |
| fused local MM + RS | replicated input -> fractured output | avoids the separate MM/RS boundary | fused API requires persistent intermediate/output buffers | needs adapted fractured residual/norm or later gather | inherited exact O/down BFP4/LoFi probes passed but were 26.63% / 14.56% slower; inspect current API before final disposition |
| gathered input + local-output MM | fractured input -> local output, then gather only if replicated boundary chosen | fused AG+MM plus optional gather | API-specific | only compelling with a carried fractured input | prior generic API hardcodes TP8 transfer accounting on this TP4 ring; preserve minimal repro/source evidence |
| packed gate/up | unchanged residual | no CCL change; one larger projection plus split/layout | packed weight | compatible | adapt packed output layout/split/SwiGLU/down under the final advisor/BF16-minimal family and compare whole-layer latency |

## Graph-rewrite disposition

- Dedicated operations already replace primitive RMSNorm, QKV head creation,
  RoPE, SDPA/FlashDecode, concat-heads, and SwiGLU activation sequences.
- Q/K/V are already one packed projection; there are not three repeated
  projection matmuls to fuse.
- Gate/up is the only legal shared-LHS peer merge.  Both packed and separate
  implementations exist; prior compatible TP4 measurement selected separate.
- The remaining highest-value spelled-out subgraph is not mathematical
  attention but the composite RS+AG all-reduce plus two layout conversions.
  The dedicated persistent minimal all-reduce is therefore the first rewrite.
- No host `to_torch`/`from_torch`, host reduction, or functional decoder
  fallback occurs in either measured forward path.

## Cumulative starting contract

| Contract item | Strongest correct starting value |
| --- | --- |
| projection topology | packed QKV; separate gate/up; row-parallel O/down |
| attention weights / math | BFP4 / LoFi |
| MLP weights / math | BFP4 / LoFi for gate, up, and down |
| cache | KV-head sharded, BFP8 or BF16 caller-owned; final benchmark uses BFP8 |
| logical batch / padded rows | batch 1 / 32 tile rows; support preserved through batch 32 |
| residual/norm | replicated across TP ranks, BF16, 16-core width-sharded L1 in decode |
| collectives | BF16, ring, two links, composite async RS+AG, shared persistent semaphores |
| matmul geometry | DRAM-sharded; QKV/O/gate/up/down full input-shard K blocks 8/2/8/8/7 |
| final unchanged latency | 0.746841 ms prefill; 0.320079 ms traced decode |
| final unchanged PCC | 0.9999998070672766 against single-chip optimized output |

## Final action and evidence

| Audited item | Action | Final disposition |
| --- | --- | --- |
| repeated Q/K/V input matmuls | verified packed QKV in source and profiler | one material QKV matmul retained |
| repeated gate/up input matmuls | adapted packed weight, 56-core packed output, L1 split/SwiGLU, and down under exact QKV32/BF16-minimal topology | separate gate/up wins: 0.246689 ms versus 0.248601 ms packed, three 100/1000 processes each |
| composite decode collective | replaced RS+AG plus surrounding layout conversions with persistent minimal all-reduce | kept; long decode 0.266609 versus 0.320035 ms before advisor compute |
| minimal CCL placement | default workers, special Llama workers, NOC1, one/two links, BF16/BFP8, one/two buffers | default workers, normal NOC, two links, BF16, one buffer kept |
| fused matmul+RS | inherited exact adapted O/down tests passed but were 26.63%/14.56% slower | rejected; new minimal all-reduce makes the separate path still stronger |
| fused AG+matmul | inherited source/repro proves the generic transfer ledger is TP8-hardcoded on this TP4 ring | rejected as inapplicable after shape/layout adaptation; minimal direct all-reduce is implemented instead |
| residual layout | adapted and retried 64-core carry chain through buffer, norm, gate/up, down, add, and layer output | correct but 0.251809 ms versus stronger 16-core family; final inter-layer contract is replicated-across-ranks BF16, 16-core width-sharded L1 |
| activation sharding | ran fresh advisor and applied 1-D L1 activation layouts; also tried exact sharded residual/norm chain | fresh 1-D projection family kept; exact residual chain rejected by full-family timing |
| DRAM-sharded versus 1-D matmul | retained phase-specific DRAM-sharded prefill weights and lazy interleaved decode weights | avoids cross-phase compromise; decode advisor family wins |
| advisor per-role advice | independently swept QKV/O/gate+up/down cores, grids, per-core N, output subblocks, matching output memory, power-of-two and non-power K blocks, then reran material combinations cumulatively under QKV32 | QKV 24 cores/block32/1x2 and O 64 cores/block32/1x2 promoted; gate/up/down retain 56/56/64 active cores, block8, 1x2; exact-default six-run median 0.246686 ms |
| prefill one-core norms | internally padded width-sharded L1 norm with both boundary conversions | correct but 39.3% slower; DRAM norm retained |
| cache/SDPA | final report shows BFP8 paged update and explicit SDPA decode, each about 5% or less | retained; contiguous, paged, boundary, trace, and watcher gates pass |
| runtime fallback | source audit plus `throw_exception_on_fallback=true` hardware suite | clean |

The replay-only final decode profiler contains two
`AllReduceAsyncDeviceOperation` rows per execution and no decode
reduce-scatter/all-gather rows.  Reshard remains only
inside the local layer where projection geometry changes; stacked correctness
passes with no inter-layer collective or gather.
