# Gemma 4 26B A4B multichip mesh plan

Status: selected before implementation on 2026-08-15.

## Target

- Hardware: four Blackhole P300C chips, physically discovered and smoke-tested
  as a `1 x 4` mesh.
- Fabric/topology: `FABRIC_1D_RING`, `ttnn.Topology.Ring`, two links where the
  collective API supports them.
- Baseline: `tt/optimized_decoder.py` at commit `8bc06c55da5`. It preserves
  active top-8 sparse experts, paged KV cache, arbitrary valid logical prompt
  lengths, and traced decode.
- Parallelism: TP=4. Residual activations are replicated at decoder boundaries.
  Every expanding projection is column parallel and every contracting
  projection is row parallel. Row-parallel partials are reduced once, after the
  contraction, so expanded activations never cross the ring.

## Tensor and cache plan

All dimensions below are logical unless a padded dimension is explicitly
shown. BF16 ring payloads are used initially; lower precision is only accepted
after real-weight PCC evidence.

| Tensor / operation | Global shape | Per-device shape | Mapping / padding |
| --- | --- | --- | --- |
| residual, norms, router input/output | `[..., 2816]` | `[..., 2816]` | replicated; local RMSNorm and identical top-8 routing |
| sliding Q | `[2816, 4096]` | `[2816, 1024]` | column parallel; 4 Q heads/device |
| sliding K, V | each `[2816, 2048]` | each `[2816, 512]` | column parallel; 2 KV heads/device |
| sliding KV cache | `[blocks, 8, 64, 256]` | `[blocks, 2, 64, 256]` | local-head paged cache; page-table rows replicated |
| full Q | `[2816, 8192]` | `[2816, 2048]` | column parallel; 4 Q heads/device |
| full K (also V) | `[2816, 1024]` | `[2816, 512]` | each 512-wide KV head is replicated on one chip pair; chips 0-1 own KV0, chips 2-3 own KV1 |
| full KV cache | `[blocks, 2, 128, 512]` | `[blocks, 1, 128, 512]` | local paged cache; duplicated within each Q-group pair |
| attention O | sliding `[4096,2816]`, full `[8192,2816]` | `[1024,2816]`, `[2048,2816]` | row parallel; one hidden-width all-reduce |
| dense gate/up | each `[2816,2112]` | each `[2816,544]` | N padded `2112 -> 2176`, then quartered; slice is unnecessary before down |
| dense down | `[2112,2816]` | `[544,2816]` | K padded `2112 -> 2176`; padded rows zero; one hidden-width all-reduce |
| expert gate/up | each `[128,2816,704]` | each `[128,2816,192]` | expert intermediate padded `704 -> 768`; top-8 sparse execution on every device |
| expert down | `[128,704,2816]` | `[128,192,2816]` | padded K rows zero; score weighting stays local; one hidden-width all-reduce |
| decoder input/output | prefill `[B,1,S,2816]`, decode `[1,1,B,2816]` | same | replicated and stack-compatible; gather exists only in test/output composition |

Padding is setup-only for weights and internal for activations. Public sequence
length, page-table, and current-position contracts remain logical; no prompt
length becomes divisible-by-32/64/128/4 as a public requirement.

## Collective topology candidates

For batch 32 decode, one BF16 hidden activation is `32 * 2816 * 2 = 180224`
bytes before ring protocol overhead.

| Candidate | Residual before / after | Next consumer | Collective bytes per boundary | Persistent buffers | Decision |
| --- | --- | --- | ---: | --- | --- |
| local row matmul + all-reduce | replicated / replicated | local RMSNorm or residual add | one 180224-byte logical tensor | ping-pong RS/AG semaphores and buffers | selected correctness baseline; measures coherent whole layer |
| reduce-scatter then immediate all-gather | replicated / replicated | local RMSNorm | same logical all-reduce, split into primitives | same | implementation mechanism, not a distinct optimization claim |
| reduce-scatter with delayed gather | replicated / hidden-sharded | distributed norm, then column projection | 3/4 of hidden payload until later gather | persistent reduced residual | must be measured through the next norm/projection boundary; not rejected by an immediate gather trial |
| fused all-gather + next matmul | hidden-sharded / projection-sharded | QKV or gate/up | moves hidden once while consuming it | fused-op output buffer | candidate if Blackhole op accepts Gemma shapes and trace contract |
| fused matmul + reduce-scatter | replicated or projection-sharded / hidden-sharded | distributed norm/residual | avoids standalone RS dispatch | fused output + semaphore buffers | candidate if supported for selected dtype/program geometry |
| fully residual-sharded layer | hidden-sharded / hidden-sharded | distributed norms and residual adds | smallest steady-state payload | persistent layer-stack residual | higher-risk candidate; accept only with end-to-end layer latency and PCC |

## MoE strategy and rejected alternatives

- Selected: replicated router/top-k followed by TP=4 active-expert sparse
  gate/up/down. The top-8 mask is identical on all chips. Each chip stores one
  quarter of every expert's intermediate columns, performs only selected-expert
  work, and contributes a row-parallel down partial to one hidden all-reduce.
- Rejected as a final design: dense all-128-expert execution. It violates the
  baseline execution contract and multiplies decode work by 16 relative to
  top-8 routing.
- Rejected: expert replication on every chip. The 30-layer expert tensors are
  the dominant model weight footprint and replication gives no single-user
  bandwidth scaling.
- Rejected: EP=4 with 32 experts/chip as the first path. Top-8 tokens would need
  dispatch/combine collectives around each sparse expert call; at batch 1 the
  communication and imbalance risk dominate, while TP fractures every selected
  expert's DRAM traffic uniformly.

## Memory and context contract

The model has 25 sliding and 5 full-attention layers. Real-shape layer stats
contain about 24.5 billion layer-weight elements before embeddings/head. The
selected optimized policy stores dominant experts/dense/full-attention weights
in BFP8; TP=4 therefore targets roughly 6.2 GiB/device for decoder-layer
weights before allocator padding and replicated norm/router metadata.

At the advertised 262144-token context and batch 1, BFP8 full-attention KV is
`262144 * 1 local head * 512 * 2(K,V) * 1 byte = 256 MiB` per full layer and
`1.25 GiB/device` over five layers. BF16 doubles that number. Sliding cache is
bounded to 1024 tokens and is much smaller. Batch 32 is a serving batch contract
with aggregate paged allocation; it is not a promise that 32 independent users
each simultaneously occupy 262144 tokens. No advertised-context reduction is
planned. Runtime capacity evidence must record weight, KV, trace, activation,
and allocator reserves before stage completion.

The resolved BF16 full-stack cache envelope is `2.5 GiB/device` for five full
layers plus `25 * 1024 * 2048 bytes = 0.048828125 GiB/device` for bounded
sliding layers, totaling `2.548828125 GiB/device` per maximum-context user.
The per-layer advertised-context sliding probe intentionally allocates a full
`262144`-token stress cache; it is not the stack allocator. TP4 rollover at
logical length 1025 verifies that slot 0 changes while slots 1..1023 retain PCC
0.9999.

Each P300C exposes 32 GiB DRAM. The conservative stack envelope is 6.8 GiB for
TP4 decoder weights, 2.548828125 GiB for the maximum-context BF16 KV caches,
0.0625 GiB for decode trace buffers, 6 GiB for live activations/workspaces, and
4 GiB for programs plus allocator fragmentation: 19.411328125 GiB/device,
leaving 12.588671875 GiB/device. The activation allowance exceeds the sum of a
maximum-prefill BF16 residual (`262144 * 2816 * 2 = 1.375 GiB`), local full-Q
(`1.0 GiB`), local K and V (`0.25 GiB` each), and temporary/output buffers.
The additional 4 GiB reserve is deliberately not counted as usable model
capacity. The real-weight 262143-token probes exercised the dominant live
buffers for both layer kinds; this calculation extends that measured layer
peak to the stack's persistent weights and bounded caches.

The batch-32 profiler also bounded the inherited sparse program choice. TP4
expert-down K is six tiles, so legal divisors are 1, 2, 3, and 6 tiles; six is
selected because it consumes local K in one block while the smaller choices
add K-loop iterations. Dense all-expert execution is invalid by contract and
16x the active-expert work. The signposted profile attributes about 53% to
sparse matmul and 26% to transpose/unary layout work, versus about 1.35% to
CCL. Thus batch-32 is sparse/layout dominated, while batch-1 scaling is more
sensitive to the three fixed hidden-width reductions. Further sparse-layout
fusion is a follow-on kernel optimization, not evidence for a different mesh
or dense-expert strategy.

`$autofix` audited the hidden-sharded alternative. A stack-compatible chain is
generic reduce-scatter, local residual/norm, then all-gather before the next
full-K projection. For physical decode `[1,1,32,2816]` BF16, a ring all-reduce
moves approximately 270336 bytes/device; reduce-scatter moves 135168 and the
mandatory later all-gather moves 135168, the same total, plus distributed RMS
statistics. QKV, dense gate/up, and expert gate/up all retain global K=2816, so
no next consumer can use only a 704-wide residual shard. Fused matmul+RS is
also disabled on Blackhole for nondeterministic race #46181. Thus this is an
exact no-lower-byte blocker for 1D TP; changing it requires a different
SUMMA-like weight decomposition rather than a decoder-local collective swap.

## Required selection evidence

The final README/work log must contain same-regime TP topology measurements,
including the replicated residual baseline and at least one stack-compatible
lower-movement residual candidate or an exact op-contract blocker established
with `$autofix`. It must also contain multi-chip versus single-chip PCC for both
layer kinds, local paged-cache probes, non-aligned lengths, trace replay,
determinism/stress, watcher, warmed latency, and `tt-perf-report` tables/CSV.
