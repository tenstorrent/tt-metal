# Llama 3.1 70B multichip decoder mesh plan

Date: 2026-07-17

This plan was frozen before writing the final multichip implementation.  The
single-chip behavioral and numerical baseline is `tt/optimized_decoder.py` at
commit `35ccb90250c`.

## Final measured selection (appended after implementation)

The compiler-matching `2x2` implementation was completed first. Its packed
variant measured 1.410 ms prefill and 1.859 ms decode against 3.187/1.853 ms:
2.26x prefill but only 0.997x decode. `MultiChipDecoder` therefore selects the
planned flat TP4 candidate on the same four ASICs, presented to TTNN as a
logical `1x4` ring. The physical system remains the four-chip cycle found by
topology discovery. A flat all-reduce over logical `2x2` was rejected because
row-major ranks 1 and 2 are not direct line neighbors; the failed candidates
are retained in `flat_tp4_{linear,ring}_correctness.xml`.

Selected logical layout:

| Tensor | Mapping | Per-device logical shape |
| --- | --- | --- |
| residual, norm weights | replicated TP4 | `[1,B,S,8192]`, `[8192]` |
| packed QKV weight | output-column TP4 | `[8192,2560]` |
| Q/K/V activations | head TP4 | `[B,16/2/2,S,128]` |
| output weight | input-row TP4 | `[2048,8192]` |
| packed gate/up weight | output-column TP4 | `[8192,14336]` |
| down weight | input-row TP4 | `[7168,8192]` |
| paged K/V cache | KV-head TP4 | `[blocks,2,64,128]` |
| page table / positions / RoPE | replicated TP4 | unchanged |

Only output and down partials all-reduce. Decode decomposes each into
persistent L1 reduce-scatter plus all-gather with two links, ring topology,
10 chunks/sync, two workers/link, and two buffers/channel. Packed QKV and
gate/up need no collective, and replicated norms need no statistics gather.
The two BF16 all-reduce inputs total 1,048,576 logical bytes/layer at batch 1.
Prefill keeps the composite operation because its sequence/chunk sizes vary.

Selected physical decode shards and geometry:

| Role | Padded input | L1 width shard | Program `(in0_block_w, per_core_N)` |
| --- | --- | --- | --- |
| QKV | `[1,1,32,8192]` | 16 cores x 512 | `(16,5)` |
| output | `[1,1,32,2048]` | 8 cores x 256 | `(8,8)` |
| gate/up | `[1,1,32,8192]` | 32 cores x 256 | `(8,14)` |
| down | `[1,1,32,7168]` | 32 cores x 224 | `(7,8)` |

O2 and G1 were selected from ten isolated real-weight candidates. O2 improved
paired eager latency 1.521315 -> 1.451859 ms and G1 1.537236 -> 1.470207 ms;
all candidates passed PCC >= 0.999855. The `8x8` prefill grid aligns local
per-core N with the eight-bank DRAM weight shard, so prefill/decode share one
BFP4 weight copy.

The decomposed-persistent candidate improved paired trace replay
0.620704 -> 0.603138 ms and eager 1.471944 -> 1.451630 ms at PCC 1.0. BF8,
one-link, linear, and persistent-BF8 candidates were rejected on end-to-end
latency: persistent BF8's 0.601112-ms trace was only 0.34% beyond persistent
BF16 but its 1.614094-ms eager path regressed 11%.

The canonical final run uses logical prefill 39 with an internal 32+7 test
chunk. It measures 3.710810/1.850049/1.844675 ms for single-chip
prefill/eager-decode/trace and 1.653101/1.556849/0.598285 ms for TP4. The
trace headline is 3.083272x speedup and 77.0818% four-device efficiency.
Prefill/decode PCC are 0.9999978879/0.9996807609; direct two-layer composition
is 0.9984936517. Dynamic positions 39/40, page-table mutation, position 64,
local cache reconstruction, and 100 watcher-enabled advancing trace replays
pass.

The non-fused hidden-sharded alternative has two large activation all-gathers
plus two output reduce-scatters, equal to the selected path's four large RS/AG
phases, and adds two norm-statistics all-gathers. It is source-dominated by the
selected replicated boundary. Fused all-gather-matmul/matmul-reduce-scatter is
the only credible lower-wire family, but current validators do not accept the
selected DRAM-sharded program/layout and the minimal fused Blackhole path has
open nondeterminism issue #46181. It was rejected until a correctness-stable
fused primitive exists.

Full-context per-device persistent bytes are 9,628,549,120 decoder weights,
295,501,824 embedding/LM-head reserve, and 10,737,418,240 BF16 KV cache. The
20,661,469,184 subtotal leaves 13,517,261,824 transient bytes. Unchunked MLP
prefill would conservatively peak at 23,689,428,992 bytes, so the selected
4096-token internal chunk bound is required. Conservative explicit live
activations are 10,737,418,240 bytes; after metadata, trace, CCL, and allocator
reserves, total peak is 33,243,717,888 of 34,178,731,008 bytes, leaving
935,013,120 bytes/device.

## Target hardware and topology

The remainder of this document preserves the pre-implementation plan.  Its
2x2 compiler-matching path was implemented and measured first; the measured
selection above supersedes it as the final target without rewriting that
frozen decision record.

The target is the complete local mesh: four Blackhole p300c ASICs, opened as a
logical `MeshShape(2, 2)`.  A serialized `tt-smi -ls --local` check reported
devices 0--3, and a TTNN mesh-open smoke reported:

```
shape=MeshShape([2, 2])
num_devices=4
compute_grid=CoreGrid(x=11, y=10)
dram_grid=CoreGrid(x=8, y=1)
```

The authoritative compiler capture in
`doc/functional_decoder/multichip_provenance.json` also used a 2x2 mesh.  The
initial implementation therefore uses the captured two-axis tensor-parallel
plan, not a newly invented mapping.  It targets this exact four-device mesh;
smaller meshes are outside the contract.

The first CCL candidate is `FABRIC_1D`, linear topology, with two Ethernet
links on each logical axis.  `models.common.modules.tt_ccl.get_num_links`
identifies this four-Blackhole system as P150x4 and selects `(2, 2)`.  One-link
linear and `FABRIC_2D` are measurement candidates if profiling shows that the
two-link setup loses to setup/synchronization overhead.  Ring is rejected for
the two-device axis collectives because it cannot improve a two-rank ring and
has no payload advantage.

## Model dimensions and tensor plan

Global dimensions are hidden 8192, Q heads 64, KV heads 8, head dimension
128, and MLP intermediate 28672.  Mesh axis 0 is the hidden/input-reduction
axis; mesh axis 1 is the head/intermediate-reduction axis.  Dimensions divide
exactly, so weights need no semantic padding.  TTNN may tile-pad physical rows
for batch 1 decode, but the logical public shapes remain unpadded.

| Tensor | Global logical shape | Mesh mapping | Per-device logical shape |
| --- | --- | --- | --- |
| residual / layer output | `[1, B, S, 8192]` | hidden on axis 0, replicated on axis 1 | `[1, B, S, 4096]` |
| RMSNorm weights | `[8192]` | hidden on axis 0, replicated on axis 1 | `[4096]` |
| packed QKV weight | `[8192, 10240]` | input hidden on axis 0; grouped Q/K/V outputs on axis 1 | `[4096, 5120]` |
| local Q/K/V activations | `[B, 64/8/8, S, 128]` global | Q/KV heads on axis 1; replicated on axis 0 after reduction | `[B, 32/4/4, S, 128]` |
| output projection weight | `[8192, 8192]` | input heads on axis 1; output hidden on axis 0 | `[4096, 4096]` |
| gate/up weights | `[8192, 28672]` each | input hidden on axis 0; intermediate on axis 1 | `[4096, 14336]` each |
| down weight | `[28672, 8192]` | intermediate on axis 1; output hidden on axis 0 | `[14336, 4096]` |
| paged K/V cache | `[blocks, 8, 64, 128]` | KV heads on axis 1; replicated on axis 0 | `[blocks, 4, 64, 128]` |
| page table | `[B, pages_per_sequence]` | replicated on all devices | unchanged |
| RoPE matrices / scalar metadata | sequence-dependent | replicated on all devices | unchanged |

The packed QKV host transformation preserves Q/K/V group boundaries before
the axis-1 split: each half contains 32 Q heads, 4 K heads, and 4 V heads.
Simply concatenating global Q, K, and V and slicing the concatenation in half
is invalid and is explicitly rejected.  There is no MoE or expert routing in
Llama 3.1 70B; its MLP is dense SwiGLU.

## Activation and collective sequence

Each distributed RMSNorm computes local squared statistics across the 4096
hidden values on axis 0, all-gathers the compact statistics on that axis, and
applies the local norm weight.  Axis 1 replicas see identical results.

The captured 2D layer executes these reductions:

1. packed QKV partials: sum on axis 0;
2. output projection partials: sum on axis 1;
3. gate partials: sum on axis 0;
4. up partials: sum on axis 0;
5. down projection partials: sum on axis 1.

Packing QKV replaces the compiler capture's separate Q, K, and V collectives
with one collective.  Gate and up remain separate initially so the optimized
single-chip kernel and numerical baseline are preserved; a fused packed
gate/up weight is an eligible graph rewrite only if it reduces warmed latency.
All collectives use the shared ping-pong semaphores owned by one `TT_CCL`
instance, which makes the path trace replay safe and avoids per-layer semaphore
allocation.

For batch-1 decode, physical tile padding makes the representative BF16
collective payloads approximately: packed QKV 327,680 bytes, output projection
262,144 bytes, gate 917,504 bytes, up 917,504 bytes, and down 262,144 bytes,
or 2.69 MB per layer plus the small norm statistics.  Prefill uses the same
logical graph with payload proportional to the real sequence length; chunking
is internal and must not change the public logical length.

## KV cache and context capacity

The paged cache block size is 64 tokens.  Arbitrary logical prefill and decode
positions are represented through replicated INT32 page tables and current
positions; neither is inferred from a tile-aligned physical extent.  Cache
heads are local (`4` per device) and are duplicated only across axis 0.  Tests
must cover non-identity page tables, nonzero positions, page-boundary updates,
and logical sequence lengths not divisible by 32 or 64.

The prior one-device context advertisement of 18 tokens is not a physical
limit for the multichip stack.  Using the selected BFP4 weight representation
(576 bytes per 32x32 tile) and BF16 KV cache, the conservative per-device
full-stack estimate is:

| Component | Per-device bytes |
| --- | ---: |
| 80 decoder layers of weights | 9,626,112,000 |
| token embedding plus LM head, untied | 295,501,824 |
| BF16 KV cache, batch 1, 131072 tokens, 80 layers, 4 local KV heads | 21,474,836,480 |
| total before activations, trace, allocator overhead | 31,396,450,304 |
| device allocator capacity observed by the baseline | 34,178,731,008 |
| remaining headroom | 2,782,280,704 |

Thus the model's full 131072-token context is physically feasible at batch 1
on the captured 2D layout, provided RoPE tables and immutable metadata are
shared by the future layer stack.  This decoder stage will advertise the full
context at batch 1 and record that larger batches trade batch for context.  A
one-layer allocation probe and full-size cache-shape validation are required;
constructing the full model is deliberately outside this goal.

## Optimization candidate and rejected alternatives

After the provenance-matching path is correct, a flat TP4 candidate may be
measured on the same four physical devices.  It replicates the full residual,
uses 16 local Q heads and 2 local KV heads, column-parallel QKV/gate/up, and
row-parallel output/down.  It needs only output and down all-reduces, about
1.05 MB total at batch-1 decode, and halves per-device KV-cache memory.  Its
communication and cache advantages make it the strongest performance
candidate, but it is not accepted without correctness, trace, and profiler
evidence against the captured 2D implementation.

Rejected before measurement:

- one or two devices: leaves available hardware unused and cannot meet the
  weight/KV-cache objective;
- pipeline parallelism: increases single-token bubbles and retains whole
  layer weights/caches per stage;
- sequence parallelism for decode: sequence length is one and provides no
  useful work partition;
- KV-head replication across all four devices: simple but doubles cache memory
  relative to the captured 2D plan and quadruples it relative to flat TP4;
- dense all-expert execution: inapplicable because this model has no experts;
- an aligned-only sequence API: forbidden even when kernels use tile padding.

## Acceptance evidence to collect

The final selection requires prefill and decode PCC against the optimized
single-chip TTNN layer for every layer kind (this architecture has one), paged
cache/page-table/current-position checks, stacked boundary layout checks,
nonaligned lengths, repeated deterministic/stress decode, warmed trace replay,
watcher-clean runs, and a source/runtime fallback audit.  Warmed single-chip
and multichip latency must be collected under the same shape/dtype policy.
`tt-perf-report` human-readable and CSV artifacts must be inspected for CCL,
DRAM, compute, and reshape/data-movement costs before accepting a topology.
