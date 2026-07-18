# Falcon3-7B multichip decoder

This stage specializes the optimized Falcon3 dense decoder layer for the full
four-chip Blackhole mesh installed in the host.  It is a decoder-layer stage;
it does not add embeddings, a layer stack, generation, or vLLM integration.

## Selected target and tensor plan

The fixed target is a `1x4` mesh with tensor parallelism on mesh axis 1.  The
host reports four Blackhole p300c devices, while TTNN's CCL helper classifies
the host wiring as P150x4 and selects two links.  The compiler provenance in
`../functional_decoder/multichip_provenance.json` independently requires the
same `1x4`, TP=4 decomposition.  Smaller mesh configurations are out of scope.

The optimized single-chip decoder is the numerical and local-kernel baseline.
The selected path preserves a replicated `[1,batch,sequence,3072]` residual at
layer boundaries and partitions every projection that has a model-parallel
dimension:

| Tensor or operation | Global shape | Shape resident/produced per device | Placement |
| --- | ---: | ---: | --- |
| residual input/output | `[1,B,S,3072]` | `[1,B,S,3072]` | replicated BF16; decode width-sharded over 32 cores inside each chip |
| input/post RMSNorm weight | `[3072]` | `[3072]` | replicated BF16 |
| packed QKV weight | `[3072,5120]` | `[3072,1280]` | column parallel; Q/K/V are packed rank-by-rank |
| local Q/K/V heads | Q=12, K=4, V=4 | Q=3, K=1, V=1, head dim 256 | rank local; no attention all-gather |
| O weight | `[3072,3072]` | `[768,3072]` | row parallel; full-width partial followed by all-reduce |
| gate and up weights | each `[3072,23040]` | prefill `[3072,5760]`; decode `[3072,6144]` with 384 zero columns | column parallel |
| down weight | `[23040,3072]` | prefill `[5760,3072]`; decode `[6144,3072]` with 384 zero rows | row parallel; padded rows multiply zero activations, then full-width partial is all-reduced |
| contiguous K or V cache | `[B,4,L,256]` | `[B,1,L,256]` | rank-local BFP8 in DRAM |
| paged K or V cache | `[blocks,4,32,256]` logically | `[blocks,1,32,256]` | rank-local BFP8; replicated `[B,pages]` page table |

There is no MoE or expert-routing subgraph in Falcon3-7B-Base.  The expert
strategy is therefore not applicable; the dense SwiGLU MLP is tensor-parallel
as shown above.

All logical tensor dimensions divide TP=4 and tile size 32.  Runtime discovery
reports eight DRAM banks per chip.  QKV is already bank- and 8-core-aligned;
the decode-only local MLP dimension is padded from 5,760 to 6,144.  This is the
least common alignment for eight DRAM banks and 24 compute cores.  Padded
gate/up columns are zero, the corresponding gated activations stay zero, and
the padded down rows are zero, so no public slice is required.  Non-aligned
logical sequence lengths are padded only inside TTNN operators and sliced back
before cache fills and public output.  Paged-cache positions and page tables
remain logical and are not rounded to a tile.

Decode weights retain the optimized decoder's DRAM width sharding.  With eight
DRAM banks, the local decode weight shards are QKV `[3072,160]`, O `[768,384]`,
gate/up `[3072,768]`, and down `[6144,384]` per bank.  Decode L1 grids are 8
cores for QKV/O, 24 cores for gate/up, and 8 cores for down; their per-core
output widths are respectively 160, 384, 256, and 384 elements.  Prefill weights are interleaved
DRAM and large matmuls run in bounded row chunks without changing the logical
sequence contract.

## Collective and topology decision

Each layer performs two BF16 collectives on the full hidden-width partial:
one after O and one after down.  The fabric and CCL topology are Ring on axis 1
with two links.  This exactly matches the compiler graph and keeps the stacked
decoder boundary replicated, so the next layer consumes the output directly.

The alternatives considered before implementation were:

| Candidate | Collective chain per layer | Decision |
| --- | --- | --- |
| local matmul + all-reduce | O AR, down AR | selected compiler-proven baseline; lowest integration risk and direct stacked boundary |
| reduce-scatter + immediate all-gather | O RS+AG, down RS+AG | rejected: same bytes as all-reduce plus extra dispatches and no residency benefit |
| residual-sharded with delayed all-gather | O RS, local residual add, distributed RMSNorm, AG before next QKV; analogous MLP boundary | rejected by complete boundary measurement: 0.109954 ms versus 0.097611 ms replicated (1.1265x slower), PCC 0.999898 through real next QKV |
| fused all-gather + matmul | delayed AG fused into next column-parallel projection | rejected for this layer baseline: needs persistent CCL semaphores and a residual-sharded stack contract absent from the compiler graph |
| fused matmul + reduce-scatter | row-parallel matmul fused with RS | rejected on this target: the repository disables the primitive on Blackhole because of a nondeterministic synchronization race |
| sequence parallelism | shard prefill rows/tokens | rejected for the final layer baseline: decode has one logical token and paged-cache ownership becomes more complex without reducing TP weight memory |

The residual-sharded and fused-boundary probes must beat the selected complete
boundary, preserve PCC, and leave no deferred communication before they can
replace it.  Otherwise the selected all-reduce path remains the final path.

## Context and per-device capacity plan

TP4 changes both weight and cache residency.  The two optimized physical
projection copies (prefill and DRAM-sharded decode), plus two BF16 norm
weights, consume 1,926,414,336 bytes per device for all 28 layers.  This is an
exact tile calculation using 576 bytes per 32x32 BFP4 tile; decode-only MLP
padding accounts for 55,738,368 bytes of that total.  At the advertised
32,768-token context, both BFP8 caches for all 28 layers consume 499,122,176
bytes per device for batch 1.  This uses the physical 1,088-byte BFP8 tile:
each K or V cache has 8,192 tiles per layer and device.  The shared BF16 rotary
cache is 33,554,432 bytes per device and the BF16 residual is 201,326,592
bytes.  Trace capture reserves another 100,000,000 bytes.  The 1,024-row
prefill chunk has a calculated 63,176,704-byte live activation set; the plan
reserves 100,663,296 bytes (96 MiB) for it and allocator headroom.  Total
batch-1 resident storage plus these reserves is 2,861,080,832 bytes per device,
which fits comfortably in the roughly 34 GB device DRAM.  The multichip
contract therefore restores the advertised 32,768 context, with batch 1 as the
fully executed capacity gate.  Batch 32 still supports the compiler/optimized
short-sequence workload; full-context batch 32 is not claimed by this
decoder-only stage because it was not executed.  Its KV cache and residual
would consume 15,971,909,632 and 6,442,450,944 bytes per device, and its
resident lower bound is 24,474,329,344 bytes before full-prefill concatenation
and attention temporaries.

## Validation and performance

Falcon3 has one meaningful dense decoder-layer kind, so real layer 14 is the
representative correctness and performance layer.  All values below are on the
selected TP4 defaults and real model weights.

| Check | Result |
| --- | ---: |
| TP4 prefill versus optimized single-chip, batch 32 / seq 17 | PCC 0.99999976 |
| TP4 decode versus optimized single-chip, batch 32 / position 17 | PCC 0.99999994 |
| TP4 K / V cache versus optimized single-chip | 0.99999672 / 0.99999931 |
| TP4 prefill versus HF, batch 1 / logical seq 31 | PCC 0.99993482 |
| TP4 decode versus HF, positions 31 / 32 | 0.99916386 / 0.99902942 |
| TP4 K / V cache versus HF, seq 31 through position 32 | minimum 0.99677188 / 0.99555507 |
| logical seq 1,025, two internal row chunks | output 0.99994438, K 0.99679253, V 0.99555432 |
| heterogeneous batch positions 17 / 31 versus independent batch-1 calls | 0.99999162 / 0.99999302 |
| full logical seq 32,768 plus decode position 32,767 | completed; sampled K 0.99644107, V 0.99534817 |

The non-aligned paged test uses a permuted page table, fills positions 0..30,
then updates logical positions 31 and 32 across a page boundary.  Its four rank
outputs are bitwise equal, the stacked output contract is `[1,1,1,3072]`, the
program-cache count is stable, and eight warmed trace replays are bitwise
deterministic.  Heterogeneous batch positions are independently checked, not
only compared between mesh ranks.  The full-context run uses 1,024 physical
pages per device with local cache shape `[1024,1,32,256]`; page tables and
positions remain replicated logical tensors.

### Warmed latency

Each decode number is the median of five samples; each sample contains 100
warmed trace replays.  The baseline is the completed single-chip
`OptimizedDecoder` using its selected DRAM-48 BFP4 path.

| Batch / phase | Single chip | TP4 | Speedup | TP efficiency |
| --- | ---: | ---: | ---: | ---: |
| 1 decode | 0.644047 ms | 0.356696 ms | 1.8056x | 45.14% |
| 32 decode | 0.768483 ms | 0.578799 ms | 1.3277x | 33.19% |
| 32 prefill, seq 17 | 3.262737 ms | 2.740996 ms | 1.1903x | 29.76% |

The precision-locked batch-1 tuning sweep selected QKV/O core counts 8/8,
MLP gate/down counts 24/8, BF16 CCL, and two links.  The resolved physical
grids and padding are recorded in each new sweep artifact.  Nearby controls
were slower: QKV16/O8 was 0.365193 ms, QKV8/O12 was 0.359111 ms, gate8/down24
was 0.379098 ms, gate8/down8 was 0.368295 ms, QKV8/O24 was 0.363258 ms, BFP8
CCL was 0.370917 ms, and one link was 0.360265 ms.  The 8-core down-only
candidate was repeatably faster than 24 cores (0.356590 versus 0.358628 ms)
and became the production default.  Gate/up still requires the shared
load-time padding from 5,760 to 6,144; padded values remain zero and do not
change the logical output.

### Profiler and topology audit

The selected Tracy capture has zero host ops.  `tt-perf-report` reports 444 us
of merged device work for short prefill.  Across three traced decode replays it
reports 1,020 us total device work, or 340 us per replay, consistent with the
0.356696-ms wall result.  Decode gate/up reaches about 231 GB/s and 44.5%
compute utilization; the selected down reaches 258 GB/s and 49.8%.  Each
all-reduce lowers to an approximately 24/23-us reduce-scatter plus 14-us
all-gather, so the two CCL pairs account for about 75 us per layer replay.  The
human tables, CSVs, raw ops CSV, and hashes are under `tracy/tp4_selected/`.

The authoritative residual-boundary experiment carries a fractured partial
through reduce-scatter, local residual add, distributed RMSNorm, hidden
all-gather, and the next real layer-14 QKV.  It produces PCC 0.999898 but takes
0.109954 ms versus 0.097611 ms for all-reduce/add/RMSNorm/QKV, so the complete
fractured boundary is 1.1265x slower.  The earlier RS/add/AG microprobe measured
only the collective boundary and is not selection evidence.  The
compiler-proven replicated boundary remains the stack baseline.

### Safety and limitations

The runtime fallback audit finds no `torch`, `from_torch`, `to_torch`, or
`OptimizedDecoder` call in any hot forward method.  The serialized final suite
includes the safe complete fractured-boundary gate; fused
matmul+reduce-scatter is intentionally excluded because the repository
disables it on Blackhole for a nondeterministic synchronization race tracked
by issue #46181.  Watcher passed the real paged/trace test on all worker
and dispatch cores with zero error matches.  Active-Ethernet Watcher injection
had to be disabled because it expands the fabric router to 27,920 bytes, beyond
the fixed 25,600-byte Blackhole kernel-config buffer; the ring fabric and both
CCL operations remained active in the passing run.  The exact log and settings
are under `watcher/`.

The implementation is deliberately fixed to this four-chip mesh.  The host
also warns that firmware 19.8 is newer than fully tested 19.5, `/dev/shm` has
only a narrow margin over the 16-MiB MPI allocation, and the pre-existing
Inspector directory has a permissions problem; none caused a test failure.
This stage remains decoder-only and does not begin full-model, generation, or
vLLM work.

## Artifacts

- `results/direct_optimized_baseline_pcc.json`: direct TP4-to-optimized TTNN PCC.
- `results/paged_non_aligned_correctness.json`: paged, boundary, stacked-layout,
  deterministic trace, and HF evidence.
- `results/{heterogeneous_positions,long_prefill_1025,max_context_batch1}.json`:
  logical-position, arbitrary-length, and largest-context gates.
- `results/final_batch{1,32}.json`: final warmed wall latency and speedup.
- `results/sweep_*.json`: rejected kernel/CCL candidates.
- `tracy/tp4_selected/{prefill,decode}_perf_report.{txt,csv}` and
  `profile_provenance.json`: human and machine profiler evidence.
- `watcher/{watcher.log,pytest_stdout.log,watcher_clean.json}`: exact Watcher
  log, pytest result, and audit.
- `work_log.md`: exact commands, failures, and stage handoff record.
