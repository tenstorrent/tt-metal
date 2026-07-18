# Qwen3-32B multichip decoder

This stage implements and validates the Qwen/Qwen3-32B decoder layer on the
complete four-device mesh in this machine. It intentionally stops at the
decoder: no full-model or vLLM work is included.

## Target and selected strategy

- Baseline: `tt/optimized_decoder.py`, `all_bfp4_lofi`, batch 32, BFP4
  projections, BF16 activations, BFP8 KV cache, and traced decode.
- Hardware: four Blackhole p300c devices as fixed `MeshShape(1,4)`, TP=4 on
  axis 1, `FABRIC_1D_RING`, Ring topology, two links.
- Model: 64 structurally identical dense decoder layers. Qwen3-32B has no MoE
  router or experts, so expert parallelism is not applicable.
- Layer boundary: BF16 hidden-width-sharded `[1,32,L,1280]` per rank. A second
  decoder consumes the first decoder's output directly, with no boundary
  gather.
- Per layer: two hidden all-gathers and two row-parallel reduce-scatters. The
  decode trace uses rotating persistent collective buffers.

This plan was selected before the final implementation. Two-device TP leaves
half the hardware idle and doubles local weight/cache ownership. Replicated
layers duplicate memory and do not accelerate one layer. Sequence parallelism
does not help single-token decode. KV-only or head-only partitioning leaves the
MLP and projection memory dominant. The compiler-provenance replicated-boundary
family was retained as a measured candidate and rejected below.

A shape-faithful BF16 CCL smoke restored `[1,1,32,5120]` with all-gather PCC
1.0 and produced the exact four-way reduce-scatter sum.

## Tensor, activation, cache, and program plan

`M=32*logical_sequence_length` for prefill and `M=32` for decode.

| Tensor | Dense shape | Per-device logical shape | Mapping |
|---|---:|---:|---|
| Layer boundary | `[M,5120]` | `[M,1280]` | hidden-width shard, BF16 |
| RMSNorm weights | `[5120]` | `[5120]` | replicated BF16 |
| Packed QKV weight | `[5120,10240]` | `[5120,2560]` | column shard; `Q16,K2,V2` |
| Q/K norm weight | `[128]` | `[128]` | replicated BF16 |
| O weight | `[8192,5120]` | `[2048,5120]` | row shard |
| Gate/up weight, each | `[5120,25600]` | `[5120,6400]` | column shard |
| Down weight | `[25600,5120]` | `[6400,5120]` | row shard |
| Contiguous K/V, each | `[32,8,S,128]` | `[32,2,S,128]` | KV-head shard, BFP8 |
| Paged K/V, each | `[blocks,8,64,128]` | `[blocks,2,64,128]` | replicated page table |
| Positions | `[32]` | `[32]` | stable replicated int32 trace input |
| RoPE cosine/sine | `[S,128]` | same | replicated BF16 TILE and ROW_MAJOR tables |

Decode uses DRAM-width-sharded weights and L1-width-sharded activations. There
is no channel padding in the projection shapes. The only decode padding is the
device-only Q-head workaround described below.

| Role | Grid | Input shard | Output shard | DRAM weight shard | `in0/per_core_N` |
|---|---:|---:|---:|---:|---:|
| Local boundary | 20 cores, 10x2 | `[32,64]` | same | n/a | n/a |
| Full residual/norm | 20 cores, 10x2 | `[32,256]` | same | n/a | norm `block_w=8,subblock_w=4` |
| QKV | 20 cores, 10x2 | `[32,256]` | `[32,128]` | `[5120,320]` | `8/4` |
| O | 8 cores, 8x1 | `[32,256]` | `[32,640]` | `[2048,640]` | `8/20` |
| Gate/up | 20 cores, 10x2 | `[32,256]` | `[32,320]` | `[5120,800]` | `8/10` |
| Down | 20 cores, 10x2 | `[32,320]` | `[32,256]` | `[6400,640]` | `10/8` |

All projections use `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.
That config class does not expose an output-subblock field; the factory chooses
it internally. `results/final_selected_o8.json` contains the machine-readable
shape, shard, grid, program, dtype, cache, padding, and collective contract.

The selected decode dataflow is:

1. `[32,1280]` local boundary -> Ring AG -> `[32,5120]` -> norm/QKV -> local
   `Q16,K2,V2`.
2. Local SDPA/O creates `[32,5120]` partial -> Ring RS -> `[32,1280]`, then
   residual add.
3. Ring AG -> `[32,5120]` -> post-attention norm/gate/up -> local `[32,6400]`.
4. Local down creates `[32,5120]` partial -> Ring RS -> `[32,1280]`, then the
   sharded layer output.

At decode, each collective transforms 81,920 input bytes/rank to 327,680 output
bytes/rank or the reverse. The estimated Ring wire traffic is 245,760
bytes/rank/collective and 983,040 bytes/rank/layer.

## Paged cache, positions, and trace contract

Contiguous and paged paths keep two K heads and two V heads per rank. The page
table and current positions are replicated, while cache blocks remain
KV-head-sharded. Prefill accepts one authoritative page table; the earlier
ambiguous `chunk_page_table` argument was removed because it did not define a
chunk-start semantic.

The public contract accepts every logical sequence length through the capacity
ceiling. Length 31 is tested. Page-block and tile padding are internal and are
never exposed as a required caller alignment.

The final paged trace gate prefills a real layer at length 64, captures decode
at position 64, replays unchanged, mutates the device page table in place,
replays the remapped table, and then advances the stable position buffers to
65. Output, K cache, and V cache PCC are all 1.0 for every replay. The eager
references are created before trace capture, so replay performs no allocation.

Blackhole SDPA decode leaves the second eight-query GQA group invalid for a
physical 32-head tile with logical 16Q/2KV. The device-only workaround
duplicates each eight-Q group to physical 32Q/2KV, runs SDPA once, and selects
one copy of each group. Logical ownership remains 16Q/2KV and no host fallback
is used.

## Topology decisions and rejected alternatives

| Family | Before collective | Collective | After / next consumer | Evidence / decision |
|---|---|---|---|---|
| Selected sharded boundary | local `[32,1280]` | AG to `[32,5120]`; O/down partial RS back to `[32,1280]` | full norm/QKV or gate/up after AG; local residual after RS | PCC 1.0 vs provenance family; 0.629466 ms traced |
| Compiler provenance | replicated `[32,5120]` | two all-reduces, each RS+AG | replicated residual and next layer | correct, 1.860160 ms warmed eager; trace replay stalled over four minutes and required reset; rejected |
| Fused matmul+RS | interleaved row-parallel partial | fused RS | local boundary | PCC 0.99997153; whole layer 0.768530 ms, 21.7% slower; rejected |
| Fused AG+O matmul | local attention shard | fused AG+matmul | full O partial | shape-valid but PCC 0.00440685; rejected |
| BFP8 CCL | same selected topology | BFP8 AG/RS | same consumers | correct but 0.675635 ms and less precise; rejected |

The provenance trace stall was isolated to that rejected family. Recovery was
`tt-smi -r all`, followed by healthy four-board discovery. The selected trace
is stable.

Independent per-role sweeps tested QKV, O, gate/up, down core counts and
`in0_block_w`, plus HiFi2 advice candidates. The 8-core O grid was the only
repeatable improvement. Three alternating 200-replay pairs measured:

| Trial | O16 default (ms) | O8 (ms) |
|---:|---:|---:|
| 1 | 0.631343 | 0.629469 |
| 2 | 0.631262 | 0.629486 |
| 3 | 0.631355 | 0.629413 |
| Pooled median | 0.631343 | 0.629469 |

Other role candidates ranged from 0.632014 to 0.650841 ms. Attention HiFi2
was 0.669230 ms and MLP HiFi2 was 0.782342 ms, so the advice did not justify an
accuracy-unneeded fidelity increase.

The current shard-advisor pass is documented in `shard_advisor_status.md`. Its
pinned environment fails to import `_ttnn.so` because of an undefined
`moe_compute` symbol. The completed single-chip report remains relevant to the
per-rank L1 skeleton, but the advisor explicitly does not own the selected
DRAM-sharded-weight strategy and cannot model the Ring CCL graph. Direct
role-specific measurement is authoritative here.

## Correctness

Qwen3-32B has one meaningful dense layer kind; all 64 source layers share its
configuration and weight shapes. Layer 32 is the representative real-weight
layer.

| Gate | PCC / result |
|---|---:|
| Synthetic non-aligned prefill vs HF, length 31 | 0.99647633 |
| Direct sharded decoder-to-decoder prefill handoff vs HF | 0.99125232 |
| Synthetic decode vs HF | 0.99659289 |
| Paged vs contiguous prefill/decode | 1.0 / 1.0 |
| Synthetic traced decode vs HF, positions 32 / 33 | 0.99657259 / 0.99657060 |
| Paged traced unchanged/remapped/advanced output and K/V | all 1.0 |
| Real query vs optimized TTNN | 0.99993715 |
| Real SDPA / concatenated attention vs optimized TTNN | 0.99966767 / 0.99966767 |
| Real attention residual vs optimized TTNN | 0.99999055 |
| Real prefill / decode vs optimized TTNN | 0.99999995 / 0.99997244 |
| Real prefill K/V cache vs optimized TTNN | 0.99998010 / 0.99996648 |
| Real decode K/V update vs optimized TTNN | 0.99993595 / 0.99990264 |

## Performance

The same real checkpoint and prompt-derived layer-32 activation are used for
both paths. Results are warmed medians; final decode is nine trials of 200
trace replays.

| Path | Prefill 17 (ms) | Traced decode (ms) | Prefill PCC | Decode PCC |
|---|---:|---:|---:|---:|
| Single-chip optimized baseline | 5.502353 | 1.217318 | reference | reference |
| Final 1x4 TP, O8 | 3.127879 | 0.629461 | 0.99999995 | 0.99997244 |
| Speedup | 1.7591x | 1.9339x | | |
| Four-device efficiency | 43.98% | 48.35% | | |

`perf_report.md` contains the human-readable Tracy and `tt-perf-report` tables
and advice dispositions. The compact CSVs are retained. The final raw CSV shows
174 merged device ops across three decode replays: about 0.554 ms device time
per replay. Matmuls are 43.2%, reduce-scatter 8.2%, all-gather 4.1%, and
explicit resharding under 0.6% of device time.

## Full-stack capacity and context

The 64-layer reservation includes duplicated prefill/decode BFP4 weight
layouts, local BFP8 K/V, both TILE and ROW_MAJOR RoPE table layouts, norm and
persistent CCL state, the trace reservation, and the ownership-aware prefill
live set.

| Logical context | Physical cache | Static bytes/device | Peak/failure bytes/device | Result |
|---:|---:|---:|---:|---|
| 12,352 | 12,352 | 23,443,308,544 | 32,550,191,104 peak | pass; 28,539,904 free |
| 12,353 | 12,416 | 23,515,693,056 | 30,617,174,016 at failed allocation | expected OOM |

At 12,353, the prefill live set requests 2,028,994,560 bytes, or 253,624,320
bytes/bank, while the largest free block is 245,194,624 bytes/bank. Therefore
the supported public context is every logical length `1..12352`. At the HF
advertised 40,960 tokens, local TP K/V alone is 45,634,027,520 bytes/device,
already beyond physical DRAM. `doc/context_contract.json` records the exact
physical limit.

## Runtime audit, Watcher, and limitations

- Static audit rejects runtime `torch`, `from_torch`, `to_torch`, or `super()`
  fallbacks in collective, prefill, cache-fill, MLP, and decode paths.
- Final Watcher ran on all four devices and retained the exact 2,243-line log;
  no `error|assert|hang|stuck|timeout` signature matched. ETH instrumentation
  is disabled because it grows active-Ethernet firmware to 27,920 bytes, above
  the 25,600-byte kernel-config buffer. Worker/data/compute paths remain watched.
- Profiler and Watcher runs are separate, as required by the device-use policy.
- Implementation is fixed to batch 32 and this exact 1x4 mesh. Smaller mesh
  configurations are not supported.
- `/dev/shm` has about 17 MB free and MPI warns while requesting a 16 MB
  segment; all acceptance runs complete.

## Authoritative artifacts

- Final wall/PCC/mesh plan: `results/final_selected_o8.json`
- Single-chip baseline: `results/baseline_single_chip.json`
- Topology comparison: `results/topology_family_benchmark.json`
- Paged trace refresh: `results/paged_trace_refresh.json`
- Role/precision sweeps: `results/role_*.json`, `results/advice_*.json`
- Fused probes: `results/fused_collective_probes.json`
- Capacity: `results/capacity_seq12352.json`, `results/capacity_seq12353.json`
- Watcher metadata and exact log: `results/watcher_clean.json`,
  `results/watcher_clean.log`
- Profiler metadata: `results/profile_run.json`
- Consolidated static/synthetic/real/paged gate: `final_gate.xml` (4 passed)
- Filtered CSVs: `decode_perf_report.csv`, `prefill_perf_report.csv`
- Raw Tracy CSV, retained locally:
  `tracy_final_o8/reports/2026_07_18_02_10_04/ops_perf_results_2026_07_18_02_10_04.csv`
- Commands, recovery, review, and commit history: `work_log.md`
