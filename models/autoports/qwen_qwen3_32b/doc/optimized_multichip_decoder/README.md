# Qwen3-32B optimized multichip decoder

This stage optimizes the existing Qwen/Qwen3-32B decoder layer on the real
four-device Blackhole mesh. It does not add full-model or vLLM code. The final
default path is TP=4 on a `1x4 FABRIC_1D_RING`; no single-chip, replicated, or
fallback measurement is used as the stage result.

## Result

The default constructor now keeps the inter-layer residual as an L1
width-sharded `[1,32,1,1280]` tensor on every rank and packs each rank's gate
and up weights into one `[5120,12800]` projection. Decode uses persistent BF16
asynchronous collectives, BFP4/LoFi projection weights, BFP8 KV cache, and
DRAM-width-sharded matmuls. Packed gate/up uses 10 cores with
`in0_block_w=8`; QKV and down use 20 cores; O uses 8 cores.

Prompt-derived layer-32 inputs from checkpoint revision
`9216db5781bf21249d130ec9da846c4624c16137` were compared to the accepted
optimized single-chip reference. Qwen3-32B has one homogeneous dense decoder
layer kind, so layer 32 covers every meaningful layer kind. This is not an MoE
model.

| Measurement | Before | Final default | Change |
|---|---:|---:|---:|
| Prefill PCC | 0.999999954 | 0.999999954 | preserved |
| Decode PCC | 0.999972436 | 0.999972436 | preserved |
| Warmed prefill, seq 17 | 4.641535 ms | 3.372121 ms | 27.349% faster |
| Traced warmed decode, 1000 replays x 9 trials | 0.629593 ms | 0.616720 ms | 2.045% faster |

The final values are from `results/final_default.json`, produced after the
winner became the source and harness default. Its decode median is independently
consistent with the long candidate run at 0.616706 ms.

## Inter-layer residual contract

The full-model stage must preserve this boundary:

- logical output/input: `[1,32,seq,1280]` per rank, TP-sharded on hidden axis;
- decode physical layout: BF16 TILE, L1 WIDTH_SHARDED over a `10x2` grid,
  shard `[32,64]` for one-token decode;
- prefill boundary: the same logical TP fracture in DRAM interleaved form;
- layer-boundary collective count: zero;
- decoder entry owns the two internal hidden all-gathers required before
  column-parallel QKV and packed gate/up; row-parallel O and down own their
  reduce-scatters;
- callers must not gather, all-reduce, restore to replicated hidden, or
  reshard between layers.

The direct two-layer synthetic test consumes one decoder's output in the next
decoder without an intervening collective and passes PCC 0.991252 for prefill
and 0.990980 for decode against two HF applications. Every decode shard is
asserted to remain L1 width-sharded at the boundary. The lower-movement family
was measured end to end; it was not penalized by an immediate restore to the
former DRAM boundary.

## Operation-topology audit

| Topology item | Audit and action | Evidence |
|---|---|---|
| Split Q/K/V | Already rank-packed into one QKV matmul; retained | source and profiler `32x5120x2560` row |
| Repeated same-input gate/up | Rewritten to one rank-packed gate/up matmul plus two slices | separate 0.629593 ms; packed final 0.616720 ms |
| Composite attention | `nlp_create_qkv_heads_decode`, SDPA decode, RoPE, and concat-heads already use dedicated ops; retained | real internal PCC and profiler |
| Row matmul + reduce-scatter | Experimental fused matmul-RS tried for both O/down; correct but slower | `candidate_packed_sharded_fused_matmul_rs.json`, 0.752589 ms |
| Hidden all-gather + matmul | Distributed RMSNorm plus fused AGMM tried after legal rank/shape/link adaptations; correct but slower | `candidate_distributed_norm_fused_agmm_1link.json`, 0.778513 ms |
| Material collectives | Two internal AG and two RS remain; async and persistent; no boundary CCL | final profiler and `mesh_plan.collectives_per_layer` |
| Decode output restore | Removed; output remains on the selected local residual shard | 0.629593 -> 0.624113 ms before packing |
| Reshard/layout conversions | Shard-advisor full rank-local family and packed split layouts measured; final keeps DRAM split because L1 split regressed | advisor 0.728389 ms; L1 split 0.624330 ms |
| SwiGLU | SiLU remains folded into the gate/up multiply | source and profiler |

## Coherent family comparisons

All rows use real TP=4 execution and accepted PCC unless noted.

| Family | Candidate | Prefill ms | Decode ms | Decision |
|---|---|---:|---:|---|
| Residual layout | former DRAM output restore | 4.641535 | 0.629593 | reject |
| Residual layout | sharded output | 4.313862 | 0.624113 | accept as family base |
| Packed projection | separate gate/up | 4.313862 | 0.624113 | reject |
| Packed projection | packed gate/up, 20 cores | 3.476185 | 0.617377 | accept rewrite |
| Packed geometry | 10 cores/in0=8, long | 3.392129 | 0.616706 | selected |
| Packed geometry | 20 cores/in0=8, long | 3.441192 | 0.617178 | reject |
| Packed geometry | 10 cores/in0=4 / 2 | 3.496187 / 3.435271 | 0.624872 / 0.645748 | reject |
| Packed split | L1, 20 / 40 cores | 3.421913 / 3.453123 | 0.624330 / 0.632892 | reject |
| Packed split | DRAM, 40 cores | 3.433956 | 0.625113 | reject |
| Persistent buffers | disabled | 3.441491 | 0.630448 | reject |
| Persistent buffers | enabled | 3.475837 | 0.616889 | selected |
| CCL dtype | BFP8 | 3.595996 | 0.635099 | reject; lower PCC too |
| CCL dtype | BF16 final default | 3.372121 | 0.616720 | selected |
| Collective placement | distributed norm + ordinary AG/matmul | 3.409621 | 0.628085 | reject |
| Fused CCL+matmul | distributed norm + one-link AGMM | 3.464048 | 0.778513 | reject |
| Fused matmul+CCL | matmul-RS | 3.447051 | 0.752589 | reject |
| Shard advisor | four emitted rank-local 1-D matmuls and reverts | 3.476971 | 0.728389 | reject |
| Attention fidelity | HiFi2 | 3.484897 | 0.653609 | reject |
| MLP fidelity | HiFi2 | 3.630294 | 0.767596 | reject |
| KV dtype | BF16 | 3.560661 | 0.618760 | reject; no PCC benefit |
| Prefill input placement | chunk-local L1 interleaved | 3.572162 | 0.616811 | reject; same-settings DRAM was 3.452574 ms |

The first 10-core packed attempt failed because the automatically selected
`in0_block_w=16` exceeded the per-bank weight shard's eight K tiles. That was
not treated as a rejection: legal widths 8, 4, and 2 were run, and 8 won.
Similarly, fused AGMM was adapted from rank-2 to rank-4 weights, from an
interleaved statistics output to a legal one-core sharded output, and from two
links to its verified one-link schedule before its performance rejection.
The profiler's final packed-graph advice to place prefill matmul input 0 in L1
was also implemented, not dismissed from the full-context size: each at-most
640-row input was copied to L1 immediately before its QKV/O/packed/down matmul.
It retained exact PCC but regressed prefill 3.452574 -> 3.572162 ms (3.464%) in
a source-identical seven-trial A/B, so the copy cost outweighs the matmul gain.
The opt-in reproducer is `QWEN3_32B_MULTICHIP_PREFILL_INPUT_L1=1`.

## Material CCL contract

All selected collectives use hidden axis `dim=3`, BF16 payload, Ring topology,
2 links, `chunks_per_sync=10`, 2 workers/link, and 2 buffers/channel. Times are
source-current device times from the retained human-readable profiler tables.

| Phase / semantic rows | Input -> output memory contract | Intermediate / output persistence | Mean row time | Decision |
|---|---|---|---:|---|
| Prefill AG x2 | rank-local DRAM interleaved `[544,1280]` (1,392,640 B) -> full DRAM interleaved `[544,5120]` (5,570,560 B) | n/a / nonpersistent | 82.3 us | selected; required immediately before QKV and packed gate/up |
| Prefill RS x2 | full DRAM interleaved `[544,5120]` -> rank-local DRAM interleaved `[544,1280]`; internal nonpersistent scratch is DRAM | none / none | 89.2 us | selected after O and down; reduces before chunk concatenation |
| Decode AG x2 | rank-local L1 width-sharded `[32,1280]` (81,920 B) -> full DRAM interleaved `[32,5120]` (327,680 B) | n/a / double-buffered persistent | 11.2 us | selected; persistent A/B is faster overall |
| Decode RS x2 | full L1 interleaved `[32,5120]` -> rank-local L1 width-sharded `[32,1280]` | double-buffered persistent DRAM / double-buffered persistent L1 | 21.6 us | selected; preserves the next-layer boundary |

Persistent decode buffers measure 0.616720 ms versus 0.630448 ms without
preallocation. Fusing row matmul+RS regresses to 0.752589 ms; distributed norm
plus one-link AGMM regresses to 0.778513 ms. BF16 CCL is both more accurate and
faster than BFP8 (0.635099 ms). Thus no material CCL, placement, fusion, link,
payload, or buffer choice is left on documentation-only evidence.

## Optimize checklist

| Item | Disposition |
|---|---|
| OPT-001 packed QKV | already present and retained |
| OPT-002 SDPA/KV contract | contiguous and paged cache, BFP8/BF16 KV, repeated and advancing trace tested |
| OPT-003 residual chain | final local L1 shard preserved across decoder boundary; distributed-norm chain also measured |
| OPT-004 DRAM matmul geometry | role-specific 8/10/20/40 core and legal in0 variants measured; DRAM-sharded winner retained |
| OPT-005 logical batch | public batch remains 32; no tile padding leaks into the API |
| OPT-006 cumulative contracts | final default reran PCC, stress, capacity, Watcher, and profiler gates |
| OPT-007 attention precision | LoFi vs HiFi2 measured on real weights |
| OPT-008 row-parallel decomposition | separate matmul+RS, fused matmul-RS, and replicated provenance evidence compared |
| OPT-009 persistent CCL buffers | enabled after direct persistent/nonpersistent A/B |
| OPT-010 packed MLP | packed versus separate, split layouts, cores, and block widths measured |
| OPT-011 phase-specific shards | DRAM-interleaved prefill and role-specific L1 decode shards retained; coherent advisor family tested |
| OPT-012 real-weight authority | synthetic stress passes, but final selection uses prompt-derived real checkpoint activations |
| OPT-013 measured dtype proof | Tracy rows show BF16 activations, BFP4 weights, LoFi, BFP8 cache updates, and BF16 CCL |
| OPT-014 precision x geometry | selected geometry was rerun under attention/MLP fidelity and KV/CCL dtype changes |
| OPT-015 shard advisor | four fresh TP-rank captures, reports, final IR, executable translation, and measured rejection |

No applicable optimization is deferred. Async CCL, fused CCL paths,
collective placement, preallocated buffers, residual layout, activation
sharding, DRAM-sharded decode matmuls, precision/fidelity, and packed
projections all have executable evidence. Dense active-expert handling is not
applicable because Qwen3-32B is not MoE.

## Correctness, stress, and context

- Real layer-32 internal PCC: query 0.999937, SDPA/concat 0.999668,
  attention residual 0.999991, final prefill 0.999999954, final decode
  0.999972436, and all KV checkpoints above 0.99990.
- Logical sequence length 31 passes without public alignment requirements;
  the decoder owns padding, slicing, causal masking, and cache placement.
- Contiguous and permuted paged cache agree exactly. A captured paged trace
  remains exact after a fixed-address page-table refresh and after advancing
  position 64 to 65 (`results/paged_trace_refresh.json`).
- Runtime fallback audit passes with `throw_exception_on_fallback=true`; the
  final forward methods contain no `torch`, `from_torch`, `to_torch`, or
  superclass runtime fallback.
- The consolidated source-current gate is retained as `final_gate.xml`: four
  tests passed, covering the static fallback audit, length-31 and direct
  two-layer stress, real layer-32 checkpoints, and paged trace refresh.
- Physical full-stack capacity remains 12,352 logical tokens. It leaves
  28,539,904 bytes/device. Logical 12,353 pads to 12,416 and fails the
  prefill-live allocation: 253,624,320 bytes/bank requested versus
  245,194,624 free. See `doc/context_contract.json` and the two capacity JSONs.
- Watcher log SHA-256 is
  `ee439baa627976def5e7063cc7eba6ca8eaf764c519eb8fd72fb27b4a8df0b9f`
  with no fault matches. ETH instrumentation alone is disabled because it
  grows ring firmware to 27,920 bytes beyond the 25,600-byte buffer.

## Profiler result and advice disposition

The fresh raw Tracy CSV is
`tracy_final_reviewed/reports/2026_07_19_13_10_14/ops_perf_results_2026_07_19_13_10_14.csv`,
SHA-256
`0fb51eaf87262710217bbc0a6c2dde35468779f58e3f028c08ec5c8515cae86b`.
The advice-enabled human-readable `*.txt` tables, row CSVs, and summary CSV/PNG
files are retained under `tracy/`.

| Final profile | Dominant groups |
|---|---|
| Prefill | matmul 29.90%; reduce-scatter 8.76%; all-gather 8.09%; overall modeled DRAM roofline 10.5% |
| Decode, three traced replays | matmul 43.03%; reduce-scatter 7.99%; all-gather 4.16%; overall modeled DRAM roofline 22.1% |

The same profiler run reconciles the latency levels rather than equating
roofline utilization with end-to-end time:

| Phase | Theoretical modeled DRAM floor | Summed device-op time | Same-run wall time | Gap explanation |
|---|---:|---:|---:|---|
| Prefill | 0.214 ms (`2.035 * 10.5%`) | 2.035 ms | 4.932 ms | 2.800 ms of serialized op gaps plus 0.097 ms signpost/sync/profiler overhead; the remaining theory-to-device gap is unmodeled transforms/CCLs and matmuls at only 34-47% modeled roofline |
| Decode / replay | 0.119 ms (`(1.618 / 3) * 22.1%`) | 0.539 ms | 0.645 ms | 0.106 ms dispatch/dependency/sync gap; the merged four-rank gap column overlaps ranks and is not additive, while theory-to-device includes transforms, SDPA, CCLs, and 44-54% roofline matmuls |

Profiler instrumentation raises prefill wall time above the uninstrumented
3.372121 ms final gate; the roofline/device/wall comparison intentionally uses
only the same profiled run. Decode's same-run 0.645 ms is likewise distinct
from the 1,000-replay 0.616720 ms final latency.

`tt-perf-report` recommends tracing for host gaps; decode is already captured
and the final latency is trace replay. It recommends HiFi2/HiFi4 for accuracy;
separate attention and MLP HiFi2 trials were correct but slower. It reports no
output subblock for dominant matmuls because TTNN's DRAM-sharded program
factory does not expose one; the actionable core/in0 geometry, advisor 1-D
programs, packed rewrite, and fused alternatives were measured instead. Its
prefill L1-input recommendation was implemented on each bounded chunk and
rejected by the measured 3.464% regression above.

## Limitations

- The layer is intentionally fixed to batch 32 and a 1x4 p300c ring.
- The supported context is 12,352, below the HF-advertised 40,960, due to hard
  per-device DRAM evidence rather than an input-alignment restriction.
- The experimental fused AGMM family is retained only as an opt-in reproducer
  and uses one link; ordinary selected collectives use both available links.
- `tt-triage` ARC/device discovery was usable during hang diagnosis, but its
  worker/NoC/Ethernet readers were incompatible with installed UMD 0.9.5.

Commands and the full artifact ledger are in `work_log.md`; fused-hang repair
evidence is in `AUTOFIX.md`.
