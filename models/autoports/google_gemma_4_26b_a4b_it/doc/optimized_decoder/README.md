# Gemma-4 26B A4B optimized decoder

This stage provides the single-device, batch-1 optimized decoder in
`tt/optimized_decoder.py`. It starts from the Stage 01b fused decoder and does
not include multichip, full-model, LM-head, CCL, or vLLM work. Batch 32 is
explicitly out of scope for this MoE optimization stage and was neither swept
nor used as a completion gate.

## Result

The default is `bfp8_experts_lofi`: attention weights remain BF16 with the
inherited, profiler-verified HiFi2 linears; dense weights remain BF16/HiFi4;
packed expert weights use BFP8 with LoFi; the active-expert sparse
geometry uses four up/gate cores with K block 8 and eight down cores with K
block 11. Batch-1 decode keeps the residual stream in an eight-core L1
width-sharded layout. The first attention RMSNorm deliberately preserves the
fused DRAM-interleaved order because sharding that reduction missed the
sliding-layer HF PCC bar.

Sequence-1,024 warmed measurements are paired, same-process medians of five.
Decode numbers include TTNN trace replay. All results are logical batch 1.

| layer kind | fused prefill ms | optimized prefill ms | prefill change | fused traced decode ms | optimized traced decode ms | decode speedup | fused-vs-optimized decode PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| sliding, layer 0 | 400.921 | 239.641 | **40.23% faster** | 2.455 | 1.341 | **1.831x** | 0.999913 |
| full, layer 5 | 401.868 | 240.887 | **40.05% faster** | 2.669 | 1.534 | **1.740x** | 0.999892 |

Real-weight HF acceptance remains above the functional threshold for both
meaningful layer kinds. The final sliding layer records prefill PCC 0.998678
and decode PCC 0.999743; the full layer also passes the same 0.995 gates.
Reduced BFP8 DRAM-interleaved KV caches are supported and selected in the
optimized context contract: BF16-vs-BFP8 decode PCC is 0.999783 sliding and
0.999959 full. Their warmed decode medians are 1.331 ms and 1.530 ms.

## Operation-topology audit

| segment | measured/current operations | opportunity | action and evidence |
|---|---|---|---|
| attention | input RMSNorm, packed QKV, head split, Q/K norm, RoPE, paged cache update, paged SDPA, head concat, output linear | reduced projection precision, DRAM-sharded decode matmul, SDPA grid/chunks | attention BFP4 and sliding BFP8 failed the HF bar; legal DRAM-sharded K-block-1 QKV was 0.479x/0.487x versus interleaved and K-block-11 exceeded exact L1 capacity; SDPA configurations were swept and the common 8x4/q32/k64 point retained |
| residual/norm | fused graph repeatedly returned to DRAM around norms and residual adds | keep the residual chain sharded | selected eight-core width sharding (32x352 shards); only the numerically sensitive first input norm remains interleaved |
| dense MLP | same-input gate/up matmuls, GELU, multiply, down matmul | packed gate/up, lower precision, large prefill programs | packed `minimal_matmul_split` and BFP4 dense policies failed HF PCC; shape-derived large prefill configs were correct but regressed both layers by about 0.05%, so separate BF16/HiFi4 projections remain |
| router | RMSNorm, scale, FP32 linear, softmax, top-k/scatter | composite/lower-precision router | retained: routing order and scale are correctness-sensitive; the fused-stage topology already uses dedicated softmax/top-k |
| experts | packed sparse up/gate, activation-bearing reshape/slices, sparse down, score multiply/reduce | lower precision and phase-specific sparse geometry | selected BFP8/LoFi and 4-core/8-core K8/K11 independently for decode and prefill; BFP4 was faster in isolation but its interaction with the sharded path missed the final HF bar |
| cache | paged fill in prefill and fused paged update in decode | reduce KV storage and movement | BFP8 cache supported by device-side fill typecasts; no host conversion; paged semantics and modulo-tail integrity preserved |
| output | branch merge, RMSNorm, residual add, scalar multiply | retain width-sharded chain | adds and norms stay width-sharded until the final DRAM output |

The final signposted profiler windows contain 57 device-op rows in each
32-token prefill, 78 in sliding decode, and 80 in full decode. There are no
host ops, host fallbacks, or `Reshard` ops. Sliding decode has ten
interleaved-to-sharded and seven sharded-to-interleaved boundary operations
(8.96 and 7.14 microseconds total); full decode has eight and seven (8.31 and
8.76 microseconds). These cross dedicated attention/dense/router/sparse helper
contracts or enter/exit the persistent residual layout. They were counted in
the whole-layer comparison and are not hidden from the headline latency.
There is no runtime `torch`, `from_torch`, or `to_torch` in the owned hot
methods. Required interleaved/sharded conversions occur only at contracts for
attention, dense, router, and sparse helpers or at the residual-chain entry
and final output. Adds and all later norms avoid a DRAM residual round trip.

The current `tt-perf-report` identifies packed expert up/gate sparse matmul as
57.8-58.9% of prefill device time and 21.6-24.8% of decode device time. Sparse
down is 18.8-19.1% of prefill and below 10% of decode. These >15% paths were
attacked with BFP4/BFP8 fidelity sweeps and phase-specific geometry. Decode
used six named points per precision; selected-policy prefill independently
measured five safe core/K/subblock combinations at sequence 1,024.
The selected BFP8/LoFi geometry is the fastest correct final policy. Overall
modeled DRAM throughput is 125 GB/s (24.4% roofline) sliding decode and
132 GB/s (25.8%) full decode; prefill reports 110/112 GB/s (21.6/21.8%). Thus end-to-end
decode remains synchronization/compute/layout limited rather than approaching
the report's aggregate DRAM roofline. Separate device-trace replay measures
1.287 ms sliding and 1.478 ms full, consistent with but lower than host trace
medians because host dispatch/replay overhead is excluded.

## Semantics and capacity

- Real-weight prefill/decode and traced mutable-buffer behavior pass for
  sliding layer 0 and full layer 5.
- Logical non-aligned prefill lengths 31, 33, and 1,025 pass for both kinds;
  there is no public divisibility restriction.
- Paged KV cache behavior, shared/full cache views, and sliding modulo-tail
  integrity remain inherited and tested.
- Batch-1 decode at current position 262,143 passes for both layer kinds, so
  the advertised 262,144 context is unchanged. BFP8 cache reduces capacity
  pressure; no advertised capability was reduced.
- Both layers pass 101 trace replays with deterministic outputs.
- A separate `TT_METAL_WATCHER=10` run of HF acceptance and traced decode
  passes 4/4 with no watcher fault.

## Rejected and adapted candidates

- BFP4 attention: HF prefill PCC 0.938930 sliding and 0.980318 full.
- BFP8 attention: sliding HF decode PCC 0.994771; full passed.
- BFP4 dense gate/up, dense-all, and packed gate/up: representative sliding
  PCC 0.985881, 0.978582, and 0.985921 respectively; all rejected.
- Combined BFP8 attention/dense/experts HiFi2: sliding decode PCC 0.994776.
- BFP4 experts passed in an interleaved isolation but missed the final sliding
  HF bar when combined with the initially all-sharded norm chain. BFP8 experts
  also showed that the first sharded input norm was the interaction: restoring
  only that norm to the established interleaved reduction recovered the bar.
- DRAM-sharded decode QKV is legal at batch 1. The legal K-block-1 program was
  correct but more than 2x slower including layout boundaries. The next large
  legal block exceeded measured L1 capacity; it was not rejected on the first
  API error.
- Full-attention SDPA 8x8/q32/k128 was 0.4% faster in one sweep, while sliding
  regressed; the established 8x4/q32/k64 common configuration was retained as
  the difference was noise-level and not consistent across kinds.
- Large prefill dense programs were correct at PCC >0.99999 but regressed
  sequence-1,024 prefill for both kinds, so they remain disabled.
- Prefill sparse geometry was tuned separately from decode. The fused-equivalent
  4/8-core K1/K1 point measured 388.999/390.107 ms; 2/4-core K4/K2 and K11/K11
  were slower at 405-433 ms; 4/8-core K22/K22 measured 242.637/244.070 ms; and
  selected K8/K11 won at 239.602/240.936 ms with PCC 0.999952/0.999957 versus
  K1. This changes prefill only; decode retains its independently selected
  geometry.
- Safe 4/8-core K22/K22 BFP8/LoFi was measured after independent review. It
  passed, but K8/K11 remained faster: 1.339 vs 1.340 ms sliding and 1.536 vs
  1.539 ms full in the 21-replay geometry run.
- One-core packed sparse geometries are host-rejected in this checkout. The
  first such trial hung because the sparse multicast factory lacks the in0
  single-core `SKIP_MCAST` fix. Live triage, reset/recovery, AutoFix isolation,
  and the pre-dispatch guard are recorded in `AUTOTRIAGE.md` and `triage/`.

## Evidence map

- `precision_*.json`: paired fused/optimized sequence-1,024 PCC and timing.
- `geometry_*.json`, `prefill_sparse_geometry_*.json`, `sdpa_*.json`, `residual_chain_*.json`,
  `dram_qkv_*.json`, `large_prefill_*.json`: candidate matrices.
- `kv_bfp8_*.json`, `advertised_context_*.json`, `stress_trace_*.json`:
  cache, capacity, and repeated-run gates.
- `tracy/`: filtered op reports, summaries, and device-trace rows; raw Tracy
  captures were intentionally not retained.
- `watcher_clean.log`, `artifacts/watcher_provenance.json`,
  `artifacts/final_suite.log`, and `artifacts/profiler_provenance.json`: final
  source-bound gates and commands.
- `work_log.md`: chronological commands, decisions, checklist, and commits.
