# Optimized decoder work log

## Baseline and audit

- Starting commit: `199234c4b31`.
- Clean-pass implementation checkpoint: `d1b837b94ea`.
- Confirmed a single dense decoder-layer kind, packed QKV, packed gate/up, paged KV cache, LongRoPE, non-aligned prefill, and batch-1/batch-32 trace paths.
- Final same-harness functional baseline: prefill b1/b32 1.761852/37.714475 ms; traced decode b1/b32 1.049735/1.269351 ms.
- Profiled the functional topology before choosing precision, sharding, packed projections, SDPA, and program candidates.

## Candidate chronology

- Added an independent optimized prefill/decode implementation and a static audit preventing functional runtime fallback or host tensor conversion.
- Moved decode weights to DRAM width sharding and the residual/norm/MLP chain to L1 width sharding.
- Swept BFP8/BFP4 and LoFi/HiFi2. BFP4/LoFi met PCC and won.
- Swept final BFP4 policy over 8/16/32 cores at both batches. Chose 16.
- Adapted the 8-core experiment after its first RMSNorm subblock error. BFP8 then hit a precise 1,652,480-byte circular-buffer requirement over the 1,572,864-byte L1 limit; crossing with BFP4 made it legal, and it was then rejected by measured latency.
- Tried implicit prefill matmuls after the first API failure established that DRAM-sharded B weights require the explicit family. Added phase-specific interleaved prefill weights; this passed. Explicit large 2D prefill was also measured and rejected.
- Compared default versus explicit SDPA and BF16 versus BFP8 KV cache at both batches. Explicit SDPA and BFP8 cache win the joint criterion.
- Added the missing batch-32 prefill control and fixed its page-table batch dimension.
- Measured split QKV after adapting its unsupported sharded concat to a legal DRAM concat: correct but slower. Split gate/up won and replaced packed gate/up.
- Swept the next smaller legal block width independently for QKV, O, split gate/up, and down at both batches.
- Final optimized performance: prefill b1/b32 1.823820/30.657629 ms; b1 decode 0.481357 ms mean / 0.480407 ms min; b32 decode 0.658103 ms mean / 0.654058 ms min.

## Correctness and capacity

- PCC evidence is tabulated in `README.md`; every meaningful path exceeds the functional 0.995 bar.
- Ten trace replays are bitwise deterministic at batch 1 and 32. Split-QKV candidate PCC is 0.999996 at both batches.
- Tested seq 33/65, nonzero chunk crossing at 32769, exact shapes at 131071/131072, LongRoPE trace position 4096, prefill-to-decode cache transition, and decode at logical context 131072.
- Updated `context_contract.json` for BFP8 paged cache and optimized sharding without reducing advertised capability.

## Profiling and hardware

- Hardware: one Blackhole p300c from a healthy four-device host; all device-facing commands were serialized.
- Tracy report source: `generated/profiler/reports/2026_07_30_12_07_33/ops_perf_results_2026_07_30_12_07_33.csv`; it includes separate b1/b32 prefill and decode signposts.
- Profiler and watcher were run separately. `tt-perf-report` findings and roofline accounting are in `README.md`.
- Watcher command: `TT_METAL_WATCHER=10 pytest -q -s .../test_optimized_decoder.py`; 14 passed in 246.41 s, with pytest and watcher artifacts preserved.

## Optimize checklist

- [x] Baseline and candidate use the same real-weight harness, shapes, synchronization, warmup, and trace replay.
- [x] Topology audit covers repeated-input matmuls, packed projections, reshards, composite SDPA, and lower-movement replacements.
- [x] Precision/fidelity, sharding, DRAM matmul geometry, block widths, prefill programs, SDPA, and KV dtype were evaluated.
- [x] Batch 1 and serving batch 32 were swept independently; batch-1 is primary and batch-32 does not regress.
- [x] Final PCC, paged-cache semantics, determinism, LongRoPE, non-aligned lengths, chunk crossing, and advertised context are gated.
- [x] Final measured runtime has no host fallback or tensor conversion; remaining device layout operations are explained.
- [x] Warmed latency, Tracy device profile, `tt-perf-report` advice, device-time/e2e distinction, and bandwidth roofline are recorded.
- [x] Stress/repeated replay and watcher-clean correctness coverage exist.
- [x] Scope excludes multichip, full model, LM head, MoE, CCL, and serving.
- [x] Fresh stage review returned clean-pass after two evidence/documentation remediation rounds.
- [x] Local stage commit: `d1b837b94ea` (never pushed).
