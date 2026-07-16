# Stage Review

Verdict: clean-pass

## Required Work

None.

## Other Concerns

- The mandatory advisor IR is intentionally a geometry seed rather than a reconstruction of the selected final policy: it records BFP8 attention/down weights and the original 11x8, `in0_block_w=2` down projection, while the selected runtime uses all-BFP4/LoFi weights and an 11x6, `in0_block_w=4` down projection. This is non-blocking because `shard_advise/advise_llama70b.py` now freezes the exact saved-IR policy, the work log explicitly distinguishes advisor capture from later precision/geometry sweeps, and the final hardware profile proves the selected runtime policy.
- `shard_advise/final_ir.mlir` retains one `ttnn.validation_unfixable` marker on `nlp_concat_heads_decode` because the advisor leaves SDPA output DRAM-interleaved while concat-heads requires sharded input. The runtime contains the targeted conversion, its cost appears in the final profile, and the lower-movement exact advisor residual chain was measured and rejected, so the marker does not invalidate the applied advisor seed.
- The final batch-1 BF16/BFP8 cache comparison is extremely close in decode (1.845752 versus 1.846026 ms). Five prefill repetitions and 500 trace replays make the selected BF16 primary policy reasonable, while the stage correctly preserves caller-owned BFP8 caches as the faster batch-32 throughput policy.

## Hard-Check Gaps

- None within the optimized single-device decoder scope. Per the review task, I did not run hardware, reset devices, start a server, or exercise full-model/generation/vLLM paths. Those paths are explicitly outside this stage; the review used the frozen real-device JUnit, watcher, Tracy, and advisor artifacts.

## Anomaly Ledger

- Observed anomaly: The early DRAM-width-sharded prefill candidate produced PCC 0.000324 with infinities.
  Evidence: `AUTODEBUG.md` and `work_log.md` identify disagreement between the eight-bank weight shards and the 11-column 2-D program's `per_core_N`.
  Affected path: Experimental DRAM-sharded prefill projection family.
  Control or comparison: The functional path and DRAM-interleaved prefill weights remain finite and pass real-weight PCC; the repaired DRAM family reached 0.9999796 prefill PCC.
  Likely subsystem: DRAM weight-shard/program geometry compatibility.
  Investigation performed: Fresh-context AutoDebug, source comparison to in-tree DRAM-sharded matmul warnings, isolated interleaved-weight control, and corrected L1 block trials.
  Resolution: fixed. Prefill uses coherent DRAM-interleaved weights with explicit 2-D configs; the selected advisor path shares interleaved weights across phases.

- Observed anomaly: All-BFP4 attention materially reduces raw appended K/V PCC from the BF16/BFP8-attention neighborhood to approximately 0.993.
  Evidence: The final 8-step JUnit records minima of 0.9999541473 output, 0.9927964013 K, and 0.9933396936 V; the final default PCC remains above the stage's 0.99 bar.
  Affected path: Prefill-to-decode cache transition and recurrent decode positions 18 through 25.
  Control or comparison: Higher-precision attention has K/V PCC near 0.99985 but slower traced decode; BFP4/LoFi is the faster projection policy.
  Likely subsystem: Expected BFP4 attention-projection quantization, not cache-indexing corruption.
  Investigation performed: Real-weight one-step output/cache checks, eight cache-consuming decode steps, five traced replays with input refresh, non-aligned sequence coverage, and a BFP8-cache cross-check.
  Resolution: controlled. Every recurrent output and cache append remains above the functional 0.99 contract, and the final profile verifies the intended BFP4/LoFi runtime rows.

- Observed anomaly: Full-phase packed gate/up exceeds Blackhole L1, while fewer-op decode-only packing is correct but does not win latency.
  Evidence: `packed_prefill_resource_failure.xml` records 2,263,296 bytes of circular buffers versus 1,572,864 bytes available. The best adapted decode-only policy records PCC 0.9999539804/0.9927964013/0.9935370450 and measures 2.122008 ms versus 2.120094 ms split over 500 replays.
  Affected path: MLP gate/up projection packing.
  Control or comparison: Split gate/up BFP4/LoFi on the same 11x8 checkpoint topology.
  Likely subsystem: Packed matmul L1 allocation and packed-output slicing/layout overhead.
  Investigation performed: Full-phase 10/11-column trials followed by phase-decoupled 100-core block-2, 100-core block-1, and 106-core decode-only adaptations, real-weight correctness, 50-replay screening, and 500-replay close-call adjudication.
  Resolution: controlled. Split gate/up is the measured whole-layer winner; rejection is not based on the first allocation error.

- Observed anomaly: The advisor-seed down projection was a dominant slow row at roughly 589 us per replay.
  Evidence: Final-family sweeps cover blocks 2/4/7/8/14/16 and grids 11x8/11x6/11x5 under all-BFP4/LoFi. The selected final profile reports roughly 423 us with 64 cores, `in0_block_w=4`, `per_core_N=4`, and a 1x4 subblock.
  Affected path: Decode MLP down projection.
  Control or comparison: The 11x8/block-4 control measures 2.120273 ms; 11x6/block-4 measures 2.114013 ms over 500 replays.
  Likely subsystem: One-dimensional matmul grid/block geometry.
  Investigation performed: Precision-locked block/grid sweep, 500-replay final adjudication, 8-step recurrent correctness, and final Tracy reproduction.
  Resolution: fixed. The selected 11x6/block-4 topology is wired into `OptimizationConfig` and appears in final profiler rows.

- Observed anomaly: An adapted fused cache-update/head split triggered a watcher NoC sanitizer fault after the initial overlapping split failed validation.
  Evidence: The topology audit records both the initial validator failure and the adapted non-overlapping, head-aligned retry fault in `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`.
  Affected path: Fused decode K/V update experiment.
  Control or comparison: Two split `paged_update_cache` operations pass correctness, trace replay, recurrence, and the final watcher gate.
  Likely subsystem: Blackhole fused head-creation/cache-update NoC contract.
  Investigation performed: Split adaptation beyond the first API failure, watcher-sanitized hardware trial, recovery, and retained split-path profile accounting.
  Resolution: controlled. The split update is the supported final topology and its two cache rows are present in the final profile.

## Scope Inspected

- Goal/skill paths: `.agents/prompts/model_bringup_multigoal/02-optimized-decoder.txt`, its hard-gate script, and the complete `$stage-review`, `$optimize`, `$graph-rewrite`, `$shard-advise`, and `$tt-device-usage` instructions applicable to this stage.
- Artifact paths: `doc/optimized_decoder/{README.md,work_log.md,AUTODEBUG.md,watcher_test_report.xml,watcher.log}`, `candidate_evidence/*.xml`, `candidate_evidence/README.md`, `tracy/*`, and `shard_advise/{report.json,report.txt,final_ir.mlir,advise_llama70b.py}`.
- Code paths: `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`, the functional decoder/reference tests used as controls, and `doc/context_contract.json`.
- Commands run: read-only source/doc inspection; JUnit status/property parsing; raw/processed Tracy row inspection and device-time summation; final-policy/program-config comparison; `python -m py_compile`; and the optimized-decoder shard-advisor hard-gate script. No device command was run by the reviewer.

## Residual Risk

- Correctness evidence is decoder-layer PCC and short recurrent cache evidence, not generated-text or full-stack quality. Full-model, qualitative, long-context, multichip, generator, and serving validation remain correctly deferred to later stages.
- The final profile is batch 32, while the primary batch-1 signoff uses the same 5/500 traced timing harness without a separate batch-1 Tracy capture. This is acceptable here because the selected program geometry is explicitly verified in the final batch-32 profile and the frozen batch-1 end-to-end artifacts reproduce the final default, but later full-model work should profile its actual single-user stack.
- The 0.30% 11x6-versus-11x8 batch-32 win and near-tied batch-1 cache decode result are small. The retained 500-replay controls reduce timing noise, but later integration should recheck the cumulative winner in the full-model execution environment.
- Stage-owned commit creation and work-log SHA recording occur after this independent review and were therefore not part of the evidence verdict.
