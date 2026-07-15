# Stage 07 Anomaly Ledger

## Multi-tile terminal shard mismatch

- Observed anomaly: the first full AIME24 prefill run failed with `Shard height 32 must match physical height 256`.
- Evidence: `run_prefill_check_pre_autofix.log` and `AUTODEBUG_PREFILL_LM_HEAD.md`.
- Affected path: readiness/full-logit terminal projection at logical M=249; decode M=1 was unaffected.
- Control or comparison: selected DRAM-sharded factory and source validation require one M tile; dynamic multi-M program configuration is not legal.
- Likely subsystem: LM-head input layout/program contract.
- Investigation performed: `$autodebug` source/contract analysis followed by isolated `$autofix` hypothesis and target-hardware regressions.
- Resolution: fixed. Logical 1–32-row normalized tiles reuse the selected projection; 33-row reduced hardware and exact 249-row full readiness pass.

## Active-trace allocation warning

- Observed anomaly: TT Metal emits its conservative `Allocating device buffers is unsafe due to the existence of an active trace` warning while the cooperating sampler trace is registered.
- Evidence: `run_teacher_forcing.log`, Stage 06 `../full_model/README.md`, and the repo-root `AUTOTRIAGE.md` disposition.
- Affected path: split model/sampler trace setup, not steady replay.
- Control or comparison: sampler outputs, local/gathered pairs, parameters, token feedback, logits, position/RoPE, and page tables are persistent and preallocated; repeated replay, reset/recreate, changed tables, normal functional, scoped watcher, and profiler paths pass.
- Likely subsystem: TT Metal trace-region registration warning, triggered because the model trace already exists when the second region is captured.
- Investigation performed: Stage 06 live triage removed actual capture-time sampler allocations and bound all-gather to a preallocated output; Stage 07 preserved that implementation and reran the controls.
- Resolution: controlled. No buffer is allocated in steady replay, and trace counters show no host fallback or stale input.

## Full Ethernet watcher instrumentation overflow

- Observed anomaly: enabling watcher on active Ethernet fabric expands the fabric program to 27,792 bytes, above the 25,600-byte active ETH configuration buffer before model execution.
- Evidence: initial watcher transcript retained in the stage run history; passing `watcher_reduced_functional.xml` uses worker watcher with `TT_METAL_WATCHER_DISABLE_ETH=1`.
- Affected path: instrumentation build only; no model kernel executes before the failure.
- Control or comparison: the same reduced full-model path passes normal execution and scoped worker watcher.
- Likely subsystem: watcher instrumentation footprint versus active ETH firmware configuration.
- Investigation performed: `$tt-device-usage` classification separated instrumentation capacity from model/runtime behavior and reran the safe worker scope.
- Resolution: controlled physical instrumentation limit; not a model failure or waived worker assertion.

## Cross-device `tt-perf-report` chronology gap

- Observed anomaly: merged report shows a 667.9 ms op-to-op gap between operations from different device streams.
- Evidence: `tt_perf_report.txt` and the losslessly compressed `profiler_raw_ops.csv.gz`.
- Affected path: CSV merge chronology, not the summed device operations or end-to-end trace benchmark.
- Control or comparison: selected signpost window has 155 device ops/2.833 ms summed device work; unprofiled warmed trace benchmarks reproduce; trace counters show zero per-token host sync/readback.
- Likely subsystem: asynchronous per-device timestamp merge in the reporting tool.
- Investigation performed: compared raw per-device ordering, signpost boundaries, device-time sums, and end-to-end source-current benchmark; the gap is not used as a latency claim.
- Resolution: controlled reporting artifact. Device-op tables and host end-to-end timing are reported separately.

## Base-checkpoint prompt behavior

- Observed anomaly: the initially selected eight-input-shard/block-3 LM-head geometry changed the Fibonacci control at token zero; independent base-checkpoint prompts also exhibit corpus-autocomplete behavior and repetition in some exact-HF controls.
- Evidence: `qualitative/vllm_qualitative_outputs.json`, `qualitative/verdict.md`, and `autoregressive/`.
- Affected path: prompt-based generated text.
- Control or comparison: same exact HF revision and tokenizer rendering; `GemmaTokenizer.chat_template=None`.
- Likely subsystem: LM-head accumulation geometry for the block-3 regression; absent chat template/base-checkpoint training distribution for shared HF/TT corpus completion behavior.
- Investigation performed: prompt-format metadata, rendered token IDs, HF/TT side-by-side output, teacher-forced top-k, and mechanical degeneration checks.
- Resolution: fixed. Four-input-shard/block-2 restores every Stage 06 TT prompt exactly and is bit-identical to the legacy LM head across all aligned logits. Mechanical checks classify the remaining repeated phrases as HF-mirrored corpus completion, not a TT-only degeneration loop; the autoregressive story has zero adjacent repetitions and zero repeated trigrams.
