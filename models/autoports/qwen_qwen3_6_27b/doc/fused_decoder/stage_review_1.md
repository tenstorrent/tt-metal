# Stage Review

Verdict: clean-pass

Independent review, round 1, of stage 2 (fused decoder), Qwen/Qwen3.6-27B.
Reviewed the live `main` worktree based on
`60c3c9e00911429d1ab6d4a519e8ae0c79be5c4a`. The delivered runtime SHA256 is
`6e2429f7440438719c1c3bd42e8dc12d34c22281018602d2eb00a291ac739a31`.

## Required Work

- None. The implementation and completed evidence satisfy the supplied stage contract. Local checkpoint commits and recording their SHAs remain the stage owner's post-review steps, as required by the stage-review skill. The run plan prohibits pushing; this reviewer made no commit or push.

## Other Concerns

- Preserve the documented serialization requirement when assembling later stages: convolution exchange scratch and semaphores belong to the layer, while recurrent/KV state belongs to requests. Concurrent forwards on the same layer instance are not validated. Prepared eager requests interleaved with one retained decode trace are validated.
- The B1/s1 native GDN prep/scan adapter uses checkout-specific internal bindings. Its current signatures, gate-tail masking, flat-head mapping and state-copy behavior match the inspected native implementation. Revalidate this adapter after upstream changes.
- The post-stop ideas in the work log are valid hand-offs, not measured rejections: native state-output alias/ring mode, residual epilogues, and broader dense sharding/geometry/precision work. The explicit stop condition is met, so these do not block this stage.
- Accuracy evidence uses real checkpoint weights and seeded BF16 input activations. Full-context HF comparison advances every cache/state token but checks selected output rows. This is the preserved functional-stage acceptance method, not evidence of full-model generation quality.

## Hard-Check Gaps

- `summarize_fused.py` verifies provenance for all 34 final JSONs but does not itself reassert every contract, mixed-request, trace-reuse and long-context numerical gate. Those gates are enforced by the individual runners and the fail-fast driver. I independently rechecked their saved values, including continuation negative controls; all pass. A future standalone summary should retain this distinction.
- The Python AST runtime audit does not enumerate calls through the local `native` alias in `_delta_chunks`, and cannot inspect C++ wrappers. I inspected the native prep/scan and public GDN paths, the convolution and rotary wrappers, reshape dispatch, and fused-cache validation. The native audit, runtime sentinels, successful tracked capture/replay, and final profiler rows cover the present delivery. The static operation count alone is not a complete native-runtime proof.
- Prefill wall-time rows are single warmed passes; 100-replay medians support the main decode comparison. Small differences between candidate prefill rows should not be interpreted as statistically resolved gains. The baseline-to-final prefill improvement is substantial across the recorded length matrix.

## Anomaly Ledger

- Observed anomaly: the first combined full-attention graph failed at high-position continuation with an L1/static-CB overlap.
  Evidence: `before_lifetime_fix_delivered_context_3_262143.log/.json`, `lifetime_fix_context_3_262143.json`, final `delivered_context_3_262143.json` and `delivered_context_3_262144.json`.
  Affected path: fused K/V update followed by large-context SDPA.
  Control or comparison: the earlier separate-update implementation supported the context; the final source releases the temporary L1 input list immediately after the ordered cache update.
  Likely subsystem: temporary tensor lifetime versus SDPA L1 reservation.
  Investigation performed: inspected the source lifetime and preserved failure; checked the targeted repair and fresh final-source context results, high-position continuation, and eager/traced decode equality.
  Resolution: fixed. No context, key-block, cache-dtype or acceptance-bar reduction.

- Observed anomaly: long-context GDN decode HF PCC decreases from 0.999986067 to 0.999797103.
  Evidence: functional full-context controls in `doc/context_contract.json` and `delivered_context_0_262143.json`; short direct functional output/contribution/state controls in final `run_decoder` artifacts.
  Affected path: accumulated recurrent arithmetic at context 262144.
  Control or comparison: both implementations use the same HF oracle method and threshold; the final selected prefill-contribution minimum is 0.999236844, and eager/traced decode is bit-equal.
  Likely subsystem: changed native convolution/GDN/normalization arithmetic and operation order; the exact delta has not been isolated to one fusion.
  Investigation performed: compared saved functional/fused values, reviewed the native arithmetic path and direct functional controls, and independently rechecked the unchanged 0.995 gates.
  Resolution: controlled within the original acceptance contract, with the unisolated numerical attribution retained as a residual risk. This is not a waived failing gate.

- Observed anomaly: allocation-tracker corruption warnings appear in correctness logs.
  Evidence: `run_decoder.py` positive-control block; final batch2/batch32 and Watcher results record `allocation_positive_control: true`.
  Affected path: deliberately unsafe allocation before replay, outside accepted model execution.
  Control or comparison: the runner requires replay rejection, frees that buffer, and then successfully replays.
  Likely subsystem: intended trace-allocation safety check.
  Investigation performed: inspected the positive control and settings (`TRACKING=1`, `SKIP_PROGRAM_CACHE=0`), plus successful retained-trace runs. No broad exemption or exact-buffer acknowledgment is present.
  Resolution: controlled.

- Observed anomaly: an early batch timing driver's discovery briefly overlapped the preceding pytest teardown.
  Evidence: work-log scheduling entry, final serialized driver, `final_functional_batch32_0.json`, and delivered batch32 timing results.
  Affected path: historical host-wall timing.
  Control or comparison: acceptance uses the remeasured functional row and final jobs serialized with `flock`.
  Likely subsystem: job scheduling.
  Investigation performed: checked the replacement baseline named by `batch32_performance.json` and the successful final driver sequence.
  Resolution: controlled; the overlapping row is not the accepted comparison.

- Observed anomaly: final logs contain topology warnings, optional Tracy viewer-copy failures, histogram overflow counts, and small AICLK settling differences.
  Evidence: final console logs; raw operation CSVs and `metrics.json`; four compressed Watcher logs.
  Affected path: device discovery/reporting and profiler environment, not a failed model gate.
  Control or comparison: topology warnings concern single-chip discovery; the optional viewer copy is separate from successful CSV/report generation; histogram overflow is a duration bucket count. The main unprofiled length 512 timing rows have no settling warning; final full-attention profile reports 1343 MHz and the length 511 timing row reports 1337 MHz against requested 1350 MHz.
  Likely subsystem: UMD labeling/clock settling and host profiler reporting.
  Investigation performed: searched final logs for fatal/error signatures, verified every raw CSV/Watcher hash, recomputed report totals, and compared profiled with unprofiled timing. No dropped-marker or Watcher error/assert/corruption signature was found.
  Resolution: controlled. The small clock variation further limits claims about sub-percent candidate differences.

## Scope Inspected

- Goal/skill paths: the supplied original stage contract; installed 0.1.23 `stage-review/SKILL.md`, `graph-fusing/SKILL.md`, `optimize/SKILL.md`, `optimize/references/perf-accounting.md`, `tt-enable-tracing/SKILL.md`, `tt-device-usage/SKILL.md`, and the model-bringup startup requirements. Plugin inventory confirms enabled tt-model-bringup 0.1.23 and tt-autodebug 0.1.8.
- Artifact paths: `doc/fused_decoder/README.md`, `work_log.md`, `pattern_assessment.md`, `AUTODEBUG.md`, `AUTOFIX.md`, `acceptance.json`, all 34 indexed final JSONs and their logs, `validate_final.sh`, `validate_final_driver.log`, `delivered_pytest.log`, final hook log, `runtime_audit.json`, `native_runtime_audit.md`, context contract, performance summaries, candidate ledger, measured-run/source archives, Watcher audit/raw logs, and four functional/final profiler directories.
- Code paths: complete `tt/fused_decoder.py`; functional runtime for comparison; `tests/run_decoder.py`, `run_contract.py`, `run_long_context.py`, `run_mixed_decode.py`, `run_trace_reuse.py`, `test_fused_decoder.py`, `evidence.py`, `summarize_fused.py`, `audit_runtime.py`, `render_perf.py`; selected candidate source archives; relevant native GDN, convolution, rotary, reshape and fused-cache sources; repository trace-correctness guide.
- Commands run: read-only `git diff/status/rev-parse`, `rg`, `cat`, `sed`, `nl`, `head/tail`, plugin inventory, and small standard-library Python scripts over existing artifacts. No TTNN import, device open, test execution, server, hardware reset, or implementation edit was performed. Only this review report was written.

The following conclusions were rederived from code and primary artifacts:

| Contract area | Independent result |
|---|---|
| Scope and provenance | Functional source remains unchanged at SHA256 `476f9d99fa90a1268f264964037dbc6837b320adc9f48d172dc1156b8b35962d`. Staged changes are confined to the target autoport. All 34 final results identify the delivered fused class, no candidate override, the current runtime hash, and the current runner hash. All 131 measured-run archive references resolve and hash correctly. |
| Runtime selection | The fused runtime imports no functional decoder. Tests explicitly select it. Functional execution in `run_decoder` is a separate numerical control outside measured windows. Mixed-request state dataclasses imported from the functional module contain tensors only and do not route execution through functional methods. |
| Correctness and boundaries | Both kinds pass HF and direct functional output/contribution/state PCC at 0.995. The 20-case contract runners cover 12 logical lengths, seven continuation starts with failing dropped-history controls, and mixed rows. Final batch 2 lengths 511/513, batch 32 length 33, and four-row mixed positions pass. |
| Context/capacity | Both kinds freshly execute every decoder token at 262143 and 262144, without diagnostic tail-only mode, chunk override or injected drains. Traced decode reaches context 262144. Full-attention high-position continuation passes. Layout/capacity changes are recorded without lowering supported context. |
| Trace and determinism | Tracked batch tests include repeated bit equality, changed input/position/RoPE, and an active allocation positive control. Each retained-trace runner uses one trace across six prepared requests, changed page maps and nonzero continuation starts, with zero request-time release/invalidation. Program-cache allocations remain included. |
| Watcher | Independently decompressed and hash-checked all four final raw logs (748615–981481 bytes); no error/assert/corruption/timeout signature found. Watcher and profiler regimes are separate. |
| Performance | Same-harness, same-reference, same-policy B1 length 512 prefill: 17.848329→5.564706 ms GDN and 7.087370 → 5.784396 ms full attention. Traced decode: 2.777220 → 2.132236 ms and 2.382522 → 2.041422 ms. Both modes also improve across 128/511/513/4097 and the recorded batch32 case. Final decode reproduces the fastest correct candidate family within 0.02%; no gain is assigned to that noise. |
| Profiler | Recomputed CSV kernel/gap totals and checked raw hashes for functional/final prefill and decode. Final op counts are 26/31 GDN and 27/36 full attention. Dense matmuls are BF16×BF16→BF16 HiFi4. No tilize/untilize/reshard appears in the signposted final windows; required norm transfers and fused K/V input shards are accounted for. |
| Stop condition | Active dense BF16 bytes independently agree with the packed projection/MLP shapes: 766771200 GDN and 744488960 full attention. Adding mandatory recurrent/KV traffic at 512 GB/s gives 1.509888 ms and 1.458184 ms, or 70.8124%/71.4298% of final unprofiled traced time. Profiled host/kernel/gap numbers are separately reconciled. Recomputed kept gains: GDN flat-Z view 0.2757%, logical gates 1.4046%; full fused-cache 0.2928%, HF rotary 0.1335%, subsequent combined norm 0.8091%. |
| Candidate decisions | Named graph-fusing patterns have measured outcomes or source-backed applicability decisions. Reviewed real-weight adapted controls include actual matmul activation fusion, packed MLP, decode head/concat layouts, native SwiGLU K-block 8/16/20, native GDN modes, and norm/core alternatives. Rejections are not based solely on a first API error or synthetic precision failure. |
| Completion evidence | Final driver records 35 zero-exit jobs, including all 34 result-producing runs and pytest; pytest records 4 passed. Final hook log passes. Post-review local commits remain intentionally pending under the required workflow. |

## Residual Risk

- This is an inspection review of recorded hardware evidence, not a hardware rerun.
- The nominal roofline is an estimate for the stated B1/context513 workload, not a claim of measured peak bandwidth, batch32 roofline, or full-model throughput.
- Full context was validated at batch1; batch32 evidence is for short requests. No batch32 full-context capability is established.
- The exact long-context GDN PCC change is not isolated to one fused operation. Full-model accuracy and real autoregressive generation remain later-stage gates.
- Shared scratch serialization, native binding compatibility, and generator-wide preparation of future execution signatures must be preserved when this decoder is integrated.
