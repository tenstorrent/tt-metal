# Stage Review

Verdict: clean-pass

Independent first-round review of Stage 11, Qwen/Qwen3.8-Flash-Next. This verdict concerns benchmark measurement, protocol and evidence integrity only. It makes no model-accuracy acceptance judgment and imposes no score or roofline threshold. The explicit user contract supersedes generic quality gates and does not reopen unchanged Stage 9/10 capabilities.

## Required Work

- None. No missing, contradictory or invalid required measurement was found.

## Other Concerns

- Each accuracy task has one frozen document. This sharply limits inference about either population; the report explicitly states the denominators and unmatched published comparison. The count follows the measured pilot, fixed seed/ranking and required performance reserve. It is not a score-based selection or a model-ability verdict.
- The reported rooflines are qualified host-observation and source-accounting estimates. Completion polling uncertainty is substantial, and the DRAM ledger is not a transaction counter. The report retains these qualifications beside the timing bounds and accounting definitions. No minimum utilization is required.
- The server has two configured slots and receives concurrency two, but all 1016 measured decode events have one active sequence. The report states this and does not claim an observed device batch of two.
- The final post-review clock entry, evidence packaging and isolated local checkpoint commit are the stage owner's planned closure steps. At 2026-11-18T05:38:16.000272+00:00, elapsed client time including review was 8049.617510 / 9000 seconds, calculated from invocation UTC minus recorded setup time. The runner itself recorded 7627.747302 client seconds and 2503.071863 / 3600 setup seconds. Preserve the final clock and local-only push policy when closing.

## Hard-Check Gaps

- The evidence checker validates structure, hashes and statistics, but cannot independently prove the physical DRAM model or exact phase occupancy. This review inspected the accounting and completion-observer implementation, reconstructed the selected events and work ledger, and verified the stated qualifications. A device profiler or a new runtime-counter requirement would exceed the supplied contract.
- The upstream client's raw `max_concurrent_requests` is an inclusive one-second-bin statistic, not exact simultaneous HTTP concurrency. It is 3/2/2 in the B2/functional/long-context raw files. An independent sweep of request start and reconstructed finish times gives 2/1/1, matching the requested limits. The main report uses configured concurrency and separate server slots; it does not promote that coarse raw field as an exact peak.

## Anomaly Ledger

- Observed anomaly: GPQA strict extraction is 0/1 while flexible extraction is 1/1.
  Evidence: `gpqa_diamond_cot_zeroshot/results.json`, `samples_gpqa_diamond_cot_zeroshot.jsonl`, `responses.jsonl`, `protocol-review.json` and the retained two-document pilot.
  Affected path: upstream strict answer-format filter.
  Control or comparison: the prompt requests step-by-step reasoning without the strict filter's literal phrase; final content explicitly supplies `(A) 0.85`, the flexible filter returns `(A)`, and the response stops normally. Pilot strict/flexible extraction was 0/2 versus 2/2.
  Likely subsystem: upstream prompt/filter recipe mismatch.
  Investigation performed: read the actual prompt, final answer, reasoning separation, strict regex and both unmodified scored sample rows. Both upstream scores and the one-document denominator remain intact; the report labels the strict row as a format diagnostic.
  Resolution: controlled; documented advisory, not a missing statistic or an accuracy failure gate.

- Observed anomaly: nominal phase attribution has wide completion bounds, including conservative summed upper bounds larger than client wall time.
  Evidence: `phase-raw-b2.json`, `phase-accounting-b2.json`, `b2-measurement-audit.json` and `REPORT.md`.
  Affected path: nonblocking host observation of asynchronous CQ0 events.
  Control or comparison: 5112 native event IDs strictly increase on one mesh/CQ; selected intervals are contiguous and nonoverlapping. Their sum is 2030.503519406 seconds versus 2030.509355852 client seconds, a 0.005836446-second difference at the HTTP boundaries.
  Likely subsystem: delayed observer polling rather than overlapping request-latency summation.
  Investigation performed: reconstructed request selection, exact prompt mapping, native FIFO validation, every interval, phase totals and the accounting ledger from the raw snapshot; each reconstructed object equals the retained artifact. The maximum poll gap is 34.357589 seconds. The report expressly calls the phase split nominal attribution and explains why conservative upper bounds need not sum to the wall interval.
  Resolution: controlled; valid qualified estimates, not precise device occupancy claims.

- Observed anomaly: raw upstream peak concurrency exceeds configured HTTP concurrency.
  Evidence: the three measured `perf*.json` files and `/home/ttuser/qwen38-flash/vllm-benchmark-client/vllm/benchmarks/serve.py:674` through the inclusive second-bin accumulation.
  Affected path: upstream derived peak statistic.
  Control or comparison: exact interval sweep gives peak two for B2 and one for each lone-user row.
  Likely subsystem: consecutive requests share the same inclusive integer-second bucket.
  Investigation performed: inspected upstream calculation and independently swept the saved starts, TTFTs and ITLs.
  Resolution: controlled; no evidence of an over-concurrency measurement.

- Observed anomaly: setup pilots initially had an incorrect token length and a failed native event export.
  Evidence: retained setup diagnostics, `AUTOFIX_token_transport.md`, `AUTODEBUG_event_abi.md`, `setup/abi-fixed/native-smoke-proof.json`, `collector.log` and `upstream-validation.json`.
  Affected path: client text retokenization and the model-local native query binding.
  Control or comparison: the replacement native smoke emits one prefill and three decode completions; the complete 4096/128 collector pilot passes before accuracy. Final measured prompts are exact token-ID payloads, with all eight B2 requests linked to runtime hashes.
  Likely subsystem: tokenizer transport mismatch and nanobind ABI mismatch.
  Investigation performed: inspected the token wrapper, native binding, live pilot evidence and final raw event stream. Final exports contain no errors or pending tickets.
  Resolution: fixed for the measured run; failed diagnostic evidence remains separate.

- Observed anomaly: a generic active-trace allocation warning appears during initial state-copy trace preparation.
  Evidence: final server log near line 2087, allocator warning implementation, unchanged Stage 10 generator/model/factored-decode sources, and Stage 10's `final_trace_async_summary.json` and accepted review.
  Affected path: inherited startup trace preparation.
  Control or comparison: the warning is also retained in earlier Stage 10 logs; the underlying preparation and model paths are unchanged. Stage 10 supplies tracked synchronous/asynchronous replay evidence. Stage 11 adds host observer bookkeeping and completion markers, without new tensor allocation or capture bodies.
  Likely subsystem: generic allocator diagnostic emitted with live traces.
  Investigation performed: located the warning between existing state-copy captures, read its allocator condition, checked the launch settings and compared model source with the Stage 10 parent.
  Resolution: controlled within the unchanged prior-stage evidence boundary; no fresh Stage 9/10 gate imposed.

- Observed anomaly: CPU performance-client teardown emits nanobind reference warnings; server shutdown force-terminates the remaining EngineCore.
  Evidence: performance client logs, `RUN_NOTES.md`, final server log, `setup/b2-1794966797189440608/cleanup.json`, and `doc/optimized_vllm/owner_decision_shutdown_2026_11_17.json`.
  Affected path: client interpreter teardown and the inherited server cleanup path.
  Control or comparison: all required HTTP requests complete with exact lengths and successful client exits. Before server shutdown both request queues are zero; final observer export has 44644 events, no errors and no pending tickets. Cleanup retains empty owned-process and device-holder inventories.
  Likely subsystem: client binding teardown and owner-accepted incomplete orderly server teardown, respectively.
  Investigation performed: distinguished these warnings from the repaired observer ABI failure and checked the accepted shutdown decision. The unsuccessful CPU help-command control is not treated as a reproduction or fix.
  Resolution: controlled for benchmark evidence; neither a claim of repaired client bindings nor complete orderly trace release.

## Scope Inspected

- Goal/skill paths: `setup/review-contract.txt`; installed tt-model-bringup 0.1.24 `stage-review/SKILL.md`, `benchmark-model/SKILL.md`, `benchmark-model/references/run-contract.md`, `tt-device-usage/SKILL.md` and the model-bringup startup instructions.
- Artifact paths: `REPORT.md`, `summary.json`, `run_config.json`, `manifest.json`, identity/configuration snapshots, protocol and B2 audits, all three measured performance files and exact-token companions, all three warmups, task results/request links/scored samples/raw responses, both phase artifacts, roofline data, evidence/context/check/source-integrity records, invocation/setup clocks, sample-sizing and pilot evidence, final export/cleanup records, root notes and evidence instructions. Also inspected the pinned local model card and relevant unchanged Stage 10 serving/trace/cleanup evidence.
- Code paths: model-local `benchmark_client.py`, `benchmark_server.py`, `benchmark_roofline.py`, `benchmark_accounting.py`, `tt/benchmark_phases.py`, `tt/benchmark_native/event_query.cpp`, observer integration in `tt/generator_vllm.py`; installed benchmark-stage subset/evaluator code; upstream vLLM 0.26 peak-statistic calculation; allocator warning implementation.
- Commands run: bounded `cat`, `sed`, `rg`, file listings, `git status`/targeted `git diff`, UTC clock reads and short CPU-only Python analyses over retained artifacts. No device access, server launch, live request, reset, reservation operation, long test, implementation edit or additional agent.
- Independently rederived: 8/8 B2 requests at 4096/128; 4/4 functional at 128/128; 2/2 long-context at 32768/128; zero failures; distinct exact prompts; warmup completion; throughput and mean/median/p95/p99 TTFT/TPOT/ITL/E2EL, with only sub-0.002-ms reconstruction differences at independently recorded stream boundaries. Summary metrics equal their raw performance fields.
- Independently rederived: all eight source hashes; configuration, server identity, performance and phase evidence hashes; manifest digest/document hashes and common-population identity; fixed leading reduction within the pilot ranking; final-content equality between API responses and upstream scoring inputs; normal stops and completion-token statistics. Both actual final answers and the separated reasoning fields were inspected.
- Independently rederived: 4096 eight-token prefill chunks, 32768 executed logical rows, 1016 consecutive decode positions, eight uniquely mapped requests, 1920.376579191 prefill seconds and 110.126940215 decode seconds. Work totals are 466552499470336 modeled prefill FLOPs and 23890985205760 modeled decode bytes. The stated formula gives 0.02057199145% and 10.59279418571%, matching the report.
- Push state: stage plan authorizes isolated local checkpoint commits only; no push is part of this review or the stage closure.

## Residual Risk

- This review validates the delivered measurement contract, not population accuracy, comprehensive output quality, production readiness or new serving capacity. One sample per task and the unmatched published recipe remain explicit limitations.
- Event polling, added completion-notification overhead, fidelity normalization and omitted physical DRAM transactions limit roofline interpretation. No profiler-derived precision is claimed.
- Archive restoration and the final local commit occur after this review. The parent must preserve the reviewed raw evidence and record the final elapsed client clock; no additional measurement is required by this verdict.
