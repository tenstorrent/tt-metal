# Stage Review

Verdict: clean-pass

Independent review of Stage 11, benchmark-model, Qwen/Qwen3.6-35B-A3B. Reviewed the live `main` worktree at `bfebe7b0d9816529ab59e13fa6a657c7cb38ef82`, including staged evidence. This verdict concerns measurement completeness, protocol, provenance and reporting. It assigns no accuracy acceptance verdict or score threshold.

## Required Work

- None. All six required performance rows, both selected upstream accuracy tasks, answer statistics, four full-phase accounting profiles and the evidence/context checks have completed evidence. The explicit third-attempt approval supersedes the default client budget. Its partial-result exception is unnecessary because all required measurements completed.

## Other Concerns

- No unresolved Stage 11 contract finding. Accuracy estimates cover only 14 MMLU-Pro questions and 18 IFEval questions; the report states their full populations and the unmatched published-reference protocol. The owner determines their usefulness.
- The enclosing runner's cached 0.1.24 checker still rejects the approved budget. This is disclosed, with its actual exit 2 retained. The repaired editable 0.1.26 checker passes and retains the measurement gates; no passing result is falsely attributed to the cached checker.
- Post-review local checkpoint and final clock refresh are the stage owner's prescribed finalization steps. At inspection the checkpoint is explicitly pending in `commits.json`; this review does not claim that the final commit already exists. The run plan records local commits only and no pushes.

## Hard-Check Gaps

- No missing required measurement/check found. `stage-check-post-hooks.log` reports benchmark evidence passed and zero critical context findings. The 61 advisory context matches include nested historical log quotations; all four observed servers retain `max_model_len=262144`, and the context-contract bytes match HEAD.
- The checker accepts an owner-approval record rather than independently authenticating the approver. For this review, the parent's explicit user-approval brief and `owner-approval.json` establish authorization for attempt 3, client 28800 seconds and setup 10800 seconds. Inspection of workflow commit `f287316` shows the budget guard was extended without removing measurement validation.
- Server-side byte traffic is modeled, as the contract permits; it is not a hardware-counter measurement. The evidence discloses expert-union/reuse assumptions, physical-row charging, omitted intermediate traffic and nominal peak assumptions. No stronger physical-traffic claim is accepted here.

## Anomaly Ledger

- Observed anomaly: Upstream `max_concurrent_requests` exceeds configured concurrency (2, 10, 18 and 34 for 1, 8, 16 and 32).
  Evidence: `attempt3/perf-b*.json`; `concurrency-metric-audit.md`; installed vLLM 0.26 `benchmarks/serve.py` and endpoint implementation.
  Affected path: Performance metric interpretation.
  Control or comparison: Reconstructed unrounded stream intervals peak at exactly 1, 8, 16 and 32; the upstream field counts requests touching inclusive one-second buckets. Client semaphore/connection limits retain the configured concurrency.
  Likely subsystem: Upstream bucket metric semantics.
  Investigation performed: Independently reconstructed all six stream-overlap peaks, checked the source-enforced bound and verified the report distinguishes requests from server slots.
  Resolution: controlled. Raw fields are retained unchanged and their interpretation is disclosed.

- Observed anomaly: Mean multi-user TPOT is much larger than median ITL; b32 reports about 18731.86 ms TPOT versus 431.17 ms median ITL.
  Evidence: Raw `ttfts`, `itls` and ordered phase events in the four profiles.
  Affected path: Interleaved prefill/decode serving latency.
  Control or comparison: Complete phase ledgers contain repeated long prefill intervals between decode intervals. At b32, prefill totals 8918.901485 seconds of the 9141.271260-second model-phase envelope.
  Likely subsystem: Serial prefill and replacement-request scheduling.
  Investigation performed: Independently recomputed TTFT, TPOT, ITL and E2EL mean/median/p95/p99 from raw stream timings, within 0.005 ms of saved metrics; reconstructed non-overlapping phase unions and all internal host gaps.
  Resolution: controlled. The report presents these measured costs; neither low utilization nor slow latency violates a Stage 11 threshold.

- Observed anomaly: Attempt 2 lost one IFEval response to a fixed 3600-second HTTP timeout.
  Evidence: Historical failure evidence, workflow timeout repair, `timeout-live-validation.json`, current IFEval document 59 and `accuracy-shared.log`.
  Affected path: Accuracy transport/completion.
  Control or comparison: Attempt 3's same frozen 18-question IFEval set completed in a shared accuracy pass of 3844.048666 seconds with the approved timeout propagated. Document 59 has a linked normal-stop response containing 16352 completion tokens and nonblank final content.
  Likely subsystem: Client timeout configuration.
  Investigation performed: Inspected deadline propagation and linked every current input/document to its raw response and scored sample. Historical partial scores are not used as current results.
  Resolution: fixed. All 32 current responses are present, normal-stop, nonblank and scored upstream.

- Observed anomaly: Normal EOF hooks could change byte-hashed raw performance JSON.
  Evidence: `.pre-commit-config.yaml`, `raw-before-hooks.json`, collector hashes and the staged blobs.
  Affected path: Raw artifact provenance.
  Control or comparison: Exact, narrow exemptions preserve the current upstream raw files.
  Likely subsystem: Repository formatting hooks.
  Investigation performed: Independently hashed all 12 current performance/warmup files in both the worktree and Git index and matched the frozen pre-hook hashes. All four accounting bindings and every referenced detail-piece hash also match.
  Resolution: controlled. Current raw bytes remain intact.

## Scope Inspected

- Goal/skill paths: `/home/ttuser/qwen36-35b-a3b/multigoal_logs_s11/11-11-benchmark.prompt.txt`; installed 0.1.24 `stage-review/SKILL.md`, `benchmark-model/SKILL.md`, `benchmark-model/references/run-contract.md`, `tt-device-usage/SKILL.md`, and model-bringup startup; the explicit approval and repaired workflow changes.
- Artifact paths: `doc/benchmark/identity.json`, `RUN_NOTES.md`, `doc/context_contract.json`; canonical `run/REPORT.md`, configuration, manifest, summary, checks, commits and links; `run/attempt3/` performance files, observed server configurations, raw responses, scored samples, request links, frozen input export and provenance; all four roofline indices and their detail pieces; relevant setup, pilot, shutdown, cleanup, clock, hook and check evidence. All 178 inspected symlinks resolve, and every local link in the canonical report exists.
- Code paths: `tt/benchmark_phases.py`; instrumentation in `tt/generator_vllm.py`; `tools/benchmark_server.py`, `collect_roofline.py`, `benchmark_work.py`, `benchmark_recover_scores.py`; launcher and EOF-hook changes; repaired workflow evaluation/plan/check/run sources; installed client metric/random-dataset sources; pinned plugin pending-output drain and existing read-event completion boundary. No serving source differs from baseline `18527ae0` in the inspected serving paths; the plugin checkout is clean.
- Commands run: Read-only `cat`, `sed`, `rg`, `git status/log/show/diff`, and bounded Python standard-library JSON/hash/arithmetic analyses. No server, model import, inference, device operation, reservation, reset, new scorer run or long test was run. This report is the only file written by the reviewer.
- Independent accuracy reconstruction: All 32 document hashes and request-instance identities match the manifest/export; each final response equals its own scored sample. MMLU-Pro is 11/14 = 78.5714%; IFEval prompt strict/loose is 15/18 = 83.3333%, and instruction strict/loose is 29/32 = 90.625%. Extraction is complete with no blank-at-cap responses. Nearest-rank median/p90/max token counts reproduce 1521/2295/2560 and 2438/6025/16352. Read actual final outputs from every response, plus retained reasoning examples; no transport/template/scoring defect was identified. Benchmark answer errors remain upstream scores, not grounds for an accuracy gate.
- Protocol reconstruction: Structured MMLU contexts have a system message, five user/assistant demonstrations and the final user question; IFEval has one user message. No native template control wrapping appears in those inputs. Server native-template application is supported by the client path, pinned server/tokenizer identity and pilot token-count comparison. Generation settings match the pinned model README's general-task thinking recipe. The pinned README also contains the 85.2 MMLU-Pro reference.
- Independent performance reconstruction: Counts 8/24/48/96 at 4096/128 plus 4 at 128/128 and 2 at 32768/128: 182 completed, zero failed, exact per-request token counts, 127 inter-token intervals each. Throughput and latency statistics agree with raw data. Warmups remain separate setup artifacts.
- Independent accounting reconstruction: All 176 required-profile lifetimes have contiguous 4096-token prefills and exactly positions 4096 through 4222 for decode. Complete call counts are 1032/440/496/608 for b1/b8/b16/b32. No unfinished or cross-phase-overlapping calls were found. Recomputed prefill/decode seconds are 671.012541/22.195171, 2120.819513/53.989998, 4283.864747/106.186722 and 8918.901485/222.369775. Term sums and bindings agree. Percentages reproduce 0.234900/10.462950, 0.222962/9.706056, 0.220765/9.434855 and 0.212072/8.687195. Decode completion is after the plugin's existing read-event synchronization; instrumentation adds no device synchronization or live profiling.
- Clock/cleanup reconstruction: Preserved runner client 21806.454038 seconds plus continuous elapsed time from its final-summary anchor reconciles exactly with the reported finalization clock; setup totals 7209.513573 seconds. Both remain below approved limits during this review. All four shutdown records show owned-process identity and prior idle checks; final cleanup records empty device holders, no reset and the unchanged hub-owned reservation.

## Residual Risk

- Small frozen samples, differing reference recipes and sampled reasoning limit statistical interpretation. This is a benchmark report, not an accuracy certification or matched TT-versus-HF quality judgment.
- Roofline estimates remain conditional on the documented source-derived work and nominal hardware models. Their arithmetic and attachment to the measured execution were checked; actual physical transactions were not measured.
- Raw logs/full snapshots omitted from Git are listed for persistent local retention; required compact measurements, responses and accounting pieces are staged. Portability of those additional local-only diagnostics is not claimed.
- The final clock must continue through the owner's finalization and local commit, as already implemented by `finalize_report.py`. No new measurement or review round is needed solely for that prescribed bookkeeping.
