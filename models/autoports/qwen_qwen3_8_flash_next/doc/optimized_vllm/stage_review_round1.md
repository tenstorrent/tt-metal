# Stage Review

Verdict: more-work-needed

Independent review, round 1 of at most 3, stage 10 optimized-vLLM for Qwen/Qwen3.8-Flash-Next. Inspected the live workspace at main `ed9fd9c4f76adf08e7c69d7cb00edff079c6f5fd` plus the supplied uncommitted evidence and diagnostic-client correction. No server, device operation, TTNN import, reset, implementation edit, or new experiment was performed.

## Required Work

- P2: The shipped server launcher still silently selects budget 256 instead of the accepted budget 9.
  Evidence: `doc/optimized_vllm/run_server.sh:24` expands `--max-num-batched-tokens ${TOKEN_BUDGET:-256}`. With that variable unset, ordinary shell expansion selects 256. `chunk_selection.json` selects 9 under the 1000 ms interference bound; the README describes budget 9 as the serving configuration. The successful measurement, replay, and acceptance drivers explicitly supply `TOKEN_BUDGET=9`, so their evidence is valid, but it does not validate the launcher's fallback. The three driver entrypoints require `--budget`; a direct invocation of the README-linked `run_server.sh` can silently run a different configuration.
  Why this matters: the delivered default launch does not reproduce the selected configuration or its mixed-traffic acceptance. This is executable configuration wiring, not a request for additional optional tuning.
  Required next step: change the launcher's fallback to 9, or require an explicit budget with a clear failure when absent. Preserve explicit overrides for historical measurements and sweeps. Verify the resolved argument without starting a server, and record that the accepted device evidence already used explicit budget 9. Do not rewrite historical source hashes. A small diff-only follow-up review is sufficient; this correction does not require repeating unchanged hardware gates.

## Other Concerns

- The four raw sampling failures are correctly retained as failures and may be separated under the installed vllm-integration skill's explicit reproducibility-only rule. I independently counted 142 cases, 63 passes, 75 skips, four failures, and zero errors in the original XML. Current standalone repeat/row-swap full-vocabulary checks pass, and I independently compared all 400 served/native token-step top20 maps and histories: every map and selected token is exactly equal. This supports the documented acceptance scope; it does not prove the original stochastic prefixes, full served logits, or arbitrary active-width equivalence. No new HF-same claim is justified for these four families. The existing summary correctly says so.
- Eval readiness is a pass of critical execution checks with explicit limitations, not release readiness. Raw P4 request records contain 32 completions and 37,315 output tokens, median 428.5; only three requests reached 6000 tokens. The 120.104851-hour planning estimate correctly uses actual wall capacity, not the packaged per-active-user rate multiplied by 32. Preserve the three advisory entries and the lack of a resident-B2 theoretical floor. The owner's earlier B1 floor acceptance does not establish a B2 floor.
- Slow/incomplete orderly teardown is accepted only under the supplied owner decision. Cleanup reports show no owned processes/device handles, while trace-release counts and forced worker termination remain visible. The disabled native profiler helper is not proven to cause these particular interrupted releases. The stage documents that distinction correctly; no plugin change is requested.
- No meaningful primary speedup was measured: 9.650107 to 9.654300 tokens/s/user is approximately 0.0435%. Retaining the state-copy traces and unused-history removal under the owner cap is documented without claiming a win. Untried optimization families remain hand-off work, not requirements for this review.
- Update closure-only wording after the passing follow-up: the README title, final acceptance manifest, checklist, and `doc/full_model/serving_contract.json` stage-10 supplement still contain review/gate-pending language. This is ordinary documentation closure, not grounds for another hardware run or an additional finding.

## Hard-Check Gaps

- The stage checker reports advisory exit 1, not an unconditional success. Its current result agrees with the raw prefill/eval records: optional dense batched prefill was not adopted, the serial burst ratio is approximately 1, and three P4 advisories remain. The goal explicitly permits recording eval findings in limitations. These are not hidden critical failures.
- The original sampling XML's three supported chunked-prefill skips are covered by a separate three-test passing XML. The earlier 24-token legacy invocation failed; the successful invocation uses existing chat/reasoning options and unchanged assertions. Remaining skips cover speculative serving and the API's all-vocabulary-logprob limit. A literal all-tests-pass claim would be incorrect, but none is made in the current disposition.
- The primary benchmark has one measured request per arm; identical TTFT/TPOT percentiles do not provide statistical confidence. The report correctly avoids a speedup claim.
- The native logit retry lacks an exit sidecar because its parent was interrupted. The completed JSON, code ordering, and recorded device closure support the checks; do not invent exit 0. This is disclosed rather than concealed.
- There is no serving profiler requirement. Existing benchmark JSON, source inspection, tracking-enabled replay, sampler tests, and route evidence satisfy the serving-specific evidence contract.

## Anomaly Ledger

- Observed anomaly: four output-repeat assertions fail under changed admission histories.
  Evidence: final `checks/sampling.xml`, `AUTODEBUG_sampling.md`, `AUTOFIX_sampling.md`, and all three current logit JSON artifacts.
  Affected path: device k32/greedy and explicit host k50 compatibility requests.
  Control or comparison: current standalone/served fixed-width repeat and row reversal; 400 exact selected-logprob maps; separate tracked sync/async controls and narrow-sampler regression.
  Likely subsystem: inherited active-width-dependent model arithmetic. Cross-width native/served histories both diverge for Text0/Number; no mid-request sampler-route switch is demonstrated in the failing windows.
  Investigation performed: admission reconstruction, source comparison, full native-logit repeat/row checks and multi-step served comparison.
  Resolution: controlled for the skill's reproducibility-only disposition, with the documented limits above; not a claim that the raw pytest suite passed.

- Observed anomaly: supported recall checks initially stop inside thinking.
  Evidence: `sampling_followup/chunked_recall.xml` and `chunked_recall_chat.xml`.
  Affected path: the test invocation's prompt format/reasoning allowance.
  Control or comparison: three unchanged assertions pass with the tests' existing low-reasoning/chat options and 1528-token allowance.
  Likely subsystem: harness invocation.
  Investigation performed: original outputs inspected and focused invocation corrected.
  Resolution: fixed.

- Observed anomaly: selected-budget launcher fallback is still 256.
  Evidence: `run_server.sh:24`, `chunk_selection.json`, actual driver environment construction.
  Affected path: direct server launch without `TOKEN_BUDGET`.
  Control or comparison: accepted launches explicitly used 9.
  Likely subsystem: launch configuration.
  Investigation performed: source inspection and CPU-only shell expansion.
  Resolution: more-work-needed as the single Required Work item.

- Observed anomaly: incomplete orderly trace release at server teardown.
  Evidence: final serving/native cleanup, tracked execution manifests, owner shutdown decision.
  Affected path: plugin-owned shutdown.
  Control or comparison: idle queues and empty owned PID/device-handle inventories after cleanup; plugin remains clean at `629f07cb385494b14d0ee3d138de039d71d09546`.
  Likely subsystem: plugin/runner teardown lifecycle; exact cause is not established by these logs.
  Investigation performed: release/kill audits and source inspection of the disabled-profiler path.
  Resolution: controlled by the explicit owner acceptance, while preserving mandatory process/handle cleanup.

- Observed anomaly: short P4 answers, missing B2 floor, and unusable packaged forecast.
  Evidence: canonical `long_answer_load.json`, raw `requests.jsonl`, `eval_forecast_wall.json`, and `limitations.json`.
  Affected path: coverage and performance planning, not request completion.
  Control or comparison: all seven critical checks pass; 32 raw records complete; actual wall capacity is 37,315/4006.0 = 9.314778 tokens/s.
  Likely subsystem: natural EOS and offered-versus-resident concurrency assumptions in the harness.
  Investigation performed: raw count/token arithmetic and corrected forecast.
  Resolution: controlled as explicitly permitted advisory limitations; no actual eval-accuracy claim.

- Observed anomaly: mixed traffic has a 1944.979 ms maximum nonempty text-stream gap despite the chunk sweep passing its 1000 ms ITL bound.
  Evidence: `final_mixed_summary.json`, including the largest-gap event window, and `chunk_selection.json`.
  Affected path: streamed text publication during mixed requests.
  Control or comparison: the gap window contains five decode submissions and two prefill submissions with no capture; selected-budget foreground windows separately measure worst client ITL 592.1/615.7 ms at 4K/32K.
  Likely subsystem: text-chunk publication and concurrent request scheduling; these are different measured quantities/workloads.
  Investigation performed: event-window classification, with no unsupported attribution to deferred capture.
  Resolution: controlled and disclosed; no universal sub-1000 ms text-gap guarantee is claimed.

- Observed anomaly: the first native logit probe fails KV allocation before generation.
  Evidence: retained failed `stage10_sampling_logits_standalone.json`, corrected `_standalone_v2.json`, diagnostic source and AutoFix record.
  Affected path: standalone diagnostic setup.
  Control or comparison: retry uses the server's actual 8866-page pool at unchanged C262144/B2; model/shared/config source hashes match.
  Likely subsystem: test pool sizing.
  Investigation performed: corrected only test allocation setup and reran; diagnostic revisions remain distinct.
  Resolution: fixed.

## Scope Inspected

- Goal/skill paths: exact `10-10-optimized-vllm.prompt.txt`; installed stage-review, model-bringup startup, vllm-integration and eval-readiness-load reference, optimize, tt-enable-tracing, qualitative-check, tt-device-usage, AutoFix; local `TraceCorrectness.md`.
- Artifact paths: the final acceptance manifest, README/work log/checklist, stage-check JSON/log, performance summary and raw before/after primary/CI/prefill-burst results; source/runtime provenance; tracked sync/async histories, mixed evidence and raw logs; final acceptance raw server log and sampling XMLs; served/native logit artifacts; quality outputs and controls; P4/P5 canonical results and raw request records; four saved P4 answer tails; chunk selection/coverage and cleanup evidence; native sampler, lifecycle, watcher, CCL stress and allocation-tracker positive-control artifacts; owner decisions and limitations.
- Code paths: model `tt/generator.py`, `generator_vllm.py`, `slot_state.py`, diagnostic `check_sampling_repro_logits.py`, async-history and serving-driver code; changed common sampler code; changed native sorted64 sampling compute/writer paths; multicast all-gather factory/reader/writer changes; selected launch scripts and serving-contract rows.
- Commands run: read-only `cat`, bounded `sed`/`rg`, `git log/show/diff/status/rev-parse`; standard-library Python analyses of JSON/XML/logs and source hashes; CPU-only shell parameter expansion. Only this report was written.
- Independently rederived: raw benchmark metrics and rates; equality of before/after server contract/shared runtime/loaded binaries; only generator/slot-state model-source differences between measured arms; current source identity against final tracked/acceptance hashes; 198/202/205/205/205 distinct capture signatures in before/after/tracked-sync/tracked-async/final-acceptance logs, each captured once with no duplicate signature; 960 exact sync/async history output tokens; 400 exact served/native logprob maps; raw sampling outcome counts; P4 token/count/median arithmetic. No unsafe-allocation/capture-write/fatal/traceback markers appeared in those inspected final logs.
- Generated outputs: read all 12 current HTTP qualitative outputs and all four saved P4 tails. The poems/explanations/story/translation/Fibonacci outputs are coherent within their caps, with the disclosed sampled-haiku meter defect. The four displayed P4 conclusions are correct; this is not a 32-answer accuracy verdict.
- Push state: the authorized run plan is local checkpoint commits only, never push. Review does not infer remote state from reflog. Final stage-owned checkpoint/SHAs remain stage-owner closure work after the passing follow-up.

## Residual Risk

- Reproducibility across arbitrary active-width/admission histories is not established. Current greedy selected-logprob diagnostics are narrower than the four original stochastic/device failure histories, and no matched HF control for those exact four cases is claimed.
- Long-answer/eval accuracy, 12K-answer throughput, and the full planning workload remain unmeasured. Resident capacity 2 queues the 32 offered users; it is not resident batch-32 proof or a largest-feasible-batch claim.
- Shared CCL and sampling changes have targeted hardware evidence on this 1x4 Blackhole configuration; CPU route enumeration does not establish every other topology's hardware behavior.
- Raw large evidence remains on persistent local storage rather than entirely in commits. Preserve it through hand-off.
- No further optimization, profiler work, plugin change, or hardware rerun is requested by this review. Close the launcher configuration finding, update closure documentation, obtain the small follow-up verdict, then checkpoint only stage-owned changes locally.
