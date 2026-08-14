# Stage Review

Verdict: clean-pass

## Required Work

- None. The prior P1 production-device-sampling gap and P2 punctuation anomaly are closed by current code and artifacts.

## Other Concerns

- The final default serving result is intentionally a reproduced no-regression result, not a speedup: primary TPOT moved from 18.927619 to 18.925413 ms and TPOT-derived decode from 52.832848 to 52.839005 token/s/user. The only serving-local candidate, rank-0-only sampled-token readback, was correctly rejected after measuring slower primary and CI results and was reverted.
- The compatibility server exposes host-only API features, but its 72-pass/1-skip suite is explicitly labeled compatibility coverage and is not used as production stochastic proof. Production stochastic proof comes from a separate host-compatibility-disabled server and opt-in route audit.
- The final CI burst omitted the baseline's explicit `--max-concurrency 32` while submitting exactly 32 requests; both raw results report observed maximum concurrency 32. This is disclosed and makes the effective 100/100/32 workload comparable, but future parity runs should retain the explicit cap to remove the command-level difference.

## Hard-Check Gaps

- No serving profiler, Tracy, `tt-perf-report`, watcher, adapter profiler, `ReadDeviceProfiler`, or live device profiler was collected, as the stage contract explicitly prohibited them. Benchmark JSON, live sampling routing, unit checks, stale-input checks, async source inspection, and cleanup evidence substitute for those tools.
- The production device-sampling evidence is a focused HTTP/routing artifact rather than a pytest transcript. It is nevertheless sufficient here: `server_production_device_sampling.log` records `perform_device_sampling=True` for stochastic prefill, stochastic decode, mixed greedy/stochastic decode, and reversed slot reuse; the companion JSON records 17/17 HTTP 200 and post-run health 200.
- Local checkpoint commits are intentionally pending the clean review. Stage-owned root changes and the nested plugin audit change are isolatable from the listed unrelated profiler outputs, cluster-descriptor files, and root's untracked nested-repository entry; committing them is the stage owner's required post-review follow-up.

## Anomaly Ledger

- Observed anomaly: The first compatibility-disabled sampling smoke ended in an EngineCore fatal.
  Evidence: `readiness_vllm/sampling_smoke_production.log` says the fatal occurred during a smoke whose first target is `test_mixed_params_batch`; current test/source classification shows that target combines top-k 100, explicit seeds, and penalties, which this adapter marks host-only. With host compatibility disabled, that unsupported mixed request cannot enter the supported top-k-32 device contract. The original server log was overwritten, so this attribution is explicitly reported as source-backed inference rather than direct log proof.
  Affected path: Unsupported host-only API use while the host compatibility path is disabled, not the supported stochastic Sampling1D route.
  Control or comparison: A later host-compatibility-disabled production server ran a 192-prompt/128-output top-k-32 stochastic request plus two reversed eight-request mixed greedy/stochastic waves. Live audit logs show `perform_device_sampling=True` at prefill and decode, all 17 requests returned 200, and health remained 200.
  Likely subsystem: Disabled host fallback for unsupported sampling parameters in the old smoke.
  Investigation performed: Compared smoke target ordering, plugin capability routing, adapter host-compatibility checks, live production route logs, HTTP evidence, and post-run health.
  Resolution: controlled; the supported production device path passes.

- Observed anomaly: The remediation server's first startup failed with a device-0 active-Ethernet heartbeat timeout.
  Evidence: `final/server_production_sampling_startup_failure.log` fails in `open_mesh_device` before model construction, trace capture, or request processing and names core 29-25. `work_log.md` records the bounded list/reset/list sequence and a successful 1x4 `FABRIC_1D` mesh-open control before the identical server subsequently passed.
  Affected path: Hardware initialization transition only.
  Control or comparison: Successful identical server startup, complete production stochastic/qualitative run, clean teardown, no leftover vLLM/EngineCore process, and all four p300c boards listed in `final/cleanup.log`.
  Likely subsystem: Recoverable ERISC/device transition.
  Investigation performed: Classified failure timing and stack, applied the `$tt-device-usage` bounded recovery, verified mesh open/close, then used the successful identical configuration.
  Resolution: controlled.

- Observed anomaly: An earlier compatibility-enabled greedy completion contained `learning,,,` while the qualitative verdict claimed no mechanical repetition.
  Evidence: The old bytes were replaced, but the first review preserved the exact symptom. `AUTODEBUG.md` records two isolated HTTP runs, two OpenAI-client runs, two matching greedy-to-stochastic transition sequences, eight concurrent identical prompts, and the full six-prompt host-compatibility-disabled suite. None reproduced the symptom. The current readiness, candidate, and final output files share SHA-256 `a4a2338f026e0baefcd69a40d41b51fc8889bcb1645fa6d878a5ac7a3c07f3f9`; prompt 2 is clean. The comma text tokenizes as two distinct IDs, not a stuck one-token feedback loop.
  Affected path: Earlier compatibility-enabled artifact only.
  Control or comparison: Prompt-correct HF/full-model controls plus repeated production greedy, mixed-transition, concurrent, and full-suite controls.
  Likely subsystem: Unreproduced compatibility-only host/device transition or close-logit qualitative variance; not the current production token/position/page-table/slot lifecycle.
  Investigation performed: `$autofix`/`$autodebug` source audit and focused runtime controls; the isolated AutoDebug runner's Bubblewrap network-namespace failure is disclosed in `AUTODEBUG.md`, and the parent-environment report records the completed evidence.
  Resolution: controlled; no functional change was justified and the final production artifact is clean.

- Observed anomaly: Final CI burst TPOT is 1.4% slower than baseline and follows the full sampling stress suite.
  Evidence: Mean TPOT moved from 19.618114 to 19.887325 ms; output throughput moved from 1026.974 to 1018.207 token/s. All 32 requests and 3200/3200 output tokens completed. Primary single-user TPOT and decode are effectively unchanged/slightly improved, and CI is correctly reported only as secondary capacity/nightly-parity evidence.
  Affected path: Burst serving variance, not headline single-user decode.
  Control or comparison: Same effective 100/100/32 workload with observed concurrency 32; primary 128/128/1 reproduction; rejected candidate was also slower.
  Likely subsystem: Run-to-run burst/scheduler variance after stress, not a retained implementation regression.
  Investigation performed: Compared raw before/after result JSON, completion counts, observed concurrency, command differences, and primary result.
  Resolution: controlled and accurately reported.

- Observed anomaly: Nanobind leak diagnostics appear when the sampling test client exits.
  Evidence: `final/sampling_tests.log` records 72 passed, 1 skipped before the binding diagnostics; `final/cleanup.log` proves the owning server terminated and no API/EngineCore/cache owner remained.
  Affected path: Python binding teardown diagnostics, not request execution or server ownership.
  Control or comparison: Successful tests, clean server shutdown, process scan, and four-board list.
  Likely subsystem: Existing binding finalization.
  Investigation performed: Checked ordering, exit result, cleanup transcript, and named limitation in `perf_summary.json`.
  Resolution: controlled residual limitation.

## Scope Inspected

- Goal/skill paths: supplied optimized-vLLM contract; `.agents/skills/stage-review/SKILL.md`; stage expectations embodied in `$vllm-integration`, `$optimize`, `$tt-device-usage`, `$tt-enable-tracing`, `$qualitative-check`, and the remediation `$autofix`/`$autodebug` report.
- Artifact paths: `doc/optimized_vllm/{README.md,work_log.md,perf_summary.json,qualitative_verdict.md,final/,candidates/shard0_read/}`; `AUTODEBUG.md`; `readiness_vllm/` benchmark, sampling, qualitative, non-aligned, exact-chat-template, unsupported-survival, and prior smoke artifacts; `doc/context_contract.json`; `doc/datatype_sweep/selected_precision_config.json`; `doc/optimized_full_model/{README.md,perf_summary.json,qualitative_suite/}`.
- Code paths: `tt/{generator_vllm.py,generator.py,model.py}`; `tests/test_vllm_adapter.py`; nested plugin `vllm_tt_plugin/{platform.py,async_decode.py,model_runner.py}` and `test_device_sampling_limits.py`.
- Commands run: read-only `sed`, `nl`, `rg`, `find`, `jq`, `sha256sum`, `wc`, `git status`, `git diff`, `git show`, branch/revision queries, and `git diff --check`. No server, vLLM process, hardware open/reset, profiler, watcher, or TT test was run during review.

## Residual Risk

- The exact selected BFP4/LoFi attention/MLP, BFP8 KV, BF16 activation/CCL/LM-head policy is visible in the production startup log. The 32,768-token served context matches the physical TP4 context contract; no context, benchmark, evaluation, or non-aligned capability was reduced.
- The production adapter and plugin implement the requested async boundary: `decode_forward(..., read_from_device=False)` returns TT device tokens; model and sampler traces replay with `blocking=False`; `read_decode_output(..., async_read=True)` schedules only sampled-token transfer; event synchronization and token formatting occur after submission. Live startup enables async scheduling.
- Persistent token, signed current-position, RoPE, page-table, sampler parameters, cache, and sampler output are source-backed. Sampling parameters and page tables use snapshots and refresh only on change; tests cover stale token/position, reset, fresh prefill, slot remap, and changed/unchanged page tables.
- Supported greedy and stochastic production routes use split Sampling1D with no adapter-local argmax or full-logit readback. Host logits, untilize, and full-vocabulary transfer remain reachable only through explicit host compatibility/diagnostic methods and are excluded from the measured production route.
- Final vLLM decode is 52.839 token/s/user versus 54.452 for comparable optimized full-model host-free token-out, a 2.96% serving gap. That is consistent with the unavoidable plugin sampled-token boundary and satisfies the “about as fast” requirement.
- All final qualitative outputs are coherent and prompt-correct. The sampled thermodynamics answer starts by calling the sequence “three laws” but enumerates zeroth through third before truncating; HF/full-model controls use the same conventional zeroth-law framing, so this is ordinary model behavior rather than serving corruption.
- Stage-owned commits and SHA logging remain procedural post-review work, not an unresolved model or evidence defect. No push is authorized.

Verdict: clean-pass
