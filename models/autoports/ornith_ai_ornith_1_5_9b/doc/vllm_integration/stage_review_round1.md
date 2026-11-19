# Stage Review

Verdict: more-work-needed

Independent reviewer, round 1 of 3. Stage 9 vLLM integration for `ornith-ai/Ornith-1.5-9B`, live tt-metal worktree based on `d1c644f35289ee46bfa4a23598aa607d5ea96522`. The only required item is an operator decision about the already-tested serving default. This review found no concrete current implementation defect, incorrect reported metric, or missing mandatory runtime gate requiring another device run.

## Required Work

- P2: Record operator acceptance of the default `top_k=20` serving recipe.
  Evidence: `work_log.md` originally describes the override at line 98 as an operating recipe, without an explicit owner acceptance. The final launch commands in `readiness_vllm/sampling64_tracked/execution/commands.sh` and the separate final64 B1/B32 launch summaries use `--override-generation-config '{"top_k":20}'`. The pinned checkpoint has no readable `generation_config.json`; requests that omit `top_k` therefore receive this chosen finite sampling limit. The stage owner confirmed that the supplied Entry 1 approval covers the plugin history interface, not this default. The 22:27 work-log entry records the exact pending decision.
  Why this matters: `$vllm-integration` requires a served-default override to be recorded as an operator decision; `$stage-review` explicitly identifies a generation-config override that keeps the default recipe on the device sampler as an operator-decision item. Passing sampling and performance evidence establishes that this configuration works, but does not supply the missing acceptance. The later instruction to freeze the best version does not explicitly select this sampling default.
  Required next step: **operator decision** on accepting the tested omitted-`top_k` default of 20. Record the response beside the exact command and scope. No implementation change or hardware rerun is needed if accepted. A follow-up review can inspect only that authorization/status delta. All remaining Required Work is of this operator-decision kind.

## Other Concerns

- Strict forced-preemption token parity remains failed in the three reduced diagnostic modes. This is not relabeled as a pass. The canonical sampled and 50-output greedy controls reproduce the served branches; the aligned151 comparison shows identical original-prefix KV, equal host/device rebuilt state, preserved inactive state, and differences beginning at the decoded suffix. This controls the observed serving-versus-canonical question without proving exact recomputation equivalence or a harmless numerical tolerance. Keep the existing next-stage investigation note; no specific remaining adapter defect was established by this review.
- Dense batched prefill is disabled. The current vLLM skill and `prefill-features.md` explicitly permit the correct serial path at Stage 9; its completed 32-request burst is present. The missing batched/serial ratio is advisory, not a missing runtime correctness gate. Batched prefill remains an optional later optimization.
- Arbitrarily many equal local maxima and precise physical-page telemetry remain the bounded follow-up ideas already listed under `Carried to next stage`. They do not invalidate the tested greedy, capacity64, page-growth, or output contracts.

## Hard-Check Gaps

- The installed stage gate exits 1 for advisories: legacy `reasoning_content` tokenization through `/tokenize`, historical 512-token diagnostic log references, and absent `serial_reference` for a serial-only prefill burst. It has no critical failure. P3 behavioral cases, including the corresponding chat history case, pass; the dedicated reasoning-field artifact distinguishes the deprecated alias from the actual chat API path. No probe was weakened.
- Sampling and qualitative artifacts originate from EngineCore 1265789; mixed/lifecycle and remaining tracked probes originate from retry EngineCore 1293243. This split is explicitly recorded. I verified all 26 promotion-manifest source/destination pairs are byte-identical. It is valid reuse of unchanged production code, not evidence of one uninterrupted combined run.
- The 262143-input plus one-output capacity acceptance predates the final sampler/horizon changes. It is labeled accordingly. Final source still serves context 262144 and passes the nine long-context accuracy requests up to approximately 131K. This review found no context allocation or capability reduction requiring another maximum-capacity run.
- The primary percentile values describe one 128/128 request, as documented; they do not estimate variability across repeated requests. No extra benchmark series is requested.

## Anomaly Ledger

- Observed anomaly: original32 P2 reasoning requests routed top-k40 to host and failed TPOT ratio.
  Evidence: `readiness_vllm/before_sampling64_b32_untracked/harness_requests.json`; final `doc/eval_readiness/harness_requests.json`; `sampling64_tracked/p2_route_audit.json`.
  Affected path: stochastic device sampling capacity.
  Control or comparison: preserved default32 reference, explicit64 primitive/device tests, final full sampling and full P2.
  Likely subsystem: 32-candidate sampling boundary.
  Investigation performed: native64 specialization, default32 exact comparison, integrated TP4 boundary/trace/history checks; tracked P2 counts 644 device and zero host decode calls.
  Resolution: fixed; final applicable P2 ratios are at most 1.009 against 1.15.

- Observed anomaly: first final64 mixed check lacked concurrent decode during continuation chunks.
  Evidence: `sampling64_tracked/first_attempt/mixed_trace.json`, `completed_sampling_reuse.json`, final `mixed_trace.json` and `lifecycle_trace.json`.
  Affected path: coverage of scheduler overlap and trace reuse.
  Control or comparison: retry admits long requests after the short request has begun streaming; the device-event concurrency assertion is retained.
  Likely subsystem: workload admission timing.
  Investigation performed: inspected admission timestamps, actual starts 0/896/1024/2048, six baseline trace signatures, and event streams. Both final workloads have 286 device and 69 explicit host-control decode calls, no new captures, and exact greedy agreement across modes/order.
  Resolution: fixed coverage; original failed attempt preserved.

- Observed anomaly: forced-preemption replay differs from uninterrupted decoding.
  Evidence: `diagnostic9_pages/preemption_analysis.json`, `replay_greedy_follow50.json`, `replay_model_state151.json`, and `replay_model_state151_interpretation.md`.
  Affected path: recomputation of generated suffixes through prefill versus incremental decode.
  Control or comparison: canonical standalone controls reproduce both served trajectories with matching sampler state; original KV rows 0–128 are exact and differences begin at row129.
  Likely subsystem: canonical prefill/decode numerical route.
  Investigation performed: logit comparison, matched consumed-prefix/state comparison, host/device rebuild equality, and repaired scheduler page replacement.
  Resolution: controlled for this serving integration review; exact parity remains unproven and explicitly failed, with the scope limitation carried forward.

- Observed anomaly: tracking-off benchmark logs warn that allocation with live traces can be unsafe; a diagnostic initially failed on 28 late slice-program allocations.
  Evidence: final64 B1/B32 server logs; `replay_model_state151_before_slice_warmup`; final tracked environment and replay artifacts; earlier `doc/full_model/trace_alloc_tracking/positive_control.log.gz`.
  Affected path: trace allocation lifetime.
  Control or comparison: tracked runs set `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`. The positive control rejects a live unsafe buffer and succeeds after release. The diagnostic failure directly confirms program-cache allocations were checked.
  Likely subsystem: diagnostic preparation; generic allocator warning during untracked timing.
  Investigation performed: inspected generator-wide preparation, exact logits acknowledgment, retained backing tensors, nonblocking split replay, and corrected diagnostic warmup. No broad allocation exemption was added.
  Resolution: controlled; the final tracked workload passes and timing is separately labeled tracking-off.

## Scope Inspected

- Goal/skill paths: `/home/ttuser/ornith-9b/multigoal_logs/09-09-vllm.prompt.txt`; installed 0.1.24 `stage-review`, `vllm-integration`, `tt-enable-tracing`, `tt-device-usage`, and `qualitative-check` skills; vLLM prefill-features reference; pinned plugin capability/interface documents.
- Artifact paths: stage README/work log/plugin records; serving/context/precision contracts; final gate and cleanup; final64 source/native snapshots; promotion manifest; full sampling log; native original32/compare/22-case logs; adapter acceptance; tracked mixed/lifecycle/page-growth/route/logit evidence; canonical P2/P3/P5 JSON and raw P5 answers; final B1/CI/prefill raw benchmark JSON; served qualitative outputs and all twelve fresh TT and twelve reused HF output texts.
- Code paths: `tt/generator_vllm.py`; changed canonical generator, prefill contract, sampling state, common sampler and native sampling code; relevant native tests; plugin changes from base `042a94863ad27d88207bcf71abdf67122c6f4c27` to `2a1504554aa7babb1d426dfbca8c7db8c0a795ef`.
- Commands run: read-only `cat`, `sed`, `rg`, `git status`, `git diff`; small Python JSON/text/gzip analysis and direct byte comparisons. No TT imports, device access, server launch, reset, or test execution. The only file written is this report.
- Re-derived metrics: primary 128/128/1, concurrency1: TTFT236.249336973ms, meanTPOT8.700078386ms, decode114.941493122tokens/s/user, output95.410488634tokens/s. CI100/100/32: completed32, output3200, aggregate519.415015532tokens/s. Serial-prefill128/32/32: completed32, output1024, aggregate197.497606763tokens/s. These agree with reported rounded values.
- Source identity: the four changed model files, two common sampler files, and all13 native sampling snapshot files compared byte-identically against final measurement snapshots. Plugin worktree is clean with the three documented local commits. At review completion, `git log -1` confirms preservation checkpoint `aa3d320cd6f0f1b64c0ddd9a5034c15b0f8b68bf`; the work log records its normal pre-commit pass and a formatting-only diagnostic-test change with AST equality. This is not a stage-pass claim. Per the supplied run plan, nothing is to be pushed.

## Residual Risk

- This is inspection of recorded execution, not a fresh device run or a full audit of every earlier bringup artifact.
- Qualitative outputs are coherent and on-topic, with correct French translation and valid visible Fibonacci implementations. Long answers end at the documented256-token cap in serving and controls. The six-prompt suite is not a statistical quality comparison; no claim of exact seeded or cross-geometry output identity follows from it.
- Reduced forced-preemption controls do not prove full32-layer exact replay parity. Future accuracy/release work should retain that limitation rather than treating it as a passed gate.
- Process cleanup evidence shows no serving processes or device owners and no reset. The outstanding default-recipe acceptance is the sole reason this round cannot return `clean-pass`.
