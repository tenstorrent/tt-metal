# Stage Review

Verdict: clean-pass

## Required Work

- None for stage 11's benchmark contract. This is a benchmark-evidence verdict, not an accuracy or release-readiness verdict. The explicit operator instruction assigns recorded generation concerns to stage 12.

## Other Concerns

- Three semantic concerns in generic sampled responses remain stage-12 handoff items: unsupported acoustic-frequency confounding, an invalid per-leg-delay feasibility guarantee, and inconsistent detector-gain application. `semantic_concerns.json` contains the evidence and routing counterexample. No matching HF samples establish whether these originate in the checkpoint or TT numerics; no such attribution is claimed.
- Four natural-stop samples generated 1,413, 1,654, 1,624 and 3,929 tokens. Their 32,768-token cap does not establish sustained generation to either eval output cap or a ten-minute soak. The report explicitly states this limitation.
- The final capacity-gap text retains the planning phrase “Concurrency1 remains scheduled.” The actual measured and budget-missing concurrency-1 rows are separately enumerated correctly; this wording does not invalidate evidence.

## Hard-Check Gaps

- Mechanical degeneration checks do not assess semantic correctness. Direct output inspection identified the three separately recorded concerns above.
- Installed serving vLLM exposes no defensible source commit. Identity evidence preserves its installed source-tree and distribution hashes; no commit is inferred from another checkout.
- Rooflines are modeled useful arithmetic/DRAM traffic divided by host-wall phase intervals. They are not device utilization measurements. Padding, cache reuse and temporary traffic approximations are documented; telemetry correctly leaves device fields null.

## Anomaly Ledger

- Observed anomaly: Initial streaming latency evidence was buffered until completion.
  Evidence: `attempts/buffered-logging/`, `setup/AUTODEBUG_logging.md`, `setup/AUTOFIX_logging.md`.
  Affected path: SSE timing and response logging.
  Control or comparison: Unbuffered measured rerun, response-logging host checks, complete response-body comparisons.
  Likely subsystem: Native buffering middleware.
  Investigation performed: Source diagnosis, model-owned streaming-forwarding repair, fresh measurements; invalid timing attempt retained and excluded.
  Resolution: fixed.

- Observed anomaly: Original supplemental orchestration threatened the native runner deadline.
  Evidence: `setup/budget_handoff.json`, `setup/native_fallback.json`, `setup/AUTODEBUG_budget.md`, `setup/AUTOFIX_budget.md`.
  Affected path: Benchmark orchestration and row admission.
  Control or comparison: Genuine all-batch native run completed in 899.973 seconds before the original absolute native boundary.
  Likely subsystem: Budget accounting/admission.
  Investigation performed: Preserved failed attempt, bounded recovery, whole-row admission, unchanged outer clock; independently checked all 392 carried-file original/copy hash comparisons.
  Resolution: fixed.

- Observed anomaly: Several delivered rates fell below predicted ranges, particularly full-concurrency long-prefill traffic.
  Evidence: `serving_rates.json` row causes, raw timings and phase events; C1 controls; `repeatability.json`.
  Affected path: Delivered token cadence and prefill latency.
  Control or comparison: Matching batch/precision/sampling C1 rows and repeated native B32 headline.
  Likely subsystem: Interleaved canonical prefill, context-dependent decode work, scheduling and fixed per-request costs.
  Investigation performed: Quantified phase shares, observed completion cadence, raw HTTP TPOT and modeled work. Short B1/128 prefill miss separately explains omitted fixed costs. No device-level attribution claimed.
  Resolution: controlled for benchmark reporting; no performance threshold is imposed by this stage.

- Observed anomaly: Coherent generic text contains the three semantic errors described above.
  Evidence: `anomaly_replies/1-response.json`, `2-response.json`, `3-response.json`, `semantic_concerns.json`.
  Affected path: Sampled reasoning at operator-authorized served defaults.
  Control or comparison: Constructive two-leg routing counterexample and internal formula/example comparison; no matching HF long-sample control.
  Likely subsystem: Undetermined model reasoning; TT precision causation is unproven.
  Investigation performed: Reviewer read all four complete responses. Owner preserved distinct semantic concerns and explicit stage-12 handoff, independently of detector count zero.
  Resolution: controlled for this stage's explicit record-only contract; semantic correctness remains unresolved for stage 12.

## Scope Inspected

- Goal/skill paths: `setup/review_contract.md`, repository `AGENTS.md`, `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`; enabled model-bringup 0.1.27 `stage-review`, `benchmark-model`, `agentic-serving`, `qualitative-check`, `tt-device-usage`, and startup instructions.
- Artifact paths: `run/REPORT.md`, native `summary.json`, `run_config.json`, raw profile/grid/supplement bundles, `roofline.json` and per-row events, `plan.json`, all three serving-rate files, release matrix, identity, timing, repeated measurements, shared qualitative outputs and pinned stage-8 HF controls, four long responses, semantic handoff, setup repair/gate/admission/ownership records, raw archive manifest and telemetry packet `00156d66-a2a8-4738-a330-600ac7418ded.json`.
- Code paths: `tt/benchmark_phases.py`, adapter observation/readback hooks in `tt/generator_vllm.py`, `tools/benchmark_work.py`, `collect_benchmark_roofline.py`, `benchmark_eval_client.py`, reporting and orchestration artifacts. Inspection found no added device synchronization in the phase observer.
- Commands run: Read-only `cat`, `sed`, `rg`, `ls`, Git status/diff inspection and small Python artifact analyses. Recomputed all 44 completed rows from raw events: 39 grid/native rows and five release rows. Verified complete request populations, zero failed requests, every output length 128, phase-duration partition, FLOP/byte totals, raw/evidence hashes, report metric arithmetic and exact policy-filtered views. Independently checked all 18 archive-part hashes and 931 current-original file hashes against the manifest. No server, endpoint, device or reset commands were run by this reviewer.
- Final closure: Native evidence, full-context and agentic-serving gates all exit zero in `setup/final_gates.json`; logs agree. Eleven missing grid rows are fully named: eight pool-capacity gaps and three budget-admission gaps. `deferred_rows.jsonl` records whole-row estimates exceeding remaining windows. No release rows are missing. The final B32 reload used observed startup plus margin and a whole-row estimate; actual work and shutdown completed inside the original clock. Report elapsed is 9,921.7 of 10,800 seconds, leaving review/checkpoint time. `setup/ownership_closed.json` records no owners, no live recorded launcher, removed owned state and no reset.
- Identity and commit scope: Live branch `v5-extended-run`, measured runtime `262d3a3cc8b9af24a048d75211a7b3e520a1790b`, pinned HF revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, full 60-layer autoport, TP4/P300x2, head_bfp4 and context 262144. Final stage checkpoint follows this review, as required by the stage-review workflow. Operator run plan requires local commits only; no push is authorized.

## Residual Risk

- Stage 12 must evaluate real task trajectories and semantic quality, including much longer generated outputs. This review does not establish eval scores.
- Missing B8 maximum-context and unseeded B32 maximum-context rows require labeled extrapolation where used; measured B1 maximum-context and seeded B32 maximum-context evidence do not replace them.
- Synthetic benchmark traffic and its concurrency admission differ from agent trajectories. Rates are evidence for planning, with the report's workload and host-wall attribution limitations.
