# Stage Review

Verdict: clean-pass

**Accepted scope: the exact isolated five-line inactive-logit repair, validated by the complete reduced-model revision5 retry1 control.** The candidate is suitable for exact source promotion and a compact, isolated local checkpoint. This is not a Stage 11, full64, serving/APC, performance, complete-context, or sustained-host-isolation pass.

Review date: 2026-09-16. Independent strict verification and host predicate checks below ran before source promotion at 08:41:13.033972 UTC. Following the reviewer's acceptance message, root applied the exact change and recorded `promotion.json`; the reviewer then independently confirmed that the live generator bytes match the accepted candidate. The earlier revision5 run remains rejected for ownership and retained as diagnostic evidence; retry1 supplies separate accepted evidence.

## Required Work

- None before promoting this exact candidate within the reviewed scope. No new source, numerical, trace, provenance, cleanup, or bounded-run ownership defect was found.
- For the promotion/checkpoint handoff, apply only `generator.patch` to the pinned live baseline and verify the resulting generator bytes equal the accepted candidate SHA below. Record baseline/candidate/result hashes and the local commit in a new transition receipt; retain existing source pins and historical receipts unchanged. Keep unrelated dirty/index state out of the compact checkpoint. This is recordkeeping for the accepted change, not a request for another device experiment.

Promotion update: the exact source application and byte check are now complete in `promotion.json` (08:41:13.033972 UTC). Local checkpoint creation/recording remains root-owned. The pre-promotion verifier was not rerun against the intentionally changed live source.

## Acceptance Evidence

The reviewer independently ran the complete artifact verifier:

```text
/usr/bin/python3 -B /home/mvasiljevic/qwen38-full-rerun/tti-release-qwen38/qb2-resume/batched-prefill-candidate/probes_revision5/retry1/verify_retry.py
exit 0
PASS_ACTUAL_REDUCED_MODEL_CONTROL
numeric_checks: 768
retry_native_binding: true
full64: false
serving_integrated: false
timing_measured: false
```

Tool chunk: `f2421c`. Five audited stdlib retry checks also passed independently (`check_host.py -v`, exit 0, tool chunk `b12d46`). Their synthetic negative cases remain host tests, not runtime evidence.

These are historical pre-promotion verification receipts from 2026-09-16, before 08:41:13.033972 UTC. The append-only source transition is recorded in [promotion.json](promotion.json); no original evidence or pin was rewritten to accommodate promotion.

| Required evidence | Retry1 result |
| --- | --- |
| Exact original matrix | Greedy/sampled, slot orders `[1,7]`/`[7,1]`, serial/grouped arms, both S2048 prefills and four decode steps complete |
| Model scope | Actual Qwen3.8-27B-FP8 checkpoint, original layers `[0,3]`, TP4 `[1,4]`, B8, context allocation 262144, block32, 33097 physical pages |
| Original numerical gate | 768/768 exact; PCC minimum 1.0; maximum absolute difference 0.0; threshold remains 0.995 |
| Canonical state | Original exact all32 tokens/seeds/history, positions/RoPE/table, inactive-state/page canaries, and allocation-identity checks pass |
| History/rebind | All eight arms exercise original recorded-history and device-table rebind checks; state/KV exact and released handles verified |
| Captures | 24 state-exact capture records, 48 native guarded captures; cache-miss guard available |
| Trace budget | Maximum recorded 2,162,688 bytes/device within 134,217,728-byte budget |
| Added diagnostics | Complete 24 warmups and 32 replay snapshots; active raw/canonical equality plus byte hashes pass on all four shards; every inactive/padded sampler row is finite zero |
| Native binding | Same pinned native binaries before/after and observed in the actual child; child PID/start ticks, wrapper identity, build receipt, environment, and artifact hashes bind correctly |
| Child/wrapper | Child PID1134061, wrapper PID1134060; both exit 0; wrapper errors empty |
| Cleanup | Device close returned, cleanup errors empty; all four owners empty before imports and after child execution |
| Ownership observer | `pass_observed_ownership`, wrapped exit0, 704 samples, zero errors, maximum gap 0.102484330534935 seconds |

The observer covers 08:38:08.755401–08:39:19.334035 UTC; child receipt covers 08:38:08.833621–08:39:19.256069 UTC. It records only the expected positive child PID during device use. The final partially empty sample at 08:39:19.032484 follows device-close logging at 08:39:18.550 and cluster-close completion at 08:39:19.028; all owner lists are empty by 08:39:19.133082. This is consistent with closure and the unchanged predicate.

## Predicate and Source Review

- Retry executes the original pinned `probes_revision5/check_equal_span.py`; the model command changes only the final `--output` path. Working directory, environment overrides, candidate, original source pins, diagnostics, and runtime matrix are unchanged.
- Direct verifier diff and audited AST checks show all original result, inactive-logit, ownership, and execution predicates preserved. Only command/wrapper receipt routing uses the fresh retry directory. `validate_native` adds PID/start-tick, mapped-library, wrapper-error, environment, build-receipt, and idle-endpoint checks. No ownership exception or reduced numerical/trace gate was added.
- Retry bindings preserve hashes of the original failed result, log, execution receipt, observer result, checker, and source pins. The host negative check still rejects that old observation with `Ownership wrapper failed`.
- This reviewer directly compared all 48 retry stage summaries with the earlier diagnostic revision5 run: **all match**, including active numerical hashes, tokens/seeds/history/metadata, and inactive canaries. The accepted run reproduces the candidate result under a valid sampled ownership window.
- The new causal diagnostics reproduce 72/288 differing inactive raw cross-arm row hashes and 0/96 differing active raw pairs. All actual raw inactive records are finite; NaN/Inf handling remains covered by source reasoning and host tests, not an observed hardware NaN case. Active raw/canonical byte equality and finite-zero canonical inputs pass in the accepted run.
- Previous preparation inspection established that the source delta is exactly the sparse branch before existing B-to-32 padding. It leaves active row order, dense/unspecified-active branches, sampling/seed advancement, token target, prefill behavior, and trace lifecycle unchanged. Exact source promotion does not require importing the probe wrappers into production.

## Anomaly Ledger

- Observed anomaly: the first revision5 run had 57 PID0 ownership-error samples despite numerical success.
  Evidence: immutable original `owner_observation.json`; prior `REVIEW_runtime_diagnostic.md`.
  Affected path: required ownership acceptance.
  Control or comparison: fresh retry1 retains the same observer and predicates, completes the same matrix, and records zero errors with exact reproduced stage summaries.
  Likely subsystem: external ownership environment; original unknown owners remain unidentified.
  Investigation performed: compared unchanged predicates, fresh owner changes, execution intervals, logs, source bindings, and all summaries.
  Resolution: **controlled for retry1's bounded execution**. The old run stays failed; the host's recurring interference risk is not resolved by one clean window.

- Observed anomaly: raw inactive logits differ while active raw logits match.
  Evidence: 72/288 inactive raw pairs differ; 0/96 active pairs differ; all canonical inactive/padding rows are finite zero; all32 feedback/metadata match.
  Affected path: undefined inactive decode output reaching sampling.
  Control or comparison: same candidate in both arms, active before/after byte equality, complete exact original numerical gates, and repeatability against the first revision5 run.
  Likely subsystem: SDPA skip-output/caller sampling contract.
  Investigation performed: original source diagnosis, preparation review, direct raw-row comparisons, and complete accepted retry verifier.
  Resolution: **fixed at the reviewed consumer boundary**. Prior device allocation contents remain unobserved and are not needed for the repaired inactive-input contract.

- Observed anomaly: zero-logit rows yield token270; greedy active output repeats token220 in the reduced model.
  Evidence: exact retry stage summaries match the earlier diagnostic run and the available revision4 active controls; sampled active outputs differ appropriately from greedy by later steps.
  Affected path: interpreting native tied-zero sampling and reduced-model outputs.
  Control or comparison: unchanged sampler, prior padded-zero token270 behavior, exact all32 cross-arm equality, preserved active hashes.
  Likely subsystem: native tied-max selection and reduced-model output.
  Investigation performed: direct token/summary comparison, previously documented tie contract.
  Resolution: **controlled for this equivalence test**. Neither token-zero semantics nor text quality is claimed.

- Observed anomaly: motherboard-discovery fallback and L1-semaphore fragmentation warnings recur.
  Evidence: retry actual.log lines17 and27–28, also present in both previous runs. Four-device ring initialization and all required calculations/close complete.
  Affected path: topology naming and existing collective allocations.
  Control or comparison: unchanged warning texts, matching logical/physical topology, stable native binding, bounded memory, no allocator/capture/cleanup exception.
  Likely subsystem: existing topology lookup and collective scratch policy.
  Investigation performed: compared logs, guard records, memory, and cleanup.
  Resolution: **controlled within the reduced run**; no long-run memory/isolation claim.

- Observed anomaly: `device_only_prefill_calls=0` despite 24 observed actual model prefill calls.
  Evidence: unchanged capture guard counter and original call records; retry reports782/782 global JIT cache hits.
  Affected path: guard-counter interpretation.
  Control or comparison: this control uses eager prefill; the counter wraps different prefill trace entrypoints. All48 required model/sampling captures use the original guarded begin/end path.
  Likely subsystem: counter scope.
  Investigation performed: prior source wiring inspection and unchanged-source/actual-record comparison.
  Resolution: **controlled**; no unsupported prefill-trace claim.

## Hard-Check Gaps

- No unmet check remains for this exact reduced candidate control. The passing verifier is bounded to its published matrix and existing canary scope, including selected KV pages/future tails and explicit inactive/null/spare sentinels rather than every inactive KV page.
- Public serving integration, full64 accuracy/quality, APC/chunked-serving, complete-context behavior, performance, and Stage11 release gates are separate unfinished work. These are not waived by the reduced repair acceptance.
- Ownership is sampled every100ms. Maximum observed gap is102.484ms; shorter events cannot be excluded. This clean approximately70-second child interval does not prove sustained host isolation for long APC/evaluation work or erase repeated earlier interference.

## Promotion Identity and Checkpoint Scope

| Item | SHA-256 |
| --- | --- |
| Live baseline generator | `a8348dac14766d88564dad335f258b3ba07981685c15c673c6819961a5693182` |
| Accepted candidate generator | `5a7bbdee4c1009fe3fdbf5673727e4a2fe21cd802503b7bf6a0fce83c17cb3c0` |
| Exact five-line patch | `b234a02b0c84bc2dd4378cc22bc1026e1b60d5a91adcbf09ca43f07135ea454e` |
| Candidate method AST | `0b2378e7ce1be5f9484c00262bb092238da6aedb06d8e5227e16f5a54cd973ae` |

Promote the exact branch into `models/autoports/qwen_qwen3_8_27b/tt/generator.py` and checkpoint that repair with compact evidence/provenance. Do not promote the experimental grouped-prefill helper or probe instrumentation on the strength of this repair review. Do not include unrelated dirty files or push the checkpoint. Since the isolated binder deliberately requires the pre-promotion live baseline, future post-promotion validation must use an explicit new source transition; do not rewrite its historical pins to make old checks appear current.

## Scope Inspected

- Already-read installed stage-review skill and prior preparation/runtime diagnostic reviews.
- Retry1 README, verifier, wrapper, host tests, commands, bindings, actual JSON/log/execution/observation/native-map receipt, and preparation manifest; immutable original revision5 verifier and actual artifacts.
- Commands: independent complete `verify_retry.py` (exit0); audited `check_host.py -v` (5/5, exit0); direct local verifier diff; stdlib JSON/hash comparison and local file reads. Artifact comparison tool chunk `0545df`.
- No model/device/server/network/process-inspection actions, source edits, or nested agents. Only this report was written. Python-only exact promotion does not require a C++ build.
- After root's promotion, a stdlib hash check confirmed the live target equals the accepted candidate, and baseline/patch hashes agree with `promotion.json`. No old live-source preflight was rerun after promotion.

| Retry artifact | SHA-256 |
| --- | --- |
| `actual.json` | `05d95feecfbd0a3816d9fdd82f0eddb50b524a7d90874949b4b745e7050e07d9` |
| `actual.log` | `a84da60b4e2654d85b238d92e355996d5c906f146b3b30afafd0ce55cbcc889f` |
| `actual_execution.json` | `78e664bbb367024a4436afd66e85af3e62d97d0596ea53e836c9430fb1bdf4eb` |
| `owner_observation.json` | `ec7fb1d341ab26e4f5d2d929ab9e3526353a3ce8d60d980f9d4cb26585951ea0` |
| `native_runtime_binding.json` | `f26ec9a3b37a2835e11493a156b51b22a73615a00c831bb7d82c0fbf4c7d6bed` |
| `bindings.json` | `fe401dc4182f7669421f96202c84bd4ef638b4a6271c1912a022fdb894b49962` |
| `verify_retry.py` | `493abd554dd43ca1613e347ade4464665aac6d0d703cd34ae87676c5a7d593d0` |

## Residual Risk

This acceptance establishes the repaired inactive-sampler-input contract and preservation of active behavior in the exact reduced matrix, with valid native/trace/cleanup and sampled ownership evidence. It supports exact source promotion and a local checkpoint. Broader model/serving validation and durable host isolation remain outside this accepted unit and must retain their current unfinished status.
