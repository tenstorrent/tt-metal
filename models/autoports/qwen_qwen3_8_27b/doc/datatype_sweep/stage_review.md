# Stage Review

Verdict: clean-pass

Independent review of Stage 8, datatype-sweep, for `Qwen/Qwen3.8-27B` on
`mvasiljevic/qwen38-full-bringup`, with uncommitted stage changes based on
`624f6352a9ca1c339870dbe517e564910bb71b15`. Reviewed the finalized evidence on
2026-09-14 UTC. Paths below are relative to this evidence directory unless
prefixed with the autoport's `tt/`, `tests/`, or `doc/` directories.

## Required Work

None before the stage checkpoint. The policy satisfies the user's numerical
selection contract and the required runtime, capability, provenance, and
qualitative-investigation checks. The selected haiku is **not** conventionally
5/7/5; its controlled disposition is explained below and is not a claim that
all shared-suite instructions were satisfied.

The coordinator must now perform the explicitly required post-review local
checkpoint and record its SHA. This review does not claim that commit already
exists. Only stage-owned files belong in that checkpoint; unrelated
`PIPELINE_BLOCKERS.md` changes must remain untouched. No push or vLLM integration
is authorized by this stage.

## Other Concerns

- The selected BFP4 head produces a stable 5/7/6 haiku where the pinned HF and
  BFP8/HiFi2 TT controls produce 5/7/5. Fixed-geometry repeated native generation,
  host-greedy generation, and a fidelity-only control identify prompt-specific
  weight-precision sensitivity. This is a real retained quality loss under the
  conventional meter audit, not a repaired output. The exact shared prompt asks
  for a haiku without explicitly specifying syllable counts. More importantly,
  the user explicitly selects by minimum full-model top-1/top-5 and traced
  teacher-forcing performance. The quality investigation now meets the
  stage-review requirement for an evidence-backed explanation; it does not
  introduce a new numerical selection gate. Both READMEs and
  `qualitative_review.md` accurately disclose the limitation.
- The initial KV candidate encountered a recoverable device startup stall.
  Its exact native cause remains unproven. The retained capture, bounded
  recovery, equivalent successful retry, and final health evidence support
  continuing this stage, not a claim that a native defect was fixed.

Acceptance was independently checked against raw candidate and confirmation
artifacts, not inferred from the README:

| Evidence | Verified result |
| --- | --- |
| Full-model matrix | 15 complete 64-layer configurations; all pass the required 90% top-1, 98% top-5 and 100% top-100 gates. JSON/CSV ranking and warmed last-two medians agree with raw results; 90 material-group rows describe consumed policies. |
| Selected candidate | `head_bfp4_lofi`: decode 99/100/100%; 40.878129 teacher-forcing t/s/u and 76.607464 ms TTFT at B1/S203/G100. Prefill is 100/100/100%. |
| Comparable policy baseline | 40.141822 teacher-forcing t/s/u. Selected is approximately 1.83% faster in this measured regime. The original optimized-path refresh is separately retained in `baseline_original.json`. |
| Normal default confirmation | `selected_confirmation.json`: 40.848672 teacher-forcing t/s/u, within 0.1% of candidate, with the same selected policy and passing accuracy; no precision-config environment override. |
| Post-selection token-out | `selected_token_out.json`: queued no-readback 41.095462 t/s/u, 58.417927 ms TTFT; deferred complete delivery 41.075005 t/s/u, 58.985590 ms TTFT, at B1/S128/G128. These are separate from teacher-forcing ranking. |
| Context and prompt shape | Full64 S262143/G2 and S262144/G1 reach position 262144. Full64 non-aligned lengths 1, 31, 32, 33, 4095, 4096, 4097 pass eager/traced, remapping, retained-output and capture checks. |
| Cache and batch contract | `selected_contract_b32.json` passes full-batch32 fixed-slot, external-cache, inactive-state, remapping, feedback, history, repeat and continuation checks; continuation PCC is 0.99928558. |
| Device and trace safety | Separate tracker/watcher run passes on real layers 0 and 3 across four devices; full64 lifecycle checks pass separately. Final bounded listing exits 0 and all four device owner lists are empty. |

The selected runtime uses BFP4 decoder/head weights with LoFi material matmuls
and FP32 destination accumulation, BF16 projection activations/residual/CCL/
logits, BFP8 paged KV, FP32 recurrent state, and uint32 tokens. Sensitive fixed
computations and the separate HiFi2 final norm remain explicit. The sweep
includes canonical mixed and BFP8 policies, material BFP4/LoFi and HiFi2
controls, head controls, KV BF16/BFP8/BFP4, activation/CCL BFP8, layer exceptions,
the selected-head/KV combination, and disabled projection FP32 accumulation.
No obvious missing legal precision candidate invalidates “fastest evaluated.”

`precision.py` precedence, generator/model construction, per-layer decoder
policy, actual uploaded/copied weight dtypes and compute attributes, CCL
workspace dtype, activation casts, residual adds, cache allocation/binding,
and terminal head/logits/sampling paths were inspected. Recorded policy fields
are consumed or explicitly validated as fixed assumptions. This is supported
by actual runtime summaries and `propagation_check.json`, not config names alone.
The default reproduces the candidate. Both final Pareto PNGs show all 15
evaluated points, the appropriate frontier, red selected point, and dotted
minimum-accuracy line.

## Hard-Check Gaps

- The 100-position readiness reference measures teacher-forced agreement with
  pinned HF outputs. It is not an AIME dataset solution score or a general
  instruction-following accuracy estimate.
- Maximum-context runs establish execution/capacity and valid final positions,
  not full-context HF numerical agreement. Batch32 evidence uses short prompts;
  it does not establish batch32 at maximum context. The memory contract makes
  this distinction and preserves the advertised 262144-token capability.
- Two warmed ranking samples support the nominal fastest evaluated policy,
  not statistical separation of close candidates or a global optimum. The
  selected-head/BFP4-KV combination is especially close. No eager or diagnostic
  haiku timing is used to rank candidates.
- Watcher/tracker evidence uses a reduced real-layer case; full64 trace/cache
  behavior is covered by separate functional guards and contract tests. These
  controls address observed warnings, not every possible future lifecycle.
- No serving/API or vLLM behavior is established. The user excluded that work.

## Anomaly Ledger

1. **Observed anomaly:** Noncanonical layer-exception keys were accepted but
   silently missed by canonical string lookup.
   **Evidence:** `tt/precision.py`, `tests/check_datatype_evidence.py`,
   `policy_validation_equivalence.json`.
   **Affected path:** Explicit precision-policy layer overrides.
   **Control or comparison:** Integer `0` and string `"00"` versus canonical
   string `"0"`; all 15 candidate policies plus selected before/after validation.
   **Likely subsystem:** Host policy validation.
   **Investigation performed:** Reviewer reproduced the ignored override,
   checked the early rejection fix and canonical success, and inspected the
   validation-only source difference and exact policy equivalence.
   **Resolution:** fixed. Accepted runtime policies and measured kernels are
   unchanged; prior hardware evidence remains applicable.

2. **Observed anomaly:** Selected native tokens agree with exact HF generated
   tokens at 98 positions although readiness reports 99% top-1.
   **Evidence:** Preserved `.refpt`, native-token arrays,
   `reference_topk_order_audit.json`, teacher-forcing scoring source.
   **Affected path:** Accuracy interpretation.
   **Control or comparison:** Stored reference top-k ordering versus HF argmax
   generation; position 15 has tied maxima with different ordering.
   **Likely subsystem:** Reference tie/scoring convention.
   **Investigation performed:** CPU-only reference inspection and recomputation
   of every stored native result against actual reference top-k entries reproduce
   reported top-1/top-5/top-100. Selected's sole scored miss is position 17,
   where its output remains top-2.
   **Resolution:** controlled. The reported metric is correct for its recorded
   scoring convention; no sampler correction or metric rewriting is justified.

3. **Observed anomaly:** `kv_bfp4_evaluated` stalled during initialization and
   exited 143 after capture and owned-process termination.
   **Evidence:** `AUTOTRIAGE_kv_startup.md`, `AUTOFIX_kv_startup.md`,
   `triage_kv_startup/`, failed run log, successful recovered candidate.
   **Affected path:** Device startup/weight upload, before KV allocation.
   **Control or comparison:** Same-policy full64 retry with actual BFP4 KV,
   passing accuracy and traces after bounded recovery.
   **Likely subsystem:** Device completion/transport startup; exact cause unknown.
   **Investigation performed:** Preserved timeout/host stack and sync-failure
   evidence; inspected failed listing, one successful reset, four-device listing,
   and exact Ring mesh open/close smoke. Triage timed out; no native root cause
   is claimed. Diagnostic faulthandler timing is disclosed.
   **Resolution:** controlled. Successful recovery/retry is evidence for
   recoverability, not a source fix or proof BFP4 caused the stall.

4. **Observed anomaly:** Recovery prose claimed empty owners while a raw JSON
   field showed PID 602116 on device 1.
   **Evidence:** `triage_kv_startup/ownership_timeline_correction.json` and
   unchanged `recovery_actions.json`.
   **Affected path:** Recovery provenance and ownership accounting.
   **Control or comparison:** Original narrow tool transcript versus field label.
   **Likely subsystem:** Evidence timestamp labeling.
   **Investigation performed:** Verified empty-owner output and model exit 143
   at 22:58:20, listing at 22:58:26–22:59:00, reset launch at 22:59:14, and JSON
   owner snapshot at 22:59:15 during reset. PID command was not captured.
   **Resolution:** fixed documentation, controlled residual uncertainty. Reset
   ownership of PID 602116 remains explicitly an inference; raw evidence is kept.

5. **Observed anomaly:** Normal logs contain an active-trace allocation warning
   mentioning possible corruption.
   **Evidence:** Candidate logs, `selected_non_aligned.*`,
   `selected_contract_b32.*`, `selected_watcher.*`, native warning source.
   **Affected path:** Trace allocation/lifetime and output/cache ownership.
   **Control or comparison:** Full64 changed-input/remapping/retention tests and
   separate allocation-tracker/watcher run with clean device checks.
   **Likely subsystem:** Generic warning in untracked trace execution.
   **Investigation performed:** Inspected warning trigger, capture guards,
   ownership assertions and result counters; watcher/profiler separation holds.
   **Resolution:** controlled for the exercised paths. No observed corruption
   is hidden behind the passing accuracy score.

6. **Observed anomaly:** Motherboard fallback and L1 semaphore-placement warnings.
   **Evidence:** Runtime logs, native physical-system discovery and multicast
   factory source, `anomaly_ledger.md`.
   **Affected path:** Topology metadata and semaphore allocation.
   **Control or comparison:** Same warnings in baseline/candidates and successful
   four-device mesh, full64 checks and watcher run.
   **Likely subsystem:** Bus-ID fallback for motherboard identification and
   normal-L1 allocation when L1_SMALL is zero.
   **Investigation performed:** Checked exact fallback/allocation source and
   absence of corresponding allocation or topology failures in completed runs.
   **Resolution:** controlled; no numerical or runtime fault demonstrated.

7. **Observed anomaly:** Short qualitative outputs truncate inside reasoning or
   before story completion.
   **Evidence:** Actual HF/TT shared-suite outputs at 256/1024 tokens and
   `tt_qualitative_story_2048.json`.
   **Affected path:** Generation budget and qualitative interpretation.
   **Control or comparison:** Exact pinned HF chat prompts; both HF and TT story
   truncate at 1024. TT reaches EOS at 1452 with a 2048-token budget.
   **Likely subsystem:** Token-budget exhaustion.
   **Investigation performed:** Read completed answers and story; verified exact
   1024-token TT prefix preservation and final EOS for all six prompts. No
   mechanical looping, wrong language, prompt leakage or cross-request leakage
   was found. Longer TT completion is not presented as equal-budget HF evidence.
   **Resolution:** controlled.

8. **Observed anomaly:** Selected haiku ends “A new truth emerges,” six syllables.
   **Evidence:** Actual final answers, `haiku_selected_retry.json`,
   `haiku_baseline.json`, `haiku_head_hifi2.json`, `AUTODEBUG_haiku.md`,
   `AUTOFIX_haiku.md`.
   **Affected path:** Precision-sensitive autoregressive answer quality.
   **Control or comparison:** Identical full64 S60/G1024, capacity1088, 34-page
   BFP8 KV, history1023 and SDPA geometry. Selected native repeats and host-greedy
   yield identical 347-token EOS-trimmed output; BFP4/HiFi2 matches it exactly;
   BFP8/HiFi2 repeats the earlier correct 418-token baseline answer. HF is also
   conventionally 5/7/5.
   **Likely subsystem:** Head-weight precision changes the generation trajectory.
   **Investigation performed:** Compared complete token streams, common source/
   reference hashes, actual dtype/fidelity and geometry. First divergence at
   generated index 31 is away from page/chunk boundaries. Repetition, isolated
   request order and host sampling controls do not implicate trace state or
   sampling delivery. Fidelity-only repair is refuted; BFP8/LoFi was not tested
   for this specific prompt.
   **Resolution:** controlled, not fixed. The numerical fastest-policy contract
   passes with this explicitly retained semantic limitation. No perfect
   instruction-following or universally equivalent creative output is claimed.

9. **Observed anomaly:** Host Fibonacci checker rejected correct code using a
   list named `sequence` instead of hard-coded `fib`.
   **Evidence:** Actual HF/TT generated functions,
   `tests/check_qualitative_artifacts.py`, `qualitative_metrics.json`.
   **Affected path:** Host qualitative code validation.
   **Control or comparison:** Both functions produce correct sequences for
   negative/zero and positive input cases.
   **Likely subsystem:** Over-specific test assumption.
   **Investigation performed:** Direct source inspection and restricted host
   execution; reviewed narrow generalization to locally initialized lists while
   retaining call/import guards and behavioral assertions.
   **Resolution:** fixed. The passing degeneracy/code checker is not used as a
   blanket semantic-quality verdict.

10. **Observed anomaly:** Two diagnostic/launcher attempts failed independently
    of model correctness: live launcher editing caused a Bash read-offset error;
    initial haiku reporter assumed the optional host page-table mirror remained.
    **Evidence:** `failed_attempts.json`, retained initial logs/exits,
    `haiku_selected_native0.json`, corrected probe and successful retries.
    **Affected path:** Experiment launcher and post-generation reporting.
    **Control or comparison:** Frozen-launcher `head_bfp8_lofi` rerun exits 0;
    selected haiku retry reads/asserts actual device page table and exits 0.
    **Likely subsystem:** Harness assumptions.
    **Investigation performed:** Inspected failure locations, invalidated host
    mirror semantics, preserved original artifacts and complete replacement runs.
    **Resolution:** fixed. Failed/incomplete attempts are excluded from ranking.

## Scope Inspected

- **Goal/skill paths:** `goal_contract.md`; repository-local
  `.agents/skills/{stage-review,datatype-sweep,tt-device-usage,qualitative-check}/SKILL.md`.
- **Artifact paths:** All 15 accepted configurations and raw results; baseline,
  selected/default/token-out/readiness/non-aligned/batch32/watcher evidence;
  commands, environments, source hashes and exit files; full runtime weight/
  fidelity summaries; memory reports and `doc/context_contract.json`; both
  Pareto plots; actual shared HF/TT qualitative outputs and focused controls;
  startup triage; host checks, precommit log, README/work log and anomaly reports.
- **Code paths:** `tt/{precision,model,generator,multichip_decoder,optimized_decoder}.py`
  and relevant inherited functional paths; candidate, summarizer, memory,
  propagation, benchmark, readiness, tracing, batch contract, deferred generation,
  qualitative/reference and focused-probe harnesses; shared teacher-forcing
  scoring; relevant canonical Qwen36 precision and native warning source.
- **Provenance checks:** All 89 final-index file hashes match. All 39 source
  archives and 987 member hashes were verified, including exact member lists.
  Historical logical source paths remain recoverable using the documented
  archive mapping. The committed-reference copies are byte-identical to the
  measured root `.refpt` and metadata. Historical policy snapshots differ from
  final validation only as recorded in the equivalence audit. Mutable review
  and checkpoint documents are intentionally outside the runtime freeze.
- **Commands run by reviewer:** Read-only Git/source searches and file reads;
  small Python JSON/CSV/hash/tar/policy analysis; CPU-only `torch.load` with
  `weights_only=True` for the reference; restricted generated-function checks;
  visual inspection of both final PNGs. No TTNN import, hardware command,
  server, reset, device experiment, or long test was run by this reviewer.
  Parent-run checks and their exit statuses were inspected, not represented as
  reviewer executions. Python/docs-only changes require no C++ build. This
  report is the only file authored by the reviewer.

## Residual Risk

- Quantization can change semantic choices outside the small numerical and
  qualitative samples; the controlled haiku error is a concrete example.
- Performance applies to the recorded four-Blackhole TP4 Ring setup, inherited
  geometry and specified timing regimes. Close rankings can change with noise
  or other shapes; no global optimality or serving throughput claim is made.
- Recovery succeeded, but the initialization stall's exact native cause remains
  unknown. The preserved evidence supports later investigation if it recurs.
- Maximum context and batch32 are separately demonstrated capabilities, not a
  promise of their Cartesian product. No advertised context reduction was used
  to obtain the selected result.
