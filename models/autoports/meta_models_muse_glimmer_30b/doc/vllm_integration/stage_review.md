# Stage Review

Verdict: clean-pass

Stage 9, vLLM integration, `meta-models/Muse-Glimmer-30B`. Round 10, independent
reviewer, read-only. No device opened, no server started, no hardware reserved.
Perturbation experiments were run against a copy of the tree in
`/tmp/mgreview/`, never in place.

## Required Work

None. Every completion requirement in the stage contract was re-derived from the
artifacts and holds. The two items below are non-blocking.

## Other Concerns

- **`work_log.md` §12 overstates the round-9 guard rewrite by exactly one
  enumerated evasion.**
  §12 states: *"All eight evasions from rounds 7-9 are now `rc=1`, and all four
  true sentences are `rc=0`."* Thirteen of the fourteen evasions enumerated in
  §10/§11/§12 do return `rc=1`, and all four true sentences return `rc=0`, but
  one does not:

  ```
  "All 21 logprobs tests, e.g. the chat ones, pass."   ->  rc=0   (want 1)
  ```

  This is the fourth of the four evasions listed by name in §11 (round 8), which
  §11 recorded as fixed. Round 9 replaced the period-tolerant character window
  with sentence scoping, and `_sentences()`
  (`doc/vllm_integration/bench/check_reported_figures.py:131-140`) protects
  filename dots but not abbreviation dots, so it splits on `e.g. ` / `i.e. ` /
  `cf. ` / `vs. `. Neither resulting fragment carries both `logprobs tests` and a
  pass-word, so the guard never sees the claim. Verified by planting the sentence
  in both `README.md` and `work_log.md` of a `/tmp` copy and running the checker
  with `--check`.

  Reproduce:
  ```
  rsync -a --exclude readiness_vllm/server.log \
    models/autoports/meta_models_muse_glimmer_30b/ /tmp/mg/models/autoports/meta_models_muse_glimmer_30b/
  touch /tmp/mg/models/autoports/meta_models_muse_glimmer_30b/readiness_vllm/server.log
  printf '\nAll 21 logprobs tests, e.g. the chat ones, pass.\n' \
    >> /tmp/mg/models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/README.md
  python3 /tmp/mg/models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/bench/check_reported_figures.py --check ; echo $?
  ```

  This is scoped to the self-check instrument, not to a stage deliverable: no
  false claim about the model, the adapter, the serving path or any measurement
  currently stands in either document as a result of it. The wrong sentence is
  the §12 self-report about the linter. Suggested fix, either order: add
  `(?<!\be\.g)(?<!\bi\.e)(?<!\bcf)(?<!\bvs)` (or protect the abbreviation dots the
  same way filename dots are protected) in `_sentences()`, **or** correct §12 to
  say which evasions are covered. Do not leave the sentence as written.

- **`work_log.md` §12 contradicts itself and `AUTODEBUG.md` on why
  `probe_full.json` is absent.** The same bullet opens with *"that arm hung before
  reaching its JSON write"* — which `logs/probe_full.log` confirms; it stops
  mid-run at 06:01:06 with no exit — and then says *"there is no
  `probe_guard.json` and no `probe_full.json`, both runs exiting at a guard before
  the write."* Only `probe_guard` exited at a guard. `AUTODEBUG.md:132-135` has it
  right. One-word fix.

- **Cosmetic**: the server log's `get_max_tokens_all_users` line reports
  `~14.83 GB/device` (derived from the 1,048,576-token budget) while the pool vLLM
  actually allocates is 16,416 blocks = 14.86 GB/device, which is what
  `README.md` and `probe_full_fixed.json -> per_device_kv_cache_bytes` report. The
  README figure is the measured one; only the log line is the estimate.

## Hard-Check Gaps

These are sentences nobody has written and no round has caught. Recording them so
the ratchet's real perimeter is on the record, not as work.

- `_sentences()` splits on `:` and `;` as well as `.`, so a claim whose filename
  and pass-word land in different clauses escapes both guards:
  `` `test_request_isolation.py` is a clean file: all tests pass. `` -> `rc=0`;
  `The isolation file is clean; every test in it passes.` -> `rc=0`.
- The logprobs guard keys on the literal `logprobs tests?`, so
  `The logprobs suite passes in full, all 21 of them.` -> `rc=0`.
- The bare-citation exemption still trusts a two-word negation adjacent to the
  name, as §12 itself states. The three constructions §12 claims are `rc=1` are
  `rc=1` (verified).
- `serving_audit.json -> marker_provenance` correctly records that both committed
  server logs predate the `DEGRADED PATH …` markers, so marker absence is not
  evidence *about those runs*. The stage substitutes timing/counter evidence,
  which is sound; a rerun of the primary benchmark under the current markers would
  close it directly. Not required — the existing evidence ties the claim to the
  measured path.
- The adapter decides `refresh_inputs` from its own `_device_inputs_current` /
  `reset_batch` / `slot_remap` state rather than from the plugin's
  `can_use_steady_decode_fast_path`. The two are consistent today (verified
  against `vllm_tt_plugin/async_decode.py:318-341`) and the stale-input probe
  drives the adapter with deliberately corrupted host values, so this is not a
  defect; it is an unasserted coupling between two repos.

## Anomaly Ledger

- Observed anomaly: `Allocating device buffers is unsafe due to the existence of
  an active trace. These buffers may be corrupted once a trace is executed.`
  Evidence: `readiness_vllm/server.log` (1 occurrence), `logs/probe_full.log`.
  Affected path: sampling-trace pre-compile between `_capture_decode_trace` and
  `_capture_sampling_trace`.
  Control or comparison: identical warning classified in
  `doc/full_model/README.md:1406-1410` and `doc/optimized_full_model/README.md:1158`.
  Likely subsystem: tt-metal allocator / trace lifecycle.
  Investigation performed: independent grep of the full 84 MB server log for
  `ERROR|Traceback|FATAL|corrupt|Disabling async` returns this one line and
  nothing else; the trace-capture code path is unchanged by this stage.
  Resolution: controlled — inherited, previously classified, not a new anomaly.

- Observed anomaly: `Using custom scheduler class vllm_tt_plugin.scheduler.TTScheduler
  … If you have subclassed Scheduler instead of AsyncScheduler, you will see
  degraded performance due to async scheduling being disabled.`
  Evidence: `async_overlap/server.log`.
  Affected path: `--async-scheduling` overlap validation arm.
  Control or comparison: `vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py:31` —
  `class TTScheduler(AsyncScheduler)`; the same log then prints
  `Asynchronous scheduling is enabled.` and never `Disabling async scheduling`.
  Likely subsystem: vLLM scheduler registration.
  Investigation performed: read the base class; the warning's conditional does not
  apply.
  Resolution: controlled — generic warning, condition not met.

- Observed anomaly: `determinism_vllm.json -> standalone_baseline.first_divergence: 2`
  and `identical: false` on all six qualitative prompts.
  Evidence: `determinism_vllm.json`, `qualitative_comparison_chat.json`,
  `qualitative_vllm_vs_datatype_sweep_chat.json`.
  Affected path: served-vs-standalone token comparison.
  Control or comparison: `determinism_baseline_recheck.json` (character-level,
  381-703 chars, `identical_over_common_prefix: true` 6/6) and
  `qualitative_stripped_divergence_chat.json` (token-level with id 200023 removed,
  `stripped_first_divergence_vs_standalone: null` 6/6).
  Likely subsystem: OpenAI API special-token stripping, not the model.
  Investigation performed: re-derived both from the committed completion sets.
  Resolution: fixed/controlled — fully explained; the HF row is *not* explained by
  it and the README says so.

- Observed anomaly: p1 diverges from the HF control at token 1 (`to=user` vs
  `to=self`).
  Evidence: `qualitative_stripped_divergence_chat.json` (`stripped_first_divergence_vs_hf: 1`).
  Affected path: prefill channel-token argmax.
  Control or comparison: the datatype-sweep stage records
  `first_divergence_from_hf: 1` for p1 with `<|message|>` present on both sides
  and no stripping; `doc/datatype_sweep/channel_margin_probe.json` scores the
  position under the shipped `c14` policy at 0.0625 logits — one BFP4 step.
  Likely subsystem: inherited precision policy.
  Resolution: controlled — pre-existing, not serving-introduced.

- Observed anomaly: one `trigram_loop_fraction` of 0.5 in the degenerate check.
  Evidence: `logs/degenerate_check_all.log`.
  Control or comparison: it is the 6-token p4 greedy completion; adjacent
  duplication 0.0; exactly one such row in 36 measurements.
  Resolution: controlled — sample-size artifact, checker treats trigram looping as
  advisory, log exits 0 with "No degenerate output detected".

- Observed anomaly: 10 sampling-suite failures.
  Evidence: `readiness_vllm/sampling_tests.log`.
  Control or comparison: 7 reproducibility-only (batch-1 seeding, top-k, greedy,
  different-seed variety all pass) and 3 correctness-class, each resolved by
  measurement — presence margin 3.0/4.5 vs the API's 2.0 clamp
  (`sampling_failure_probe.json`), device-path flip at 0.475 on a 0.5-margin
  prompt (`presence_flip_probe.json` phase B), and all 5 `allowed_token_ids`
  requests emitting 10 in-set ids with the printable set returning 10 characters.
  Resolution: fixed (correctness class) / controlled and disclosed in Limitations 1
  (reproducibility class).

## Scope Inspected

Goal/skill paths:
- `.agents/skills/stage-review/SKILL.md`
- stage contract as supplied in the review prompt

Artifact paths (all absolute under
`/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b/`):
- `doc/vllm_integration/README.md`, `work_log.md`, `AUTOFIX.md`, `AUTODEBUG.md`
- `readiness_vllm/vllm_benchmark.json`, `vllm_result.json`,
  `vllm_ci_serving_benchmark.json`, `vllm_ci_serving_result.json`,
  `sampling_tests.log`, `server.log`, `vllm_qualitative_outputs.json`
- `doc/vllm_integration/probe_full_fixed.json`, `serving_audit.json`,
  `determinism_vllm.json`, `determinism_baseline_recheck.json`,
  `logit_determinism.json`, `kv_budget_probe.json`,
  `sampling_failure_probe.json`, `presence_flip_probe.json`
- `doc/vllm_integration/qualitative/*` (all 8 files, completions read in full)
- `doc/vllm_integration/async_overlap/*`
- `doc/vllm_integration/logs/server_excerpt.log`, `degenerate_check_all.log`,
  `probe_full.log`, `probe_guard.log`
- `doc/context_contract.json`, `doc/datatype_sweep/evidence_perf.json`,
  `doc/datatype_sweep/evidence_accuracy.json`, `doc/datatype_sweep/README.md`,
  `doc/optimized_full_model/prefill_trace_probe.json`, `..._8192.json`

Code paths:
- `tt/generator_vllm.py` (whole file)
- `tt/generator.py`, `tt/model.py`, `tests/test_full_model.py` diffs
- `models/common/readiness_check/run_vllm_server.py` diff
- `doc/vllm_integration/bench/check_reported_figures.py`, `adapter_probe.py`
- `/home/ttuser/dev/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`, `worker.py`
  (diffs), `scheduler.py`, `async_decode.py`

Commands run (read-only, plus perturbation in `/tmp/mgreview`):
- `git status --porcelain` / `git diff` in both repos
- `check_reported_figures.py` (68 checks, `rc=0`) and 40+ perturbation runs
  against the `/tmp` copy
- independent re-derivation of sampling per-file counts from the ANSI-stripped
  log, of every cited path (own scanner, wider than the checker's), of the
  degenerate-check measurement count and maxima, of the trigram bands and
  direction, and of the standalone/teacher-forcing comparison ratios
- greps of the 84 MB `server.log` for error/corruption/async markers

## Residual Risk

- **Prefill is eager.** A measured, disclosed 1.33x at 128 rows is left on the
  table, blocked by `prefill_trace_max_entries = 1` (confirmed on the built
  serving model). Correctly scoped to optimized-vLLM.
- **Seeded reproducibility at batch > 1** stays open. Bounded from both sides:
  batch-1 seeding is reproducible and the 160-candidate logit distribution is
  bitwise identical run-to-run and across 8 batch rows, so what differs is the
  seed stream, not the model.
- **KV pool ships at 16,416 blocks against a proven-feasible 28,672.** A margin
  choice, measured rather than budgeted, and it does not touch `max_model_len`
  (131072 = the contract, no reduction). The probe stops at the first feasible
  rung, so 28,672 is a lower bound on the ceiling, which the artifact states.
- **Uniform `FullAttentionSpec` over 39 sliding layers** costs DRAM. The blocker
  (vLLM's `SlidingWindowSpec` page-table zero-padding vs this port's absolute
  decode positions) is stated, matches the choice `tt_transformers` makes for
  Gemma3/GPT-OSS, and costs memory rather than correctness.
- **The stale-claim checker is a ratchet, not a proof**, and now has one
  documented hole in a claim it reports as closed (Other Concerns 1). The
  underlying stage evidence does not depend on it: every headline figure,
  taxonomy, path and qualitative number in this review was re-derived
  independently of the checker.
