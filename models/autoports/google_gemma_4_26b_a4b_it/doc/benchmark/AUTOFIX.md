# AutoFix report

## Starting evidence
Fresh isolated diagnosis: AUTODEBUG.md. Failing contract: final upstream lm-eval
accuracy and two vLLM profiles require a provisioned accuracy client, live
server profiles and a validated full-phase collector.

## Hypothesis experiments
- Hypothesis: the active environment merely hides an existing evaluation client.
  Experiment: active Python module lookup and `python -m lm_eval --help`;
  isolated agent inspected /opt/venv and bounded existing environments/caches.
  Result: no lm_eval in either environment and no alternate harness found.
  Verdict: local existing-client recovery refuted. See run/lm_eval_probe.log
  and AUTODEBUG.md. No source fix can supply the missing dependency.
- Hypothesis: Stage 10's working server remains attachable.
  Experiment: process inventory, localhost:8000 socket connect, Stage 10 cleanup.
  Result: connection errno111 and explicit stopped owned process records.
  Verdict: refuted. Relaunch would count in the eventual timed runner.
- Hypothesis: existing phase evidence can satisfy both required profiles.
  Experiment: source inspection and Stage 10 launch configuration review.
  Result: no full-phase collector; historical concurrency1 has32 server slots.
  Verdict: refuted. Reusing those results would violate the benchmark contract.

## Final status
Blocked by an unavailable prerequisite under the repository's explicit dependency
installation prohibition. No speculative source edit or hardware run was made.
No device failure was observed and reset/recovery is inapplicable. No unrelated
process was stopped. Provision the upstream client, implement/validate phase
collection, and run the genuine profiles on resume. Stage remains incomplete.
Evidence/context checks and exit statuses are in run/check_results.json.

Checker integration: a null generator_module caused an AttributeError. Omitted the unobserved optional value (no module identity invented) and reran; checker now exits2 with the intended missing-generated-implementation evidence failure. Initial log retained.

## Continued setup checks

The offline tokenizer probe initially compared a BatchEncoding directly with a
list. Inspecting the actual return proved this was a probe error. The retained
benchmark_tokenizer_check.py extracts input_ids for Mapping returns and passes
both modes, preserving the original native template.

Source-only phase diagnosis and reducer tests are in PHASE_COLLECTION_DESIGN.md
and run/phase_reducer_tests.log. No runtime instrumentation is claimed complete.
Native top_k64 mismatch is independently documented in
AUTODEBUG_sampling_policy.md and run/sampling_policy_probe.json. A host-sampling
route is a candidate to validate, not an applied fix or a completed benchmark.
