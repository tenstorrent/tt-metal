# Stage Review

This is a user-directed follow-up using targeted skills and independent review
agents, not an end-to-end invocation of the agentic-research/model-bringup
pipeline or every stage gate. `STAGE_REVIEW.md` records bounded source/evidence
reviews requested through the stage-review skill; it is not a pipeline stage
completion certificate. The automatic acceptance gates proposed here have not
been installed into the pipeline.

Verdict: more-work-needed

## Current Review: Warmed HTTP Trace Reuse

Independent source/artifact review on 2026-09-11. Measured checkpoint
`4a02bf62cf510f3807f177ad826555bc9b648c0d` adds startup compilation and checked
reuse of resident decode traces to the previously reviewed native prefill,
seed policy, explicit reload contract and device-filled logit rows. The
requested warm-HTTP optimization is demonstrated: median 128/252/C1 TTFT falls
from 728.482 to 177.361 ms, a 551.120 ms reduction and 4.107× speedup. This is
1.487× the separate 119.301 ms B1 generator prefill measurement; it is not equality of those
measurement boundaries. Full-model adapter state comparisons and the completed
HTTP smoke, text and benchmarks support the measured change. No new concrete
source defect was found in that valid-request path. Three required-point CI
runs remain active, and the 60/500 ms targets remain unmet. Historical findings
below retain their original wording; these current dispositions take precedence.

## Required Work

- P1: Complete the required 13-point OSL252 CI execution and report its results.
  Evidence: runs34617909914 (`decode_only`),34618127149 (`all`) and34622652794
  (warmed `all`) are active on separate runners, with build jobs skipped. Their
  dispatch identities and source overlays are verified; no complete matrix
  is available. The third run tests the measured warm-serving checkpoint.
  Two successful local benchmark points do not constitute that matrix.
  Required next step: inspect the executed rows, requested output counts and
  long-context/concurrency results, and resolve any failures. Preserve all
  runs; no run's dispatch or intermediate progress is a pass.

- P1: The performance requirements remain unmet.
  Evidence: full64 B1 generator medians are119.301ms S128 and1299.242ms S4096
  against60/500ms. Selected local median HTTP TTFT at128/252/C1 is177.361ms,
  still2.956 times60ms. The4096/252/C8 burst is a different admission/concurrency
  regime and must not be compared to an isolated B1 S4096 target.
  Required next step: keep these targets open while the already-dispatched
  matrix establishes the current serving behavior. Report actual results and
  remaining bottlenecks; neither explaining the gap nor the achieved relative
  speedup satisfies the target.

## Closed Findings and Verified Scope

- **Warm HTTP result and source identity:** raw
  `warm_serving/readiness_vllm/vllm_result.json` confirms4/4 requests, zero
  failures and all1008 requested output tokens at128/252/C1. Median TTFT is
  177.361ms, mean180.004ms and p99195.039ms. The preceding device-fill run used
  the same benchmark configuration and measured728.482ms median. Mean TPOT
  is essentially unchanged,88.822 to88.740ms. Server logs show one eligible
  startup capture and four setup reuses during this short benchmark. That
  counter is not an all-mode capture total for the entire serving suite.
  All five source hashes in `artifacts/warm_serving_source.json` match the
  committed4a02 checkpoint, including the runner. Later invalid-input cleanup
  and capture-counter hardening must not be represented as already measured
  by this run.

- **Full-model state and reuse evidence:** the reviewer independently compared
  all four alternating request pairs in `production_trace_reuse_full64.json`.
  Active tokens match on all four ranks, all512 cache/rank digests match, and
  every position tensor matches exactly. The envelope is full64, B32 allocated,
  C1 slot0, S128, context256 and three decode steps per request. Eight captures
  become one capture plus seven reuses across correctness, warmup and timing
  requests. Median adapter prefill plus first decode falls798.811 to248.156ms;
  first decode falls631.824 to90.822ms. Both arms use complete allocation
  checks without GC, isolating the reuse change. These include decode readback
  and are not HTTP TTFT. The independent four-layer comparison also passes.

- **Production ownership contract:** startup compilation precedes capture,
  temporary decode state is restored/released, and real scheduler inputs still
  require authoritative reload. Reuse checks warmed request shape/slot/sampling
  signatures, stable cache and trace owners, and unchanged program-cache
  counts. Complete C++ allocation maps, including program-owned allocations,
  are checked before model state advances and again before sampler replay.
  A late failure aborts without retrying a partially executed token. Both
  active masks, tokens, positions and page-table contents refresh in place.
  All live bucketed sampler namespaces with corruptible exemptions are
  rejected. Shared sampler callers without the optional executor retain their
  existing behavior. Omitting GC conservatively leaves uncollected allocations
  unsafe; this does not disable allocation tracking. One serialized device
  submission owner is required because allocation query and replay are not
  atomic against unrelated external mesh submissions.

- **Warm serving correctness and text:** the unchanged canonical smoke reports
  3passed/1skipped in24.30s, covering mixed-request isolation, top-1 greedy and
  host-path min-p. The all-vocabulary logprob skip proves no logprob accuracy.
  The reviewer read all12 shared chat outputs and verified that all six greedy
  strings are byte-identical to the previous device-fill run. Sampled text is
  coherent, with no recurrence of the dropped words or inconsistent variables.
  Both learning explanations and translations, the greedy haiku and greedy
  Fibonacci answer are complete. Both stories and thermodynamics answers
  reach the fixed256-token limit; the sampled haiku stops within reasoning.
  The sampled Fibonacci function is complete, but its example comment is
  truncated. Both complete function bodies pass reviewer host checks for
  n=0,1,10. These limits do not constitute a complete-answer or release-quality
  pass. The run reports exit0; abort-mode server shutdown remains distinct from
  a graceful device-teardown test.

- **Earlier scatter serving integration:** the unchanged canonical smoke
  in `device_fill_serving/readiness_vllm/sampling_tests.log` reports3passed/
  1skipped in23.86s. The reviewer read all12 generated strings and independently
  confirmed all six greedy outputs are byte-identical to the preceding
  seed-policy run. Sampled outputs have no observed dropped-word or variable-
  name corruption. The sampled haiku, learning explanation, thermodynamics
  answer and translation are complete; both story outputs and the greedy
  thermodynamics answer are limited by the fixed256-token budget. The sampled
  Python's requested Fibonacci function and example are complete and correct
  by inspection; an optional second function begins when the budget ends.
  This closes the demonstrated integration regression without claiming all
  answers finish at this budget. Raw benchmark JSONs confirm4/4 short requests
  with1008 output tokens and8/8 burst requests with2016 output tokens, exactly
  the requested counts. Median HTTP TTFT is728.482ms and11443.820ms respectively.
  The source summary records checkpoint4ea57c41431, and the actual server log
  confirms B32, max_model_len262144, sampling mode `all`, and FABRIC_1D_RING.
  That earlier run did not use trace reuse; the warm run above does.

- **Slot-logit scatter source/static boundary:** production's method body is
  AST-identical to the measured probe candidate, excluding its docstring.
  B1 returns the original object without device work. For B32, one
  `zeros_like(logits)` row is reused for inactive slots; concat puts the
  original local vocabulary row at the absolute slot, then releases the
  temporary zero row and consumed logits. The caller supplies one active
  terminal-logit row and a valid slot derived from the prompt lengths.
  `slot_logits_scatter.json` derives vocab248320/local62080 from the pinned
  checkpoint, uses four distinct rank inputs, and checks all32 logical rows
  on each rank at slots0/7/31. All are exact against both the oracle and old
  implementation. The canonical input remains allocated and unchanged; the
  private working input is consumed, preserving the old ownership contract.
  Output remains tiled BFP8 DRAM with local shape `[1,32,1,62080]`.
  Synchronized method medians are175.957→0.506ms,184.647→0.997ms, and
  175.891→0.521ms. Both arms exclude cloning/readback/output release equally,
  discard two warmups, alternate measurement order and show no program-cache
  growth. These are static operation measurements, not full-model or HTTP
  speedups. Repeated zero-row references still produce a full output; this
  is not a zero-copy claim. The original method is preserved in the JSON;
  the updated probe can load it with `--baseline-artifact`, preventing a
  future run from silently using the new production method as the old arm.

- **Unseeded full-model rank divergence:** production source keeps the shared
  SeedManager policy off by default and enables it only in Qwen SamplingArgs.
  Explicit and mixed seeded paths retain their existing seed calculations;
  all-unseeded draws push fresh shared entropy without setting explicit-seed
  activity, so the sampler remains eligible for internal tracing. The reviewer
  reran the actual AST-only host suite:12 tests pass. In
  `sampling_rank_full64_production.json`, the probe override is false and
  the production policy is true. The four full64/B32 cases contain
  131/147/148/148 decode steps (574 total), all with rank agreement; the
  source sidecar records the loaded files. All four actual completions are
  coherent and end at EOS. Both seed42 arms match each other and the earlier
  seeded baseline exactly. This closes the demonstrated failure without
  claiming the exact intervening RNG-state-changing kernel was identified.

- **Actual HTTP sampled corruption:** the pre-scatter production run in
  `final_serving/readiness_vllm/` passes the unchanged canonical tests
  (3passed/1skipped in30.37s). The reviewer read all12 shared completions:
  the prior dropped words and inconsistent Python variables are absent.
  Learning explanations and translations are complete and coherent; both
  Fibonacci function bodies are correct by inspection. The sampled haiku
  stops during coherent reasoning; stories, thermodynamics answers and the
  sampled Python example output hit the fixed256-token budget. These are
  retained completion limits, not a complete-answer pass. Min-p is a host
  compatibility test and the all-vocabulary logprob skip proves no logprob
  accuracy. The completed selected scatter rerun above supplies subsequent
  integration evidence.

- **Warmup penalty history:** `penalty_recapture_b32.json` records zero
  mismatches on all four ranks for all three resident history buffers after
  setup, and zero mismatches against the expected update after one real traced
  sampling replay. This closes the source finding with four-layer B32 device
  evidence. Canonical v1 resets restore authoritative host history after
  setup; they opt out of snapshots. The full64 serving test must not be
  relabeled as full64 coverage of the separate snapshot-preserving branch.

- Earlier flat-native accuracy, fully active B32 recurrence capacity, shared
  sampler geometry, request-order seed salting and explicit reload semantics
  remain closed as detailed in the review history.

## Other Concerns

The selected local C8 burst completes8/8 requests and all2016 output tokens,
with11285.561ms median HTTP TTFT,90.355ms mean TPOT and59.502 output tokens/s.
This is only1.38% below the previous11443.820ms result. C8 lies outside the
current C1 reuse envelope, and this small difference is not attributed to
reuse. Seeded sampling, non-greedy sampling, penalties and logprobs likewise
use normal capture rather than cross-request reuse. The full64 digest probe
covers slot0 with fixed page mappings; the broader source support for warmed
slot/mapping changes is not relabeled as independently measured digest coverage.

The short result measures actual HTTP delivery with prefix caching disabled,
B32, max_model_len262144 and FABRIC_1D_RING. Logs record startup prefill spans
381.696ms at128 and1498.708ms at4096, decode compilation286.473ms, and retained
capture537.079ms. Those costs move to startup; model loading and KV-pool
allocation are additional startup costs. The177.361ms HTTP result and248.156ms
adapter result use different publication/readback boundaries. Their difference
does not establish a measured server-span breakdown. The B1 generator result
remains119.301/1299.242ms from native validation. These are measured boundaries,
not estimates formed by subtracting isolated kernel or operation timing.

The latest benchmark summary correctly generates its ISL4096/OSL252/C8 label
from the configuration. Earlier historical artifacts retain the stale
100/100/32 label, whose command/config/token counts already proved the actual
executed point. `PIPELINE_GAPS.md` records this metadata error and distinguishes
available-operator discovery misses from defects exposed by later plugin and
sampler dependencies. Its new automatic gates are proposals, not claims that
the pipeline was modified. No material overclaim was found in that ledger.

## Hard-Check Gaps

The CI provenance is internally consistent. All three runs reuse the earlier
image built from native checkpoint6f60917b27b63908f85fc982fe010b1d6ad76523
with plugin c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42. The reviewer independently
verified no changes outside `models/` through source checkpoint
4a02bf62cf510f3807f177ad826555bc9b648c0d, so the native build inputs are unchanged.
The first two runs' fixed tag `mvasiljevic/qwen38-perf-4ea57c41431` resolves to
4ea57c41431a1e80318c4d32d249d5552c4f165f. Their three read-only source overlays include the autoport,
`models/autoports/vllm_bundles` and `models/common/sampling`; omitting the last
path would have retained the old shared sampler in the image. The helper's
`git clone --branch` accepts the fixed tag, while the dispatch's metal-ref
input alone does not select the overlay when image reuse skips SHA resolution.

- [Run34617909914](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/34617909914)
  pins inference-server8d9f3084853883999aec235293d58472f6d783f4 and retains
  `decode_only`, preserving the prior CI sampling configuration. Its prefill
  uses host logits and bypasses device scatter.
- [Run34618127149](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/34618127149)
  pins inference-serverdf501912ccadd1bf4ccd7131042430357933c086. Independent
  source comparison confirms its only change from the first configuration is
  `sample_on_device_mode: all`, exercising device scatter with the same13
  required points. Both CI modes retain FABRIC_1D; local serving used
  FABRIC_1D_RING, so local results do not certify that CI configuration.
- [Run34622652794](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/34622652794)
  pins runtime4a02bf62cf510f3807f177ad826555bc9b648c0d and inference-server
  ed2ef012fd91ff6f8b49119d216786609dbc6b68. The reviewer inspected its YAML
  change: the fixed overlay tag becomes `mvasiljevic/qwen38-warm-4a02bf62cf5`,
  with five tracking/reuse/report/warmup environment settings added. The
  required13-point matrix remains unchanged. It warms all eight requested
  input lengths,128 through261892, and retains sampling `all`/FABRIC_1D.

At review, direct GitHub job queries report all three benchmark jobs in
progress, respectively on `120-qb2-p04t07`, `qb2-120-p01t03` and
`qb2-120-p03t06`, with all image-build jobs skipped.
The inspected dispatch/reusable workflow chain at
ba2f03318608be52c5cb2085469598a89dcfa0fc contains no concurrency cancellation
group; both earlier runs remain active. Durable inputs, matrix and observed
statuses are in `artifacts/ci_dispatch*.json`, `required_ci_matrix.json` and
`ci_overlay_validation.json`. They prove dispatch/provenance, not completed CI.

Maximum-context/current native behavior and the complete13-point CI matrix
are not certified by the short local smokes. The unsafe binary-SiLU candidate
and earlier instance-hook trace-reuse prototype remain unselected. The current
production reuse policy has separate full64 and HTTP evidence; those results
do not retroactively select the earlier prototype. The selected changes are
Python-only, so the supplied AGENTS.md
requires no C++ build. The owner reports formatting checks; this reviewer
independently ran only standard-library AST/JSON inspection and the12 host
regressions, with no Torch/TTNN imports or device operations.

## Anomaly Ledger

- Rank-divergent unseeded feedback and malformed sampled text: fixed for the
  production seed policy by full64 rank controls and the reviewed HTTP runs,
  including device-fill and warm-serving implementations.
- Synthetic warmup token contaminated preserved penalty history: fixed;
  source ordering,12 host regressions and reduced B32 device evidence agree.
- Excess serving TTFT: improved but target unmet. Device scatter first reduced
  short HTTP TTFT by182.516ms; production warm trace reuse then reduced it by
  551.120ms to177.361ms. C1 reuse is now measured, while C8 continues to recapture
  and the60ms requirement remains open.
- Fixed-budget incomplete shared answers: controlled as truncation, with
  coherent text and explicit case-level limits; no blanket task-success or
  release-quality claim is made.
- Server shutdown reports abort-mode engine termination and nanobind leaks of
  operation/config/type bindings after the completed requests. This log does
  not demonstrate graceful device teardown or establish a new scatter/cache
  leak. The separate ownership/lifecycle probes remain the cleanup evidence;
  the HTTP smoke is not relabeled as a teardown test.

## Scope Inspected and Residual Risk

Source: `tt/generator.py`, `tt/trace_reuse.py`, vLLM startup/reload consumers,
common sampling generator, allocator/tracker/trace bindings, pinned c9 plugin
submission/readback source, creation-op source and `probe_slot_logits_scatter.py`.
Evidence: both production adapter JSONs, warm-serving source hashes compared
with checkpoint4a02, actual warm text/raw benchmark/smoke/server logs, static scatter
JSON and preserved baseline, production full64 sampling/source sidecar,
penalty-history JSON, all shared text and test/benchmark outputs under
`final_serving/`, `device_fill_serving/` and `warm_serving/`, dispatch/overlay JSONs, exact
inference-server/workflow source diffs, and current GitHub run/job statuses.
Commands were read-only
`rg`, `sed`, `git diff`, small standard-library AST/JSON scripts, and
`python3 -m unittest discover -s models/autoports/qwen_qwen3_6_27b/tests
-p test_vllm_decode_reload.py` (12passed in the preceding source review).
This warm-serving update used no device operations or Torch/TTNN imports.
It compared artifacts, used read-only `gh api` queries, and executed only the
two reviewed generated Fibonacci function bodies with standard Python inputs.
Only this report was edited. Source hardening after measured checkpoint4a02
requires its own clearly labeled host checks; this review does not silently
extend the hardware result to later edits.

No additional valid-request source bug was found in the measured reuse path.
The requested warm-HTTP improvement and local serving integration are proven
within the documented C1 envelope. The unfinished CI matrices
and unmet targets retain the more-work-needed verdict. This targeted review
is not formal pipeline completion, release certification or a target pass.

## Earlier Review History (Superseded by the Current Dispositions Above)

Latest source inspection includes the serving seed/reload repair and the exact
CI plugin `c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`. The reviewer ran the seven
AST-only `test_vllm_decode_reload.py` tests successfully, with no Torch/TTNN
imports or hardware operations. Qwen's duplicate-seed salting is disabled;
explicit reload commands now determine token/position authority, seed-counter
alignment, parameter refresh and page-table-only updates. Resident overlap
does not inspect stale host token/position/page tensors. Transition penalty
history restoration occurs after warmup and writes the stable penalty buffers
in place, so the new restoration does not replace captured tensor addresses.

**Closed P2 source finding: recapture without sampling-state reset counted a
synthetic warmup token.** A legacy page-table change sets
`will_setup=True` with `reset_batch=False`; legal explicit
`reload_inputs=True, reset_sampling_state=False` has the same effect. The
generator's setup eagerly samples and counts its warmup token, but the
adapter previously restored history only when `reset_batch` was true. Read-only
adjudication using the real AST method and its existing fake generator showed
history `[21,22,700,999]` at the next replay after both triggers (`999` is the
fake setup token). The final implementation now defaults
`preserve_sampling_history=True` on generator setup/capture. When penalties
are active, it clones all three mutable output-history buffers, retains the
original counting warmup, restores the resident buffers, synchronizes and
frees the snapshots before beginning model trace capture. The penalty update
source writes those same three buffers in place; prompt history is not changed
by warmup. The adapter opts out only when it will restore authoritative
history after setup. Source inspection and nine reviewer-run AST tests pass,
including the actual capture function's restore/free/capture ordering and the
opt-out path's absence of snapshots. No further source defect found.

The running canonical server imported the adapter before this final legacy
edge extension. Its transition plan already resets authoritative history and
follows the same operation sequence as the new opt-out path. Its live results
must not be described as full64-layer validation of snapshot preservation.
The owner has planned a separate reduced B32 hardware history probe for that
path's device/resource envelope; this reviewer has not run it.

The initial live canonical mixed-seed isolation failure is **closed** by
`readiness_vllm/sampling_tests.log`: the unchanged `test_mixed_params_batch`,
top1 greedy test and min-p smoke pass; the all-vocabulary chat-logprobs test
skips. The reviewer checked those log rows (3 passed,1 skipped in30.29s).
Min-p exercises host sampling and the skip supplies no logprob correctness
claim. Shared chat outputs, serving benchmarks and the separate snapshot
preservation probe remain pending.

The final native gated-tail source and new sampler evidence were inspected
on 2026-09-11. The initial findings below are retained as review history;
their current disposition is:

- **Closed: final flat-branch full-model accuracy.**
  `artifacts/native_final_full_validation.json` records the production
  default, without experimental norm/program patches, 1872 flat calls and
  zero rank-4 calls. Logical303/physical320 prefill and logical203/physical224
  teacher/generation inputs preserve logical lengths. Both prefill and
  teacher forcing achieve99/100 top1 and100/100 top5/top100. The actual TT
  generation remains coherent English problem setup, as does the100-token
  HF control. Traced generation records99 replays and zero host
  token/position/page-table refreshes.
- **Closed: B32 sampler allocation and adapter lifecycle failure.**
  `artifacts/sampler_followup_validation.json` records successful split-TopK
  boundary IDs and B32 prefill, decode, stale-input replay, remap and mesh
  close. `generator.py` now assembles per-request rows on device, retaining
  rank-local vocabulary ownership. Shared sampling tiles logits before
  padding and restores the requested local-chunk path. The current Qwen
  local62080/padded65536 shape has no16-bit local-index overflow.
- **Still required: planned final live serving and native shared qualitative
  suite.** The reduced adapter output is lifecycle evidence, not full-model
  text quality. `qualitative_prompt_format.json` names the correct checkpoint,
  chat template and six shared prompts; outputs and the comparison to the
  prior Qwen3.8 control still need inspection. Live checks must use the chosen
  CI plugin revision, whose source differs from the locally installed plugin.
- **Closed: native capacity for a fully active B32 prefill.** The newly exposed
  `B*HV <= compute_cores` limit is addressed inside both native entry paths.
  `_forward_batches` derives the limit from the actual grid, slices all six
  inputs on the batch axis, and concatenates head-major output and batch-major
  state on axis0. For110 cores and12 heads,9+9+9+5 preserves slot/head ordering.
  B1 passes the original tensors directly to the unchanged native call.
  `native_batched_b32_s128.json` and `native_batched_b32_s65.json`, supported
  by their full logs, record exact lifecycle checks and normal mesh closure
  with per-request mode disabled. The aligned run covers the flat path and
  the65-token run covers rank4. Five host tests cover partition boundaries,
  paired input/state slices and ordering. These checks resolve the native
  capacity defect; they are not full64-layer B32 quality measurements.
- **Closed: unsupported shared-sampler geometry.** Explicit local chunks now
  reject padded local vocabularies above65536, single-device vocabulary
  sampling and chunks narrower than `max_top_k` before device uploads. The
  existing default single-device path is unchanged. The recorded final host
  log contains nine passing tests, including all three guards.

No concrete bug was found in the dedicated gated-norm tail. Its input
`[B*12,T,128]`, gate `[B,T,1536]`, shared128-element weight and tile-aligned T
match the C++ operator contract. The result is token-major and multiplying it
by z supplies SiLU's remaining factor. Unaligned chunks retain the existing
composite tail. Model-only K-block limit8 selection preserves standalone
decoder defaults. The unaccepted binary-SiLU candidate's whole-graph
non-finiteness remains unresolved, but that candidate is absent from runtime.

The final default warmed medians are119.301ms S128 and1299.242ms S4096. These
replace the earlier numbers for the selected result and remain above60/500ms.

The former shared-API geometry concern is now resolved by the guards above.
The final host-test log also prints nanobind shutdown warnings for Python
binding objects (CoreRangeSet/MemoryConfig and registered types/functions).
This run uses CPU op doubles and performs no device allocation; the warning
does not establish a native-state lifetime defect. The separate hardware
lifecycle logs show successful normal device closure.

## Initial Review (Superseded Finding Dispositions Above)

Independent inspection on 2026-09-11 of the live worktree based on
`761f5e2ff138f173f80c4681bb087b77a08f7e08`, branch
`mvasiljevic/qwen38-deltanet-kda`. Scope is the follow-up full-model prefill
implementation and CI readiness for pinned Qwen/Qwen3.8-27B, not release
certification or a claim that the 60/500 ms targets have been reached. This is
an interim verdict: the owner was running follow-up validation while this
review was written.

## Required Work

- P1: Complete full-model accuracy evidence for the measured flat GDN and fused
  convolution branch.
  Evidence: `artifacts/native_full_validation.json` uses an unpadded 203-token
  AIME prompt for generation/teacher forcing and 303 tokens for the all-logit
  prefill check. `generator.generate` and `prefill_logits` pass those physical
  lengths through. `MultichipDecoder._linear_attention_prefill_chunk_impl`
  selects `flat_forward` only when the physical chunk length is divisible by
  32. These HF checks therefore take the rank-4 adapter and composite
  convolution, while S128/S4096 timing takes flat GDN and fused convolution.
  Why this matters: the flat branch also changes normalization and decay
  preprocessing; the prior identical-input FP64 oracle does not cover those
  differences. Reduced-layer PCC is useful, but the claimed full-model quality
  gate must cover the selected faster branch.
  Required next step: finish the already-running `validate_runtime.py
  --pad-prefill` run, inspect its text/accuracy and physical/logical shapes,
  and retain `native_flat_full_validation.json`. Preserve logical prompt
  lengths through masking and decode positions. No new test framework is
  needed.

- P1: Resolve the actual B32 adapter failure before declaring CI readiness.
  Evidence: `/tmp/qwen_native_vllm_adapter.log` fails in row-major `ttnn.pad`
  during sampling. The native slot-lifecycle JSON passes exact reset/remap/
  peer-state checks, but its runner does not execute the failing sampler or
  subsequent decode. The existing readiness benchmark JSON is an older
  eight-request result and is not current native serving evidence.
  Why this matters: the production serving boundary currently has a concrete
  failure despite standalone prefill succeeding.
  Required next step: complete the sampler repair already being investigated,
  rerun the B32 adapter through prefill, traced decode, stale-input handling
  and remap, then complete the planned live serving smoke. Keep the failure
  and repair in the follow-up ledger. Inspect the shared sampler change
  independently before including it in this stage's checkpoint.

- P2: Run the shared qualitative suite after selecting the native runtime.
  Evidence: current native full-model quality evidence contains only the AIME
  prompt. `doc/qwen38_checkpoint_swap/full_model_qualitative.json` supplies an
  earlier Qwen3.8 suite/control, but predates this change; the checked-in
  readiness chat suite still names Qwen3.6. Both `optimize` and
  `qualitative-check` explicitly require the shared suite after optimization.
  Why this matters: one coherent math continuation does not establish that
  the selected graph and serving sampler preserve other prompt behavior.
  Required next step: complete the planned native shared-suite run, retain
  exact checkpoint/prompt-format metadata, and compare the actual text to a
  matching HF or prior-stage control. The existing pinned Qwen3.8 controls
  can be reused when rendering and generation settings match.

## Other Concerns

- The documentation's earlier statement that the generator itself pads S33 to
  outer128 is incorrect. The probe supplies a padded physical tensor. Direct
  unaligned generator calls take the rank-4 fallback, whose native C++ adapter
  handles internal time padding. Correct the description and distinguish
  logical from physical lengths.
- The native op defaults to HiFi4 with FP32 accumulation. The policy summary's
  `linear_recurrent_fidelity=HiFi2` describes the existing decode setting; it
  does not configure the new native prefill operator. Label the distinction
  in the profile/precision narrative.
- The new profile still contains material projection/reshape/dispatch costs.
  Historical packed-projection and CCL losses on the sequential graph do not
  reject those candidates for this graph. The bounded follow-up can report
  its measured improvement and remaining work; it should not claim the graph
  has exhausted its optimization opportunities.

## Hard-Check Gaps

- The required 13-point OSL252 CI matrix is recovered in `PIPELINE_GAPS.md`.
  Dispatch and results remain pending. This review does not turn preparation
  into a claim that the matrix passes.
- Updated `doc/context_contract.json::native_prefill_followup` records the
  Qwen3.8 identity, outer512/internal32 split and 24,576 bytes/device of shared
  constants. It correctly preserves historical Qwen3.6 capacity evidence as
  historical. Current maximum-context native execution remains a CI risk,
  not a fresh capacity result.
- The reviewer ran no builds, tests, servers, or hardware jobs. Python-only
  changes do not require a C++ build under the supplied AGENTS.md; final
  formatting and the stage owner's checks still need to be recorded.

## Anomaly Ledger

- Observed anomaly: original/native long-context recurrent states disagree.
  Evidence: `native_full4096_oracle.json` and `native_gdn_followup.md`.
  Affected path: rank-4 native recurrence on identical captured real inputs.
  Control or comparison: full-sequence FP64 recurrence; original relative L2
  26.10%, native 2.03%.
  Likely subsystem: algorithm/intermediate rounding and recurrent precision.
  Investigation performed: identical-input oracle comparison, not merely
  incumbent-cache PCC.
  Resolution: controlled for that boundary; flat preprocessing requires the
  separate full-model check listed above.

- Observed anomaly: AIME generations stop while restating the problem.
  Evidence: actual `generation.tt_text` and `hf_text` in
  `native_full_validation.json`.
  Affected path: 100-token greedy chat continuations.
  Control or comparison: same pinned HF checkpoint also stops during the
  problem setup at this budget.
  Likely subsystem: generation budget/checkpoint reasoning behavior.
  Investigation performed: direct text inspection. Both are coherent English,
  without mechanical repetition or apparent cross-request leakage.
  Resolution: controlled; no solved-answer accuracy is claimed.

- Observed anomaly: first flat S128 measurement inflated the unchanged
  full-attention control.
  Evidence: `native_flat_s128.json` versus
  `native_flat_s128_uncontended.json`.
  Affected path: reduced eager timing.
  Control or comparison: concurrent HF CPU work removed; full attention
  returned to its prior latency range.
  Likely subsystem: host contention.
  Investigation performed: repeated uncontended measurement.
  Resolution: controlled; contaminated timing is excluded from the result.

- Observed anomaly: adapter prefill fails in sampler padding.
  Evidence: `/tmp/qwen_native_vllm_adapter.log`.
  Affected path: B32 serving prefill/sampling.
  Control or comparison: standalone native prefill and exact slot lifecycle
  pass, narrowing the failure to the serving sampler boundary.
  Likely subsystem: shared sampler local-chunk selection/layout.
  Investigation performed: owner and AutoFix agent are repairing it.
  Resolution: more-work-needed until the actual adapter rerun succeeds.

## Scope Inspected

- Goal/skills: supplied follow-up contract; `stage-review`, `optimize`,
  `graph-fusing`, `tt-enable-tracing`, `qualitative-check`, and
  `vllm-integration` under `.agents/skills`.
- Artifacts: `README.md`, `PIPELINE_GAPS.md`, `native_gdn_followup.md`, native
  full-model/reference/lifecycle JSONs, matched eager baseline, fresh native
  reduced profiles, context contract, prior Qwen3.8 qualitative controls and
  readiness outputs.
- Code: native runtime diffs in `tt/model.py`, `generator.py`,
  `functional_decoder.py`, `multichip_decoder.py`, `prefill_recurrence.py`;
  relevant optimized-decoder and vLLM adapter consumers; native GDN C++
  implementation; validation/probe and lifecycle runners; current shared
  sampler diff for failure context only.
- Commands: read-only `git status/diff/rev-parse`, `rg`, `cat`, `sed`, `nl`,
  `head`/`tail`, and small Python JSON inspection. One initial `python` command
  was unavailable; inspection continued with `python3`. Only this report was
  written by the reviewer.

## Residual Risk

No additional concrete native shape/state bug was found by inspection. The
4-to-12 head mapping, K-by-V state orientation, three-token fused-convolution
history, masked recurrence, selector-based convolution state, narrowed slots,
and constant teardown order agree with their consumers. Fresh reduced
profiles show the actual native GDN/fused-convolution operators and
BFP4/LoFi projection kernels.

The matched full64 S128 result supports a large improvement (2480.102 to
136.498 ms); the S4096 result is 1834.551 ms without a matched eager full64
baseline. Both remain above their 60/500 ms targets. Reduced profiling is not
a full-model latency floor, and neither this review nor the short quality
smokes certifies release accuracy or all 13 CI points.
