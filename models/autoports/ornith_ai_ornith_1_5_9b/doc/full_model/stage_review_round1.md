# Stage Review

Verdict: clean-pass

Independent review, round 1, of stage 6 (`full-model`) for
`ornith-ai/Ornith-1.5-9B`, performed on 2026-11-13. The reviewed checkout is the
live worktree on `main`, based on
`c2254f91c38840cd4069a1ecbbd3f4874e27d557`. Paths below are relative to this
directory unless stated otherwise. The original contract is
[review_request.md](review_request.md).

## Required Work

None. Inspection found no unresolved correctness defect, missing stage gate,
unsupported stage capability reduction, or contradicted performance claim.
The evidence supports completing this full-model stage. Local stage-owned
checkpoint commits and recording their SHAs remain the stage owner's authorized
closure steps; this review did not commit or push anything.

The decisive checks were independently examined as follows:

| Contract | Evidence inspected and conclusion |
|---|---|
| Complete optimized TP4 text stack | Read all production model/generator, prefill, cache, batch, decode and sampling-state modules, and the decoder diff against the base commit. `tt/model.py` assembles all 32 layers, embedding, final norm and vocabulary-sharded head. The decoder diff changes pooled cache addressing and masked positions; it does not replace the selected projection, fidelity, activation/CCL or inter-layer residual policy. Public prefill uses the prepared device runtime. No host activation computation or less-optimized decoder fallback was found. |
| AIME24 reference and accuracy | Inspected reference generation provenance, the original-tokenizer/chat-template contract, `accuracy_summary.json`, `final_gate_commands.json`, and actual accepted prefill/teacher-forcing logs. Prefill is 99/100 top-1 and 100/100 top-5/top-100. Teacher forcing is 98/100 top-1 and 100/100 top-5/top-100, reproduced in `logs/delivered_teacher_forcing.log.gz` after the final readiness-extras change. These meet the explicit top-5 >=98% and top-100=100% gates. |
| Resumed and arbitrary-length prefill | `logs/accepted_prefill_resume.log.gz` and `../../readiness/prefill_resume_check.json` show full-stack length 513 resumes at 32, 68 and 512 with PCC and top-1 agreement 1.0. Both wrong-history controls fail where run. `serving_prefill_probe.json` tests mixed lengths, reordered slots, starts `[68,0,32]`, inactive peers and implicit horizons 33 -> 68 -> 513 against solo execution. Logits and state are bit-equal. The implementation preserves canonical physical profiles and aligned recurrent snapshots rather than retaining the diagnostic BF16/fixed-32 alternatives. |
| Context and batch capacity | `capacity_full_final.json`, its raw log and runner prove all-layer 262143-token prefill, finite decode at position 262143 and full 262144-token prefill, with 820 programs unchanged after preparation. `../context_contract.json` separately accounts for weights, KV, recurrent/conv state, staging and reserve using measured allocator data. `batch32_full.json` and its runner prove full-stack B32 at context 512, including solo comparisons, row permutation, resident feedback and inactive-state preservation. The README correctly separates these capacities. |
| Split traced device feedback and lifetime | `tt/generator.py:607` prepares all supported binding state and paths before capture; `:707` acknowledges only the exact producer logits; `:952` replays model and sampler without token reconstruction; `:963` queues owning host transfers before overwrite. Model and sampling trace variants cover masking and penalties. Persistent device token output, position advancement, seed progression and scoped history restoration are implemented explicitly. Runtime, pending-output and repeated-request probes support these mechanisms. |
| Allocation tracking and cross-request reuse | Accepted correctness/quality commands enable `TT_METAL_TRACE_ALLOC_TRACKING=1` with program-cache exclusion disabled. `trace_alloc_tracking/positive_control.log.gz` actually rejects one deliberately live unsafe device allocation and succeeds after its release. `watcher_trace.json`, `external_cache.json`, `serving_decode_probe_final2.json` and capacity artifacts record stable programs and/or trace maps, capture/release counts and changed request shapes. No blanket corruptible allocation scope appears in the delivered generator. |
| Serving-facing contracts and external ownership | Checked `serving_contract.json` against pinned plugin `042a94863ad27d88207bcf71abdf67122c6f4c27`, including its interface and decode-reload documents. The inventory has 114 rows: 76 supported, 38 explicitly not supported, zero missing; supported evidence paths exist. Source and probes cover arbitrary physical page pools, constructor injection, atomic rejection, scoped release/reset, resident remap, writable host-logits output and two pending asynchronous outputs. The declared recurrent-state prefix-cache exclusion is valid; no caching-enabled server is claimed. |
| Common sampler and terminal cost | Inspected both common `SamplingGenerator` and pinned `Sampling1D` source interfaces, plus `sampler_selection.md`. The selection accurately identifies existing common penalty/history management as the integration advantage; it does not falsely claim the alternative lacks TP or feedback. `sampler_probe3.json` compares semantic greedy against host argmax and common forced argmax on real logits. `sampling_transition2.json` exercises seeded continuation, greedy/sampled transitions, all three penalties, scoped resets and inactive rows. The measured terminal sampling/feedback cost is about 0.605 ms, about 6.8% of whole-model token-out latency; it is not dominant. |
| Primary performance | Recomputed medians from `performance_primary.json` and inspected the runner, commands, raw log and reduced operator profile. The 128-input/128-output, B1, configured-context-2048 warmed medians are 92.579 ms TTFT and 8.936 ms/token, or 111.904 tokens/s/user. Each primary run records 127 model and sampler replays, no per-step token/page refresh, one initial position refresh, 128 caller readbacks, no capture/release and 712 stable programs. Teacher forcing is separately 8.953 ms/token; the 8.314 ms model-only diagnostic is correctly labeled fixed-device-input rather than teacher forcing. |
| Original-HF qualitative control | Read the actual final HF/TT autoregressive strings, all 12 six-prompt seed-0 outputs and all 20 thermodynamics control outputs. Inspected prompt rendering, original HF snapshot identity, generation settings, rubric, blind export/scoring and unblinding. Recomputed the fixed score counts: HF 4/10 substantive failures, TT 1/10; all 20 are cap-limited. No TT excess or uncontrolled mechanical/wrong-language defect was established. |
| Delivered source and final checks | Independently matched all 50 `delivered_source_snapshot.json` source hashes and all 174 decompressed `archive_manifest.json` entries, including lengths. Examined the two generator hash transitions in `import_format_provenance.json` and `readiness_extras_change.json`, current qualified calls/kwargs handling, focused validation and delivered full-stack teacher forcing. The changes do not invalidate the measured no-extras model/sampler path. `logs/final_host_contract_tests2.log.gz` records 40 passing tests; `logs/final_hooks_review.log.gz` records passing applicable hooks. |

## Other Concerns

- **Shared model limitations are real.** The matched thermodynamics set contains
  overbroad second-law/efficiency claims, and every sample reaches the 256-token
  cap without a completed final answer. The fixed blind score file has SHA256
  `bb0c1bf3c69695f5472dae6f6be38071c4d9746a50981e45ee8486f637275c10`, matching
  `qualitative_entropy_control/unblinded_results.json`. Both limitations are
  recorded in `../eval_readiness/limitations.json` with
  `original_model: "same"` and the matched evidence. They are controlled model
  limitations, not Required Work or an owner decision. The observed 1/10 versus
  4/10 does not establish a general TT quality improvement.
- **A DRAM-sharded LM-head candidate remains unmeasured.**
  `profile/layer_stack_accounting.json` identifies the real head geometry and
  reports about 0.749 ms, 345 GB/s and 67.4% of the stated 512 GB/s model. Its
  ideal bandwidth-bound saving is approximately 0.244 ms, or 2.7% of primary
  whole-model latency. The report does not claim this candidate was rejected or
  that the selected head is globally optimal. This is a reasonable optimization
  candidate for the next stage, not a missing named stage-6 gate.

## Hard-Check Gaps

No missing required hard check was identified. The following boundaries limit
the conclusions and must remain attached to the results:

- Maximum-context evidence checks allocation, execution, finite outputs and
  program reuse. It is not a 262144-token HF accuracy comparison. B32 is tested
  at context 512, not at 32 simultaneous maximum-length requests; the latter's
  KV allocation alone exceeds physical DRAM.
- The three-run performance result is a warmed generator result on the stated
  128/128 workload, with token readback. It is not cold-start latency or a vLLM
  serving benchmark. Allocation-tracked correctness timings and concurrent-HF
  diagnostic timings are not interchangeable with the quiet primary timing.
- The operator profile contains one linear and one full-attention layer plus
  the actual embedding, norm, head and sampler. Multiplying layer costs gives
  an estimate, not a measured 32-layer trace breakdown; the report explicitly
  exposes the approximately 9.381 ms estimate versus 8.936 ms whole-model result.
- Several contract-transition and watcher probes intentionally use layers 0/3
  and small batches. Full-stack accuracy, capacity, ownership and B32 probes
  complement them, but do not turn every reduced probe into an exhaustive
  all-shape/all-layer test.
- The serving inventory establishes a generator boundary against the pinned
  plugin. Adapter wiring, real scheduler concurrency and API/evaluation gates
  remain for the later authorized vLLM stage. No server was required or run for
  this stage, and the unsupported recurrent prefix-sharing capability is explicit.

## Anomaly Ledger

- Observed anomaly: A changed logical request shape initially created unsafe
  program allocations while decode traces were live.
  Evidence: `autofix/AUTODEBUG.md`, `autofix/AUTOFIX.md`, retained failure logs,
  `watcher_trace.json`, `capacity_full_final.json`,
  `trace_alloc_tracking/positive_control.log.gz`.
  Affected path: Eager prefill followed by reused model/sampling traces.
  Control or comparison: Requests 33/34/31/65, serving mixed/resumed requests,
  stable prepared program counts, and deliberate unsafe-buffer rejection.
  Likely subsystem: Shape/offset-dependent program preparation and allocation
  lifetime.
  Investigation performed: Inspected preparation/profile/runtime code and exact
  logits acknowledgment; checked tracked command environments and successful
  replay evidence. The positive control proves checking was active.
  Resolution: fixed. Generator-wide preparation precedes capture; no blanket
  exemption or per-request trace eviction is retained.

- Observed anomaly: Early resumed-prefill implementations diverged at starts
  32 and 68 for a 513-token request; a reduced BF16 control did not establish a
  valid all-layer repair.
  Evidence: `resume_hypothesis.md`, `full_resume_bf16.json`,
  `full_resume_overlap.json`, `full_resume_profiles.json`,
  `logs/accepted_prefill_resume.log.gz`.
  Affected path: Shape-conditioned prefill precision/projection families and
  recurrent block phase across a resume boundary.
  Control or comparison: Full versus resumed same-cache execution; both
  wrong-history controls; retained diagnostic alternatives and final all-layer
  starts 32/68/512.
  Likely subsystem: Prefill execution profile and recurrent continuation state.
  Investigation performed: Read the failed hypotheses and selected profile,
  padding, snapshot and overlap implementation. Final accepted logs report
  PCC/top-1 1.0, with controls failing as expected.
  Resolution: fixed. The final path preserves optimized policies and bounded
  aligned overlap; diagnostic BF16/fixed-32 fallbacks are not selected.

- Observed anomaly: Inactive output values and asynchronous shared output
  ownership required explicit treatment; earlier probes are not the final
  output contract.
  Evidence: `serving_decode_probe_final2.json`,
  `tests/full_model_serving_decode_probe.py` under the model root,
  `tt/generator.py:963` under the model root, `cache_ownership_full.json`.
  Affected path: Fixed-slot sampled tokens, host logits and multiple pending
  submissions.
  Control or comparison: Two submissions before read, later host submission
  versus saved earlier output, inactive cache/state preservation, immutable
  submission masks and resident whole-pool remap.
  Likely subsystem: Output transfer ownership and inactive-row presentation.
  Investigation performed: Inspected queued owning transfers/event ordering,
  final sentinel handling and probe assertions for output and state equality.
  Resolution: fixed. Final token/logit sentinels are -1/zero and reads are
  observational; external backing remains owned by the caller.

- Observed anomaly: Free-running HF/TT story tokens diverge early despite high
  teacher-forced agreement.
  Evidence: `../../readiness_autoregressive_final/` completion archives and
  metadata, `autoregressive_review.json`, `degenerate_output.json`,
  `logs/delivered_teacher_forcing.log.gz`.
  Affected path: Free-running generation.
  Control or comparison: Same raw story prompt, complete saved HF/TT texts and
  separate matched AIME logits/teacher-forcing gate.
  Likely subsystem: Autoregressive branch divergence after numerical choice
  differences; no cache or mechanical-collapse evidence was found.
  Investigation performed: Read both complete 128-token outputs. Both continue
  the tree-shadow story coherently in English and terminate mid-explanation at
  the fixed cap; the first differing token is index 8. Checked that raw story
  continuation supplements the original-chat-template quality suite.
  Resolution: controlled. Exact free-running token identity is not claimed.

- Observed anomaly: Thermodynamics overstatements and cap-limited thinking
  appeared in the shared qualitative screening.
  Evidence: `qualitative_shared_seed0/`, `qualitative_entropy_control/`,
  `qualitative_verdict.json`, `../eval_readiness/limitations.json`.
  Affected path: Chat-template sampled generation, temperature 1, top-p 0.95,
  top-k 20, presence penalty 1.5, seeds 0..9, 256 output tokens.
  Control or comparison: Original HF snapshot
  `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, original tokenizer/template and
  matched settings; ten blinded samples per side, fixed rubric and scores.
  Likely subsystem: Original-model content and the chosen generation cap.
  Investigation performed: Read all output texts and score notes, checked
  generation manifests/signatures and pre-unblinding score hash, and recomputed
  failure counts. The six-prompt single-seed run is treated as screening only.
  Resolution: controlled. HF 4/10 versus TT 1/10 substantive failures and
  10/10 incompleteness on both sides support the documented shared limitations,
  not completed-answer or broad-accuracy claims.

- Observed anomaly: Some accepted measurement hashes precede the delivered
  generator hash, and earlier HF control attempts timed out.
  Evidence: `final_gate_commands.json`, `final_performance_commands.json`,
  `import_format_provenance.json`, `readiness_extras_change.json`,
  `readiness_extras_commands.json`, `delivered_source_snapshot.json`,
  `qualitative_entropy_hf_retry_commands.json` and per-worker retry/completion
  records.
  Affected path: Artifact identity and completeness.
  Control or comparison: Recorded pre/post source hashes, normalized import
  change, targeted extras probe, delivered all-layer teacher forcing and all
  twenty completed matched-control sample records.
  Likely subsystem: Formatting/readiness adapter compatibility and bounded
  CPU-reference job duration.
  Investigation performed: Verified delivered hashes and every archived-content
  hash; inspected final source branches, command completion and the completed
  control outputs. Failed attempts remain distinguishable from successful retries.
  Resolution: controlled. The import rewrite and ignored unknown high-level
  extras do not alter recognized/no-extras execution used by primary performance;
  low-level validation remains strict. Final evidence does not depend on an
  incomplete HF worker run.

## Scope Inspected

- Goal/skill paths: `review_request.md`; installed
  `/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/`
  `stage-review/SKILL.md`, `full-model/SKILL.md`,
  `tt-enable-tracing/SKILL.md`, `qualitative-check/SKILL.md`,
  `optimize/SKILL.md` and its LM-head/sampling reference, applicable
  `tt-device-usage/SKILL.md` and `model-bringup/SKILL.md` instructions;
  repository `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifact paths: Model README, this directory's README/work log, context and
  serving contracts; final gate/performance commands; source and archive
  manifests; accepted accuracy, resume, runtime, batch/capacity, cache/ownership,
  sampling and serving JSON/logs; trace positive control and watcher evidence;
  `profile/` reports/accounting/provenance; final autoregressive completion
  archives; shared six-prompt and matched twenty-output quality sets, rubric,
  blind scores, manifests and verdicts. Historical failed hypotheses and final
  replacement evidence were distinguished rather than pooled as one run.
- Code paths: Complete model-root `tt/model.py`, `tt/generator.py`,
  `tt/prefill_runtime.py`, `tt/prefill_ops.py`, `tt/prefill_contract.py`,
  `tt/prefill_plan.py`, `tt/cache_contract.py`, `tt/decode_contract.py`,
  `tt/batch_state.py`, `tt/sampling_state.py`; base-to-worktree
  `tt/multichip_decoder.py` diff. Read relevant runner/probe assertions,
  including full-model performance, capacity, B32, qualitative generation,
  serving decode, external ownership and fallback checks. Also inspected common
  sampler source and the pinned alternative sampler interface.
- External read-only interface scope:
  `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin` at
  `042a94863ad27d88207bcf71abdf67122c6f4c27`, especially
  `MODEL_INTERFACE.md`, `DECODE_RELOAD_CONTRACT.md` and execution requirements;
  `/home/ttuser/ornith-9b/dependencies/tt-transformers` sampler at
  `25ebcdb3311bb86d1cb4ceff77bb8bf481f4cc73`. The plugin checkout is clean.
- Commands run: Read-only `rg`, `rg --files`, `sed`, `cat`, `git diff`,
  `git status`, `git rev-parse`; small standard-library Python analyses using
  `json`, `gzip`, `hashlib`, `pathlib`, statistics/counting and CSV inspection.
  These verified source/archive hashes, supported-row evidence paths, matched
  control counts and signatures, and performance medians. No TTNN import,
  device open/reset, hardware test, server, long test, implementation mutation
  or additional reviewer was used. The only file authored by this reviewer is
  this report.

## Residual Risk

This is an independent source-and-artifact review, not an independent hardware
rerun. Existing tests cover representative boundaries rather than every valid
request ordering, length, sampling combination or hardware state. Current
evidence supports the stated stage-6 contract and limits; it does not establish
production-serving behavior, exhaustive maximum-context numerical accuracy,
completed answers at the short quality cap, or optimal terminal performance.
The next stage should preserve those distinctions and revalidate any paths it
changes. The authorized closure plan is local stage-owned commits only, never a
push.
