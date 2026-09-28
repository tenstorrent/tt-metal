# Stage Review

Verdict: clean-pass

Independent review of the **local pre-publication gate** for the post-pipeline
Gemma-4 TTFT optimization, completed 2026-09-28. This is not a release-accuracy
verdict, remote qualification, or completion of the user's overall task.
No required work remains within this local gate. The authorized remote build,
dispatch, terminal benchmark result, exact-reference audit and final remote
cleanup remain mandatory after publication.

The review covered the live candidate on `mvasiljevic/gemma4-ttft-opt`, based on
`e7ae5f184a`, before its TT checkpoint. It was performed independently and
read-only, except for writing this report. No server, vLLM request, hardware
operation, reset, build, push or dispatch was performed by the reviewer.

## Required Work

None within the local pre-publication gate. The remaining remote qualification
is a subsequent gate, not an implicitly passed check or a waived user requirement.

## Other Concerns

- The selected deployment is the explicit **synchronous latency-priority
  profile**, not the inherited async launcher default. The new
  `gemma4-autoport` catalog emits `--no-async-scheduling`; its launcher test
  checks the negative flag and absence of the positive flag. Canonical Gemma
  catalog defaults are unchanged. The documented local latency command also
  selects sync explicitly.
- The primary S128/O128/C1 medians re-derived from native request arrays are
  **95.241, 95.357 and 95.028 ms** across the initial matrix and two complete C1
  repeats. The original baseline is 419.105 ms. The matched-warmup legacy async
  control is 443.567 ms; thus the larger warmup count does not explain the
  improvement. The first selected cohort has mean TPOT 20.373 ms, mean E2EL
  2682.509 ms and output throughput 47.712 tokens/s. These are HTTP end-to-end
  results, not device-only estimates.
- This is a narrow warmed-primary target result, not a universal sub-100 ms
  guarantee. S128/O16 initially measured 107.186 ms, with later complete-cohort
  repeats at 97.385/95.592 ms and P99 100.276/102.068 ms. S129 and S256 remain
  above 100 ms. All cohorts are retained. O128 has only three measured requests
  per cohort; its P99 is descriptive, not a statistical tail guarantee.
- Async remains a supported throughput-oriented alternative. Its S128/O16 and
  S128/O128 TTFTs are 103.379/104.294 ms, not target-met results. Sync increases
  C1 TPOT approximately 6–9%. At C8/C32 it largely shifts first-decode capture
  from TTFT into the first ITL: whole-request latency increases approximately
  0.85%/0.40% versus optimized async, despite much lower TTFT. These costs are
  disclosed in the README, comparison reports and performance summary.
- Both final serving profiles have full shared qualitative/benchmark/sampling
  evidence, each with **72 passed, 1 skipped** sampling tests. Selected sync
  completed in 1114.96 s; async in 1135.72 s. The skip is the configured
  all-vocabulary logprobs limit. Ordinary TP4 logprobs tests use the explicit
  host compatibility path; they are not proof of device logprobs support.
- I read the generated outputs, not just their verdict fields. All six selected
  greedy suite outputs match the async, legacy and previous-stage controls.
  The 18 selected greedy replays and six seed-71 sampled dictionaries match
  legacy exactly, including finish reason and usage. Rendered prompt IDs agree
  across the profiles and with reported prompt-token counts. Inherited wording,
  factual-quality and capped-output limitations remain; this review makes no
  blanket accuracy or natural-language quality claim.
- Source inspection found no unresolved trace/cache/page-table/sampling defect.
  Exact-length/cache-identity/shape invalidation, per-request token refresh,
  changed-table refresh, canonical sampler ownership, prefill-only lifecycle,
  and fallback paths have both code and affected-boundary evidence. The selected
  precision, expert grouping 32, full page-table columns and 262144 context
  contract are preserved. The 1..1024 trace bound is not a public context cap.

## Hard-Check Gaps

- The actual optimized-source image build and remote QB2/Shield benchmarks on
  `bh-qb-ge`, `p300x2` have not run at this gate. Local execution uses the
  existing image's compiled TTNN binary, identified as image commit `7aee192`.
  Static/mock build tests cannot establish remote compilation, installation,
  registry/authentication, scheduling or performance success. Record exact
  pushed references, dispatch inputs/run URL, runtime provenance, terminal
  result, artifacts and cleanup before overall closure.
- Publication update after the local verdict: the stage owner reports successful
  normal inference/Shield pushes, but a 403 on publishing the tested vLLM plugin
  source and no repository push permission; the required commit is not remotely
  resolvable there. These external responses were reported to this reviewer,
  not independently reissued. An authorized writable source location or access
  correction is needed before an honest exact-source dispatch. This external
  blocker does not change the local code verdict and does not waive the remote
  gate. No new fork/repository is authorized by this report.
- Actual local vLLM is installed `0.26.0+empty`, with only the plugin overlaid
  from `tenstorrent/vllm@7f72b1c6e905f5137fe3377f2e7b42738d3f271d`. The image
  tag's apparent 0.21 version and the monorepo engine are not the tested engine.
  The new build helper reconstructs the pinned source-built empty-target
  engine recipe and installs only the nested plugin. Installer/override hashes,
  version constraints, origin checks, source propagation and labels are
  inspected; successful construction of that image remains a remote gate.
- Full maximum-context execution is inherited evidence, not a new 262144-token
  rerun in this stage. Current changes retain the full 8192-column cache tables,
  allocate the full-context direct-probe cache and cover changed trace boundary
  cases 1023/1024/1025. Additional persistent prefill payload is bounded at
  5,181,568 bytes/chip before allocator/trace overhead; inherited trace and
  reserve budgets are unchanged.
- Full Ethernet Watcher instrumentation could not fit the image firmware
  configuration buffer. Worker Watcher and allocation tracking passed after
  bounded recovery; full Ethernet Watcher coverage is not claimed.
- Serving device time/roofline and detailed shared-suite ITL arrays were not
  collected. The request arrays support the occupancy timing redistribution,
  but exact first-use latency attribution is not proven. No new profiling run
  is required merely to strengthen this already bounded warmed-latency claim.

## Anomaly Ledger

- Observed anomaly: async misses 100 ms; initial selected sync O16 also misses,
  and sync has a decode tradeoff.
  Evidence: `acceptance_matrix`, `acceptance_sync_matrix`, both complete sync
  repeat directories and final scheduler/original comparison reports.
  Affected path: HTTP scheduling and first-token delivery.
  Control or comparison: original baseline plus identical-warmup legacy async;
  three retained sync O128 cohorts; complete low-ISL and C8/C32 rows.
  Likely subsystem: scheduler handoff and decode-trace lifecycle, not a 4x
  kernel speedup.
  Investigation performed: re-derived medians, lengths, request counts, errors,
  full texts, TPOT/E2EL/throughput and occupancy first/rest ITLs from raw data.
  Resolution: controlled. Select the measured sync latency profile, preserve
  async as a throughput option and constrain the claim to the measured warmed
  primary workload. No missed row is relabeled target-met.

- Observed anomaly: shared-suite primary TTFT is 711.193 ms async and 147.951 ms
  sync, unlike the warmed matrices; early prose inferred an initial probe.
  Evidence: suite native CLI logs and `native_protocol_source.log`.
  Affected path: benchmark protocol and first-use state.
  Control or comparison: explicit warmup-10 native matrices; matched legacy
  warmups; repeated complete C1 cohorts.
  Likely subsystem: first-use/capture and process history; exact contribution
  is unresolved.
  Investigation performed: checked the installed native source: its apparent
  initial-test banner does not send a request when readiness timeout is zero.
  Shared suites and original occupancy actually have zero warmups/no probe.
  Resolution: fixed protocol description; controlled result scope. Both slow
  zero-warmup outcomes remain in the final performance summary. No cold/first-use
  target is claimed or substituted with warmed results.

- Observed anomaly: unusual compound wording, closed-versus-isolated
  thermodynamics error, capped answers and a Fibonacci wording mismatch.
  Evidence: actual selected/unselected suite outputs, chat replay JSONs,
  previous-stage qualitative controls, HF reference and `qualitative_verdict.md`.
  Affected path: generated natural-language quality.
  Control or comparison: exact greedy and fixed-seed parity across legacy,
  optimized async and selected sync; previous controls show the same wording
  and thermodynamics error classes.
  Likely subsystem: inherited model/selected-precision behavior and sampling,
  rather than an observed optimization regression.
  Investigation performed: read all final selected outputs and control texts;
  checked prompt rendering, tokenizer identity, IDs, seeds, caps, usage and
  finish reasons. Unseeded exact strings are not claimed reproduced.
  Resolution: controlled for this optimization's no-observed-regression claim.
  Existing accuracy/quality limitations remain, and release accuracy is not
  waived or completed by this review.

- Observed anomaly: initial full Watcher attempt failed before model execution;
  firmware reported ACTIVE_ETH configuration size 28464 exceeding 26624.
  Evidence: Watcher failure/recovery logs and `watcher_down44.*`.
  Affected path: instrumentation startup/device health.
  Control or comparison: bounded reset/list/mesh-smoke recovery; healthy four
  chips; subsequent worker Watcher/allocation checks.
  Likely subsystem: image Ethernet firmware instrumentation capacity.
  Investigation performed: inspected the failure boundary, recovery and final
  12 eager, 24 replay/fallback, 12 mode/reset and four prefill-only results.
  Resolution: controlled. Ethernet instrumentation disabled explicitly, worker
  instrumentation passed; no full-Ethernet coverage claim. Inherited CCL packet
  and semaphore warnings produced no observed corruption or execution failure.

- Observed anomaly: candidate shared BF16 geometry changed full-model tokens;
  wider expert grouping and host-yield candidates did not establish a win.
  Evidence: shared geometry diagnostics, full-model geometry/grouping sweeps,
  archived disabled yield experiments and experiment ledger.
  Affected path: experimental projections, expert grouping and host transport.
  Control or comparison: real layer-0/layer-5 activation/weight diagnostics and
  complete-model first/decode token comparisons under explicit policies.
  Likely subsystem: custom BF16 projection numerical behavior; wider-group
  stack cost; unproven host handoff hypotheses.
  Investigation performed: checked adapted diagnostic retries, numerical
  differences, full-model timing and actual selected source. Selected gate K22
  and down 44-core geometry retain the existing precision; 68 real expert rows
  are bit-exact, and the full geometry sweep has 78 matching rows.
  Resolution: controlled by rejection. No shared-precision change, wider default
  grouping or host-yield hook is accepted. Experimental alternatives are opt-in
  or archived disabled, not silently included in the final measured path.

- Observed anomaly: original workflow resolved a tested monorepo SHA in the
  standalone plugin repository; an early build plan assumed the wrong engine.
  Evidence: runtime versions/source hashes, original installer/override hash
  comparison, workflow audit and changed build/source-routing code.
  Affected path: remote image provenance and model registration.
  Control or comparison: actual installed engine and byte-identical original
  installer recipe; preserved standalone default behavior.
  Likely subsystem: wrapper repository selection and engine/plugin conflation.
  Investigation performed: traced QB2 input through Shield SHA resolution and
  Docker build; inspected full source identity checks, nested-plugin install,
  runtime dependency pins, labels and CPU test evidence (114 inference, six
  Shield, one QB2).
  Resolution: fixed locally. Remote build/runtime proof remains explicitly
  pending, not inferred from those tests.

- Observed anomaly: shutdown logs report nanobind leaks and forced EngineCore
  termination; a holder initially remained after serving stopped.
  Evidence: final server log, baseline/prior-stage teardown controls,
  `final_server_cleanup.log` and `local_cleanup.md`.
  Affected path: process teardown/device ownership.
  Control or comparison: same inherited teardown warning family; final launcher
  exit zero, no API/engine owner, and empty device/port checks after holder stop.
  Likely subsystem: inherited binding teardown, not proven runtime request leakage.
  Investigation performed: checked completion before SIGINT, owned PID/container
  identities, final device nodes 0–3 and port 8000 records; foreign containers
  were left untouched. Holder exit 143 corresponds to its deliberate stop.
  Resolution: controlled and locally cleaned up. No active owned device/server
  process remains in the recorded final checks.

- Observed anomaly: repository hooks normalize raw logs/JSON, and three observer
  JSONs exceed the publication size limit; initial normalization hit root-owned
  evidence files.
  Evidence: finalized `publication_manifest.json`, deterministic gzip archives,
  snapshot utility, publication work log and clean precommit log.
  Affected path: evidence integrity and publication.
  Control or comparison: preserved original/compressed hashes and sizes,
  separate normalized hashes, type-sensitive JSON equality and log
  whitespace-only equality. Ownership repair was limited to manifest-listed
  browsable stage artifacts.
  Investigation performed: independent gzip/decompressed/current hash checks
  for all 537 entries, comparison-report hash checks, source hashes against launch
  metadata, and final utility verification. The three oversized originals are
  gzip-only for publication; runtime caches and tensors are excluded.
  Resolution: fixed. Verification reports `Verified 537 artifacts;
  status=finalized`; precommit and staged/worktree whitespace checks pass.
  The original bytes remain recoverable, not silently replaced by normalized
  evidence. This report is outside the sealed raw tree.

## Scope Inspected

- Goal/skill paths: the parent's copied user contract for warmed C1 low-ISL TTFT,
  selected precision, full context, canonical sampling, C8/C32, measured wins,
  exact-ref remote qualification and cleanup. Fully read `stage-review`,
  `optimize`, `vllm-integration`, `tt-enable-tracing`, `qualitative-check`,
  `tt-device-usage` and prerequisite `model-bringup` under the installed
  `tt-model-bringup/0.1.14/skills` root, plus applicable repository instructions.
- Artifact paths: this documentation directory's README, ledger, work log,
  AUTODEBUG/AUTOFIX, comparisons, qualitative verdict, performance summary,
  context contract, workflow audit, cleanup and publication utility; the model's
  `readiness_vllm/ttft_optimization` raw baseline/legacy/async/sync matrices,
  repeats, suites, chat replays, launches/server logs, geometry/Watcher evidence,
  diagnostic experiments, runtime/protocol provenance and publication manifest;
  relevant inherited `doc/optimized_vllm/after`, `readiness_vllm/qualitative_*`
  and `doc/full_model/qualitative_hf.json` controls. Also inspected the final
  host precommit output `/home/mvasiljevic/gemma4-ttft-precommit-clean.log`.
- Code paths: model `tt/generator.py`, `tt/generator_vllm.py`,
  `tt/multichip_decoder.py`, affected/new tests and TTFT tools; inference build
  script, bundled-plugin installer/manifest, Dockerfile, catalog/model spec and
  tests; QB2 dispatch wrapper/test; Shield dispatch, resolver, build workflow
  and contracts in the three owned compatibility checkouts.
- Commands run: read-only `rg`, `sed`, git status/diff/show/log, shell syntax
  checks and artifact-only Python calculations/comparisons; the snapshot
  utility's read-only `verify` mode; final `git diff --cached --check` and
  `git diff --check`. No serving, sampling or hardware tests were initiated by
  the reviewer. Test outcomes above were checked in saved execution evidence.
- Runtime provenance: selected launch metadata matches current implementation
  SHA256s: generator
  `115787fff21d672efdaaa6af4c6dcb40a5aa7104bad4a1f5f311466a5dd00246`,
  adapter `0f1712be8d82638279fefc38f864f56fa36bef9f606a066961b874417478dd89`,
  decoder `ef1716fb826e81ac7ecdf7d6c54ae18eb4c511d095f774a09fbcf87f795c0d75`.
  No accepted runtime source changed during final evidence formatting.
- Post-verdict mechanical pin follow-up: inspected the new inference checkpoint
  `b6c06944f6d099350f8e9dac3b3e72c58f146112` and Shield checkpoint
  `4f900886b74b82f8fd028d40f1c011c0dc322070`. QB2 now references exactly
  `tenstorrent/tt-shield/.github/workflows/on-dispatch.yml@4f900886b74b82f8fd028d40f1c011c0dc322070`;
  its test asserts that full string and retains source repository/ref forwarding.
  This mechanical pin introduces no new blocker. TT/QB2 checkpoint and remote
  publication identifiers belong in the subsequent work-log update.

## Residual Risk

- Remote build and benchmark behavior remain unproven until the required
  exact-reference workflow reaches a terminal result. This local clean-pass
  authorizes the next reviewed stage; it does not establish remote success.
- Short benchmark cohorts, repeated process history and prompt-0 warmups do not
  establish a production latency distribution for arbitrary new prompts. Cold
  starts, nonaligned longer prompts and O16 tails can exceed 100 ms.
- Selected sync prioritizes TTFT over decode/whole-request throughput. Do not
  present occupancy TTFT phase redistribution as eliminated compute or a
  throughput gain.
- Inherited quality, capped output, host-logprobs compatibility and binding
  teardown limitations remain explicitly scoped. No release accuracy waiver,
  new maximum-context soak, full Ethernet Watcher pass or roofline claim is
  implied by the successful local optimization review.
