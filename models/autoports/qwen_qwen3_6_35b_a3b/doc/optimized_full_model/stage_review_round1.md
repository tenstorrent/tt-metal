# Stage Review

Verdict: clean-pass

Independent round 1 review of stage 7, optimized-full-model, for
Qwen/Qwen3.6-35B-A3B revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`.
Reviewed the live stage-owned worktree on `main`, starting/current HEAD
`ece17d228671dbc1375e45c6e73f90156707aa30`. All 309 source entries in
`final/measurement_source_provenance.json` match the files inspected. This is
an inspection of completed evidence; no hardware experiment was performed.

## Required Work

None. No P1/P2 correctness, contract, or missing required-evidence finding was
established. The explicit stage-cap instruction and optimize Stop Rule 3 bound
further performance experiments; they do not waive the gates checked here.

The passing review permits the owner to perform the prescribed local
checkpoint and record its SHA. That post-review closure step remains pending
at review time; this verdict does not claim it has already happened. The run
plan specifies local commits only and no push.

The principal acceptance conclusions are supported as follows:

- **Accuracy and generated output:** `final/prefill_accuracy.json` and
  `final/teacher_accuracy.json` each contain 100 predictions, top-1 98%, top-5
  100%, and top-100 100%. Their production-source sidecars match current
  bytes. The exact 161-ID AIME autoregressive pair and all twelve shared-suite
  HF/TT text files were read through their saved paired presentation, with
  membership checked against the raw files. All six fresh shared TT files
  are byte-equal to their accepted stage-6 TT controls. They support the
  bounded smoke verdict, not a completed-answer accuracy claim.
- **Trace and generator contract:** `tt/generator.py` prepares persistent
  state and eager/helper programs before capturing model and sampling traces.
  `_sampling_forward` supplies `tt_out_tok=self.tokens`; `tt/model.py::decode`
  advances persistent positions on device. The measured resident branch
  queues both traces nonblocking, with no token/logit readback or host refresh.
  No new broad tracker exemption or corruptible allocation scope was added.
  `final/replay_all40.json` uses tracking=1, program-cache skip=0 and Watcher=10:
  17 cases, 32 zero-growth cache checkpoints, two captures, zero releases
  before teardown, and the same trace IDs. Cases cover fresh and resumed
  nonaligned lengths, mixed-versus-solo rows, inactive middle slots and
  per-row top-k/top-p/seed changes. The deliberate unsafe allocation in
  `final/positive_control.json` is rejected.
- **Serving state and larger batches:** `final/serving_commands_all40_v3.json`
  passes nine checkpoints, including asynchronous output ownership, dormant
  remapping, RNG preservation and 30 changed/zero unchanged page-table copies
  across ten full-attention layers and three slots. The test derives the
  correct count from actual cache groups and checks every saved table.
  `final/b32.json` exercises all 40 layers and 32 fixed slots, reordered
  continuation, inactive state and stable program cache. Its detailed state
  readback is explicitly scoped to chip 0. The unchanged serving contract has
  56 supported and 30 declared unsupported rows, no missing rows, and existing
  evidence for every supported row. Prefix caching is excluded explicitly
  for `recurrent_state`; no vLLM integration is claimed.
- **Changed kernels preserve the selected policy:** inspected guards and
  dispatch for the new head, cached/narrow coefficient scans, direct/product
  contractions and full state-slab recurrence. The selected slab preserves
  separate FP32 arithmetic, optional BF16 consumption rounding and FP32 state
  commit. All four rounded/unrounded and persistent/temporary 128-step
  Watcher controls contain 12,288 all-rank comparisons each, with zero
  differences. The production slab's reader, compute and writer literals
  independently match the qualified probe. Nonaligned B3 integration and
  the fresh all40 gates cover integration. Cached-scan no-clear and poisoned
  padding controls pass with tracking and Watcher; unsupported geometries
  retain device implementations. Native rows confirm BFP8/HiFi2 head and
  ordinary projections, BFP8/LoFi active experts, BF16 KV, FP32 sensitive
  work and the existing residual/collective contract. No rejected precision
  or residual family was promoted.
- **Comparable performance and gap accounting:** the clean before/final
  records use the same 128-in/128-out workload, mesh, context and disabled
  diagnostics. Queued token-out changes from 25.081180984 to
  21.416868244 ms/token, a rederived 14.6098% reduction and 46.6922 t/s/u.
  The final interval contains 127 model/sampling pairs and zero refresh,
  readback, capture or release deltas. Caller delivery, fixed-token model-only
  replay and actual HF teacher inputs remain separate measurements. Executing
  the CPU-only byte model over the actual positions 128..254 reproduces
  1,107,522,496 useful bytes/rank and the 21.63129875 ms ceiling; the final
  measured floor is 10.1001%. Raw queued layer samples reproduce medians
  0.586715590 and 0.306684235 ms and the 20.668310051 ms stack estimate. The
  whole token-out excess is 3.6218% before terminal allowance, below the
  requested gap trigger.
- **Native evidence:** compact token-out CSVs independently sum to 199 model
  and 31 sampler operations per replay on each rank, reproducing the reported
  per-rank work totals. Native head attributes show reader3/block4/M1/N32,
  BFP8 DRAM-sharded weights and HiFi2/FP32 destination accumulation. The
  selected sampler groups contain neither generic vocabulary TopK nor argmax;
  candidate gathers are width32 to128. The clean sampler is about 2.28% of
  full40 token-out. Summing the saved ordered per-operation rows independently
  reproduces the maximum complete same-rank extrapolation of 20.631607125 ms.
  The report correctly distinguishes this extrapolation, the isolated-layer
  estimate, native operation work and clean wall time. Advice, CSVs, metadata,
  commands and raw capture paths are retained.
- **Context and tails:** the original 262144 aggregate-token contract is
  unchanged. `final/prefill_boundaries.json` covers 64/255/256/257/1025/2049;
  `final/prefill_context32769.json` passes 32769 tokens with context65536 and
  two resident feedback steps to position32771, stable cache/traces and zero
  resident host-boundary counters. Its prepared/final DRAM allocations agree
  at 21,024,343,552 bytes/device. BF16 KV layout is unchanged, so the prescribed
  stage-7 smoke is sufficient without repeating maximum-context acceptance.

## Other Concerns

- The FUNCTIONAL margin is small: final latency is about 0.2144 ms, or 0.99%,
  below the analytical ceiling. This is a passing observed result, not a
  guaranteed margin under other routing, context or load. No extra optional
  performance rerun is requested after the cap.
- Prefill is not established as improved. Five of six single-sample boundary
  timings are 0.885–3.428% slower; the short 128-token TTFT is 2.07% lower.
  `prefill_performance_analysis.md` correctly avoids a robust gain or causal
  attribution claim. Broader prefill and B32 optimization remain the already
  recorded Stop Rule handoffs.
- The 100/128-token qualitative samples stop during reasoning on both HF and
  TT. They do not establish final-answer correctness, broad instruction
  following or statistical HF equivalence. No TT-worse content finding is
  supported by this set. The shared q03 doubled word is present on both sides
  and is recorded with `original_model: "same"` in the limitations ledger.
- The inherited resumed-prefill numerical limitation remains explicitly
  scoped to its matched HF control and prior owner acceptance. This review
  does not enlarge that acceptance. Fresh replay checks establish state and
  trace behavior rather than erase those historical PCC results.
- Explicit sampler persistent gather outputs remain disabled. The candidate
  has a concrete raw-gather failure and a source-backed packet-tail diagnosis;
  decoder persistent scratch is still enabled. Padding/compaction and native
  repair remain untried proposals, already listed in the work log. Stable
  trace addresses must not be relabeled as explicit sampler buffer support.
- Retaining the legacy head adds 135,380,992 bytes/device in the matched
  constructor census. Maximum-context headroom is a disclosed projection
  from prior acceptance, not a new-source maximum run or runtime peak.

These are bounded risks and existing handoff notes, not unresolved Required
Work. No additional performance experiment is imposed by this round.

## Hard-Check Gaps

- A benchmark's top-level execution `PASS` is insufficient to enforce the
  FUNCTIONAL floor. The separate final `functional_floor.passed` was checked
  and independently recalculated here; the failing baseline and candidate1
  are not treated as accepted performance.
- The original queue and source snapshot retain `passed:false` after the
  serving assertion failure. `final/correctness_acceptance.json` is the
  explicit reconciliation with the passing v3 rerun. The original job exit
  codes, failure artifacts and corrected test source support that
  reconciliation. Consumers must not treat either historical false status
  or a source snapshot as the aggregate final result.
- Installed stage07 check exit0 is present, but its output also traverses
  prior-stage qualitative artifacts. The fresh stage7 accuracy, generation,
  tracker/Watcher and context artifacts were inspected independently rather
  than inferred from that exit code.
- Current B32 state snapshots are chip0-scoped; changed-kernel exact controls
  and dormant remap evidence cover all ranks separately. These establish the
  stated bounded capability, not an all-rank, all-cache B32 long soak.
- Cached embedding-gather metadata retains a height64 shape from another use
  of the program. The audit discloses this, uses the captured producer/source
  for singleton ownership, and does not use that cached height to inflate
  useful bytes. Finer prefill-role attribution remains a future evidence
  improvement, not a missing required gate.

## Anomaly Ledger

- Observed anomaly: all40 serving expected three changed page tables but saw30;
  a subsequent verification loop shadowed `state()`.
  Evidence: `final/serving_commands_all40{,_v2}.json`, their logs, test diff and
  `final/serving_commands_all40_v3.json`.
  Affected path: integration-test assertions, with unchanged production code.
  Control or comparison: ten full-attention groups times three slots; v3
  passes changed30/unchanged0 and all nine checkpoints.
  Likely subsystem: test assumptions and Python local binding.
  Investigation performed: reviewed expected-count derivation, per-table
  equality checks, rerun result and frozen source manifest.
  Resolution: fixed.
- Observed anomaly: experimental persistent sampler broadcast leaves remote
  columns poisoned and produces invalid greedy choices.
  Evidence: `persistent_sampler_l1_watched_v3.diagnostics.json`,
  `sampler_persistent_broadcast_rejection.md` and final sampler/native records.
  Affected path: rejected experimental broadcast candidate only.
  Control or comparison: each rank's local32 columns are correct; remote96
  remain poison, 3072/4096 mismatches/rank; selected non-broadcast split
  sampler passes its final fixtures and full-model gates.
  Likely subsystem: unflushed native scatter partial packet.
  Investigation performed: reviewed exact 2048-byte tile/4096-byte packet
  analysis, isolated eager repro and unchanged production selection.
  Resolution: controlled by rejection; future repair remains unpromoted.
- Observed anomaly: isolated blocking-layer sums exceeded a model-only
  measurement, and candidate2 TTFT was higher than baseline.
  Evidence: `anomaly_ledger.md`, final queued-layer JSONs,
  `final/performance.json` and `native_accounting_derivation.json`.
  Affected path: cross-regime performance interpretation.
  Control or comparison: retained-carrier nonblocking layer medians and final
  same-harness full40 timing; independent same-rank native sums.
  Likely subsystem: fencing/measurement regime and run variation.
  Investigation performed: rederived layer budget, useful-byte floor and
  extrapolation; checked separate timing boundaries.
  Resolution: controlled; no exact additive or measured-host-overhead claim.
- Observed anomaly: q03 begins with “thinking thinking”; all bounded samples
  remain in reasoning at their token limit.
  Evidence: raw paired texts, `final/qualitative/verdict.json` and
  `../eval_readiness/limitations.json`.
  Affected path: qualitative model output.
  Control or comparison: matching HF phrase/behavior and unchanged stage6 TT
  continuations; installed degeneracy check passes.
  Likely subsystem: original-model generation behavior under this recipe.
  Investigation performed: read all paired continuations and checked raw
  current/prior TT equality.
  Resolution: controlled; no statistical or completed-task quality claim.
- Observed anomaly: optional router tracked probe retained a trace-local alias.
  Evidence: `router_advice_comparison.md`, initial failure and six v2 artifacts.
  Affected path: unselected probe, outside frozen production/tests.
  Control or comparison: explicit `del out,x`, no tracker exemption, separate
  tracking1/skip0/Watcher and clean-timing v2 controls.
  Likely subsystem: probe-owned Python tensor lifetime.
  Investigation performed: reviewed failure/fix disposition and confirmed
  unchanged final source manifest.
  Resolution: fixed in the probe; no production candidate was promoted.

## Scope Inspected

- Goal/skill paths: `stage_goal.txt`; the explicit stage-cap instruction in
  the review brief and work log; installed
  `/home/ttuser/qwen36-35b-a3b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/`
  `stage-review`, `multichip`, `optimize`, `tt-device-usage`, `full-model`,
  `tt-enable-tracing` and qualitative-check requirements.
- Artifact paths: this stage's README, work log, Stop Rule assessment,
  optimization/topology/runtime/profile/context/prefill audits, anomaly and
  AutoDebug/AutoFix records; `final/` acceptance, provenance, accuracy,
  generation, B32, replay, serving, sampling, performance, boundary/context,
  queued-layer and compact native results; the selected head/structural/scan/
  slab controls; inherited policy ledger, serving/context contracts and
  limitations. Inspected saved hook/check log tails and selected failure/pass
  markers. Large native execution files were not read wholesale.
- Code paths: changed `tt/model.py`, `generator.py`, `multichip_decoder.py`,
  `prefill.py`; new `lm_head.py`, `mesh_cached_scan.py`,
  `mesh_narrow_scan.py`, `mesh_direct_recurrent_sum.py`,
  `mesh_product_recurrent_sum.py`, `mesh_state_slab_recurrence.py`; relevant
  generator preparation/replay/position/slot paths; changed measurement,
  replay, sampler and serving tests; public boundary harness and CPU byte
  model; slab probe literals and preserved baseline behavior.
- Commands run: read-only `git status --short`, `git diff --stat`, targeted
  `git diff`; bounded `rg`, `cat`, `sed`, `head`, `tail`, `wc`; standard-library
  Python JSON/CSV extraction, source-manifest comparison, AST literal
  comparison, text byte comparison and arithmetic. `runpy.run_path` was used
  only for the inspected CPU-only `tests/multichip_functional.py` byte model.
  No TTNN import, devices, reset, server, hardware test, long test or child
  reviewer was started. The only written file is this report.

## Residual Risk

The review qualifies the frozen selected implementation and the measured
scopes above. It does not prove arbitrary future tensor/config mutations,
long-form task quality, maximum-context B32 concurrency, or statistical
performance stability. Inline custom arithmetic remains a maintenance risk:
future dtype/layout/native-API changes must preserve the exact guards and
rerun affected controls. The existing Stop Rule handoffs should remain visible
for later optimization; they are not grounds to restart optional stage7 work.

After this passing review, record the local stage-owned checkpoint SHA and
retain the persistent raw evidence. No push is authorized by the stage plan.
