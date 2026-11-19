# Stage Review

Verdict: clean-pass

Independent round 1, stage 8 datatype-sweep, Qwen/Qwen3.6-35B-A3B. Reviewed the
live `main` checkout at `c873e26136c8140fe419e523525265ee7d3939dc` and the
stage-owned working tree on 2026-11-16. This verdict applies to the controlling
stage cap and explicit owner acceptance, not to the original uncapped search or
to a claim that every raw numerical gate passes.

## Required Work

- None under the controlling owner instructions. The selected policy is
  `baseline_bfp8`; the original broad candidate search is explicitly capped.
  The resumed-prefill PCC gate remains **FAIL** and is accepted by the owner
  for stage closure. This review does not turn that failure into a numerical
  pass or authorize further resumed-prefill investigation.

## Other Concerns

- Only one policy completed ranked full-model measurements. The two baseline
  processes establish repeatability, not a precision optimum. The README,
  plots and work-log hand-off correctly identify BFP4, BFP8 fidelity, KV/CCL
  and canonical-policy candidates as untried rather than rejected. The stage
  cap supersedes completing those experiments now.
- The accepted resumed-prefill limitation includes TT-only differences. HF
  fails case-level PCC at starts 68 and 2528, but passes 2559; positional
  failures do not fully overlap. The current limitations entry correctly uses
  `original_model: "better"` and retains raw FAIL. The owner instruction,
  rather than a blanket HF equivalence claim, supports closure.
- P1 has 12,288 positions from two reasoning texts. Its block standard error
  and fitted mixed BF16 reference support the reported comparison, not broad
  task accuracy. The advisory direct HF distribution check was skipped by
  the installed host-memory preflight. No direct P1 agreement with HF is
  established.
- The six qualitative pairs are a bounded regression smoke. All final TT
  token sequences exactly equal the stage-7 controls; the 128-token thinking
  continuations do not establish completed-task quality or a statistical
  failure rate. No new material degradation was observed in the actual text.
- No vLLM adapter or serving-path verification belongs to this stage. Later
  integration must preserve the common factory's selected-policy resolution
  and perform its own propagation and serving checks. The recorded plan is
  local stage commits only, with no push; the owner will checkpoint the final
  acceptance/report files after this review, including the globally ignored
  required CSV.

## Hard-Check Gaps

- The installed stage-08 shell gate checks P1 only. Its exit 0 was not treated
  as proof of the other acceptance requirements; those were reviewed
  individually below.
- The final context wrapper exit code was lost across interruption. Its
  attempt record transparently contains `exit_code: null`. The complete
  passing artifact, final counters and allocator census, and normal
  device/cluster closure in the log establish that this smoke completed.
  A rerun is not required by the available evidence.

## Anomaly Ledger

- Observed anomaly: selected default resumed-prefill PCC misses at starts
  68, 2528 and 2559.
  Evidence: `final_acceptance/resume_default.failed.json`,
  `final_acceptance/resume_baseline_equivalence.json`,
  `hf_resume_control/failure_classification.json`, `AUTOFIX_resume.md`,
  `resume_wrapup_assessment.md`, `owner_acceptance.json`, and
  `../eval_readiness/limitations.json`.
  Affected path: fresh versus resumed full-model prefill.
  Control or comparison: all final top-1 and final-argmax criteria pass;
  the four applicable negative controls fail as expected. Failed-case
  final tensors are recorded as bitwise equal to the existing baseline
  diagnostic tensors. Matched HF results only partly explain the failures.
  Likely subsystem: shape-dependent numerical sensitivity; a general repair
  is not established.
  Investigation performed: inspected existing raw results and classification,
  the recorded repair attempts, and the exact latest owner instruction.
  Resolution: controlled for stage disposition by explicit owner acceptance;
  numerically unresolved, raw FAIL preserved. No further investigation was run.

- Observed anomaly: qualitative outputs exhaust the token budget in thinking;
  q03 starts with “thinking thinking” and TT has an odd short list label.
  Evidence: canonical final `tt_greedy_outputs.json`, original HF controls,
  stage-7 TT controls, rendered prompts and all twelve completion strings.
  Affected path: chat-template greedy generation.
  Control or comparison: HF shows the doubled phrase and thinking-budget
  behavior; all six TT token sequences are unchanged from stage 7.
  Likely subsystem: bounded generation/model behavior, with no new stage
  regression demonstrated.
  Investigation performed: independently read every pair, checked matching
  model/revision/tokenizer/prompt/generation metadata and HF control hash,
  and compared exact TT token arrays against stage 7.
  Resolution: controlled bounded smoke; no claim of completed-task correctness.

- Observed anomaly: generic active-trace allocation warning in untracked
  timing/context logs.
  Evidence: `logs/final_unchanged_kv_long_smoke.log`,
  `run_final_acceptance.sh`, final qualitative trace counters, and
  `tt/generator.py` preparation/capture code.
  Affected path: preparation of model and sampler traces.
  Control or comparison: the final selected-policy qualitative run uses
  `TT_METAL_TRACE_ALLOC_TRACKING=1` and
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, completes 763 model and 770
  sampling replays with two captures and stable trace IDs/program cache,
  and has no tracker failure. The stage adds no broad corruptible scope or
  capture exemption.
  Likely subsystem: generic warning emitted when tracking is disabled.
  Investigation performed: inspected the warning, tracked-run command,
  counters, log and unchanged capture/preallocation structure.
  Resolution: controlled by representative repeated tracked replay evidence.

## Scope Inspected

- Goal/skill paths: original
  `/home/ttuser/qwen36-35b-a3b/multigoal_logs/08-08-datatype-sweep.prompt.txt`;
  installed 0.1.24 `stage-review/SKILL.md`, `datatype-sweep/SKILL.md`,
  `datatype-sweep/references/precision-reasoning.md`,
  `qualitative-check/SKILL.md`, and model-bringup startup instructions;
  `stage_cap_instruction.txt`, `owner_decision.md`, `owner_acceptance.json`.
- Full-model results: `baseline_bfp8_v5.json`, `baseline_bfp8_v6.json`, their
  provenance, and `sweep_results.json/csv`. Both runs score AIME prefill and
  teacher forcing at 98/100 top-1, 100/100 top-5 and top-100. Natural text
  scores 752/768 prefill top-1 and 754/768 teacher top-1; the minimum top-5
  is 767/768 and top-100 is 768/768. All per-reference acceptance bars pass.
  Both runtime policies equal the selected JSON and contain 94 actual public
  boundary observations. Current bytes of all 155 recorded source/input
  files per baseline were independently rehashed and matched.
- Performance: independently recomputed the four traced repeats as mean
  **47.760901203 tokens/s/user**, range **47.686910083–47.827742397**, spread
  **0.140832313**, and resident-logit TTFT mean **3847.110845 ms**. Each timed
  repeat has 99 model trace replays and no readbacks/recaptures. The final
  default-construction token-out artifact gives **46.690388244 tokens/s/user**
  from 127/2.720045919 seconds, **2799.036538 ms** warmed TTFT, 127 model and
  sampler replays, and zero timed refresh/readback/capture operations.
  The caller-observed timing remains separately labeled.
- P1: inspected the installed numerical runner, stage wrapper, immutable
  bindings/manifest, process identities, canonical gate and underlying CPU
  score tensors. Recomputed paired ΔNLL **0.001005595721**, block SE
  **0.000377738707**, KL×1000 **0.903216098**, and top-1 agreement
  **0.994059245** over **12,288** matched targets. Decode scores are 464/512
  selected versus 465/512 reference. Required P1 thresholds pass; the final
  canonical result names `baseline_bfp8`.
- Capability: selected KV remains BF16 TILE, with no capability reduction.
  Checked context arithmetic, the stronger reference's full-capacity
  allocation/preparation record, and the final nonaligned/context artifacts.
  Logical lengths 1, 33, 2049 and 32769 complete with two traced decode
  steps each. The 32769 smoke uses 65536 allocation, advances to 32771,
  retains identical prepared/final allocator totals, and leaves
  **12,954,371,584 bytes/device** free. The advertised 262144 aggregate
  context remains unchanged. Later boundary-instrumentation edits preserve
  the memory-fit run's fixed BF16 storage shapes and allocations.
- Runtime code: stage diffs from `e03236d2` and `a8fdbbc1`; `tt/precision.py`,
  `model.py`, `generator.py`, `multichip_decoder.py`, `prefill.py`,
  `lm_head.py`; candidate, P1, identity, resume, reporting and performance
  harnesses; qualitative runner. Default resolution consumes the selected
  artifact, and explicit `precision_config="baseline"` remains available.
  Selected weights/fidelities are wired into the actual projection kernels;
  unsupported public boundary choices fail validation. Native grouped-MoE
  fixed mixed intermediates are explicitly distinguished from public BF16.
- Reporting: README, work log, both plots, CSV, acceptance attempt records,
  context contract, limitations, stage gate log and final pre-commit log.
  Both plots visibly show the one evaluated policy, red selection, repeat
  range, and dotted accuracy threshold without claiming an unmeasured win.
- Commands run: read-only `git status`, `git log`, `git diff`, `git show`,
  `rg`, `cat`, `sed`, log tails; small CPU Python JSON/CSV, SHA-256 and tensor
  statistics checks; local image viewing. No devices, servers or hardware
  tests were opened or run. The only review write is this report.

## Residual Risk

- This is a clean pass of the explicitly capped stage with the owner's
  resumed-prefill acceptance. It is not a clean raw resumed-prefill result,
  exhaustive precision search, broad quality evaluation, or serving release.
  Preserve those distinctions and use the selected no-readback token-out
  number for later comparisons.
