# Stage Review

Verdict: more-work-needed

Independent early review of Stage 02, fused-decoder, for
`google/gemma-4-26B-A4B-it`. Evidence snapshot: 2026-09-25 21:26 UTC.
The live branch is `gemma-4-26b-a4b-it`, starting at
`37c8b975c83712bbdf10d873962d3cc396439710`; stage changes are uncommitted.
Final hardware gates are in progress. This verdict records unfinished required
work, not a claim that those running gates have failed. No additional static
cache-addressing or trace-capture correctness defect was found in the inspected
selected runtime.

Paths below are relative to `models/autoports/google_gemma_4_26b_a4b_it` unless
otherwise stated.

## Required Work

- P1: Complete the full-attention context and final watcher gates.
  Evidence: `doc/fused_decoder/final_validation_commands.json:64` through the
  end of the inspected journal records completed sliding 262144 and 262143
  runs, but no completed full-attention long-context or watcher run. At the
  snapshot, the fused evidence root has no watcher artifact and no
  `long_full_*.json`. `doc/context_contract.json:4` preserves 262144 for both
  kinds, with its existing evidence explicitly belonging to the functional
  stage. `tt/fused_decoder.py:49` selects a changed full-attention graph.
  Why this matters: full attention exercises much larger batched score and
  probability tensors than the 4096 headline; prior-stage context evidence
  does not validate the changed graph. Watcher-clean behavior is an explicit
  stage requirement.
  Required next step: finish and inspect fused full-attention prefill/traced
  decode at 262144 and 262143, and the planned separate watcher runs for both
  kinds. Record actual exit status, PCC results and watcher inspection. Keep
  watcher and profiler collection separate. The two sliding long-context
  results already present need not be rerun without another relevant change.

- P2: Finish the whole-layer device performance and telemetry evidence before
  declaring the selected graph faster.
  Evidence: `doc/fused_decoder/README.md:64` correctly distinguishes candidate
  host-wall timing from device timing, and `README.md:76` leaves final
  performance tables pending. Both `doc/fused_decoder/tracy` directories were
  empty at the snapshot. `doc/fused_decoder/candidate_summary.json` contains
  traced host timing only; it cannot supply warmed prefill device latency or
  the required complete-layer rooflines. The functional comparison basis is
  `doc/functional_decoder/PERFORMANCE.md:15` and its two `whole_layer.json`
  files.
  Why this matters: the stage contract requires before/after warmed prefill
  and traced decode, final-default reproduction of the selected candidate,
  tt-perf-report tables/CSV, and exact 4096/128/B1/C1 device-window metrics.
  Required next step: finish both default-policy Tracy runs, inspect the
  uncached invocation metadata, render the required reports, and generate
  whole-layer summaries and the compact telemetry packet. Compare each kind
  independently against the functional baseline. Report the reproduced
  default result and the evidence-backed best-candidate selection; do not
  substitute a host timing or a sum of selected kernels for the complete
  layer duration. Finalize README/work-log links and measured claims only
  after these artifacts exist.

- P2: Close the queued equivalence and remaining candidate checks.
  Evidence: `tests/probe_fused_equivalence.py:51` implements direct functional
  versus fused PCC over one complete prefill and all 128 decode outputs, but
  no paired result exists at the snapshot. The retained component checks
  cover individual changes and integrated checks compare against HF; neither
  alone proves the direct whole-layer TTNN equivalence relation.
  `doc/fused_decoder/AUTODEBUG_precision.md:521` identifies Q/K RoPE peer
  merging, and `tests/probe_fused_rope_merge.py` supplies the queued test, but
  its result is also absent. During this review the owner repaired the
  decode-GELU probe wiring at `tt/fused_decoder.py:186` and
  `tests/probe_fused_components.py:110`; the fresh reruns are pending.
  Why this matters: graph-fusing requires device equivalence and an actual
  measured outcome for applicable candidates. Independently passing PCC to
  HF does not mathematically imply passing PCC between two implementations.
  Required next step: complete the queued paired checks for both kinds and
  the RoPE peer-merge control. If the latter is faster and correct, measure
  it in the complete selected graph before deciding. Rerun the corrected
  explicit decode-GELU component comparison and retain its exact command and
  samples. Then update the fusion ledger and request a completed-evidence
  rereview. Local stage checkpoint commits belong after clean-pass; do not
  create them in response to this early verdict.

## Other Concerns

- Small host-timing differences should remain narrowly described. The
  full-kind KV-normalization-only and KV-normalization-plus-cache candidates
  differ by about 0.97 us in the recorded median, while complete replay costs
  about 5622 us (`full_boundary_kvnorm.json` and
  `full_boundary_combined.json`). Choosing the simpler path is reasonable;
  those numbers do not establish a substantial independent cache-writer
  speed penalty. This is not an additional gate.
- The selected graph still uses the inherited tiled-cache embedding path.
  Its full-cache layout work remains a cost, but rejection of the available
  flattened and page-axis gather alternatives is supported by the adapted
  same-input measurements. I found no basis to require a new cache ownership
  protocol or kernel in this fusion-only stage.

## Hard-Check Gaps

- No hardware, reset, server, or long test was run by this reviewer. This is
  artifact/source review and host-only analysis.
- The inspected long-context oracle compares sampled queries while computing
  every TT query and cache position. This is explicit in
  `tests/long_context.py:3` and is consistent with the inherited contract.
- Final runtime evidence must be read again after the serial queue completes.
  The README was being edited during review; future artifact entries must
  remain tied to completed commands rather than the intended queue.
- No C++ or CMake change is present. A build is not required for this Python
  and documentation stage. The existing pre-commit log was inspected;
  host-only AST parsing succeeded for the fused runtime and 28 test modules.

## Anomaly Ledger

- Observed anomaly: native FP32 RMSNorm, RoPE and softmax variants can pass a
  small/headline case and fail combined real-weight requests.
  Evidence: `norm_batch_sliding.json`, `norm_external_head_reuse_sliding.json`,
  `rope_hf_prefill_headline_sliding.json`, `guarded_softmax_sliding.json`, and
  the adaptation records in `AUTOFIX.md` and `AUTODEBUG_precision.md`.
  Affected path: close routed-expert decisions downstream of sensitive
  normalization/attention boundaries.
  Control or comparison: real-weight, isolated-site and phase controls;
  selected sliding reuse minimum PCC 0.99772459 and batch-32 minimum 0.99755985.
  Likely subsystem: precision at head normalization, rotary and softmax.
  Investigation performed: inspected the actual default policies, probe code,
  failed controls and passing selected-policy JSON.
  Resolution: controlled for the inspected workloads; outstanding final
  contexts and watcher coverage remain Required Work.

- Observed anomaly: the original embedding timing was about 659 us, versus
  about 139 us after longer warming.
  Evidence: `tiled_cache_gather.json`, `cache_gather_adapted.json`, and
  `AUTODEBUG_precision.md:582`.
  Affected path: the same-input cache-gather selection probe.
  Control or comparison: adapted warm samples; flattened gather about 24392
  us, page-axis gather about 3980 us, and dynamic-index page gather about
  4107 us, all with exact output/replay equality.
  Likely subsystem: insufficient initial warmup.
  Investigation performed: compared the retained results with the probe and
  the corrected limitation in the ledger.
  Resolution: controlled; use the warmed 139-us reference for this comparison.

- Observed anomaly: the retained component probe initially named two
  identical decode implementations `packed` and `packed_gelu` after runtime
  selection disabled decode merging.
  Evidence: the inspected initial `_chunk` branch required `not decode`;
  the owner added an explicit override during review at
  `tt/fused_decoder.py:186` and used it at
  `tests/probe_fused_components.py:110`.
  Affected path: reproducibility of a rejected optimization, not default
  runtime semantics.
  Control or comparison: corrected forced-merge component reruns are queued.
  Likely subsystem: experiment-to-runtime wiring.
  Investigation performed: traced both labeled candidates to the same branch
  and re-read the owner's source correction.
  Resolution: source fixed; fresh measured confirmation is Required Work.

## Scope Inspected

- Goal/skill paths: the supplied Stage 02 contract;
  `/workspace/tt-metal/.agents/skills/{stage-review,graph-fusing,tt-device-usage}/SKILL.md`;
  installed tt-review core/router, model-bringup and trace review guidance.
- Artifact paths: context contract; fused README, work log, PATTERNS,
  AUTOFIX, AUTODEBUG_precision, command journals, all 28 summarized candidate
  records, component probes, selected/final reuse, batch and continuation
  results, sliding long-context results, pytest/pre-commit logs, and the
  functional performance report. The final profile/watcher/paired artifacts
  were checked for presence and were absent at the snapshot.
- Code paths: complete fused/functional decoders, decode/precise attention,
  runtime audit, run_decoder, batch/reuse/continuation/long-context tests,
  fused regression, component/precision/equivalence/RoPE probes, experiment
  runners, summarize_perf; relevant Gemma4 weight tying and paged-fused-cache
  argument definitions.
- Commands run: read-only `git status`, `git diff`, `git branch`, `git rev-parse`,
  `cat`, `sed`, `nl`, `find`, and `grep` (rg unavailable); small standard-library
  Python scripts to parse JSON, recompute medians/pass status, list artifacts,
  and AST-parse source. Only this review report was written by the reviewer.
- Re-derived results: all 28 candidate-summary pass states and medians agree
  with source JSON. Both selected kinds pass nine reused requests, all 32
  batch slots, deterministic batch replay, and prefix/other-slot preservation.
  Full-kind reuse minimum PCC is 0.99926886; batch minimum is 0.99733190.
  Sliding 262144/262143 sampled-prefill PCC is 0.99818597/0.99818395; all six
  recorded long-context decode checks pass.

## Residual Risk

- This is a live, uncommitted worktree; later edits or artifacts require a
  rereview of the final state. Outstanding required evidence prevents pass.
- Layer-only, seeded real-weight checks do not establish full-model text
  quality. That is a later stage and was not imposed as a new gate here.
- No acceptance, checkpoint or telemetry-completion claim is made by this
  early review.
