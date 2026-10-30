# Stage Review

Verdict: more-work-needed

Independent inspection of stage 03, optimized decoder for `google/gemma-4-31B-it`,
pinned HF revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. Reviewed the live
worktree at HEAD `b16d01158eb873deaa5a2bd8b4ee08b8fea947bb`; stage files remain
untracked. This is a pre-review, not final signoff. The owner was already running
final acceptance, and made several repairs while this inspection was in progress.

## Required Work

- **P2: Bound sliding-prefill attention work to the sliding window.**
  Evidence: `tt/optimized_decoder.py:184` selects FP32 SDPA accumulation and
  `tt/optimized_decoder.py:577` passes the complete prefix page table with the
  absolute chunk start. In
  `tracy_candidates_v2/sliding_attention/batch1/ops.csv:178`, `:206`, `:234`, and
  `:262`, the four 1024-token SDPA chunks take 321.121, 794.676, 1271.675, and
  1756.815 microseconds at starts 0, 1024, 2048, and 3072, despite a fixed
  1024-token window. This is not just a profiler attribution suspicion:
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:81`
  enables streaming only without FP32 accumulation, while
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp:575`
  starts its K loop at zero and narrows it for sliding attention only in streaming
  mode. The inspected attention candidate also fixes FP32 accumulation on.
  Why this matters: masked-out historical K/V still costs work; total sliding
  prefill attention grows quadratically with prompt length. The visible cost is
  already material at 4096 tokens and is especially relevant to the advertised
  262144 context. This is applicable decoder optimization under the optimize
  skill's topology/data-movement and anomaly requirements.
  Required next step: compare prefill-only streaming SDPA on real activations,
  or preserve FP32 while slicing/rebasing the SDPA page-table view to retain the
  needed preceding window. Keep absolute RoPE and the original cache-write
  ownership. Validate continuation starts, nonaligned tails, and a later-prefix
  chunk; measure the complete prefill path. Retain the faster correct path or
  provide an exact blocker/adapted failing comparison.

- **P1: Validate the page-aligned continuation repair.**
  Evidence: the initial runtime accepted all 32-token-aligned continuation starts
  at `tt/optimized_decoder.py:514` and `:556`, but used Q/K chunk size 128.
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:388`
  explicitly rejects scalar starts not divisible by both chunk sizes. Thus
  valid starts 32, 64, and 96 were rejected after the prefill optimization;
  earlier acceptance exercised continuation only at 1024. During this review,
  the owner changed `tt/optimized_decoder.py:586` to select 32-token SDPA chunks
  for starts not divisible by 128 and added resume cases 32, 96, 128, and 1024.
  Why this matters: this was a concrete regression of the existing continuation
  contract, not a request for arbitrary extra boundary coverage.
  Required next step: complete the newly queued real-input continuation checks
  for both layer kinds and retain their results. Source inspection confirms the
  validator mismatch is repaired; passing runtime evidence was not yet available
  to this review. If the sliding-prefix adaptation changes this rule, validate
  that final version instead.

- **P2: Prove batch/page-table ownership with distinct requests.**
  Evidence: original `tests/run_optimized_decoder.py:56` expanded a single
  recorded activation sequence across every batch row. Existing
  `final_validation/layer{0,5}_batch*_correctness.json` therefore tests different
  page addresses but identical prefix contents. Reversing page rows and current
  positions does not expose reading another request's identical cached prefix.
  Why this matters: this stage changes packed head splitting, cache dtype, and
  larger-batch execution. Incorrect cross-request prefix reads could survive
  the reported PCC and sentinel-page checks.
  Required next step: complete the owner's new distinct-tokenized-request HF
  fixtures and rerun the affected batch checks, including changed positions and
  reordered page rows. The harness now accepts native batch fixtures and the
  owner reports that pytest will assert distinct request inputs. Existing
  repeated-input results remain useful shape coverage, but do not close this
  ownership gap.

- **P2: Complete the explicit DRAM-reader comparison contract.**
  Evidence: `candidate_ledger.json`, `geometry_0_bfp4`, `combined_0`,
  `full_geometry`, `output_padding`, and `output_padding6144` contain useful
  integrated one/two/three-reader experiments and adaptation after divisibility
  or L1 failures. Their timing evidence is whole-layer fixed-position host
  timing. No isolated alternating-order reader measurement was present.
  `optimize/SKILL.md`, OPT-015, explicitly requires that first measurement layer,
  including device time, logical/physical bandwidth, correctness, and stability.
  Why this matters: the selected reader count is part of every dominant final
  projection, and the broad whole-layer sweeps do not establish the requested
  row-level stability/traffic evidence. This finding does not invalidate the
  already substantial precision-locked geometry exploration.
  Required next step: complete the owner's planned focused isolated comparison
  using actual projection inputs and weights, compare all legal reader counts,
  and retain integrated final-path measurements. Preserve precise divisibility
  blockers for illegal counts; no exhaustive new geometry sweep is requested.

## Other Concerns

- **Fixed during inspection: profiler capture cleanup.** Initially
  `tests/profile_optimized_decoder.py:93` used unguarded begin/forward/end capture
  and closed the mesh directly on a body exception, repeating the failure pattern
  diagnosed in `AUTOFIX.md:1`. The current code at `:94` uses
  `tests/optimized_trace.py` and releases a successful trace in `finally` at
  `:125`. This source repair reuses the already tested cleanup helper; a new
  deliberate hardware-failure test is unnecessary.
- The default batch-1 host ranking reproduces the selected candidate closely.
  `final_validation/layer0_extended_timing.json` is about 907.83 microseconds
  versus 907.50 for the selected finalist. Sub-microsecond differences between
  otherwise equivalent finalists are not treated as a performance regression.
- The candidate profiles really contain BFP4/LoFi projection rows, rather than
  merely naming that policy. They predate the final padded/packed implementation,
  so they are diagnostic evidence and cannot substitute for final profiles.

## Hard-Check Gaps

These were already declared pending by the owner and are not additional surprise
findings: final maximum-context acceptance, watcher, delivered pytest, final
advice audit/checklist, shared context/memory evidence, final independent review,
and the local checkpoint commit.

The inspected candidate ledger had no timing row with batch greater than one;
the larger `final_validation` rows were correctness-only. The owner confirmed
that final profiles are planned for 1, 32, 16, and 8. Complete those rows before
claiming optimization against the agentic-serving batch set, and investigate any
new material bottleneck or regression they reveal. This is particularly relevant
because the L1 prefill policy and decode SDPA grid differ across those batches.

Final evidence must identify the final implementation after the repairs above.
The initial final-validation runtime hash was
`0c94ec6cd532cb985d8154ea66a0a26665d0b3a0ca72d23e42fc86b53601d97c`;
the continuation repair changes it. Reuse unaffected evidence and rerun affected
paths; a blanket repeat of every historical candidate is unnecessary.

## Anomaly Ledger

- Observed anomaly: sliding-prefill SDPA time grows with absolute prefix.
  Evidence: four raw profile rows and K-loop/streaming source cited above.
  Affected path: sliding `prefill_chunk_forward`, especially long context.
  Control or comparison: equal 1024-token query chunks and window, changing only
  prefix position within the same measured forward.
  Likely subsystem: standard FP32 SDPA reads/computes masked historical chunks.
  Investigation performed: re-derived timings from raw CSV; traced factory and
  reader selection; checked candidate source for an existing adaptation.
  Resolution: more-work-needed.
- Observed anomaly: page-aligned continuation can violate SDPA alignment.
  Evidence: public checks permit 32; SDPA validator required 128.
  Affected path: both layer kinds with continuation start 32 modulo 128.
  Control or comparison: previous fused configuration used 32-token chunks.
  Likely subsystem: optimized program configuration versus API contract.
  Investigation performed: source proof; owner added a conditional configuration
  and focused continuation cases.
  Resolution: source fixed; runtime verification pending.
- Observed anomaly: earlier capture-body error hung while closing the device.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, `capture_cleanup.log` and profiler code.
  Affected path: profiling exception cleanup.
  Control or comparison: correctness/timing already used the balanced helper.
  Likely subsystem: trace lifecycle.
  Investigation performed: compared all three capture callers.
  Resolution: fixed in profiler source during review.

## Scope Inspected

- Goal/skill paths: `AGENTS.md`, `bringup/run_contract.json`,
  `bringup/agentic_evals/inputs.json`; installed stage-review, model-bringup,
  optimize, agentic-serving, tt-device-usage, and tt-review-core instructions.
- Artifact paths: optimized README/work log, candidate ledger and matrix
  manifests/results, final-validation results, candidate raw Tracy CSVs and
  advice-enabled reports, AutoDebug/AutoFix evidence; fused before measurements,
  `doc/context_contract.json`, and `doc/agentic_serving.json`.
- Code paths: complete `tt/optimized_decoder.py`; correctness/timing/profiling
  harnesses, trace helper, candidate classes and sweep/matrix runners; profile
  summarization; relevant fused decoder interfaces and generic SDPA source.
- Commands run: read-only `cat`, `sed`, `nl`, `rg`, `git status/rev-parse`, and
  small Python standard-library artifact parsing/hash calculations. No TTNN
  import, hardware access, tests, servers, resets, or implementation edits.
  The only file written by this reviewer is this report.

## Residual Risk

The review covers a live worktree and only the evidence available during
inspection. It is not evidence of a clean final maximum-context run or the
subsequent fixes' numerical/performance results. Stage 03 should receive a fresh
final independent review after the required work and declared pending gates.
