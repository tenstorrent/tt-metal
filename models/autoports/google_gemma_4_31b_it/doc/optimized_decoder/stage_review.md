# Stage Review

Verdict: clean-pass

Independent review of stage 03, `optimized-decoder`, for
`google/gemma-4-31B-it` at HF revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. This verdict applies to the live,
staged implementation on branch `mvasiljevic/gemma4-31b-fresh-pipeline-v0811`,
with HEAD `b16d01158eb873deaa5a2bd8b4ee08b8fea947bb` and formatted optimized
runtime SHA-256
`5267f1ee57d0f077e71357445e96e15b3851b022c5846eeb10b77566eb4b6ff6`.
Measured evidence uses the equivalent pre-format runtime SHA-256
`820e9840990a7a16c90b2678f00e2ceaa446afee40bb493ee63d049ecab64ba3`;
the post-review hook changes are inspected below.
The fused comparison baseline is `b5284798`. Local checkpoint commits and
completion-status bookkeeping follow this review; they are not claimed to have
already occurred.

## Required Work

- None. The concrete findings raised during this review have been resolved by
  source inspection, adapted experiments, or affected-path reruns. No known
  material decoder optimization is left deferred by the stage evidence.

### Post-review formatting equivalence

The first local commit attempt ran formatting hooks without creating a commit.
A narrow independent rereview compared the resulting 20 Python files against
the still-reviewed index. All non-import AST structure is identical, including
the conditional weight-policy expression and reader-trial iteration expression.
The remaining changes are whitespace, contiguous import ordering, and ordering
of names within `from` imports. Inspection of the reordered local modules found
no initialization dependency or changed side effect. Importing `ttnn.types`
first still initializes the `ttnn` package before resolving its submodule.

The formatted runtime hash above therefore extends this clean-pass verdict
without invalidating hardware evidence or requiring a hardware rerun. Historical
run/source hashes remain the hashes of the code that actually produced those
runs. The narrow check used `git diff`, indexed source reads, Python AST
comparison, SHA-256 calculation, and static import inspection; it did not import
TTNN or execute model code. Retaining oversized provisional 128-replay text
tables locally while preserving representative reports and the final tables and
CSVs does not remove the evidence supporting this verdict.

## Other Concerns

- The selected 4096-token workload policy has a measured short-prompt tradeoff.
  At 33 logical tokens, median warmed host prefill increases from 3.027 to
  3.575 ms for sliding attention and from 3.338 to 4.121 ms for full attention.
  `prefill_lengths.json` and `PERFORMANCE.md` disclose this directly. The
  measured 1024, 1025, 2049, and 4096-token cases improve. This does not support
  a universal prefill-speedup claim, and the final report makes no such claim.
- The final profiles use `tracy_final/sliding_attention` and
  `tracy_final_k32/full_attention`. Earlier full-attention K128 profiles are
  retained experiment evidence and are explicitly superseded. Completion
  bookkeeping must continue to point to the final directories.

## Hard-Check Gaps

- None that prevent this decoder-stage pass. The final delivered pytest log
  records 14 passes in 53.15 seconds, including both layer kinds, batches
  1/8/16/32, and odd page capacities. Source and log hashes in
  `final_acceptance.json` match the files inspected.
- Four final watcher runs cover both layer kinds at batches 1 and 32. Their
  manifests enable trace allocation tracking without excluding program-cache
  allocations. The copied watcher logs are clean. Additional final-source
  runs cover sliding batches 33/56 and full batches 33/34/36, with distinct
  requests and 129 replays. The extended B1 runs cover boundary lengths,
  continuation offsets, input preservation, sentinel pages, changed page rows
  and positions, and prefill while a decode trace remains live.
- Advertised context is preserved at 262144. Sliding evidence in
  `final_context_v2/layer0_context262144_correctness.json` remains applicable
  to the unchanged sliding path. Full evidence was rerun on the delivered K32
  policy in `final_runtime/layer5_context262144_correctness.json`. Nonaligned
  262143-token tails, the exact maximum, and decode pass the 0.995 PCC gate.
  Full attention is the tighter observed case: tail PCC 0.9975234393,
  exact-maximum final-row PCC 0.9980639152, and decode PCC 0.9999324123.
- Before/after measurements cover warmed 4096-token prefill and 128 advancing
  traced decode steps at batches 1/8/16/32 for both kinds. Recomputing complete
  replay windows from raw profiler timestamps agrees with the final JSON.
  Advice-enabled prefill/decode reports and CSVs exist for all final cases.
  Runtime projection rows show the selected BFP4/LoFi policy, rather than
  merely declaring it in configuration metadata.

## Anomaly Ledger

- Observed anomaly: the earlier selected full-attention K128 policy did not
  preserve the strongest measured decode candidate.
  Evidence: `full_refined`, `final_decode_k`, and
  `final_runtime/layer5_final_default_timing.json`.
  Affected path: full-attention decode SDPA.
  Control or comparison: final-policy K128, K64, and minimum-legal K32 trials
  with real inputs; K32 correctness at B1/B32 and 4223 tokens.
  Likely subsystem: SDPA chunk configuration and default selection.
  Investigation performed: targeted tuning was repeated on the final policy;
  affected full profiles, watcher runs, larger batches, maximum context, and
  delivered tests were rerun. K32 measured 1017.571 us versus K64 1019.474 us
  at B1; the delivered default reproduced 1017.634 us. Sliding K32 lost to
  K64, so the two layer kinds retain different selected chunks.
  Resolution: fixed. Final defaults are full K32 and sliding K64, with the
  capacity guard retained.

- Observed anomaly: short page-table capacity could produce severe,
  process-order-dependent full-attention PCC failure.
  Evidence: `pytest_AUTODEBUG.md`, `pytest_capacity_fix.log`,
  `pytest_delivered_final.log`, and the decode chunk selection in
  `../../tt/optimized_decoder.py`.
  Affected path: paged decode when the final SDPA chunk exceeds the valid
  page-table capacity.
  Control or comparison: isolated versus combined-process tests and odd
  capacities for both layer kinds.
  Likely subsystem: masked SDPA readers touching page IDs outside the valid
  logical page-table width.
  Investigation performed: selected chunk size now divides the actual cache
  capacity; final mixed-process tests and final-source larger-batch runs pass.
  The earlier larger-batch evidence was recognized as stale and replaced for
  affected cases.
  Resolution: fixed. The runtime no longer relies on unspecified padded page
  IDs, and the regression cases are delivered tests.

- Observed anomaly: sliding prefill performed increasing masked historical
  work, and page-aligned continuation offsets not divisible by 128 needed
  separate handling.
  Evidence: `pre_review.md`, final runtime code, final profiler rows, and
  extended `final_runtime` correctness artifacts.
  Affected path: chunked sliding prefill and continuation offsets 32/96.
  Control or comparison: fused and optimized 4096-token profiles; continuation
  references and boundary/tail checks.
  Likely subsystem: paged SDPA view and query/key chunk alignment.
  Investigation performed: the page view is bounded to the preceding sliding
  window plus the current chunk while cache writes and RoPE remain absolute;
  non-128-aligned local starts use legal 32-token SDPA chunks. Final sliding
  chunk SDPA durations cease growing with the full prefix.
  Resolution: fixed. Final continuation and long-context evidence passes.

- Observed anomaly: early larger-batch fixtures did not establish distinct
  request behavior; logical batch 34 was also described as a new physical
  bucket despite sharing the 64-row bucket with batch 33.
  Evidence: `pre_review.md`, `../../tests/run_optimized_decoder.py`, final
  larger-batch JSONs, and corrected `work_log.md`.
  Affected path: padding, page-row ownership, and batched decode validation.
  Control or comparison: distinct tokenized request inputs, changed page rows
  and positions, and batches on both sides of the 32-row boundary.
  Likely subsystem: test coverage and evidence description.
  Investigation performed: inspected fixture construction and final reruns;
  checked the actual physical padding rule against the documented cases.
  Resolution: fixed. Final distinct-request results pass; documentation no
  longer claims a new physical bucket at logical batch 34.

- Observed anomaly: an early trace cleanup run hung while an eager L1 output
  remained live through capture cleanup.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, `capture_cleanup.log`,
  `../../tests/optimized_trace.py`, and final replay artifacts.
  Affected path: test harness trace capture and buffer lifetime.
  Control or comparison: cleanup reproducer and repeated tracked replays with
  interleaved prefill and changed inputs/page metadata.
  Likely subsystem: live buffer ownership around trace capture.
  Investigation performed: consumed eager output is released; capture helpers
  preserve balanced cleanup. Final evidence enables allocation tracking and
  includes program-cache allocations, without whole-capture exemptions.
  Resolution: fixed. Final watcher and delivered-test runs complete cleanly.

- Observed anomaly: a maximum-context run experienced a transient FetchQ stall.
  Evidence: retained `triage` artifacts, work-log investigation, and successful
  final maximum-context runs for both kinds.
  Affected path: device execution during long-context validation.
  Control or comparison: owner-checked recovery followed by successful targeted
  reruns, including full attention on the final delivered runtime.
  Likely subsystem: device/runtime execution; an exact generic root cause was
  not established.
  Investigation performed: triage and host-stack evidence were captured before
  recovery; capability was revalidated rather than reduced.
  Resolution: controlled. The stall is not represented as a proven generic
  runtime defect or used to waive the context requirement.

- Observed anomaly: profiler advice undercounted reader workers and did not
  recognize the explicit MinimalMatmulConfig; literal B32 L1-input advice
  exceeded physical L1 capacity.
  Evidence: final advice reports, `reader_microbench`, `minimal_geometry`,
  `prefill_refined`, and `batch_chunks.json`.
  Affected path: optimization advice interpretation and candidate rejection.
  Control or comparison: exact real-input reader microbenchmarks, explicit
  precision-locked geometry trials, and a legal B32 32-token L1 chunk family.
  Likely subsystem: report interpretation and memory-capacity constraints.
  Investigation performed: inspected program attributes and raw timing rather
  than accepting misleading per-op utilization. The L1 adaptation passes
  correctness but loses whole-prefill timing: sliding 886.711 versus 717.013 ms,
  full 1690.298 versus 1024.191 ms. The full B32 SDPA factory also proves that
  8x8 and 11x10 grids both use 64 active workers for this head geometry.
  Resolution: controlled. Advice dispositions are evidence-backed; rejection
  does not rest on a first API error or unsupported parallelism assumption.

- Observed anomaly: some synthetic reduced-precision trials failed PCC despite
  real target-model candidates passing.
  Evidence: precision/geometry trials, recorded activation tooling, candidate
  ledger, and final real-weight correctness/profile rows.
  Affected path: precision selection.
  Control or comparison: pinned target-model weights and recorded layer inputs.
  Likely subsystem: synthetic input distribution and numerical sensitivity.
  Investigation performed: material geometry and reader comparisons use the
  selected BFP4/LoFi policy; real-model correctness governs the final selection.
  Resolution: controlled. Synthetic failures do not veto a faster passing
  real-model candidate or justify an unmeasured higher-precision default.

- Observed anomaly: final logs contain motherboard/topology and subset-device
  notices, plus Python SWIG deprecation warnings.
  Evidence: `pytest_delivered_final.log` and the final watcher logs.
  Affected path: host device discovery and Python bindings.
  Control or comparison: completed single-chip tests, inspected topology, and
  clean watcher runs.
  Likely subsystem: platform metadata and bindings, not tensor fallback.
  Investigation performed: distinguished discovery's bus-ID fallback from
  model computation and inspected runtime conversion guards.
  Resolution: controlled. No warning indicates corrupt outputs, unsafe trace
  state, or an unclassified model/device failure in the accepted runs.

## Scope Inspected

- Goal/skill paths: checkout `AGENTS.md`, `bringup/run_contract.json`,
  `bringup/agentic_evals/inputs.json`, the stage 03 prompt under
  `bringup/artifacts/multigoal-runs/20261030T163027Z/continuation-02-20261030T182350Z`,
  and installed model-bringup, stage-review, optimize, agentic-serving,
  tt-device-usage, and tt-enable-tracing instructions. Relevant model-code,
  review-core, and comment-hygiene review guidance was also inspected.
- Artifact paths: this directory's README, work log, checklist, candidate and
  memory ledgers, topology/precision/packing/geometry/reader trials, final
  acceptance and runtime manifests, final context evidence, final profiles,
  short/tail and B32 chunk measurements; adjacent `context_contract.json` and
  `agentic_serving.json`; fused baseline evidence; and telemetry packet
  `ed13a689-7a39-4f2c-ae97-d55f889e7ffc.json` in the active continuation.
- Code paths: complete `../../tt/optimized_decoder.py`; delivered tests;
  correctness, timing, profiling, trace, reader, candidate-selection and
  summary harnesses; pinned reference and recorded-activation preparation;
  relevant generic SDPA decode factory. The runtime is standalone, with no
  candidate/fused/functional runtime import or host tensor fallback. No
  multichip, full-model, or vLLM implementation is included in this verdict.
- Commands run: read-only git status/diff/branch/history inspection, `rg`, file
  reads, and Python standard-library JSON/CSV/hash analysis. Important
  numerical claims were checked against raw evidence rather than accepted
  from README prose. No tests, servers, TT device operations, or resets were
  launched by this reviewer. The only reviewer-written artifact is this report.

The final whole-layer device decode results improve on the fused baseline at
every required batch: sliding speedups are 3.125x/2.770x/2.566x/2.266x and full
speedups are 3.037x/2.711x/2.456x/2.228x for batches 1/8/16/32. Required 4096-token
prefill improves by 3.080x to 3.354x. These are device-window measurements with
internal gaps included; same-run host timing is reported separately. The
selected default reproduces its winning tuning result, and the final source,
acceptance records, profile summaries, context contract, serving assumptions,
and telemetry agree.

## Residual Risk

- Maximum-context references validate selected tail rows and the exact final
  row against the complete prefix KV, plus decode; they do not compare every
  output row of a 262144-token HF execution. The TT path executes the full
  prefill. This scope is explicit in the artifacts and is not represented as
  all-row maximum-context equivalence.
- Layer PCC and teacher-forced decode do not establish full-model generation,
  task accuracy, sampling, serving throughput, or mixed-request scheduling.
  Those later-stage responsibilities remain outside this decoder verdict.
- Roofline FLOPs/DRAM traffic and memory reservations are reasoned estimates,
  not hardware traffic counters or measured full-model peak allocation. The
  reports disclose the theoretical peak basis and absent allocator-peak data.
- Performance is established for the inspected single-chip hardware and
  measured workloads. The short-prompt tradeoff and controlled historical
  stall remain disclosed; neither supports a universal performance or
  infrastructure-reliability claim.
