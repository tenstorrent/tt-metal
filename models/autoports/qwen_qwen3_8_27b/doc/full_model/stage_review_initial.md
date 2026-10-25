# Stage Review

Verdict: more-work-needed

Independent initial inspection of stage 06, Qwen/Qwen3.8-27B, on the live
`exp26-qwen38-bringup` worktree based on
`cdcf1091d55e793f68fd46acdbe80372337e1c34`. This is not the final stage review.
The owner explicitly identified larger-batch benchmarks, reduced profiling, and
final documentation as ongoing. Implementation changed during this inspection;
findings below distinguish an observed defect from its subsequently applied,
still-unvalidated repair. No device was opened and no benchmark or reset was run
by this reviewer.

## Required Work

- **P2 — Validate the repaired page-window boundary.**
  **Evidence:** The inspected `_fill_page_window` in `tt/generator.py` generated
  `ceil(physical/256)*8` page indices from `start//32` without bounding them by
  the page-table row width. For context 512, start 384, and physical length 32,
  it requested columns 12–19 from a width-16 row. For native context 262144,
  start 262016, and physical length 32, it requested 8188–8195 from width 8192.
  Even context-512 preparation at start 128/physical 320 requested columns 4–19.
  `ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_rm_single_row_single_core.cpp`
  reads each supplied index directly from its input stick without checking the
  width. Existing continuation evidence covers starts 128 and 256 with 33-token
  tails, not the near-end odd-128 case.
  **Why this matters:** A valid public continuation could read beyond the
  page-table input stick. Unused fill-table padding can conceal this defect in
  output comparisons; trace allocation tracking does not establish index bounds.
  **Required next step:** The owner has now added an assertion that required
  physical pages fit and clamps only unused padded indices. Validate this repair
  with the queued 384+127 versus 511 continuation/control and exact fill-window
  bounds, including reuse of the existing traces. Reuse unaffected accuracy
  evidence; a new full AIME run is not required solely for this index repair.

- **P2 — Produce performance measurements without trace-allocation diagnostics.**
  **Evidence:** `benchmark_b1.json` records tracking `1`, skip-program-cache `0`;
  all three headline repetitions use this configuration. `sampling_b1.json` and
  `sampling_b32.json` also record tracking enabled. The existing environment
  wrapper unconditionally exports tracking=1. The tracing skill explicitly says
  to collect performance separately with tracking and diagnostics disabled.
  **Why this matters:** These are valid tracked acceptance/timing observations,
  but do not satisfy the required uninstrumented performance measurement.
  **Required next step:** Preserve the tracker-on evidence and record separate
  tracker-off headline/configured-batch timings. Make summaries identify the
  actual environment rather than hard-coding “tracker enabled.” The owner has
  queued these measurements.

- **P2 — Measure reserve usage rather than leaving the memory ledger at planned reserves.**
  **Evidence:** `memory_capacity_plan.json` and `tests/summarize_full_model.py`
  currently leave `measured_transient_peak_bytes_per_device` and
  `measured_trace_bytes_per_device` null. The ledger retains 12 GiB for
  activation/CCL and 1 GiB for trace. B1 setup is usefully reconciled: 8,993,135,104
  allocated bytes/device before preparation and 9,189,253,632 after preparation,
  with a 196,118,528-byte combined program-cache/trace/persistent increment.
  **Why this matters:** Those snapshots do not measure transient peak or isolate
  trace storage. Agentic-serving requires reserves against measured peak use
  from stage 06; explanatory missing-reasons text does not supply that evidence.
  **Required next step:** Record measured allocation high-water/trace use or a
  defensible measured upper bound for the representative B1 and target-batch
  paths, then reconcile the reserve ledger. The owner is investigating allocation
  events and trace-release deltas in separate diagnostic runs.

- **P2 — Demonstrate logit reproducibility across repetitions and fixed slots.**
  **Evidence:** `probe_mixed_slots.py` compares repeated sampled token sequences
  and cache isolation. `benchmark_full_model.py` compares sampled tokens across
  identical slots/repetitions. `probe_terminal_packing.py` compares packed and
  unpacked terminal logits, but does not compare corresponding logical requests
  after slot reordering or repeated prefill/decode. The full-model skill explicitly
  asks for logits determinism across runs and batch positions.
  **Why this matters:** Token equality can hide slot-specific logit changes; this
  stage changes fixed-slot grouping, masks, and terminal packing.
  **Required next step:** Add a small real reduced-stack diagnostic comparing
  corresponding prefill and traced-decode logits after repeat and slot reorder.
  Combine it with the queued mixed-slot probe; no broad new accuracy sweep is
  necessary. The owner has agreed to add this check.

- **P2 — Finish the already-known stage deliverables and obtain final review.**
  **Evidence:** At the inspection point, `benchmark_b8.json`,
  `benchmark_b16.json`, and `benchmark_b32.json` were absent, while the B8 pipeline
  was running. The target pool allocation and reduced `tt-perf-report` therefore
  remained pending. The README still described previously completed work as
  incomplete and all headline values as unmeasured. Runtime/sampling documents
  were being reconciled during review.
  **Why this matters:** The contract requires B1/8/16/32, the 32×32768 target,
  reduced profiling, consistent final evidence, and independent clean-pass.
  **Required next step:** Complete those previously planned tasks, run the stage
  hard checks, and request a final independent review on the completed artifacts.
  Do not checkpoint the stage as passed on this initial report.

## Other Concerns

- External page-table binding checks logical coverage (`width*32 >= context`)
  but does not check the inherited flexible-prefill SDPA requirement that a row
  contain a multiple of eight int32 page IDs. For example, context 65 with a
  width-three external map passes constructor validation but reaches the factory
  assertion in `sdpa_program_factory.cpp` requiring a 32-byte stick. Document and
  validate that physical contract at binding, or adapt the map internally while
  preserving ownership. This does not justify restricting logical prompt lengths.
  Update during review: the owner reports adding the eight-entry binding check
  and a negative-width test; runtime validation remains queued.
- The exact end-of-sequence token appears in raw `structured_tt.json` because
  evidence intentionally decodes with `skip_special_tokens=False`. Inspection
  shows it is the final token, with no following template continuation. This is
  resolved EOS behavior, not an unresolved leakage finding.
- The B1 headline is currently a tracked host-wall measurement: median TTFT
  433.080411 ms and token-out decode 35.167404 tokens/s/user for 4096/128,
  concurrency one. It has one output-token readback per generated token and no
  feedback reconstruction. The caller-visible EOS/output readbacks are distinct
  from forbidden host token feedback.

## Hard-Check Gaps

- The current seed-bound change was exercised by `probe_contract_05.log`, while
  the full-validation provenance still names the previous generator hash.
  During review, the page-window patch changed that source again. Final evidence
  should state exactly which targeted checks validate these limited changes;
  do not describe old hashes as the current source.
- `preparation_contract.json` records 129 buckets and fixed program-cache counts
  across 16 lengths, but was generated before the odd-128 continuation addition.
  Subsequent complete preparation in full validation/benchmark and the mixed-slot
  continuation probe provide supporting evidence for that addition. The final
  preparation summary should join these artifacts accurately rather than imply
  the older JSON alone proves every final-source path.
- Benchmark/result JSON does not itself assert all its environment fields.
  The inspected launch wrapper supplies tracker=1/skip=0, and the benchmark
  explicitly enables fallback exceptions. The final runtime audit should cite
  those execution facts and classify the retained runtime warnings.

## Anomaly Ledger

- **Observed anomaly:** Padded page-window gather indexes exceed the bound map.
  **Evidence:** Arithmetic examples and direct gather kernel inspection above.
  **Affected path:** Odd-128 continuation near context end; some reduced warmup
  configurations.
  **Control or comparison:** Existing start-128/start-256 tests do not cross the
  row boundary.
  **Likely subsystem:** Wrapper page-window indexing.
  **Investigation performed:** Inspected generator construction/indexing,
  decoder cache allocation, gather selection, and device reader; ran host-only
  index arithmetic.
  **Resolution:** More-work-needed; repair is present, focused hardware validation
  is queued.

- **Observed anomaly:** TT structured generation previously continued after EOS.
  **Evidence:** Work log, final `structured_tt.json`, and EOS cases in
  `reduced_trace_contract.json`.
  **Affected path:** High-level generation stopping.
  **Control or comparison:** HF returns the correct JSON; final TT output ends
  with EOS 248046 at token 114.
  **Likely subsystem:** Generator EOS policy.
  **Investigation performed:** Read raw final tokens/text and stopping code;
  checked forced-stop/fixed-length/teacher probe coverage.
  **Resolution:** Fixed in the inspected evidence.

- **Observed anomaly:** AIME free-running positional agreement is 15/100.
  **Evidence:** Both completion text files and `autoregressive_meta.json`.
  **Affected path:** Free-running text comparison.
  **Control or comparison:** TT omits “carefully” at index 15, shifting the
  otherwise coherent restatement. Teacher forcing is 98% top1 and 100%
  top5/top100; the mechanical degeneracy check passes.
  **Likely subsystem:** Normal greedy trajectory divergence under the retained
  precision policy.
  **Investigation performed:** Read both complete 100-token outputs directly.
  **Resolution:** Controlled; neither result claims the problem was solved.

- **Observed anomaly:** Explanation/code qualitative outputs truncate at 128
  tokens.
  **Evidence:** All six HF/TT qualitative JSON files and prompt metadata.
  **Affected path:** Bounded qualitative generation.
  **Control or comparison:** HF also truncates; TT text is coherent, in English,
  and reaches comparable or more final-answer content.
  **Likely subsystem:** Evaluation token budget.
  **Investigation performed:** Read actual completions and prompt rendering.
  **Resolution:** Controlled for the explicitly bounded stage check.

## Scope Inspected

- **Goal/skill paths:** Parent-provided stage-06 contract; installed
  `stage-review`, `full-model`, `agentic-serving`, `tt-enable-tracing`,
  `qualitative-check`, and model-bringup startup; checkout
  `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- **Artifact paths:** Full-model README/work log/runtime/sampling documents;
  B1 benchmark and native-context results; readiness reference metadata and
  provenance; prefill/teacher accuracy; raw autoregressive and all six qualitative
  outputs; reduced trace, mixed-slot, bucket-equivalence, preparation, sampling,
  and terminal-packing artifacts; selected successful logs; context and agentic
  memory ledgers; layer-stack lower bound; mechanical degeneracy result.
- **Code paths:** `tt/model.py`, `tt/generator.py`, changed
  `tt/multichip_decoder.py` and `tt/optimized_decoder.py`; full validation,
  benchmark, evidence inventory/summarizer, HF loader/reference, trace,
  preparation, mixed-slot, bucket, sampler and terminal probes; both common
  sampling implementations; relevant gather and SDPA allocation/index contracts.
- **Commands run:** Read-only `rg`, `cat`, `sed`, `nl`, `git status`, `git diff`,
  `git diff --check`, and small `python3` JSON/hash/index-arithmetic analyses.
  An initial `python` inspection command failed because that binary is absent
  in the host shell; it was rerun with `python3`. Only this report was written.

## Residual Risk

- This is inspection evidence, not a new hardware result. B1 native-context
  validation proves capacity and finite output, not new full-model HF accuracy
  at 262144 tokens; the artifacts correctly retain that distinction.
- The owner is modifying the live worktree and producing new evidence. This
  initial verdict cannot certify the eventual final source or artifacts.
- No precision-policy relaxation, inter-layer residual-layout regression, host
  argmax on the optimized path, broad trace-allocation exemption, or unresolved
  visible output degeneration was found in the inspected source/evidence.
