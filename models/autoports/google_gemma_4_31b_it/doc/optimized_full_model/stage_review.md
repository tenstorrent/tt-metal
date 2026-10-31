# Stage Review

Verdict: clean-pass

Stage 07, optimized full model: `google/gemma-4-31B-it`, pinned revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`, TP4 Blackhole. Independent,
inspection-only review of the live worktree based on
`d3dcd354e6df404724721b2e0e3af2e1ff7db71d`, completed 2026-10-31.
No device commands, tests, servers, resets, or implementation edits were run by
the reviewer. This report is the only reviewer-written file.

## Required Work

- None. The final implementation and evidence satisfy this stage's contract.
  The local checkpoint and recording its SHA follow this review; they are not
  represented as already completed here.

## Other Concerns

- The observed throughput changes are small: 0.222%, 0.260%, 0.275%, and 0.377%
  for batches 1, 32, 16, and 8. These are measured differences, not proof of a
  statistically robust speedup. The substantive orchestration result is removal
  of per-token host reads from the fixed-step path. The README describes both
  facts accurately.
- The 1024-token history capacity is a construction-time buffer setting.
  Longer outputs can configure a larger buffer before capture or use the
  streaming API. It does not reduce the 262144-token context contract.
- Worker watcher checks passed with Ethernet instrumentation disabled under
  the inherited configuration-buffer limitation. This is scoped watcher
  coverage, not a claim that Ethernet watcher instrumentation passed.

## Hard-Check Gaps

- No outstanding required check was found. Existing stage, context,
  degeneration, and source-check logs were inspected. The agentic checker has
  only the explicit tied embedding/head representation advisory; the two
  incompatible physical layouts and their bytes are accounted for.
- No full 60-layer device profile or full-model measured roofline is available,
  as required by the operator's profiling restriction. Reduced layer-0/5
  attribution is clearly separated from full-model host timing. The 9.489 ms
  analytical DRAM lower bound is explicitly limited to the named weight/KV/head
  traffic; it is not a measured full-model latency or utilization value.

## Evidence Reconciliation

- **Source and precision.** All four production hashes match
  `source_manifest.json`. The changed files are `tt/model.py` and
  `tt/generator.py`; decoder sources remain unchanged. The final head retains
  all K elements and ascending vocabulary order in eight 8192-column local
  splits. Inspection of the exported raw profile confirms BFP4/LoFi decoder
  projections and BFP8/HiFi2 terminal projections, including explicit
  `MinimalMatmulConfig` for prefill. Every changed terminal matmul receives
  `Optimized` advice. The larger-width, sender-grid, reader-count and K-block
  trials and their adapted L1 failures support the selected head geometry.
- **Accuracy and text.** Fresh full-model prefill and teacher-forcing checks
  have top-5/top-100 agreement of 100%/100%; top-1 is 94%/93%. The reused HF
  reference has matching pinned model/tokenizer revision, AIME/chat-template
  provenance, 100-token length, top-k and generation command. The actual AIME
  completions and all six shared TT/HF pairs were read. All six streamed TT
  records exactly equal stage 06, including token IDs. The buffered prefixes
  match those accepted outputs. No new unexplained model-output anomaly remains.
- **Changed boundaries.** Real-activation head comparisons at physical M
  32/512/544/768/800/1024 have zero maximum absolute logit difference on all four
  devices. Generator tests cover 1023/1024/1025 logical prompts, mixed lengths,
  fixed slots, inactive rows, page remapping, changed seeds, greedy/sample/greedy
  transitions and buffered-versus-streaming equality at every supported batch.
  The 1024-output-token boundary passes across three successive requests.
  Full-stack execution at prompts 262143 and 262144 and decode input position
  262143 preserves the advertised context.
- **Trace lifetime.** All-batch state tests ran with allocation tracking enabled
  and program-cache allocation skipping disabled. Their assertions cover stable
  three-trace identities, unchanged program-cache entry counts, sampled-token
  feedback and device position increments. No corruptible-allocation exemption
  is used. The buffered path prepares storage and both sampler variants before
  capture; it does not allocate, refresh inputs, synchronize or read back in
  its steady loop. Every exact 4096/128 timing sample records 127 model replays,
  128 sampler replays, one initial position refresh, zero token/page refreshes,
  zero captures, and only the first-token/final-history reads.
- **Full-model timing.** All reported medians recompute from the three clean
  samples in each `performance_b*.json`; tracking, watcher and profiler are
  disabled. Complete generated sequences match the explicit streaming control.
  Batch-1 is 29.162054 ms/token. The independently checked engineering budget
  is 26.103836 ms of selected layer times + 1.210804 ms terminal + 0.482175 ms
  sampler = 27.796815 ms, leaving 4.9115%. Other batches are 4.67%–6.26% above
  their layer sums before terminal credit. These are cross-run budgets, as
  documented, and meet the 10–15% closure criterion.
- **Sampling.** The summary matches all 96 raw semantic-greedy records and all
  eight timing records. Split greedy returns maximum-valued tokens and takes
  0.482–0.526 ms versus 2.776–2.894 ms for the padded force-argmax control.
  Production uses the same split defaults. The measured decode window contains
  fast `TopkLargeIndicesDeviceOperation`, candidate-width-32 gathers and no
  generic `TopKDeviceOperation`, `ArgMaxDeviceOperation` or full-vocabulary
  gather. Sampling does not dominate the complete model.
- **Profile export.** Independently parsed the raw CSV named in
  `terminal_attribution.json`, without invoking the stage summarizer. Every
  measured row has firmware timestamps. All four per-device counts, trace IDs
  and spans reproduce the saved results: reduced decode 2991.232 us (162 ops
  per device, traces 1/3), model-only 2441.101 us (130, trace 1), terminal
  1237.016 us (22, trace 0), buffered sampler 539.649 us (32, trace 3), and
  diagnostic prefill 1498632.647 us (467, eager/trace 3). All five advice-enabled
  text, CSV and console reports were inspected. The prefill span includes
  explicit profiler drains and is not production TTFT.
- **Memory.** Recomputed full-stack constructed/prepared/usable/free/headroom
  arithmetic at batches 1/32/16/8 from allocator snapshots. KV inventory equals
  `10880 * pool_tokens + 222822400 * batch` bytes per ASIC. Reduced cold/warm
  allocation events reconcile exactly to allocator deltas. The maximum measured
  incremental peak plus 31 retained selected-logit tiles is
  `560168960 + 31 * 4194304 = 690192384` bytes, below the 805306368-byte analytical
  reserve. Batch-32 retains 7990269440 bytes per ASIC after that reserve, with
  the required 1048576-token aggregate pool. The ledger distinguishes this
  conservative bound from a measured full-stack transient peak.

## Anomaly Ledger

- **Observed anomaly:** larger head splits failed L1 validation; a temporary
  output-block-height-1 repair risked partial-core reads. **Evidence:**
  `AUTODEBUG.md`, `AUTOFIX.md`, candidate error JSON and archived
  `rejected_block1/`. **Affected path:** terminal projection/preparation.
  **Control or comparison:** final 8K head versus interleaved head at the
  affected boundaries. **Likely subsystem:** matmul CB geometry/reader bounds.
  **Investigation performed:** source diagnosis, adapted splits, exact all-chip
  logit comparison, full-model gates and watcher. **Resolution:** fixed; the
  unsafe temporary configuration is absent from final source.
- **Observed anomaly:** buffered `shared_4` continues into channel/turn markers
  after EOS. **Evidence:** `qualitative_buffered.json` and
  `qualitative_verdict.md`. **Affected path:** explicitly fixed-step generation.
  **Control or comparison:** its first 11 tokens equal the complete streamed
  answer; separate 128-token controls compare identical ignore-EOS semantics.
  **Likely subsystem:** intentional stopping policy. **Investigation performed:**
  inspected output tokens and runner length/stop behavior; the initial overly
  broad quality wording was corrected. **Resolution:** controlled; only the
  EOS-truncated prefix is conversational-quality evidence.
- **Observed anomaly:** prior thermodynamics wording differs from free-running
  HF output. **Evidence:** shared prompt 3 and stage-06
  `thermodynamics_control.json`. **Affected path:** qualitative generation.
  **Control or comparison:** exact stage-06 TT output and pinned HF continuation
  on the TT prefix. **Likely subsystem:** ordinary autoregressive branching.
  **Investigation performed:** checked exact previous-stage equality and retained
  control provenance. **Resolution:** controlled; no stage-07 regression.
- **Observed anomaly:** report advice calls prefill configs missing and suggests
  tracing already traced boundaries. **Evidence:** all five perf reports,
  raw runtime attributes and `runtime_policy_rows.json`. **Affected path:**
  report interpretation. **Control or comparison:** explicit MinimalMatmul
  config, measured trace IDs, inherited precision-matched geometry/input-L1/
  topology trials. **Likely subsystem:** generic report parsing/advice.
  **Investigation performed:** inspected actual configs, inherited experiments,
  and all measured trace bindings. **Resolution:** controlled; advice is
  retained, with no unaddressed material changed-path optimization identified.
- **Observed anomaly:** unused streaming-trace, mixed metadata, unknown-op
  taxonomy and optional WASM-copy warnings; isolated FLOPs estimates exceed
  100%. **Evidence:** profile/export logs, five report consoles and
  `profile_analysis.md`. **Affected path:** profiler enrichment/reporting.
  **Control or comparison:** stage-local trace exists; raw CSV has complete
  all-device measured windows and excludes unused trace 2. **Likely subsystem:**
  diagnostic tooling. **Investigation performed:** independently recomputed
  complete windows, inspected report rows and verified precision/trace IDs.
  **Resolution:** controlled; none is used as full-model device timing or
  measured roofline evidence.

## Scope Inspected

- Goal/skill paths: `bringup/run_contract.json`,
  `bringup/agentic_evals/inputs.json`, the supplied stage-07 contract, and installed
  `model-bringup`, `stage-review`, `multichip`, `optimize`, `agentic-serving`,
  `tt-device-usage`, `tt-enable-tracing`, `full-model`, `qualitative-check` skills
  under `/home/mvasiljevic/.codex-personal-gemma4/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.22/skills/`.
  Also used review-core/router, model-bringup-review and trace-review from the
  installed review package.
- Artifact paths: this directory's final README, work log, runtime/optimization
  audits, source manifest, AutoDebug/AutoFix reports, candidate tables, accuracy,
  qualitative and autoregressive outputs, all-batch performance/state/memory
  evidence, capacity probe, sampler controls, watcher and checker logs, all five
  perf reports and consoles, runtime policy rows, raw exported CSV and reduced
  attribution. Inspected global context/agentic ledgers and relevant stage-05/06
  controls and optimization dispositions.
- Code paths: complete `tt/model.py`, `tt/generator.py`, `tt/prefill.py`; changed
  and new state, boundary, readiness, performance, memory, sampling and profile
  runners; relevant generic sampler and indexed-fill interfaces. No removed
  Gemma implementation or Gemma-specific framework branch was consulted.
- Commands run: read-only `git status`, `git diff`, `git rev-parse`, `rg`, `cat`,
  `sed`, `tail`, `wc`; standard-library Python artifact parsing, arithmetic,
  raw CSV reconstruction and SHA-256 verification. No TTNN imports or hardware
  execution. The report was written with `apply_patch`.

## Residual Risk

- AIME evidence covers 100 positions from one prompt; qualitative outputs are
  bounded continuations, not complete task-score or long-context accuracy
  claims. Maximum-context tests establish execution/capacity, with historical
  layer PCC controls retained separately.
- Reduced instrumentation cannot establish a same-run 60-layer device peak or
  latency. The final documentation correctly preserves this limitation.
- The stage validates the standalone generator. Serving registration, scheduler
  overlap, prefix reuse and vLLM/API behavior remain later-stage work; this
  review does not approve or imply a serving adapter.
