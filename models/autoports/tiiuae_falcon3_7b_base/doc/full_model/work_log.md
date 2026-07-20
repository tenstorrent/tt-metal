# Full-model work log

## 2026-07-19 — baseline and implementation

- Started from optimized-multichip-decoder commit
  `4b1bb58f3ca85b424e807d462f2ed52e176e4cc1`.
- Confirmed four local Blackhole P300c devices, reset devices 0-3, reopened a
  `1x4` mesh with `FABRIC_1D_RING`, and verified clean open/close with a
  512,000,000-byte trace region per DRAM bank.
- Loaded the exact local HF revision
  `bf3d7ed586cb22a921520e2d681a9d3d7642cde8` and all four BF16 safetensor
  shards. Architecture: LlamaForCausalLM, 28 layers, hidden 3072, intermediate
  23040, 12 Q heads, 4 KV heads, head dimension 256, vocab 131072, untied LM
  head, max position 32768.
- Added `tt/model.py` and `tt/generator.py`; retained the optimized TP4 decoder
  as the sole block implementation. Extended chunked paged prefill in
  `tt/multichip_decoder.py` and exact TP greedy in the common `Sampling1D`.
- Added the `P300X2` readiness mesh label for reproducible standard runner
  commands on the four-die QuietBox 2 target.

## 2026-07-19 — correctness and qualitative gates

- Generated a fresh one-entry AIME24 reference with 155 prompt tokens, 100 HF
  tokens, and top-100 candidates. SHA256:
  `45637a3d5f7c41146d267fe6eb0678d48e588fb9d3a6d7d3a2ca32838de720fa`.
- Exact tokenizer metadata has no chat template. `apply_chat_template` raises;
  native completion prompting is recorded instead of inventing a wrapper.
- `run_prefill_check`: top-1/top-5/top-100 = 92/100, 100/100, 100/100.
- `run_teacher_forcing`: top-1/top-5/top-100 = 93/100, 100/100, 100/100;
  final standard-runner TTFT 2972.89 ms; callback decode 63.82 t/s/u.
- `run_autoregressive`: HF and TT match five initial generated tokens and
  diverge at index five. Both 100-token outputs are coherent English without
  repetition or language drift; verdict recorded under `results/autoregressive`.

## 2026-07-19 — sampler, trace, performance, and capacity

- Compared both common sampler abstractions and selected `Sampling1D`.
  Hardware semantic control: exact split greedy, common full gather
  force-argmax, and actual `TTSampling` all return token 2107. Final
  split/force/`TTSampling` costs: 0.799903/1.004767/1.010715 ms.
- Full 28-layer trace evidence: one capture, four token-out replays, direct
  feedback match, current/rotary position 132, unchanged page-table copy delta
  zero, changed/expanded delta one each. Changed-page replay changes logits,
  repeated state is exact, and restoring the mapping restores logits exactly.
  One hundred twenty-eight queued trace pairs reach 75.6894 t/s/u; the
  caller-visible token-out loop reaches 75.4356 t/s/u; forced-token pairs
  reach 75.6760 t/s/u. Model/sampler cost is 12.415675/0.793731 ms, sampler
  fraction 6.009%.
- Cold/warm TTFT on the standard 128/128 workload: 79.345/26.719 ms.
- Full weights/runtime allocate 2,231,753,216 bytes/device. All 28 32K BFP8
  paged caches add 499,122,176, leaving 27,351,855,616 allocatable bytes/device.
- Public maximum-boundary gate: 32,767 logical prompt tokens, generator padding
  to 32,768, sixteen 2K chunks, all 1,024 pages populated, token sampled, pass.
- Serving lifecycle gate: mixed 33/47-token prompts in 32 fixed slots, inactive
  cache positions/page rows preserved, reset synchronized and released traces
  before clearing the same cache/input buffers, the next optimized request
  safely recaptured once, explicit host-sampling compatibility returned two
  tokens, and mixed
  2,049/2,079-token prompts crossed the 2,048-token chunk boundary, pass.
- Seeded top-k/top-p/temperature split sampling checkpoints/restores RNG around
  capture, reproduces the same tokens with the same seed on a reused trace,
  uses zero token host copies, and feeds the sample back directly, pass.
- Full active-32 factorial gate: all layers, prompt lengths 33--64, exact
  repeat, reversed slots preserving logical page rows, same slots with
  disjoint remapped pages, reversed slots plus remapped pages,
  active-16/inactive-row controls, and representative batch-1 runs pass. Every
  split token equals the same run's host argmax, first/last-layer K/V pages are
  exact, and all cross-run rows remain mutual-top-5 with minimum PCC 0.997666.
- Six shared qualitative prompts ran with the exact base tokenizer completion
  transform. Five are clean; the haiku is fluent but repeats its stanza and is
  retained as a limitation. A focused 100-token control proves host-eager,
  traced, safe-recapture traced, and teacher-forced greedy equality, ruling out
  trace/sampler/cache/reset/position causes. A seeded top-k/top-p run is
  coherent and non-repetitive. Fibonacci matches HF for all 100 tokens.
- Reduced real-weight Tracy (embedding + layer 0 + final norm + TP LM head +
  split sampler) records 4.538-ms prefill and 1.441-ms/token decode. Matmul,
  local argmax, and local-max reduction are the leading device categories;
  sampling remains only 6.009% of the full 28-layer traced pair.
- All dedicated evidence gates ran with
  `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`.
- Final `tt-smi -s` after serialized hardware runs reports all four P300c
  devices with DRAM healthy, matching live heartbeat 23220, and 45.7--50.1 C;
  the mesh closed cleanly (`results/hardware_health.json`).

## 2026-07-20 — AutoFix closure after independent review

- A fresh stage review returned more-work-needed on allocator-safe trace
  lifecycle, exact batch-position evidence, the repeating haiku, caller-visible
  performance attribution/roofline, profiler command provenance, and a final
  full-stack Watcher run. Per the stage contract, these findings entered the
  `$autofix` loop rather than being waived.
- Trace lifecycle: `reset()` now synchronizes and releases both address-bound
  traces before any cache or persistent-input fill. It clears the same buffers
  and preserves their identities; the next optimized request captures exactly
  one replacement pair. Seeded stochastic capture checkpoints/restores RNG,
  so warm-up/capture is invisible to the generated sequence. Contract and
  batch-factorial gates pass with no allocator warning.
- Persistent async CCL: each trace capture resets the same two shared BFP8 CCL
  buffers and their semaphores once to capture epoch zero. There is no
  per-token reset, rebuild, buffer replacement, or policy change.
- Batch/position isolation: replaced the confounded reverse-only comparison
  with A/A-prime/B/C/D factorial runs. A second review correctly rejected the
  initial mutual-top-5 gate: dynamic paged SDPA rounded the position-64 user's
  three-page sequence to a four-page read before causal masking, while the
  generator mapped only three pages. The masked tail entry was `-1`, producing
  an invalid physical read. AutoFix now rounds mappings to the kernel's exact
  power-of-two/eight-page window and rejects missing, out-of-range, aliased,
  or cross-slot mappings. Identical runs, preserved-page slot reversals,
  remapped pages, and reverse-plus-remap are now bit-exact in logits and tokens
  across all 28 layers. The gate captures every rounded cache page plus the
  prompt-final and decode-update rows on the first and last device ranks.
- Qualitative isolation: host-eager greedy, traced free greedy, traced greedy
  after safe release/recapture, and traced teacher-forcing produce exactly the
  same repeating 100-token haiku. Positions/feedback are exact and token H2D
  copies are zero. Seeded device top-k 8/top-p 0.9/temperature 0.7 is coherent
  and non-repetitive. The greedy result remains a documented low-precision
  model limitation.
- Performance closure: the final standard 128/128 workload measures 78.073-ms
  cold TTFT, 25.083-ms warm TTFT, 75.411 caller-visible token-out t/s/u, and
  75.687 queued device-only trace-pair t/s/u. A 1/2/4/8/16/28-layer real-weight
  sweep at `max_context=256` fits `0.2697 + 0.285363 * layers` ms with R-squared
  0.999998. The 292-GB/s analytical matrix/KV/sampler floor is 4.193 ms versus
  13.256 ms measured caller-visible; assumptions and omissions are explicit in
  `results/perf_summary.json`.
- Watcher initially exposed two source bugs in the generic async all-gather
  writer. A one-tile packet unconditionally initialized a scatter header whose
  valid chunk-count range starts at two; that initialization is now compiled
  only for multi-tile packets. A Linear endpoint with no outward targets then
  unconditionally requested the absent forward/backward fabric connection;
  connection lookup is now guarded by the existing valid-target predicate.
  The exact reduced Ring-hidden/split-greedy/full-vocabulary-force regression
  passed; the subsequent complete-stack run exposed the trace-state and
  packet-header-pool issues closed in the frozen-source section below.
  Ethernet Watcher was
  disabled because firmware image size 27,920 exceeds its 25,600-byte Watcher
  kernel buffer; TENSIX Watcher remained enabled. Watcher and Tracy were run
  separately.
- Exact commands for the canonical runners, dedicated gates, reduced Tracy,
  advice-enabled `tt-perf-report`, and Watcher are in `README.md`. Watcher
  evidence is in `results/full_model_sampler_watcher.json` and
  `results/full_model_watcher.json`; timing from Watcher is not used as a
  performance claim.
- Full-context closure: the public generator executed all 28 layers for a
  32,767-token prompt, owned padding and sixteen 2,048-token chunks, generated
  two tokens, and replayed decode at position 32,767. All 1,024 pages were
  mapped; final K/V pages were nonzero on four ranks in all 28 layers (224
  tensors), and both device position tensors advanced to 32,768.

## 2026-07-20 — final command provenance

- Canonical runner commands, exit codes, exact revision, fallback policy, raw
  log paths, and SHA256 values are in `results/runner_provenance.json`.
- Dedicated commands are reproduced in `README.md`. Principal final artifacts:
  `results/full_model_evidence.json`, `full_model_contract_coverage.json`,
  `full_model_batch32.json`, `full_context_coverage.json`,
  `full_model_profile.json`, `qualitative_suite/`, and `profile/reduced/`.
- First independent stage review returned more-work-needed: ordinary reset
  released traces, stochastic trace warm-up consumed RNG invisibly, the
  six-prompt qualitative suite and profiling were absent, batch-32 evidence
  was incomplete, and changed-page evidence copied without replay. Each item
  was implemented and rerun on hardware before the final rereview.
- A subsequent review returned more-work-needed on exact page-boundary batch
  determinism, final-source provenance, and the one-layer maximum-context
  shortcut. The rounded SDPA page-table repair and full 28-layer context gate
  close the first and third findings; every required artifact is rerun from
  the frozen source before the next review.

## 2026-07-20 — frozen-source final evidence

- Froze the final runtime sources after the AutoFix trace repair, rebuilt and
  installed `libtt_metal.so`, and ran a 600-replay CCL-free Watcher control.
  The control and the complete 28-layer Watcher run both pass. The complete
  run covers cold/warm prefill, all three greedy sampler comparisons, changed/
  unchanged/restored page tables, direct token feedback, reset, and 128 trace
  pairs. Its steady-state token, position, rotary-position, page-table, and
  sampling-parameter host-copy deltas are all zero.
- The final complete-stack Watcher investigation closed two independent
  runtime defects beyond the earlier scatter/direction guards. Firmware uses
  `0xf0` for the valid `RUN_MSG_REPLAY_TRACE` run state, so Watcher now
  validates primary run messages separately from subordinate sync messages.
  The traced minimal async all-gather writer also now resets its packet-header
  pool at every kernel invocation, matching the trace-compatible upstream
  contract. The final logs contain no Watcher corruption, NoC error, assert,
  device stall, or fatal termination.
- Frozen-source official runners: prefill top-1/top-5/top-100 is
  92/100, 100/100, 100/100; teacher forcing is 93/100, 100/100,
  100/100. The readiness teacher-forcing process reports 3,686.51-ms TTFT and
  40.85 t/s/u including callback compatibility overhead. The fresh
  autoregressive HF/TT pair each contains 100 tokens, matches through index
  four, diverges at index five, remains coherent English, and passes the
  degeneracy checker.
- Frozen-source canonical performance: 78.073-ms cold TTFT, 25.083-ms warm
  TTFT, 75.411 caller-visible token-out t/s/u, 75.687 queued device-only trace
  t/s/u, and 75.676 teacher-forcing trace t/s/u. Model/sampler traces measure
  12.416047/0.793900 ms, so sampling is 6.010% and does not dominate.
  Split/force-argmax/`TTSampling` measure 0.806366/1.008621/1.008905 ms and all
  return token 2107.
- The final 1/2/4/8/16/28 depth sweep fits model-trace latency to
  `0.269728 + 0.285363 * layers` ms with R-squared 0.999998453. At 28 layers it
  measures 8.261744-ms model trace, 0.793823-ms sampling trace, 110.419
  device-only t/s/u, and 109.753 caller-visible t/s/u.
- The corrected Tracy command writes into `profile/reduced` and generated the
  timestamped v2.1 report at
  `profile/reduced/reports/2026_07_20_02_47_40/`. The reduced real-weight graph
  measures 4.637-ms prefill and 1.442-ms/token decode over three replays.
  Compact `tt-perf-report` summaries were regenerated from that timestamped
  ops CSV.
- The all-28-layer 32,767-token gate passes with all 1,024 pages and 224 K/V
  rank-layer tensors checked. Weight/runtime plus full-context K/V allocates
  2,730,875,392 bytes/device; maximum-context execution peaks at
  2,734,023,168 bytes/device and leaves 27,348,707,840 bytes/device free.
  Batch-32 factorial and serving-contract gates also pass exactly.
- Reran all six qualitative prompts and read every TT completion. Five are
  clean; the retained haiku loop is exactly reproduced by host eager, traced
  greedy, safe recapture, and teacher forcing. Seeded device sampling remains
  coherent. The final health snapshot reports four DRAM-healthy P300c devices,
  live heartbeat 35300, 48.7--52.9 C, and zero corrected/uncorrected GDDR
  errors on firmware 19.8.0/KMD 2.8.0.
- Exact final source, runner, artifact, raw-log, HF revision, hardware, and
  environment hashes are recorded in `results/final_evidence_manifest.json`.

## Review and commits

- Final fresh `$stage-review`: `clean-pass`; required work: none; material
  hard-check gaps: none. The reviewer independently matched source, runner,
  artifact, profile, raw-log, and rebuilt `libtt_metal.so` hashes to the final
  manifest and reproduced the static gates without touching hardware.
- Stage implementation commit: recorded in the post-commit entry below.
