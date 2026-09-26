# Stage Review

Verdict: clean-pass

Independent final review of Stage 02, fused-decoder, for
google/gemma-4-26B-A4B-it, completed 2026-09-26. This supersedes the early
incomplete-evidence verdict preserved in STAGE_REVIEW_INITIAL.md.

Reviewed branch: gemma-4-26b-a4b-it; base:
37c8b975c83712bbdf10d873962d3cc396439710. Frozen fused_decoder.py SHA256:
0c8892be32e04202fdbddd2b06850848c63ae57660c49d03c4d149a5a88b041c.
Paths below are relative to models/autoports/google_gemma_4_26b_a4b_it unless
otherwise stated. The reviewer used source/artifact inspection and host-only
analysis, opened no device or server, and changed only this report.

## Required Work

None for the reviewed implementation and evidence. Stage-only local checkpoint
commits and their SHA records follow this clean-pass; they are not claimed
complete here.

The additional S65 sliding FP32-HF failure remains explicitly failed. Direct
functional equivalence passes at the original 0.995 threshold, and independent
CPU controls establish the inherited BF16-cache cause. This is a controlled
numerical limitation, not an unconditional claim that all FP32-HF checks pass.

## Other Concerns

- BF16 cache rounding can change close MoE expert ranks. The S65 position-68
  outlier is reproduced in the functional decoder and a CPU cache-only
  control. The fusion stage preserves that existing cache contract.
- Small candidate differences retain their measured scope. Shared-down LoFi
  beats HiFi2 by approximately 0.16/0.25 us; the larger approximately 17-us
  improvement comes from direct sharding through the consuming norm.
  No materially faster, broadly correct tested candidate remains disconnected
  from the final defaults.

## Hard-Check Gaps

No unresolved required gap was found.

- All 17 verified_validation_commands.json entries complete with exit 0 and
  the frozen runtime hash. The new mixed-tail test has separate completed
  evidence in verified_mixed_tail_pytest.json. Earlier batch64/selected
  measurements were not substituted for final modified-runtime evidence.
- Headline compares every prefill output and all 128 decode outputs. Recomputing
  Pearson correlation from the four saved headline tensor files reproduced the
  paired JSON; every inspected value was finite.

| Layer kind | Fused HF prefill / minimum decode PCC | Direct functional/fused prefill / minimum decode PCC |
| --- | ---: | ---: |
| sliding_attention | 0.9985529033 / 0.9989850246 | 0.9996620521 / 0.9996588301 |
| full_attention | 0.9994185025 / 0.9997355920 | 0.9996741208 / 0.9998033554 |

- Both kinds pass nine changing requests, all 32 cache slots, deterministic
  replay and partial-page continuation with prefix/other-slot preservation.
  Runtime audits reject host computation, and regression tests replace
  FunctionalDecoder._forward with a raising function during fused execution.
  Source inspection confirms inherited orchestration with actual fused
  computation, without a functional computation fallback.
- The changed expert batching now has fresh S65 coverage of physical 64+32
  tails for both kinds. The durable regression retains all eight decode
  positions, finite tensors, fixed-state replay equality, program-cache
  guards, runtime audits, fallback prohibition and direct PCC >= 0.995.
  Both cases pass; unchanged S33 cases retain their verified passing run.
- All four final context cases pass at 262144 and 262143. Sliding prefill PCC
  is 0.9981859749/0.9981839549; full is 0.9962342265/0.9961807402.
  Every end-context decode check passes. TT computes all query/cache rows;
  the HF oracle explicitly samples 291 rows. No capacity or valid logical
  length was reduced.
- Separate final watcher logs match their recorded SHA256s, contain six
  completed dumps each, and have no error/assert/overflow/corruption/timeout/
  invalid-access diagnostic in independent scans.
- Complete device windows were independently rederived from final CSVs:
  one prefill and 128 decode windows per kind, all on one physical ASIC.
  Sliding has 1141 native prefill operations and 152 per decode; full has
  1077 and 142. Every operation and internal gap remains in each denominator.

| Layer kind | Final prefill device us | Final decode device us | Prefill / decode speedup over functional |
| --- | ---: | ---: | ---: |
| sliding_attention | 2812752.08697 | 5146.69761 | 1.777x / 1.906x |
| full_attention | 2825397.16222 | 5584.80889 | 1.773x / 1.951x |

- Final unprofiled defaults reproduce chosen combined candidates within
  0.03%. Both profile CSV checksums/source hashes match provenance. Raw
  host/device captures, tt-perf-report text/CSV, per-replay CSV and report
  commands exist. Roofline formulae reproduce the recorded values.
- The compact packet at
  bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/e73d7165-fc65-44f4-828b-1f979f0dcf75.json
  records exact 4096/128/B1/C1, one device, both layer kinds and final device
  metrics. All ten accuracy entries match their raw evidence, including the
  failed S65 HF row. No host timing or kernel subtotal replaces device time.
- Final pre-commit output passes applicable hooks. Independent AST parsing
  succeeds for the runtime and all 31 test modules. No C++/CMake change is
  present, so a build is not required.

## Anomaly Ledger

1. **S65 sliding FP32-HF outlier.**
   Observed: position 68 PCC is 0.9813712959 fused and 0.9809274231 functional.
   Evidence: tail65_sliding_*_paired.{json,pt}, tail65_functional_stages.json,
   tail65_cpu_precision.json, AUTODEBUG_tail65.md and mixed-tail pytest.
   Affected path/likely subsystem: inherited BF16 K/V before near-tied routing.
   Control: CPU FP32 HF with only K/V rounded through BF16 reproduces the sole
   failure (0.9814915924) and expert 10-to-70 substitution. CPU on the actual
   TT residual chooses the TT set; same-Q/K/V attention PCC is 0.9999998501.
   Investigation: inspected control source, checked input/fixture hashes and
   independently recomputed all paired outputs. Direct prefill PCC is
   0.9991649499; all eight decode PCCs exceed 0.9997196428, including
   0.9998401882 at position 68. Full-kind S65 HF checks all pass.
   Resolution: **controlled inherited limitation**. No discarded position,
   lowered threshold, runtime precision change or relabeled HF result.

2. **Shared-down fidelity was mislabeled HiFi2.**
   Evidence: final decode_example_ops.csv, shared_fidelity_* JSON/journal,
   and TTNN matmul default selection at
   ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2808.
   Affected path/likely subsystem: explicit program config changes inferred
   fidelity from baseline HiFi2 to LoFi.
   Control: seven legal K divisors measured with explicit LoFi and HiFi2,
   identical BF16 operands, consumer norm and layout; all pass PCC/replay.
   K6 is fastest for both policies; selected LoFi measures 44.718/44.764 us
   versus HiFi2 44.880/45.018 us.
   Investigation: traced runtime rows to defaults, inspected probe configs
   and recomputed medians.
   Resolution: **fixed attribution and controlled selection**. Final docs and
   summary assumptions name actual LoFi; existing gates validate that unchanged
   runtime.

3. **Native precision/layout candidates fail after plausible adaptations.**
   Evidence: precision/reuse matrices, sdpa_exact_* results, boundary cast/I2S/
   unary results, AUTOFIX.md, AUTODEBUG_precision.md and PATTERNS.md.
   Affected paths: normalization, rotary, softmax/SDPA and cache casting.
   Controls: isolated-site/phase real-weight checks; external FP32 gamma;
   precise sliding head norms; BF16-Q/HiFi4/FP32-accumulation SDPA; and
   unary-chain cast-to-shard after dedicated typecast rejected the layout.
   I2S conversion fails full/sliding PCC, while adapted unary passes.
   Full keeps its faster original cast/shard after coherent combined controls.
   Likely subsystem: validated op constraints and precision before discrete
   expert selection. Investigation compared source, failed controls and final
   broad gates; first API failures were not accepted as sufficient rejection.
   Resolution: **controlled rejection**. All applicable skill patterns and all
   five reviewer-identified movement boundaries have accepted or adapted,
   measured rejection evidence.

4. **Early probe warmup/wiring errors.**
   Evidence: cache_gather_adapted.json, corrected component reruns, shared
   sharding retry logs and PATTERNS.md.
   Affected path/likely subsystem: experiment selection, insufficient warmup
   and stale wrapper interception.
   Controls: warmed embedding approximately 139 us versus 3980/4107/24392 us
   for exact alternatives; explicitly forced decode-GELU merge approximately
   932-933 us versus 930-931 us separate; corrected shared-down interception.
   Investigation inspected candidate wiring and fresh samples/replay equality.
   Resolution: **fixed**; initial errors and obsolete timings do not supply
   final selection claims.

5. **Earlier profiler teardown and incomplete-export failures.**
   Evidence: AUTOFIX_profiler.md, failed logs and final tracy/*_verified
   captures/provenance.
   Affected path/likely subsystem: marker pairing, plausibly clock rollover;
   producer/export ordering for the incomplete CSV.
   Control: fresh captures complete with unchanged instrumentation and marker
   validation enabled. Investigation checked exit codes, raw files, hashes
   and every complete replay window.
   Resolution: **controlled**; invalid/incomplete captures are excluded.

6. **Different PCC on a repeated long-context position; interrupted jobs.**
   Evidence: long-context source/JSON, functional controls, verified journal
   and resume work log.
   Affected path/likely subsystem: deliberate intervening cache mutation and
   execution interruption.
   Control: functional tests show the same state-dependent revisit behavior;
   fixed-state headline/batch replay is separately equal. The three missing
   final context jobs subsequently completed.
   Investigation checked test ordering, hashes and exit codes.
   Resolution: **controlled**; no interrupted job is counted as passing, and
   unchanged valid gates were retained.

7. **Post-capture allocation and topology/parser warnings.**
   Evidence: request-reuse logs, functional RUNTIME_AUDIT.md, watcher logs and
   final profiles.
   Affected path/likely subsystem: caller tensor lifetime and host diagnostics.
   Control: stable trace-owned buffers persist; temporary request tensors are
   released before replay; nine requests and separate watcher runs pass.
   Every final window consistently uses one ASIC.
   Investigation compared lifetime code and warning classification with the
   actual runtime records.
   Resolution: **controlled for the documented lifetime contract**, not
   permission for arbitrary allocations while a trace is live.

## Scope Inspected

- Contract/skills: supplied Stage 02 goal; repository stage-review,
  graph-fusing and tt-device-usage; installed tt-model-bringup 0.1.14
  stage-review/startup; relevant TT review core, model and trace guidance.
  Only missing or invalidated boundary checks were requested.
- Code: complete fused/functional decoders; attention, precision and routing
  helpers; relevant imported Gemma4 operations and TTNN validators/defaults;
  modified runners, probes, regression tests and performance summarizer.
- Evidence: README, PATTERNS, ACCURACY, PERFORMANCE, work log, context
  contract, AutoFix/AutoDebug reports, initial review, all 66 summarized
  complete-headline candidate rows, final verified artifacts, explicit
  fidelity/mixed-tail controls, watcher raw logs, final Tracy directories,
  functional baseline and resumed telemetry packet.
- Analysis: read-only git/text/file inspection; JSON/CSV/hash/AST scripts;
  CPU-only loading and Pearson recomputation of saved tensors. Candidate
  medians, pass states, device windows, rooflines, packet accuracy and input
  hashes were checked independently.
- No reviewer hardware use, reset, server, implementation edit, new subagent,
  checkpoint or push.

## Residual Risk

- These are pinned real-weight single-layer checks, not full-model generation
  or later serving validation. The controlled BF16-cache outlier remains an
  actual FP32-HF disagreement and is visible in final evidence.
- Long-context HF coverage is sampled. DRAM bytes are operand estimates,
  not controller counters. Useful-FLOPs and mixed-fidelity rooflines are
  reference percentages, not measured homogeneous hardware utilization.
- This verdict applies to the frozen runtime and inspected tests/artifacts.
  The owner should record this clean-pass and stage-only local checkpoint
  SHAs. Later runtime changes require focused revalidation of affected paths;
  no push is authorized by this review.
