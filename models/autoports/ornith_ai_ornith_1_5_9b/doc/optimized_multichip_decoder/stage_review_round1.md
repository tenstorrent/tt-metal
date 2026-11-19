# Stage Review

Verdict: more-work-needed

Round 1 of 3. Independent, host-only review of stage 5, optimized-multichip-decoder, for `ornith-ai/Ornith-1.5-9B`, revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. Reviewed the live `main` worktree at HEAD `b8d8d43d8e3deb6fc772000adbb6f160cf81e863`; production decoder SHA256 is `b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8`.

There is one required correction, concerning performance measurement. I found no failing selected correctness artifact or demonstrated remaining implementation defect. Independent artifact analysis checked all 82 selected jobs, 200,526 recorded PCC values, 26 production source hashes, 20 watcher captures, and five raw profile CSV hashes. All checked PCC values are finite and at least 0.995; the minimum is 0.9957183599472046. This result does not by itself clear the measurement-contract finding below.

Paths without a prefix are relative to this report's evidence directory. `tt/` and `tests/` paths are relative to the model autoport directory.

## Required Work

- **P2 — R1: Final performance evidence must use separate diagnostics-off runs, with matched before/after settings.**
  
  **Evidence:** `tt-enable-tracing/SKILL.md`, Trace Allocation Safety Gate, explicitly requires: “Collect performance numbers separately with tracking and diagnostics disabled, recording the settings for each run.” The final timing artifacts do not satisfy that requirement. `final_v2_answer1024_32.json`, `final_v2_boundary_{1,32}_raw.json`, `final_v2_perf_{0,3}.json`, `final_v2_{large_boundaries,short_prefill_perf}_{0,3}.json`, `final_v2_stress_{0,3}.json`, and `final_v2_context128k_answer1024_32.json` report `tracking: "1"`. Their benchmark loops call the ordinary `ttnn.execute_trace`; they do not disable tracking for timing (`tests/multichip_packed_stack.py:372`, `tests/multichip_probe.py:265`). Several paired before rows use tracking `"0"`: `answer1024_stage4_32.json`, `packed_io_stage4_{1,32}.json`, and `baseline/perf_{0,3}.json`. The short/large prefill before rows and the 128K before row instead also use tracking `"1"`. `final_performance.csv` combines these as final optimized numbers. The current summarizer checks shapes and some workload fields but does not enforce diagnostics-off or identical diagnostic settings.
  
  **Why this matters:** These are required final latency, speedup, and FUNCTIONAL/roofline inputs. Instrumentation is in the measured replay API, even though this checkout's tracker has a cheap empty-unsafe-set path. Its impact cannot be assumed zero from that implementation. The performance evidence is therefore not in the required clean regime, and several comparisons use different settings. The passing tracked runs remain valid correctness evidence; this finding does not assert that the large optimization gains disappear.
  
  **Required next step:** Keep the existing tracking-enabled correctness gates. Run fresh, serialized timing jobs on the unchanged final default with `TT_METAL_TRACE_ALLOC_TRACKING=0`, trace-allocation diagnostics/tracebacks disabled, watcher off, and device profiler/Tracy off. Match batch, prefix, capacity, answer length, inputs/reference, I/O contract, timing method, and warmup between each before/after pair. Repeat the frozen-stage-4 timing rows that currently have tracking enabled, or use already-existing evidence only if it proves the same clean regime and workload. Reuse the already-clean 64K pair if its provenance remains unchanged. Update final performance/selection/FUNCTIONAL/roofline summaries and README numbers from these runs, and add a host-side check that rejects diagnostic timing rows or mismatched settings. Only affected timing work is required; no repeat of the entire 82-job correctness matrix or the valid profiler collections is requested. Have a fresh follow-up reviewer check this finding and the changed artifacts after those jobs finish.

## Other Concerns

- The repaired SDPA policy has an empirical accuracy margin, not a proof for every activation distribution. The original 128K failure is repaired under the unchanged full 1,024-step workload; K128 failed late, while K256 and K512 passed. The selected K256 minimum is only about 0.000718 above the acceptance threshold. Preserve the capacity-aware policy, its captured controls, and this limitation for later full-model accuracy work. The report correctly avoids claiming that FP32 destination registers make online statistics FP32 or that the 200-update bound mathematically guarantees PCC.
- The eight-hour stop rule is supported. I independently merged the completed hardware/recovery intervals through the frozen cutoff and obtained exactly 8.001707033316295 hours. The ledger explicitly includes setup, compilation, validation, teardown, recovery, and profiler host processing; it does not claim kernel execution hours. Remaining optimization candidates listed under `Not tried (stop rule)` are handoff notes, not blockers. In particular, this review does not reopen split attention projections, further fused-CCL staging, exhaustive geometry, or FP32 SDPA-statistics experiments.
- Two small prefill rows currently show roughly 1% and 0.7% regressions (`layer0_timing_511_prefill` and `layer3_timing_513_prefill`). These small differences do not establish a material regression from the present samples. Keep them visible in the corrected timing table rather than making an across-the-board prefill-improvement claim.
- Checkpoint commits are appropriately pending the review. The supplied run plan authorizes local stage-owned commits only and says never push. No push is required for this stage, and no push state was inferred from reflog history.

## Hard-Check Gaps

- `tests/audit_stage5_evidence.py` checks the selected source hashes, successful jobs, PCCs, watcher data, and profile provenance, but its pass does not certify clean performance settings. `tests/summarize_stage5_final.py` currently permits the diagnostic mismatch in R1. A performance-regime check belongs with the timing summary.
- The final manifest does not index the earlier short-prefill boundary evidence. This initially looked like missing coverage, but inspection found `fusion_default_grid_{0,3}.json` covering 64 and 127/128/129, `phase_boundaries_phase_{0,3}.json` covering 65 and awkward 95/96/97, and `prefill_trace_{0,3}.json` covering 127/128/129. AST comparison shows the selected `short_prefill_mlp`, its `finish` dispatch, and `reduce` unchanged from integrated source `da8f4bd47467…`; the candidate's short-MLP configs match the selected helper. Current final tests cover both layer kinds' short paths, the later 512/1024 precision/geometry boundaries, 4095/4096/4097, and the large staging/chunk boundaries. I accept the retained short-path evidence rather than require another sweep. Add a compact evidence mapping when maintaining the final manifest so future reviewers do not mistake these controls for absent checks.
- Top-level `doc/context_contract.json.evidence` still ends with the v1 max-context entries, while the stage-specific `optimized_multichip_decoder` section correctly binds the v2 artifacts and current hash. The current stage claim is supported; aligning the top-level index would reduce stale-evidence confusion.

## Anomaly Ledger

- **Observed anomaly:** Diagnostic timing rows are presented as final clean optimized measurements, with several before/after tracker settings differing.
  **Evidence:** R1 artifacts, benchmark loops, and `final_performance.csv`.
  **Affected path:** Required warmed prefill/traced decode timing, FUNCTIONAL and roofline summaries.
  **Control or comparison:** JSON settings compared across every unique before/after artifact pair; 64K is already tracking-off on both sides.
  **Likely subsystem:** Measurement configuration and summary acceptance.
  **Investigation performed:** Inspected command metadata, replay loops, tracker implementation, and summarizer checks.
  **Resolution:** more-work-needed — R1.

- **Observed anomaly:** Fixed-K64 SDPA fails the additional 128K/B32/1,024-step accuracy workload; K128 passes early but fails at step 1,013.
  **Evidence:** `context128k_autofix/{AUTODEBUG.md,AUTOFIX.md,controls.json}`, retained failure artifacts, and `final_v2_context128k_answer1024_32.json`.
  **Affected path:** Packed full-attention decode in the two-kind stack.
  **Control or comparison:** Same seed/capacity/page permutations and unchanged gates; HiFi4-only fails, K256/K512 pass, corrected frozen-stage-4 control passes.
  **Likely subsystem:** SDPA chunk policy and accumulated BF16 intermediate/statistic rounding; precise arithmetic attribution remains inferential.
  **Investigation performed:** Inspected selector, scalar/native callers, local-before-global page padding, TTNN core-allocation formula, original diagnosis and control matrix. Independently executed the extracted pure selector over 270,336 batch/capacity cases, deriving worker counts from the factory formula; all bounds and smallest-chunk checks passed. Rechecked all final recorded PCCs.
  **Resolution:** fixed for the unchanged required workload and validated paths; empirical extrapolation remains a residual risk.

- **Observed anomaly:** Legacy direct convolution comparator fails a per-head PCC gate at B13/wave104/step5.
  **Evidence:** `conv_gates_current_audit/anomaly_resolution.json` and the three retained `*.B13.wave104.step5.rank0.output_vs_ttnn.pt` captures for ordinary fusion, fast-zero fusion, and production substitution.
  **Affected path:** Direct synthetic convolution/recurrence comparator.
  **Control or comparison:** Preserved production convolution plus exact separate TTNN gates reproduces the legacy discrepancy. Final actual-production direct tests use the matched production arithmetic.
  **Likely subsystem:** Different convolution rounding in the legacy comparator, after the genuine gate-arithmetic issues were repaired.
  **Investigation performed:** Loaded the three small captures on CPU, verified their recorded SHA256 values, and independently confirmed that actual tensors are bitwise identical across all three runs and expected tensors are also bitwise identical across all three runs. Final direct source-bound evidence covers 2,048 records, all B1–32, waves104/7, with bitwise convolution/history/state/output controls.
  **Resolution:** controlled; no threshold relaxation found.

- **Observed anomaly:** Historical native attention request ordering, pooled-state padding, custom-kernel NoC alignment, and CB producer ownership failures.
  **Evidence:** Work-log investigation entries, retained direct/watcher/triage evidence, current `tt/multichip_decoder.py`, `tt/kernels/flat_ordered_conv_gates/`, and final batch/pool/direct watcher artifacts.
  **Affected path:** Independent request routing, masked page tails, recurrent helper reads and CB publication.
  **Control or comparison:** All logical batches1–32; ordered/reversed/subset/mixed pools and private-state controls; interleaved tracked traces; current direct recurrence controls at waves104/7.
  **Likely subsystem:** Core enumeration, local versus pooled page offsets, NoC alignment, and circular-buffer ownership.
  **Investigation performed:** Inspected rectangular Q-first core enumeration, padding local page zero before pool offsets, disjoint pooled regions and writes, aligned DRAM row-pair reads, head-loop assignment, and separate reader/compute CB ownership. Rechecked final watcher captures and current-source gates.
  **Resolution:** fixed by the current implementation and controlled by final evidence; no surviving contradiction found.

- **Observed anomaly:** Invalid phase-prefill RNG/capacity fixture, frozen-stage-4 BF16/BF8 workspace mismatch, and virtual-table width exceeding physical cache capacity.
  **Evidence:** `phase_prefill_fixture_audit.json`, the AutoFix control records, `final_v2_resume_jobs.json`, and corrected `final_v2_capacity_extra_*_watcher.json`.
  **Affected path:** Test comparisons and candidate controls.
  **Control or comparison:** Exact-list references, frozen workspace selected with the frozen decoder, and physically matching cache allocations replace invalid runs; failed artifacts remain retained.
  **Likely subsystem:** Harness construction and provenance.
  **Investigation performed:** Inspected harness class/workspace selection, capacity calculations, original/current command metadata, and sampled-state declarations.
  **Resolution:** controlled. Invalid fixtures are not used as production correctness passes or optimization rejections.

- **Observed anomaly:** Allocation-corruption warnings in untracked jobs and expected positive-control rejection messages in tracked contracts.
  **Evidence:** `final_v2_extended_context64k_32.log`, untracked watcher logs, and `final_v2_contracts_{0,3}.log`.
  **Affected path:** Trace output allocations and harness temporaries.
  **Control or comparison:** Corresponding tracked current-source workloads and pool interleaving checks pass with program-cache allocations included; each contract artifact records11 positive-control rejections.
  **Likely subsystem:** Generic allocator warning when tracking is off, plus deliberately unsafe positive-control allocations.
  **Investigation performed:** Inspected exact-output acknowledgments and surrounding lifetimes. The stack has one live trace/output, consumed before replay; pool traces overwrite their own outputs before consumption, with separate persistent inputs/state allocated before capture and helper programs prewarmed. No whole-forward or whole-capture exemption was found in the selected harnesses.
  **Resolution:** controlled; warnings are not unexplained evidence of surviving trace corruption.

- **Observed anomaly:** Profiler percentages above100, Cores=8 reader interpretation, CCL aggregation differences, and optional Tracy viewer-copy/pandas dtype warnings.
  **Evidence:** `final_advice.md`, `final_profile_summary.json`, final profile logs, source CSVs, `selected_runtime_rows.json`, and report CSVs.
  **Affected path:** Performance interpretation and profile export.
  **Control or comparison:** Actual runtime attributes and complete retained raw CSVs, with per-chip FW spans separated from merged report accounting.
  **Likely subsystem:** Generic performance modeling and optional export, not decoder arithmetic.
  **Investigation performed:** Verified all five raw CSV hashes; independently summed55 target rows to771.478μs kernels and46.060μs gaps. Runtime rows show ten BFP8-weight/BF16-input/output matmuls, four HiFi2 and six LoFi, with2 workers/bank for input/gate/up and4 for output/down. Ordinary-op selected timings match the report; CCL report means need not equal a selected rank. Required tables and raw evidence exist despite optional viewer-copy warnings.
  **Resolution:** controlled. No generic percentage is accepted as whole-path utilization or as the stop criterion.

## Scope Inspected

- **Goal/skill paths:** Original `goal_contract.txt`; installed `stage-review`, `optimize`, `tt-device-usage`, `tt-enable-tracing`, and model-bringup startup instructions; optimization method reference. Used the owner's eight-hour alternative and preserved all correctness/context gates.
- **Artifact paths:** README, work log, manifest, audit, accuracy/performance tables, selection, stop ledger, device-time accounting, topology decisions, final advice/profile summary, style evidence, context contract, all82 selected result JSONs and corresponding command/log/provenance records; five profile source CSVs; target report/runtime rows; AutoDebug/AutoFix controls; convolution discrepancy tensors; retained short-prefill evidence and source snapshots. Reviewed candidate decisions and representative adaptation evidence; did not re-audit every historical search result.
- **Code paths:** Production decoder initialization, collectives, state allocation/read/write, prefill dispatch, norms/MLP, scalar and native batch attention, SDPA chunk selection, pool routing and inter-layer residual contract; custom convolution, recurrence, history-write and partial-norm/RoPE wrappers and representative reader/compute/writer ownership; packed-stack, probe, pool-contract, recorded-HF and direct-kernel harnesses; final runner/auditor/summarizer/time-accounting code; TTNN trace allocator and relevant SDPA scheduling source.
- **Commands run:** Read-only `git status`, `git diff`, `git rev-parse`, `rg`, `sed`, `cat`, and small Python standard-library artifact analyses. CPU-only PyTorch loaded three109KB discrepancy captures. Independent analyses recomputed hashes/PCC finiteness and minima/watcher hashes, timing-setting pair comparisons, profile sums, the interval union, AST equivalence, and selector scheduling bounds. No TTNN import, device access, reset, server, vLLM job, implementation edit, or child reviewer was performed. The only review write is this report.
- **Independently confirmed contract points:** TP4 projection/head sharding on mesh1×4; shared serial CCL workspace; BF16 local L1 residual chaining without an inter-layer gather/reshard/all-reduce; final-default source binding; nonaligned public prefill handling; complete per-kind max-context execution at262139/262143/262144 and terminal decode262143; B1/B32 and long-answer evidence; current-source watcher and tracked replay controls. No generated model text is expected or required at this decoder-only stage.

## Residual Risk

- Inspection and finite test coverage do not prove all possible activations, page mappings, or long-context numerical cases. No new hardware confirmation was performed by the reviewer.
- B32 maximum-capacity transition tests run at short positions and explicitly sample state pages. They are not populated B32 maximum-context numerical tests. The all-layer B1 memory total is analytic, and no all-layer allocation or full-model execution is claimed. These limitations are accurately disclosed and do not narrow the tested B1 context contract.
- Trace preparation and serial workspace lifetimes remain integration obligations for the next stage. A full generator must prepare its supported signatures and resources before capture; this stage does not establish serving trace reuse, token feedback, sampling, prefix caching, or generation quality.
- The sole remaining stage-closure blocker from this review is R1. After corrected timing evidence and its summaries are complete, the next review should be a fresh diff-only follow-up, not another review of the entire stage.
