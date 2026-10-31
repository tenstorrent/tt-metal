# Stage Review

Verdict: clean-pass

Independent inspection of stage05, optimized-multichip-decoder, for
`google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Reviewed the live worktree on
`mvasiljevic/gemma4-31b-fresh-pipeline-v0811`, based on
`bdbdc9e17c3f5dbe0ac12ac1e44bc408c81f8da0`. The reviewed production file is
`tt/multichip_decoder.py`, SHA256
`7c21e0d36423a4305bd8b07cf8a805909b3145168ecba3703500cdabb9304837`.
The review used read-only inspection and artifact analysis; it ran no device
commands, tests, servers or resets and changed no implementation files.

## Required Work

None. The implementation, measured optimization dispositions and final
acceptance evidence satisfy this stage's contract. Local checkpoint creation
and recording its SHA follow this review, as required by the stage workflow.

## Other Concerns

No additional defect found. The changes preserve the accepted precision and
residual contract, remove avoidable movement, and make the shared L1 workspace
ownership explicit. The prefill norm uses private converted storage before
in-place execution; its shard planner covers occupied rows at awkward tails.
The pooled gather backings are consumed by non-inplace normalization before
reuse on the serialized stream. Stack outputs require no inter-layer collective
or layout conversion.

The optimization evidence covers topology and residual carry, fused CCL/matmul
with adapted packing and dtype, projection packing, precision-locked geometry,
activation/CCL precision, persistent buffers, readers, norm sharding and all
applicable final report advice. Prior stage04 controls were checked as measured
controls, including carried-sharded consumers, rather than accepted from prose.
The additional input-L1 advice found during review is now resolved by runnable
adaptations, exact capacity evidence and both streamed families across B8/B16/B32.

## Hard-Check Gaps

None identified within the decoder-stage contract.

- Independently counted 54 final correctness cases. Minimum prefill PCC is
  0.99989195, decode PCC 0.99991480, cache PCC 0.99984076 and advancing-stress PCC
  0.99986447, all above 0.995. Cases cover both layer kinds, B1/B8/B16/B32,
  padding/norm boundaries, cache wrap and logical lengths 262143/262144.
  Maximum-context runs execute the full prefill and compare its last 32 rows
  plus decode; that comparison scope is disclosed.
- Checked fallback guards, unused-cache-page checks, 128 advancing stress
  replays, worker-watcher logs and two-layer composition. Final-source
  `shared_workspace.json` has matching source hashes, four live traces,
  32 alternating replays and all-device bitwise equality. Allocation tracking
  includes program-cache allocations; only exact captured output tensors are
  acknowledged, with immediate consumption after their own replay.
- Checked eight complete final profiles, runtime BFP4/LoFi rows and 128-replay
  capture/body agreement. Independently recomputed a complete raw device
  timeline and all eight reported roofline percentages. The denominator is
  the complete layer window, not a matmul sum. Final B1 prefill/decode device
  times are 11150.01/475.22 us for sliding and 13488.43/527.82 us for full
  attention, versus 11422.54/484.77 and 13748.47/534.77 us respectively.
  All measured batch rows improve; host candidate timings remain separate.
- Checked final memory/context contracts and the exact required telemetry
  packet against its template and referenced B1 performance artifacts. Its
  stage, attempt, workload and numerical fields agree with the evidence.

## Anomaly Ledger

- Observed anomaly: prefill norm API/configuration errors and seven-column L1
  scratch overlap.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, norm candidate logs and final boundary
  artifacts. Affected path: sharded prefill normalization. Control: accepted
  interleaved norm and adapted eight-column norm. Investigation: legal FP32
  subblock, in-place retry, generic scratch accounting and occupied-row
  planning. Likely subsystem: norm program configuration/L1 allocation.
  Resolution: fixed; final default and affected boundary cases pass.
- Observed anomaly: output-sharded fused AG-MM prefill PCC 0.984939 with BFP8
  fused input.
  Evidence: output-AG candidate artifacts and work log. Affected path: fused
  output projections. Control: BF16 fused input restores PCC above 0.9999;
  smaller K/N and carried-sharded residual variants also run. Investigation:
  padded local weights, input dtype and downstream norm/residual consumers.
  Likely subsystem: fused-op numerical contract. Resolution: controlled;
  correct adapted families are slower and are not selected.
- Observed anomaly: distinct per-signature L1 collective outputs exhaust
  available space.
  Evidence: CCL candidate logs, `memory_plan.json`, `shared_workspace.json`.
  Affected path: persistent decode gathers. Control: two shared role backings
  with independent RS staging. Investigation: allocation inventory, alias
  identity, downstream nonaliasing and alternating live-trace replay.
  Likely subsystem: L1 workspace lifetime/capacity. Resolution: fixed by the
  documented shared-pool contract, verified on final source.
- Observed anomaly: direct DRAM slicing improves sliding but regresses full
  decode. Evidence: direct-row controls and `candidate_timings.md`. Affected
  path: row-parallel O/down movement. Control: L1 slice plus pooled L1 gather.
  Investigation: repeated full-layer comparisons and both layer kinds.
  Likely subsystem: layout/data movement. Resolution: controlled; the faster
  cumulative family is the reproduced final default.
- Observed anomaly: larger-batch profiles recommend L1 matmul inputs while
  the default uses DRAM.
  Evidence: advice tables, `input_l1_trials.md`, `input_l1_failures.json` and
  `input_l1_capacity.json`. Affected path: batched prefill projections.
  Control: role-specific inputs, shared gate/up input and two complete streamed
  families. Investigation: smaller M/K/N blocks, actual allocator/CB overlap,
  independent tile-rounded capacity calculation and 1024/2048-row adaptations
  for both kinds at B8/B16/B32. Likely subsystem: L1 capacity and movement cost.
  Resolution: controlled; correct adapted paths are slower. The warmed
  128-MiB O-input failure is correctly classified as insufficient contiguous
  space, not insufficient total free bytes.
- Observed anomaly: an early shared-workspace result did not have a retained
  source bridge to the final file. Evidence: old source hashes and replacement
  `shared_workspace.json`. Affected path: provenance of pooled trace lifetime.
  Control/investigation: rerun the focused workspace test on the final source;
  both recorded hashes now match. Likely subsystem: evidence provenance.
  Resolution: fixed; no reliance on the earlier prose bridge remains.
- Observed anomaly: the first final full-attention B8 profile has an unmatched
  device marker. Evidence: `profiler_anomaly.json`, preserved failed capture
  and successful retry. Affected path: profiler postprocessing. Control:
  identical bounded retry supplies complete 128-replay windows. Investigation:
  completed workload, device ownership/visibility and preserved failure history.
  Likely subsystem: profiler marker collection. Resolution: controlled;
  `acceptance_completion.json` reconciles the original nonzero queue exit.
- Observed anomaly: the first eager prefill-trace timing sample is an outlier.
  Evidence: both `prefill_trace_layer*.json`. Affected path: host timing summary.
  Control/investigation: retained raw samples and steady-sample medians versus
  eight traced replays, with effectively unit PCC on all devices. Likely
  subsystem: host/warm timing. Resolution: controlled; final analysis reports
  the modest sliding benefit and absent full-attention gain without changing
  the comparable headline regime.
- Observed anomaly: Ethernet watcher instrumentation exceeds its configuration
  buffer by 2640 bytes. Evidence: stage04 `watcher_configuration.json` and
  final worker-watcher logs. Affected path: instrumented Ethernet firmware.
  Control/investigation: documented capacity failure and the skill-prescribed
  worker-only watcher run, separate from profiling. Likely subsystem:
  instrumentation capacity. Resolution: controlled; no Ethernet watcher
  coverage is claimed, and numerical/trace tests exercise real collectives.

## Scope Inspected

- Goal/skills: expanded stage05 prompt, `bringup/run_contract.json`,
  `bringup/agentic_evals/inputs.json`, stage-review, optimize, agentic-serving,
  tt-device-usage, tracing lifetime requirements, and the relevant CCL/trace
  review guidance.
- Artifacts: final README/work log, acceptance/stack/context/watcher results,
  source manifests and formatting bridge, candidate controls, AutoDebug/AutoFix,
  raw profile CSVs and advice tables, reader and prefill-trace controls,
  input-L1 adaptations, memory/context/agentic-serving contracts, stage04
  controls and telemetry packet
  `a40cd227-5628-4d84-b89b-edd36b6c8bb0.json` under the required continuation.
- Code: production `tt/multichip_decoder.py` and inherited norm/projection
  consumers; validation, stress, stack, shared-workspace, candidate, profiler,
  reader, prefill-trace and summarization harnesses; relevant generic norm and
  minimal-matmul allocation code. No removed Gemma source was consulted.
- Commands: read-only `git diff/status`, `rg`, file reads, and standard-library
  Python analyses of hashes, ASTs, JSON, CSV windows, PCC minima and memory
  arithmetic. Only this review report was written.

## Residual Risk

- This validates representative sliding/full TP4 decoder layers and their
  composition. It does not claim full-model generation, serving quality or a
  measured 60-layer memory fit; those belong to later stages.
- Later stacks must pass one shared gather dictionary per serialized mesh
  stream and prepare intended signatures before trace capture. Concurrent
  unsynchronized reuse is outside the documented ownership contract.
- Rooflines use stated theoretical peaks and transfer estimates, not hardware
  traffic counters. Small measured speedups should retain that measurement
  scope. Ethernet watcher coverage remains limited as documented above.
