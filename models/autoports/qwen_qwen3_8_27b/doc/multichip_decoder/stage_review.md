# Stage Review

Verdict: clean-pass

Independent inspection of stage04, multichip-decoder, for `Qwen/Qwen3.8-27B`, completed on 2026-10-23. Reviewed the live `exp26-qwen38-bringup` worktree based on commit `7c620da56e3475f64e8656d70cde0a9b6464bb20`, before the required local checkpoint. Final decoder SHA256: `4b8b89ab05452f6c9194049ed6baa155265a3d1d1288d7043f8fe25c731eca90`.

## Required Work

- None. The findings raised during review were resolved before this verdict. The final source, validation, profiles and capability/memory records agree.

## Other Concerns

- No unresolved correctness or performance-selection finding. Source inspection supports the documented TP4 head ownership, local KV/recurrent state, distributed prefill normalization, packed projections, replicated decode interface and compatible layer-stack interfaces.
- The independent artifact audit verified all 43 required cases against the final decoder hash: batch 6, boundary 25, continuation 4, stack 2, context 2 and watcher 4. Every recorded output/state PCC is finite and exceeds 0.995. Minimum output PCC is **0.9999053588**; minimum reconstructed-state PCC is **0.9998227069**. Both layer kinds retain native context through 262,142-position prefill plus two decodes reaching context 262,144. The report correctly distinguishes this tested prefill length from the supported limit.
- Runtime audit and tracked replay evidence cover advancing hidden inputs, RoPE/current positions, randomized page tables, live page remapping, local state reconstruction, replica equality and three exact restored-state replays. Program-cache allocations remain included; there are no allocation exemptions. The capture preparation and input-copy ordering match the executed harness.
- The repeated packed/separate prefill comparisons support the selected packed representation. Removing unused separate weights saves 26,542,080 bytes per layer per device. Fused RS-only and compatible 1D AG+RS candidates were measured through sharded residual consumers on both layer kinds and were slower. Precision-locked decode and prefill geometry comparisons investigate the material SLOW rows. The apparent full-attention K20 improvement did not reproduce over 20 warmed replays: control/K20/combined medians are 10707.73/10725.69/10731.59 microseconds.

The final device results were independently recomputed from the complete measured-window operation CSVs, using each ASIC's own clock and then taking the maximum envelope. All four devices, all operations and internal gaps are included. Runtime projection rows show BF16 activations, BFP4 weights, LoFi, packed prefill width 8704, and the selected decode K blocks 4/3/2/8 with two readers per bank.

| Layer kind | Prefill device time | Decode device time | Prefill speedup | Decode speedup |
|---|---:|---:|---:|---:|
| linear_attention | 8326.151 us | 508.853 us | 1.9285x | 1.6022x |
| full_attention | 10628.844 us | 424.104 us | 1.6776x | 1.6601x |

Speedup, efficiency and roofline arithmetic match `performance.json`. The single-chip comparisons use the same 4096+128 layer workload and whole-layer device-time convention. Communication, movement and attention/recurrence costs are retained in the measurement. `profile_analysis.md` correctly separates actual CCL operations from local distributed-normalization kernels.

## Hard-Check Gaps

- No open gap required for this stage. The NaN weakness in a `min(scores)` gate was corrected to explicit finite-value checks; the final artifacts were also checked independently.
- Initial cached watcher runs only dumped blank startup state. Replacement runs use 100ms polling with `TT_METAL_WATCHER_NOINLINE=1`, retain all watcher features, and contain 53/59/53/55 completed scans with nonzero kernel mappings. Copied logs contain no failure indicators. The completed-scan count was corrected to avoid counting both start and completion lines.
- Final profiles and all required suite cases match the selected decoder hash. Historical candidates and pre-packed validation remain identifiable as comparison evidence. Compact report and measured-window CSVs are no longer excluded by the repository's general CSV ignore rule.

## Anomaly Ledger

- Observed anomaly: fused gather/matmul stalled.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, `triage_fused/`, final fused candidate JSONs.
  Affected path: experimental fused CCL decoder candidate.
  Control or comparison: unfused TP4 passed; adapted fused candidates pass PCC/state/replay checks but lose on latency.
  Likely subsystem: fused worker-index/core-coordinate ordering across a core row.
  Investigation performed: inspected the diagnosis and source contracts; verified the one-worker-per-link adaptation and coherent RS/AG+RS measurements.
  Resolution: fixed in the candidate; candidate rejected by measured latency. The selected path uses unfused collectives.

- Observed anomaly: full-to-linear single-chip stack capture attempted new compilation.
  Evidence: `stack_l3_first_failure.log`, `triage_stack_l3/`, `stack_l3_lifetime.json`, final stack manifest.
  Affected path: comparison harness warmup/capture lifetime.
  Control or comparison: retaining versus releasing the consumed eager output.
  Likely subsystem: L1 availability influencing a program-cache signature.
  Investigation performed: compared baseline output placement and harness ordering; checked the corrected repeated-replay artifacts for both stack orders.
  Resolution: fixed by releasing the eager output before capture; no exemption or baseline source change.

- Observed anomaly: watcher fabric binary exceeded ACTIVE_ETH program storage.
  Evidence: `watcher_l0_first_failure.log`, final watcher console logs and copied watcher dumps.
  Affected path: instrumented fabric startup, before model construction.
  Control or comparison: documented no-inlining option with all watcher features retained.
  Likely subsystem: instrumented kernel binary size.
  Investigation performed: checked `rtoptions.cpp` and `risc_attribs.h`; verified successful model execution and post-startup scans.
  Resolution: fixed by the documented no-inlining configuration and adequate polling. No watcher check was disabled.

- Observed anomaly: profiler presentation suggests UINT8 reduction output or an eight-core whole-chip denominator.
  Evidence: measured operation CSVs, reduce-scatter output specifications, runtime matmul attributes and device headers.
  Affected path: interpretation of profiler output.
  Control or comparison: UINT8 internal staging versus BF16 final activation; reader-worker estimate versus physical chip peak.
  Likely subsystem: profiler field selection and heuristics.
  Investigation performed: inspected the source output contracts and rederived whole-layer timing/rooflines independently.
  Resolution: controlled. Final arithmetic uses BF16 collective semantics and all four ASICs' stated physical peaks.

- Observed anomaly: a three-reader short-run sample reached 1029 us; a five-sample prefill K20 result initially appeared faster.
  Evidence: geometry results, 128-step reader controls, `prefill_selection.json` and its 20-replay controls.
  Affected path: optimization selection.
  Control or comparison: repeated measurements under the same precision and compatible topology.
  Likely subsystem: measurement variability; no model-output failure observed.
  Investigation performed: checked complete distributions and independently recomputed the prefill medians.
  Resolution: controlled; the anomalies are not used to justify an unsupported performance claim.

- Earlier cache-path, local-output-width, norm-rank and late-copy-compilation failures are recorded in `anomaly_ledger.md` and `work_log.md`. Their corrections are present in source and exercised by the final current-source suites. Resolution: fixed.

## Scope Inspected

- Goal/skill paths: supplied stage04 contract; installed `stage-review`, `multichip`, `agentic-serving`, `tt-device-usage`, `tt-enable-tracing` and relevant `optimize`/startup requirements under `.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/skills/`; repository multi-device and trace-correctness guides.
- Artifact paths: this directory's README, mesh plan, work log, anomaly ledger, suite manifests and per-case JSON/logs, watcher dumps, geometry/packed/fused comparisons, final profile CSV/text/JSON files, memory plan, performance and profile audit; adjacent `context_contract.json`, `agentic_serving.json`; optimized-stage baseline performance; supplied stage04 telemetry packet.
- Code paths: `tt/multichip_decoder.py`, inherited `tt/optimized_decoder.py`, recurrence wrapper/kernel references, multichip test/profile/report helpers, common CCL helpers and focused watcher/reduce-scatter implementation references.
- Commands run: read-only `rg`, `sed`, `cat`, Git status/diff checks and small Python scripts for AST parsing, source hashes, finite PCC/count checks, watcher scan inspection, memory/statistical arithmetic, CSV envelope reconstruction and runtime-policy verification. No hardware, reset, server or model execution was performed by this reviewer. The only file written was this report.

## Residual Risk

- This is a decoder-stage pass. Fixtures use real weights and supplied recorded hidden activations, including repeated activations for long-context checks. Full-model generation, serving quality, cross-request multi-trace ownership and full-stack allocation peaks remain later-stage responsibilities.
- Native context is validated at batch 1; batches 8/16/32 are separately validated at 4096+128. This does not claim every batch/context combination fits.
- The 18.47 GB per-device plan for 32x32768 is a feasibility ledger with estimated activation/trace reserves, not a measured full-model pool. Decode traffic rooflines are explicitly lower-bound estimates rather than hardware bus counters. These limitations are recorded consistently and do not reduce the advertised single-request context.
