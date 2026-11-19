# Anomaly ledger

## Cold B32 nonfinite prefill

- Observed anomaly: first cold GDN B32 prefill contained NaNs; instrumentation or a warm retry masked it.
- Evidence: `correctness_v1_sharded_l0_b32_s33`, `watcher_v1_sharded_l0_b32_s33`, `debug_b32/`, `AUTOFIX.md`.
- Affected path: default CCL workers may extend beyond the initial8x8 semaphore allocation.
- Control or comparison: full11x10 semaphore ownership; `watcher_cold_b32_v2` uses a fresh cache,0/476 JIT hits.
- Likely subsystem: uninitialized global semaphore storage on otherwise valid CCL worker cores.
- Investigation performed: independent source diagnosis, owner-authorized reset, cold watcher and repeated correctness/trace tests.
- Resolution: fixed by model-local full-grid CCL manager; no shared-source changes.

## Fused Linear all-gather repeated-launch stall

- Observed anomaly: first replay passed, subsequent queued replays stalled.
- Evidence: `triage_agminimal/`, `triage_agminimal_async/`, `AUTOFIX.md` and topology JSONs.
- Affected path: optional native fused AG-matmul candidate.
- Control or comparison: synchronized replay, persistent scratch, dedicated semaphores, then in-trace ordinary Linear AG admission fence.
- Likely subsystem: inter-launch admission/reset ordering in the fused Linear path.
- Investigation performed: live TT-triage before termination/reset, independent AutoTriage, repaired API/layout and queued-replay controls.
- Resolution: controlled; fenced candidate passes50 queued replays, but final packed-SwiGLU measurement is slower (0.753429 vs0.496683ms). Production uses ordinary AG plus native SwiGLU. This is not a shared-kernel root-cause fix claim.

## BFP4 setup and cache accuracy

- Observed anomaly: native tile padding rejects BFP4; after setup adaptation full-attention BFP4 cache PCC is below0.995.
- Evidence: `bfp4_v1*`, `bfp4_v2_attention4_lofi_l3_32x33.log`, `reset_bfp4_accuracy.log`.
- Affected path: optional attention-weight BFP4 candidate, not the BFP8 default.
- Control or comparison: exact BF16-representable unpack/pad/repack during setup; real recorded activations, unchanged BFP8 cache and baseline.
- Likely subsystem: padding op contract, then quantization error in projected K/V.
- Investigation performed: repaired setup, reran candidate, inspected output versus cache PCC. Cache0.993778/0.994654 while output0.999888.
- Resolution: padding fixed; attention BFP4 rejected on real cache accuracy. Passing GDN/MLP BFP4 candidates are recorded for stage8 precision selection.

## Synthetic repeated-layer stack drift

- Observed anomaly: artificial chain[0,3,0] with random activations passes prefill but decode PCC0.993135 fails0.995 under attention LoFi.
- Evidence: `final_stack_l0_b2_s65_tp4.log`, `stack_synthetic_control_{baseline,hifi4}`, `stack_real_{baseline,lofi}_b{2,32}`.
- Affected path: precision accumulation in that artificial stack/input distribution.
- Control or comparison: matched HiFi4 chain passes decode0.999925. Consecutive checkpoint layers[0,1,2,3,4] with recorded model activations pass selected LoFi at B2/S65 and B32/S33, both decode>0.99995.
- Likely subsystem: cumulative low-fidelity rounding, rather than a layout/CCL ownership fault.
- Investigation performed: same-input higher-fidelity control and actual consecutive-layer/next-token fixtures; identical layout and trace interfaces.
- Resolution: controlled. Keep the faster precision policy on real model evidence, per optimize policy; retain the synthetic limitation. This is not an end-to-end model-quality verdict.

## Harness/setup errors and source-freeze stop

- Observed anomaly: initial physical-ID/exchange-core setup arguments, rank3 distributed-norm input, and later a baseline `from_torch` positional-layout argument rejected by APIs. Formatting also triggered a suite source-freeze assertion between completed jobs.
- Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, `final_context_smoke_v2*`, `preformat_correctness*`, `format_provenance.json`.
- Affected path: setup adapters, baseline test upload, host orchestration.
- Control or comparison: corrected overload/rank adapters,630 host partition cases and14 norm shapes, complete reruns after repairs.
- Likely subsystem: Python/native interface contracts; deliberate provenance guard.
- Investigation performed: source inspection and smallest failing fixtures. Formatting changes are recorded exactly; production import ordering is the only difference between resource/topology evidence and final source.
- Resolution: fixed; final correctness/performance/profile/stress/context/stack evidence uses the frozen formatted source.

## Native prefill MMRS watcher (fixed on selected target)

Two B32/S33 watcher aborts identify a dedicated-NoC ownership mismatch.
`debug_narrow_watcher/AUTODEBUG.md` gives source/compiled-argument evidence
and the precise uncertainty about unrecorded hardware counters. Numerical
passes cannot close this gate. Separate matmul/RS and 128-row fused pieces
both pass the focused watcher test; the latter records NoC0 plans. Candidate
comparison and final-source validation are in progress. Historical
`final_perf*` and `final_profile*` describe the numerically correct but
watcher-rejected unchunked default; they are not final acceptance evidence.

Two associated harness errors are distinct: the private native plan getter
required its lower-level module namespace; a global K16 prefill setting was
invalid for down-projection K136 tiles and was replaced with K8 there. Both
exited cleanly and retained failure logs; neither was classified as a
numerical rejection or a device hang.

Resolution: fixed by model-local shape dispatch and128-row native pieces.
All six final_safe watcher decoder cases and both64-owner pressure/trace
cases pass on source63d2e53a. See AUTOFIX.md for the exact artifacts and
remaining native API limitation. Historical unsafe-source results remain
explicitly historical; final accepted-path measurements use final_safe.

## B31/S31 random-activation PCC (controlled precision sensitivity)

Observed anomaly: GDN decode user25 PCC0.994696745, below0.995, although
prefill and state scores pass and watcher reports no device assertion.
Evidence: `pcc31_default_synthetic.json`, `pcc31_initial_sharded.log`.
Affected path: LoFi attention under this seeded synthetic activation.
Control: separate MMRS preserves all reported decode/state PCCs; changing
attention to baseline HiFi4 raises minimum user decode PCC to0.999210842.
Default LoFi with recorded model activations at the same shape passes both
layer kinds, all users, state/cache and exact trace checks.
Likely subsystem: attention precision sensitivity; exact dominant operation
is not isolated. Investigation: `pcc31_control_summary.json` and independent
`debug_pcc31/AUTODEBUG.md`; the native96-row-tail hypothesis is contradicted
by the separate-path control and the state/output dependency structure.
Resolution: controlled. No threshold change or production precision fallback.
The real-input tail gate uses the same0.995 bar; the failing synthetic control
stays explicit. This does not prove full-model reasoning accuracy, which is
not implemented or claimed in this decoder-only stage.

## Transient profiler orphan marker

- Observed anomaly: GDN B32/S33 profiler aborts on an unmatched `pp_intra` END
  after numerical/trace success; device CSV is not emitted.
- Evidence: `debug_profiler_marker/failed.log`, `failed.command.json`,
  `AUTODEBUG.md`, preserved raw host capture under its `tracy/` directory.
- Affected path: legacy device-profiler timestamp collection and final pairing.
- Control or comparison: identical-source fresh-process
  `delivery_profile_retry_l0_b32` completes normal pairing and produces intact
  four-rank CSV; independent profiler-off latency and watcher runs pass.
- Likely subsystem: legacy Blackhole live-high-before-low clock sample; the
  observed low word10 fits rollover, but missing raw START makes attribution
  inferential.
- Investigation performed: fresh source-only AutoDebug, parent code checks,
  rejection of MID_RUN_DUMP because it skips pairing, retained failed capture,
  fresh collection with unchanged source/settings.
- Resolution: controlled collection failure. Shared clock-sampler repair remains
  a separate infrastructure task; no failed timing is used as accepted evidence.

## Profile shell wrapper EOF error

- Observed anomaly: parent profile script exits127 (`f: command not found`)
  after its sixth completed job/report.
- Evidence: `delivery_profiles_retry.console.log`, six wait_for exit0 records.
- Affected path: shell orchestration only, after the loop.
- Control or comparison: all six profiles pass raw hashes/ranks/precision/
  marker-integrity checks; host accounting/renderer/aggregate rerun exits0.
- Likely subsystem: live wrapper edit changed Bash's file-reading offset.
- Investigation performed: final script has no `f` command and passes `bash -n`;
  checked every job completion and artifact, with devices idle.
- Resolution: fixed orchestration source; do not edit running wrappers. No
  numerical or timing artifact was replaced by a failed wrapper's exit status.
