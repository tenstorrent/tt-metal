# AutoFix — trace preparation and cleanup

Diagnosis: AUTODEBUG.md. Contract: capture/replay must use precompiled programs with compatible allocation lifetimes; a failed capture must not leave capture active during mesh destruction.

Observations: eager recorded-input PCC passed; retaining its sharded L1 output made capture fail with a cache miss. Releasing only that output before capture restored deterministic tracked replay. The likely cache-key mechanism is the L1-sensitive history untilize path; the exact failed hash was not logged and is not claimed proven.

Repairs in tests/run_optimized_decoder.py:

1. `del warm` after the eager host comparison, before restore/capture. No runtime model precision or layout fallback; no tracker exemptions.
2. Track the live trace ID and whether capture is active. On exceptional exit, end and release the partial trace without replay, then close the mesh. Also release a completed trace if later validation fails.

Validation:

- released_warm_l0.json and record_control_l0/l3.json: PCC, changed inputs, deterministic replay, allocation tracking including program cache pass. Source versions are recorded in those JSONs.
- `optimized_run.sh check cleanup_injected --layer 0 --seq 32 --recorded --inject-capture-failure` with the sharded policy: intentionally exits 1 with INJECTED_CAPTURE_FAILURE, prints TRACE_CLEANUP_OK, and completes normal device close in cleanup_injected.log. This is a successful negative test, not a model correctness failure.
- Final candidate replay/Watcher validation remains required after ongoing optimization changes.

The cleanup guard is best effort if end/release itself fails; it does not promise recovery from every driver fault. This repair succeeded; stage work continues.

## Prefill norm AutoFix

Independent diagnosis: `AUTODEBUG_prefill_norm.md` (fresh official launcher completed). The shared norm factory accepts nonrectangular width-shard grids, but secondary reduction workers traverse the bounding rectangle and include holes. B1 decode uses only one reducer and avoids that failure. This is a concrete source defect; earlier NaN PCC alone did not locate the first nonfinite tensor.

Decoder-local fixes, within requested scope:

- Use a complete rectangular8x8 BLOCK_SHARDED layout for prefill norms when up to512 logical token rows fit the selected L1 contract. Larger token batches retain the measured interleaved norm path. This preserves capability and avoids requesting an invalid peer set.
- Any experimental width-sharded prefill layout now constructs a complete rectangular factorization of the requested core count, rather than a prefix of the11-wide grid. No upstream shared kernel was edited.
- `probe_prefill_norm.py` isolates original-snapshot versus corrected-grid norms on the same real512-token input, checks finite values and PCC, and records counts. `composite_normblock_retry_l{0,3}.json` validates the adapted complete layer, including tracked trace and deterministic prefill.
- Final watcher, awkward-tail, batch and full-context validation remains the signoff gate for these changes.

The first block-layout retry hit a Python `ttnn.Shape` slicing API error before execution. Converting the shape to a list fixed the caller; the adapted8x8 layout then passed and improved prefill. Explicit SDPA block trials were repeated with the inherited approximate-exp policy after source inspection showed that default; exact-exp and approximate-exp timings are separate evidence, not mixed rankings.

The final norm-only probe resolves the NaN-PCC ambiguity: original nonrectangular outputs were finite but reached3.0971e38 and PCC0.419/0.575 (40/80cores); FP32 statistics overflowed. Corrected rectangular grids reached max50.25 and PCC0.99999842. The FP64 probe statistics are in `prefill_norm_probe.json`. The original failure is a model-visible numerical explosion from an invalid peer set, not evidence against BF16/FP32 precision. Complete-layer block-sharded controls and watcher boundary runs validate the selected workaround.

Additional boundary repairs are recorded in `work_log.md`: internal norm row padding bounds L1, compact MLP decode batch rows prevent L1 overflow, and request fixtures now allocate room for their decode token. The former512-token fixture incorrectly allocated only512 cache positions, so its decode at512 was out of range. The corrected harness computes capacity from prompt length plus the decode token and reruns both kinds; this is not a model capability change.
