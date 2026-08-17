# Functional-decoder work log

## 2026-08-17

- Resolved `Qwen/Qwen3.8-27B` from the local HF cache at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; inspected its Qwen3.5 hybrid text configuration and canonical
  layer-0/layer-3 checkpoint keys.
- Generated `real_weight_stats.json` for all 25 canonical tensors consumed by
  representative layer 0 and layer 3. The artifact includes name, shape, dtype,
  float32 mean/population std, source shard, checkpoint revision, snapshot path,
  and index SHA-256 (`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`).
- Added the target-local `FunctionalDecoder` wrapper and state-dict adapter for
  the 48 DeltaNet and 16 full-attention layer kinds.
- AutoFix investigation found that persistent DeltaNet recurrent/conv state
  returned in L1 could overlap the following RMSNorm circular buffer. Only the
  persistent state was moved to DRAM; intermediate outputs remain in their
  kernel-selected memory configurations. Reports are under `autofix/`.
- AutoFix isolated a batch-shape defect in the reused experimental paged decode
  boundary. The target-local path now uses explicit `[B,H,1,D] <-> [1,B,H,D]`
  device permutes and passes batch-two disjoint/permuted page-table coverage.
- Corrected the DeltaNet test oracle to carry causal convolution history. This
  was an oracle issue, not a TTNN change; the investigation is recorded in
  `autofix/AUTOFIX_GDN_TRACE_PCC.md`.
- Real-weight HF-vs-TTNN prefill/decode PCC passed for both layer kinds. Full
  attention: 0.9972974494 prefill and 0.9976662041 traced decode. DeltaNet:
  0.9977131745 prefill and 0.9993015776 traced decode.
- Full advertised-context layer harness passed at 262,144 tokens, including a
  decode at position 262,143. No context reduction was needed.
- The original full test command passed 24/24 in 86.57 seconds. After stage-review
  remediation, the expanded suite passed 28/28 in 84.83 seconds; see
  `logs/full_suite.log`.
- Watcher-only command passed the two real-weight tests in 64.11 seconds. The
  watcher log audit was clean. Shutdown printed known nanobind binding leak
  diagnostics after clean device close; no device-runtime fault was present.
- AutoFix refuted stage ownership of that shutdown diagnostic with no-device
  controls. Bare TTNN and direct module imports exit cleanly; an unrelated
  pre-existing CPU-only gated-attention pytest emits the identical two-instance,
  20-type, 250-function nanobind warning and exits 0. See
  `autofix/AUTOFIX_NANOBIND_TEARDOWN.md` and its raw `nanobind_*control.log`
  artifacts.
- AutoFix replaced the overflowing combined profiler process with four bounded,
  individually selected capture processes. All passed without a buffer-full or
  dropped-event warning. `tt-perf-report` device totals are full 2.584/1.528 ms
  and DeltaNet 5.086/1.563 ms for prefill/traced-decode. Each source ops CSV,
  capture console, filtered CSV, and human table is under `perf/captures/`.
- Post-remediation watcher rerun passed 4/4 in 163.45 seconds, covering both
  real-weight B=1 paths, B=2 page routing/current positions, and the non-aligned
  262,143-token context harness. Its watcher log audit is clean; see
  `watcher_rereview/`.

Local stage commit SHAs are appended after review and commit.
