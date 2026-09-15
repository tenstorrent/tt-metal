# Native GDN runtime evidence review

Verdict: **clean-pass for the completed, isolated B8 `_delta` component experiment only**. The recorded build/install, numerical comparisons, timing and observed ownership support `verified_summary.json`. They do not establish full64/FP8 model equivalence, a serving gain, a CSV target pass, or Stage11 readiness. Production model code still does not opt into the new default-false native flag.

This read-only review follows the prior source/build-readiness review (`REVIEW_source.md`, SHA470d1df96b1bd42298027eee57d3fd2f409de326f77870e43efd0050aefa7cc9). It checks completed evidence rather than repeating that source diagnosis.

## Build and native runtime binding

- The required `.github/scripts/copilot-build.sh` was actually attempted and exited1 at2026-09-15T16:35:04 UTC: Docker was unavailable. This was not a successful Docker build and is retained as a failure.
- The bounded fallback ran the three frozen generated unity compiler commands followed by both generated linker commands, sequentially from16:44:16.829939 to16:44:39.533698 UTC. All five actual child exits were0. Their argv arrays exactly match the source-pinned plan, allowing only removal of the verified linker shell no-ops. I checked the command/program/plan bindings, each step's real PID/time/exit/log hash, and the five current output hashes against the execution record.
- Actual `cmake --install build --component ttnn-runtime` and `--component tt_pybinds` both exited0 at16:46:35. Their retained installation logs record the expected RPATH changes. Consequently the installed-library hashes differ from the pre-install link outputs; the two stages are not conflated.
- All seven installed candidate native source files and their archived baseline/candidate copies match the source ledger. The loaded Python `_ttnn.so` hash is `8f5b58e74dd5cdbaeb2355ea216b9c9db6316cac40e47e9cc4a9ad97d8f2258c`; the actually mapped `_ttnncpp.so` hash is `d9d8d553dd3889f0c3e1f31b6b384a2bacd485974b11ef1dc5b36519917839eb`. Both completed child reports record exactly those installed paths/hashes. The probe checks `ttnn._ttnn.__file__` and `/proc/self/maps` inside the actual child; this reviewer did not import TTNN or read live process mappings.
- The host build alone did not prove device-JIT compilation. The subsequent real native calls, explicit candidate calls and traced executions completed successfully with those installed sources/libraries.

**Ninja metadata incident:** root recorded that system Ninja1.11 rejected and removed the newer top-level `build/.ninja_log` during a query. The historical command hashes/durations were unavailable and were not reconstructed. The fallback used the retained dependency database, generated graph and audited object identities; it cannot retroactively prove the historical compiler flags of every retained unrelated object. Its post-build receipt states that unrelated output identities and protected files remained unchanged. This is a scoped rebuild, not a claim of a fresh full-tree build or repaired Ninja history.

## Numerical and lifetime evidence

The actual scope is base checkpoint revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, layer0, `head_bfp4_lofi` policy, TT mesh1x4, logical B8/t1, context262144; B4 is a fallback control. It is not the final FP8/all-BFP8 serving policy. The actual compute grid is11x10, yielding one B8 native partition.

Both the tracked correctness process and the uninstrumented timing process ran the same eleven native controls: finite, poisoned finite padding, second beta, second g, NaN/Inf K padding, NaN active beta, Inf active g, NaN Q padding, NaN V padding and NaN state. For each, the flagged and default native paths had identical four-rank output and recurrent-state readback byte hashes; inputs were checked unchanged. These are hashes of complete logical host tensors returned by TTNN, including all32 native output rows, not a claim to have read raw packed DRAM allocations. Nonfinite controls compare readback bytes without a finite-only shortcut.

The unchanged actual `layer._delta` then ran three evolving B8 steps and one B4 step on independent persistent state allocations. Complete projected BF16 outputs, BF16 convolution history and FP32 recurrent history matched exactly on all four ranks. The recorded native call metadata confirms the intended flag on the three B8 candidate calls and the unchanged B4 default path. Both processes' entire native-control and evolving-comparison records also match each other.

The tracked process used the public allocation tracker and wrapped public `execute_trace`; no Watcher or profiler was enabled. Each arm's three traced states/outputs exactly matched its own eager trajectory. I additionally compared all three stored cross-arm trajectories and the final records: they are equal. Persistent state addresses remain stable/nonaliasing and frozen input addresses/hashes are unchanged. A single trace is live at a time; each current output is copied to host before the opposing trace is captured. This avoids reading a later allocation after an older trace overwrites scratch.

The prior source-review limitations remain explicit: these controls do not directly inject negative-zero or subnormal g tails, or toggle eligible/fallback data within one captured trace. They do not invalidate the completed stated finite/nonfinite and repeated-input controls, and are not silently reported as covered.

## Timing recomputation

The timer surrounds64 nonblocking replays of the complete `_delta` trace plus final device synchronization. State reset, warm execution, capture and host readback are outside the timed interval. Each arm resets from the same immutable template and has its own state allocation. All six pairs alternate AB/BA order; all twelve final output/state records pass exact paired equality. This includes actual convolution/recurrent transitions and output projection, not just the identity-inverse region.

| Quantity | Independently recomputed result |
| --- | ---: |
| Default inverse median |646.580421875µs/call|
| Guarded candidate median |637.575437500µs/call|
| Median of six paired savings |**9.057429687500019µs/call**|
| Range of paired savings |8.878640625 to9.376953125µs/call|

All six paired savings are positive. Their exact values are9.160421875,9.144,9.376953125,8.878640625,8.970859375 and8.8885µs. The difference of the two independent medians is9.004984375µs; the summary correctly labels9.05743µs as the **paired median**, rather than confusing these statistics. This is one process's six paired component measurements, not a cross-run stability estimate or per-layer/full-model extrapolation.

Timing instrumentation is absent in both the execution receipt and actual child report, including allocation tracking; the child additionally checks the effective public tracker state. The actual timing wrapper hash was checked separately because its external validator does not itself assert that timing-wrapper field.

## Real exits and sampled ownership

Correctness child1104423 exited0 at16:48:58.112795 UTC; timing child1105020 exited0 at16:50:07.413299 UTC. Their logs end with the successful pending-cleanup status, without tracebacks. Both report successful mesh close and no cleanup errors. Duplicate self-PID entries after internal mesh close are recorded honestly as `own_pid_only_pending_process_exit`; the external four-device owner receipts are empty after each actual process exit.

The independently pinned passive owner observer (`3533695b4d0053fbe29c182e604ffd30ba65a339dbee1999534493ef323d82d2`) records151 correctness and109 timing samples at100ms cadence, no errors, and maximum gaps0.10159117s and0.10127194s respectively. Each recorded nonempty transition contains only that run's verified positive native child PID, including legitimate duplicate FD entries; no zero or foreign PID is present. Observer intervals enclose the actual child intervals and bind the exact execution-record hashes. Sampling cannot exclude an event shorter than its sampling interval; no continuous-ownership claim is made.

## Verification and evidence pins

I ran the existing host-only `verified(timing.json, 'timing')`, which recursively validates the actual correctness receipt, under an audit hook refusing target imports, network/process actions and live `/proc`/device reads. It returned `PASS_REAL_EXIT_AND_CLEANUP`. Separate standard-library checks rehashed all11 summary evidence references, actual build/install/native-library bindings, both wrapper hashes and passive observer source, recomputed all paired timings and compared numerical records. No model, device, serving, build or install command was rerun. No production file was edited; this report is the only review-authored artifact.

- `verified_summary.json`: `71e7a6853e38570c89a7e7d8780c5ed423820a77de6401f552b6110cd8dfbea4`
- `build_receipt.json`: `8d6c0be555a04c9702f0203183b1bd27051f53347023e808b8419572ecf28580`
- `correctness.json`: `5cfe239b3c9c94e121c093a5bced83c0693b2e148f28b92e0a455636e6e1ca65`
- `correctness_execution.json`: `fa778dfbbe3af435bfd0a1e9c318e96a16bc5c7703fe8873936237653608444d`
- `timing.json`: `cfbeb2bcf2d4b99098155a7d5bc5a380ce411695a5a837c0e62c4ed3374c42c0`
- `timing_execution.json`: `ecebadc69fbf28f2b6c212361d0d62134a93170eab15a982c276a8e1e9f88e7c`
- Probe: `efd46a7953f921452dc212f58cfc46a54e7fc7c9ee5d0b9edf01f441c618e25c`
- External validator: `27dff09e23b2c023424df43f41d3cd4b8ece4c27fddf89375e83b972c5d96b11`

No blocking contradiction was found in this bounded runtime result. The seven native files are applied, but a source search of the production Qwen autoport finds no `padded_single_token_inverse` opt-in. An explicit model caller guard, final-source numerical controls and actual serving measurements remain separate work before claiming or activating a serving benefit.
