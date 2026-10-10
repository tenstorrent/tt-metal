# Prefill numerical isolation, October 10

This is a queued diagnostic, not a prefill optimization or model qualification.
The unchanged numerical gate remains PCC >= 0.999 and relative RMS <= 0.02.

The earlier batching experiment stopped in its existing serial before-arm at
B3, prefix 32, chunk 65: PCC 0.999736-0.999795, relative RMS 2.25-2.40%.
All four ranks agreed and the expected quantized KV cache hashes matched.
The batched candidate had not run for that case. Its failure does not establish
a batched-cache defect. See the preserved
[original result](../projection-l1-results-v1/README.md).

## Controlled experiment

- Reproduce the original seeded case and three nearby controls: B3/32/64,
  B3/0/65 and B2/0/128. Retain shuffled physical pages and inactive/future data.
- Compare native, accurate exponentiation, FP32 accumulation, both together,
  HiFi4 with both, and a repeated native control. Only the serial boundary runs.
  No model policy or serving default changes.
- Use downloaded BFP8 KV and BF16 Q for a selected-row FP64 causal reference,
  independently checked against Torch SDPA. Record all users and all four ranks,
  per-row errors, immutable input/cache hashes and native before/after identity.
- The report may complete with a failing native numerical result. Completion
  explicitly does not qualify a model or claim a performance gain. Wall times
  include compilation and are not benchmark evidence.

The pinned native Metal revision is
`a08819ddbe23077f8037d3802303939064868ff6`.
The retained native SDPA sources show that the omitted compute configuration
selects HiFi2, approximate math, and no FP32 destination accumulation. The
program defaults to approximate exponentiation; FP32 accumulation also changes
streaming-kernel selection. Approximation/intermediate precision is a plausible
cause, not a demonstrated diagnosis. These public configuration controls avoid
editing the native installation.

## Validation and persistence

Frozen preflight: **722 passed, one skipped, 104 subtests passed** in 4.38 s.
The physical test collected successfully. All **288 model-source hashes** match
the local publication. Native source copies and launch/preflight receipts are
retained here; no private eval responses are included.

Unit: `qwen38-prefill-numerics-v1-20261010.service`, invocation
`7d953cfc6f7f46f98bc757a57330341c`, PID 519518 at capture.
It waits for clean completion of the exact epilogue-padding-v2 invocation
`2cb03803508147cca833c466023b1a52`, preserving combined model/GPQA priority.
At 22:38 UTC it was active, waiting, with `hardware_started=false`.

- Source: `/home/ttuser/qwen38-artifacts-20261007/prefill-numerics-source-v1`
- Output: `/home/ttuser/qwen38-artifacts-20261007/prefill-numerics-v1`
- Host: `10.228.203.98`; shared device lock; controller bound 28 h, 32 GiB,
  eight CPUs; hardware bound 1800 s.
- Estimated hardware time: 3-10 minutes once dependencies finish. Disconnect-
  persistent, not reboot-persistent. Source mismatch or predecessor failure
  stops the diagnostic. No reset was manually requested for this launch.

The combined GDN candidate completed its first two arms while this diagnostic
was prepared: at 32K, 19.996 -> 20.319 TSU; at 16K, 22.722 -> 23.142 TSU.
Both arms used B16/TP4, three repeats and identical generated-token hashes.
The after-control was still running at capture, so these are preliminary
measurements, not an accepted gain or new GPQA qualification. The current
qualified reference remains 20.035 TSU at 32K and GPQA 177/198.

The separate prefix/offload branch remains at
`fab35e060a581789c95072fd6120e8edfbc6dcb6`, with 21 CPU tests passing and no
TT capture/restore or serving integration. Its remote branch was verified
while collecting this run; the diagnostic does not implement prefix reuse.
