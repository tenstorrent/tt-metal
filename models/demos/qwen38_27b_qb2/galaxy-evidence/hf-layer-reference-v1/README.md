# Conditional HF layer diagnostic, Oct 9 2026 UTC

This experiment has passed CPU preflight and is **queued, not hardware-tested**.
It is intended to locate numerical differences if the BFP8/HiFi2-head full GPQA
control still misses 177/198. It does not supply a benchmark score or change the
accuracy gate.

The frozen CPU reference runs first on original BF16 HF weights and saves eight
teacher-forced steps on the public G0 prompt, including all decoder inputs,
final normalized hidden state and logits. The diagnostic then runs one physical
TP4 replica at B1 with the queued head-control policy and those same input tokens.
It captures each of the 64 layer inputs on all four ranks, then final norm and
full logits. It also injects the HF normalized hidden state into the actual device
head, separating head-only differences from accumulated decoder differences.

Transformers 5.12.1's `capture_outputs` replaces its final hidden-state entry with
the final normalized state. The comparison explicitly respects that boundary;
it does not compare the raw last decoder output to the normalized reference.
PCC, cosine, relative RMS, rank agreement, top-1, top-20 overlap and logits KL are
diagnostics. BF16 HF decoder weights differ from the device's BFP4 weights, so a
layer difference does not by itself establish a kernel bug. B1 eager execution
does not qualify traced serving, high concurrency or long context.

## Persistent execution

- Host: `ttuser@10.228.203.98`.
- Unit: `qwen38-hf-layer-v1-20261009.service`.
- Verified live PID 1255121, invocation `aa7a2d34fdd74d538ba2725b99d43311`.
- Source: `/home/ttuser/qwen38-artifacts-20261007/hf-layer-source-v1`.
- Results: `/home/ttuser/qwen38-artifacts-20261007/hf-layer-v1`.
- Source manifest SHA-256:
  `a362be1e846e8cb66a219f00bdc1d68637c5880ff131e2a7c3d89ecfe4c5acfc`.

The controller waits for the exact post-head CPU service invocation to stop and
its HF reference to complete. It skips the hardware diagnostic if the complete
head-control GPQA meets 177/198. Otherwise it verifies the frozen source, actual
head G0 source/precision/group, reference tensor hash and token continuity before
hardware use. It then runs through the existing cooperative device-lock runner.
The unit has a 22-hour total bound, 20-hour dependency wait, 50-minute diagnostic
bound, 160-GiB memory cap and eight-CPU quota. It survives client disconnection;
automatic resume after host reboot is not configured. Other timing work must
wait for this service too if it reaches the hardware stage.

## Validation and staging issues

Sixteen CPU tests passed locally and on the allocated host. The hardware entrypoint
was intentionally skipped during preflight. Native Torch/TTNN/model imports also
passed without opening a device. The first remote preflight failed because the
real `expect_error` fixture requires an error-message pattern while the initial
local lightweight fixture did not. Both were corrected; failed logs/JUnit remain
beside the successful results. No failed hardware attempt was hidden or retried.

The first SSH staging command was denied by the local sandbox. A later approved
retry validated the six partially copied new files before completing the new
source directory; no existing job's source was modified.

See [launch](launch.json), [source manifest](source-manifest.json),
[preflight](preflight.log), [CPU tests](unit.xml), and the
[queue snapshot](queue-snapshot.json). The live unit and receipts supersede this
point-in-time snapshot.
