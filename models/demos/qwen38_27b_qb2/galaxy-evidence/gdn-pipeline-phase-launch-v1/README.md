# Current GDN pipeline phase diagnostic

Snapshot: 2026-10-11T03:08:20.775013+00:00. Persistent service is active and waiting
for the exact combined-padding qualification follower. This is launch/CPU
evidence, not a successful physical phase capture or throughput result.

The diagnostic covers resident FP32 recurrence, packed-L1 BF16 gates and
compact-output epilogue at B16 and B32, comparing zero/skip padding and
uninstrumented/instrumented kernels. Eight cases run three updates each:
24 pipeline invocations, 48 targeted recurrence/epilogue calls. All four
physical ranks use distinct inputs. Passing requires dense recurrence checks,
finite epilogue output, unchanged input/state/output hashes between variants,
zero public output padding, complete phase markers and clean teardown.
The independent epilogue numerical qualification remains the earlier
[padding experiment](../gdn-epilogue-padding-results-v3/README.md).

Diagnostic annotations preserve kernel source tokens and ordering; line-anchored
matching avoids matching the inlined helper's acquire operation. No production
kernel source, precision, serving defaults or active source snapshot changed.
The new phase probe passed753 CPU tests, one skipped, and104 subtests; one
hardware test collected. Nine focused local tests passed; pre-commit passed.

- Unit: `qwen38-gdn-pipeline-phase-v1-20261011.service`.
- Invocation: `592686881e474eddb69f6282a8849ec7`; launch PID: 807101.
- Predecessor: combined-padding followup, invocation
  `3b91a375acef4a09b5cc4a9835bcc9ff`.
- Waits for the exact predecessor to terminate cleanly, then checks immutable
  source hashes and obtains `/tmp/tt-device.lock` through the existing wrapper.
- Bounded to1200 seconds inside pytest;5400 seconds includes lock/capture/export.
  Artifact budget4GiB total/1GiB per file; minimum free disk16GiB.
- User systemd service survives client disconnect; it does not resume on reboot.
  Host-local128GiB/8CPU caps; no NFS/native installation/firmware changes.
- Hardware counters are disabled for this phase capture. A counter multipass
  and calibrated streaming control remain outstanding. Phase lifetimes alone
  do not prove physical DRAM utilization or an additive critical path.

`capture.json` retains raw file hashes and snapshot timestamps. Logs and JUnit
are gzip-compressed without modifying their bytes. Launch commands, frozen
source manifest and predecessor snapshots are included. Image/video work is
separate; this launch contains no multimodal implementation or AgentX run.
