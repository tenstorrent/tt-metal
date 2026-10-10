# Compact GDN front-end simulator launch

CPU-only bounded service on the allocated host, launched October 10, 2026 UTC.
Unit `qwen38-gdn-frontend-sim-v1-20261010.service`, invocation
`eec01e6cb8374e278d0eca85641f38cd`, PID 3632342 at launch.
One virtual Blackhole chip; no physical accelerator access or hardware timing.
Source and simulator hashes, exact environment and resource limits are retained.
Runtime limit 60 minutes, memory limit 16 GiB, CPU quota two cores. Disconnect
persistent; not reboot persistent. No serving/model precision policy changes.

The probe compares compact convolution and preparation with native convolution
plus the previously qualified direct-preparation path. It requires exact Q/K/V,
normalized Q/K, FP32 value/gate buffers and independently computed history
chronology. Alternating allocations and changed inputs exercise address rebinding.
B16 public/compact and B32 compact cases cover DRAM/L1 and upper/lower tile faces.
Synthetic taps/activations do not establish model accuracy. Slow-dispatch
simulation does not qualify hardware trace replay, alignment or performance;
the persistent physical TP4 experiment remains required.

This directory records launch, not completion. Host results are under
`/home/ttuser/qwen38-artifacts-20261007/gdn-frontend-sim-v1`.

## Terminal outcome

All four attempts exited with code 1 and no completed comparison. The pinned
simulator rejects `SETDVALID` with implied source format as unsupported and
terminates below Python's exception handling. Raw `probe.json` files therefore
remain `running`; the separately observed failed services are authoritative.
No cleanup success, numerical match or physical performance pass is claimed.

Attempt 2 added phase markers; attempt 3 built the reference layout on the host.
Those markers alone did not locate execution because queued device work can run
later. Attempt 4 added explicit synchronization: compact convolution returned
through its fence, and failure occurred in the following compact-preparation
phase. This narrows the simulator failure but does not establish that outputs
were correct, or that the same behavior fails on hardware. Earlier attribution
to native reference layout conversion was premature and is not retained as a
root-cause claim. The simulator and native runtime were not modified.

Exact probe sources, source manifests, logs and terminal observations are retained
per attempt. The physical TP4 comparison remains queued behind the original
fusion/GPQA and B16 prefill experiments; no live hardware workload was interrupted.
