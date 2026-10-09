# Performance-first queue, Oct 9 2026

The user requested faster progress toward B16/B32 throughput. Packaging and
OpenBench qualification now run **after** performance work. The prior jobs were
deliberately interrupted, not reclassified as test failures or passing evals.
Their receipts remain intact. Only the task-owned container was removed; its
compiled/weight cache is retained for the requeued container.

Current order on `10.228.203.98`:

1. Four BFP8 two-layer profiles, **32K/B32 native and shared-QK first**, then B16.
2. Physical fused-epilogue correctness, trace replay and DRAM/L1 timing.
3. Full-model native/shared-QK/native sweeps, B32 before B16, at 32K then 16K.
4. Fresh Galaxy G0 and full 198-question GPQA for optimized BFP8.
5. Pinned native BFP8 container, API/tool checks and full OpenBench qualification.

The new controller passed **457 CPU tests and 40 subtests**, one skipped. The
first hardware profile acquired the common lock, reset the dirty device and
completed its test/Tracy report; the shared-QK B32 profile started next.
These are eager stage diagnostics with real weights and synthetic caches,
not traced full-model latency or model-quality results.

## Persistent services

- `qwen38-perf-priority-v1-20261009.service`, invocation
  `4a07251dcbda49239e3b017f7fe8ef47`: profiles, epilogue, sweeps, G0/GPQA.
- `qwen38-image-hardware-v4-20261009.service`, invocation
  `d8abae607661465980454d2a58a81ca4`: waits for the preceding completed queue
  and clean device release, then performs container qualification.

The reorder itself ran as persistent
`qwen38-perf-reorder-v1-20261009.service`, invocation
`73778f407fd14344862be169fe5bef93`. Its completed release audit proves exact
owned invocations stopped, no container with the recorded ID/label remains,
and the global lock was available. It explicitly does **not** certify hardware
health or eval success. The next safe runner performs reset under the lock.

Normal container-followup checks remain unchanged. The new
`--after-clean-release` mode accepts only the explicit performance-priority
audit; it cannot turn that audit into a container/API/eval pass.

## Paths and preserved state

Remote root `/home/ttuser/qwen38-artifacts-20261007`:

- `perf-priority-source-v1`: frozen runtime and test snapshot.
- `perf-priority-control-v1`: source manifest, preflight, launch and release audit.
- `perf-priority-v1/queue.json`: current stage, per-stage deadlines and results.
- `image-hardware-control-v4`: requeued container controller/launch manifest.
- `image-hardware-v4`: new container receipts, distinct from interrupted v3.

The container keeps the original pinned image/source/checkpoint. The one-off
controller's cache path points explicitly to the existing
`image-hardware-v3/cache`, with `mkdir(exist_ok=True)`; those two changes preserve
the completed compilation work. Its exact controller bytes are archived here.
Earlier failed/canceled directories are not overwritten. Transient services
survive SSH/client disconnects but do not resume after host reboot.

The direct-input preparation prototype is **not** in this hardware snapshot.
Its simulator limitation is recorded separately; no unqualified arithmetic or
precision change was inserted into the BFP8 comparison.
