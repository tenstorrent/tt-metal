# Separate-workspace CCL initialization repair

The tiled-history prefill failure also occurs with a second ordinary RM-history
decoder, so layout was not the root cause. First normalization/projections match;
the first reduction diverges. Sharing a workspace happens to pass. Per-request
sync and output cloning do not repair the failure. Split RS/AG downloads alter
behavior, but explicit RS-only synchronization does not repair it reliably.

Cold isolated collectives expose the initialization issue without a decoder:
first B in BAB fails at RS rank0 (PCC0.901058) or AG rank3 (PCC0.935514), both with
and without program caching. Fixed-input later calls can conceal stale output,
so acceptance uses changing inputs in the same input allocation.

Independent source diagnosis: inherited `models/demos/gpt_oss/tt/ccl.py`
initializes global semaphores on8x8. Default two-link RS/AG workers on the target
11x10 grid include `(8,0)` and `(10,0)`, outside that initialized allocation.
Initialization is blocking; adding host synchronization cannot initialize cores
outside the supplied range. Cached buffer/semaphore callbacks refresh their
addresses, so no unsupported stale-address patch was made.

Repair: model-local CCLManager overrides its setup grid with
`mesh_device.compute_with_storage_grid_size()` before inherited semaphore
allocation. It does not change collective geometry, add a warmup dependency,
reuse a global singleton, or modify unrelated GPT-OSS source.

Hardware confirmation: all eight cold-BAB RS/AG combinations pass with changing
inputs, cache enabled/disabled and fixed/automatic buffers. The original
separate-workspace decoder now matches every observed operation (minimum
PCC0.99999988), and separate-workspace direct-history prefill passes all
32/33/1024/1025 lengths. Commands/snapshots and sidecar grid provenance are in
`ccl_grid_fix_jobs.json` and its `grid_fixed_*` artifacts. Ping-pong/bank1 and
watcher stress are recorded separately as they complete.

Production SHA after the minimal grid repair:
`8d1f4c57229f29e23703059b21355e4da9336a77c4df7f019981d56bad907063`.
The serial-stack shared-workspace contract remains necessary for bounded L1
residency. Independent workspaces now have initialized semaphore coverage.
Final-default performance/context/stress gates remain pending; this report is
not stage completion.
