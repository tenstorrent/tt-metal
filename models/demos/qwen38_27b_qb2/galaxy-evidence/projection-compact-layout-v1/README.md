# Projection tuning with the actual producer layouts

October 10, 2026 UTC. The sweep is queued and has no new physical timing yet.

The older projection sweep uploaded GDN output operands as public
`[B,1,1536]` DRAM tensors. Compact GDN instead produces interleaved L1
`[1,1,B,1536]` and retains sharded projection output. At B16 the former
physically pads to 16 token tiles, while the latter uses one. The old test was
valid for its declared boundary, but cannot rank complete compact boundaries.

The revised sweep measures three separate boundaries at B16 and B32:

1. Compact L1 GDN output projection, matching the new epilogue.
2. Compact L1 MLP down projection, matching the selected MLP path.
3. Expanded DRAM GDN output projection, retaining the old control coverage.

Each boundary has ten configurations and a repeated control: 66 total
measurements and 54 candidate comparisons. Input shape, padded shape, memory
placement and output retention are explicit and checked at upload. Raw report
validation rejects missing boundaries, changed geometry, mixed-layout timing
brackets and inconsistent recomputed comparisons. Each bracket includes the
projection, required layout conversions and TP reduction. Input uploads,
weight repacking and readbacks are outside timing.

Existing gates remain: exact BFP8 values after reader-padding changes,
rank-distinct inputs, local dense GEMM and summed-output references,
changing-input A/B/A trace checks, five timing samples, and <=3% control
drift. A full-model saving is an extrapolation using 48 GDN output or 64 MLP
down calls. Alternative public/compact savings must not be added together.
The existing 1.5-3.5 ms full-step tuning target is unchanged and unmeasured.
Expected physical sweep time is 15-45 minutes after predecessors, with a
6000-second hard bound, 64 GiB host-memory cap and 8-CPU quota.

Frozen CPU preflight: **572 tests and 73 subtests passed**, one unrelated skip.
The physical test collected successfully. All model/config source hashes
match compact-gdn-source-v3; only the projection harness/controller changed.
The inherited published-versus-frozen reader formatting difference remains
documented in the [4K validation record](../compact-long-horizon-v2/README.md).

The requeue audit verifies the exact old projection-v5, prefill-v2 and
long-horizon-v2 invocations were waiting with hardware_started=false before
they were stopped. Their receipts and post-stop status are preserved.
The active compact run and its qualification/profile follower were untouched.
The new persistent chain is:

- Compact-v3, then conditional qualification/profile-v3, unchanged.
- Projection-v6: PID 268165, invocation `960d702be457402a92d32d88b4a4a591`.
- Prefill-v3: PID 268168, invocation `d7501d6753824d31846a13ecef289c32`;
  its source remains prefill-attention-source-v1.
- Long-horizon-v3: PID 268172, invocation `6f0ee1599ae144d6a6843b49bf22455b`;
  its source remains compact-long-horizon-source-v2.

Each successor requires its exact predecessor's clean terminal receipt and
takes the shared hardware lock. Jobs survive disconnect, not reboot. No native
installation, firmware, NFS, precision or serving configuration changed.

The retained before-control finished both contexts with clean teardown:
32K/B16 **16.583725 TSU / 60.300082 ms**, 16K/B16 **18.405926 TSU / 54.330329 ms**.
Compact then started its full-model arm. These are baseline measurements;
there is no completed compact full-model comparison or 30-TSU claim here.
