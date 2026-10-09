# Persistent BFP8 GDN performance and accuracy queue

Launched Oct 9, 2026 on `10.228.203.98`. The live user service is
`qwen38-bfp8-gdn-v1-20261009.service`, PID 2479417, invocation
`ebc4580116294d7f803953362980a502`. The captured state is **waiting**, with no
hardware opened by this queue. This launch is not a performance or eval pass.

The user target is 20 output tok/s/user at B16/B32 per TP4. At 32K, the
current native BFP8 sweep measures 11.749/7.354 TSU, or 85.11/135.99 ms per
step, against the 50-ms target. Historical optimized BFP4 measurements cannot
establish the optimized BFP8 speed or accuracy. This queue closes that gap.

## Ordered work

1. Wait for the exact current container/API/OpenBench job to finish successfully
   and remove its owned container. That job already follows the original
   GPQA/performance queue. All device tests acquire `/tmp/tt-device.lock`.
2. Measure **native, shared-Q/K, native again**, each at 32K and 16K,
   B16/B32 per TP4. Each cell uses one warmup and three fresh-prefill measured
   repeats, 128 output tokens. Weight/KV BFP8, recurrent state FP32,
   HiFi2 projections, accurate attention and prefill budget 32,768 stay fixed.
   Separate prefill input TPS, decode output TPS and TTFT are retained.
3. Publish both candidate/control comparisons and a native control-drift report.
   More than 3% control drift or changed control output disqualifies a stable
   speedup claim. Native/candidate output hashes are recorded; equality alone
   is not an accuracy qualification, and different greedy tokens do not replace
   evaluation with an automatic failure.
4. Capture four bounded two-layer profiles: 32K B16/B32, native/shared-QK.
   These use real weights and synthetic caches. They do not claim full traced
   critical-path attribution; all ranks, cleanup and timing columns are required.
5. Run fresh eight-replica G0 and the full corrected 198-question GPQA for the
   shared-Q/K BFP8 policy, with 65,536 output budget, T1/p.95/k20/seed42.
   Its original 177/198 gate remains recorded; the user's acceptance of the
   current native 176/198 baseline is separate. No automatic deployment.

The reused accuracy-only controller calls this final arm `native-control` in
its directory names. Here it receives the explicit
`precision_single_step_shared_qk_bfp8_all.json` policy; the precision and worker
receipts, rather than that historical directory label, identify the candidate.

## Persistence, bounds and receipts

- CPU preflight: **456 passed, one skipped, 40 subtests passed** using the
  actual host Metal environment and repository fixtures; no device tests in
  this preflight. The complete JUnit and log are included.
- Task-owned immutable source snapshot:
  `/home/ttuser/qwen38-artifacts-20261007/bfp8-gdn-source-v1`.
- Results: `/home/ttuser/qwen38-artifacts-20261007/bfp8-gdn-v1`.
- Control/launch manifest: sibling `bfp8-gdn-control-v1`.
- Survives terminal/session disconnection. Automatic reboot resume is not
  configured. Estimated 3-5 hours of work once the predecessor finishes;
  per-stage bounds are deliberately larger than this estimate.
- Wait bound 18 hours, each sweep 1 hour, each profile 20 minutes, G0/evaluation
  stage 6 hours; outer service bound 32 hours, 256-GiB host RAM and 16 CPUs.
  Each stage enforces artifact size/free-space limits and process-group cleanup.
- Failed predecessor or changed invocation prevents hardware access. Stop only
  this service to cancel the follower; it does not stop either predecessor:

```bash
systemctl --user status qwen38-bfp8-gdn-v1-20261009.service
systemctl --user stop qwen38-bfp8-gdn-v1-20261009.service
```

`launch.json` records the complete executable command and `manifest.json` pins
the frozen source. The captured preceding native sweeps are included for
comparison, not relabeled as results of this new queue. At capture the original
queue had completed the 128K/near256K sweep and advanced to the 16K prefill-budget
experiment. Near256K/B4 measured **14.454 TSU**, with clean device shutdown.
