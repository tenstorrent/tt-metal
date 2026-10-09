# Persistent physical GDN epilogue experiment

Queued at approximately **22:07 UTC Oct 9, 2026**, behind the existing BFP8
comparison and qualification job on **10.228.203.98**. This folder records
launch/preflight evidence, not hardware results. At capture all three services
were active; only the container qualification was using hardware. Its workers
had reached the final model-loading layers.

## Queue order

1. `qwen38-image-hardware-v3-20261009.service`: pinned BFP8 image, API/streaming/
   multiturn/tool checks and full 198-question OpenBench run with explicit
   client timeout/retry settings.
2. `qwen38-bfp8-gdn-v2-20261009.service`: native/shared-QK/native BFP8 comparisons
   at 16K/32K and B16/B32 (12 cells), four stage profiles, then fresh Galaxy G0
   and full 198-question GPQA for the optimized recurrence.
3. `qwen38-gdn-epilogue-hardware-v1-20261009.service`: physical TP4 arithmetic,
   trace and placement checks for the standalone fused GDN epilogue.

The epilogue follower requires the exact predecessor invocation
`c3f1a7cad8e64f04bf68976b09a505fc`, terminal process state, successful unit exit
and a completed queue receipt with device cleanup. Observation failures do not
release hardware. It then obtains the common `/tmp/tt-device.lock` through the
safe test runner. It never changes serving defaults or promotes a candidate.

## Coverage and acceptance

- B32/B16 first, then B8/B1; DRAM followed by interleaved L1. Eight cases total.
- FP32 recurrence-output input, BF16 gate/weight/output, preserved BF16
  intermediate before the final z multiplication. No precision reduction.
- All four chips compared against actual TTNN layout + gated RMSNorm + multiply.
  Per-head PCC at least 0.99999 and relative RMS at most 0.001; bit identity
  reported separately. Zero/near-zero inputs and extreme gates are included.
- Two live independent allocations, 0/1/0 cache rebinding, zero output padding,
  input preservation, and trace replay after copying new values into unchanged
  buffer addresses. Pre-z normalized output is checked as well.
- Native/fused/native latency brackets. Each uses five warm samples of 100
  trace replays. More than 3% native drift withholds qualified speedup.
- A 48-layer latency saving is explicitly a linear projection; it does not
  establish a full-model TSU result or qualify the operator for model integration.

CPU preflight: **450 passed, one skipped, 40 subtests passed**; the physical test
collected successfully. Pre-commit checks passed. This source snapshot inherits
the frozen BFP8 follow-up source and overlays the epilogue and its test files;
its manifest records all runtime/test hashes. This is a different frozen test
set from the earlier 456-test follow-up preflight.

The standalone kernel previously passed the simulator screen. Physical
instruction execution, this placement sweep and later real-weight/model
integration still require their own passing evidence.

## Persistence and inspection

The new systemd user service has invocation
`16c50a2f3e4e428fbefe4c87c662cacb`, 32-GiB host RAM/8-CPU limits, a 32-hour
predecessor-wait limit and a 34-hour overall limit. Its hardware subprocess has
a 90-minute bound and process-group cleanup; individual pytest timeout is
60 minutes. These are failure bounds, not expected runtimes. The services
survive SSH/client disconnects but do not resume after a host reboot.

Remote root: `/home/ttuser/qwen38-artifacts-20261007`.

```bash
systemctl --user status qwen38-gdn-epilogue-hardware-v1-20261009.service
cat /home/ttuser/qwen38-artifacts-20261007/gdn-epilogue-hardware-v1/queue.json
tail -n 50 /home/ttuser/qwen38-artifacts-20261007/gdn-epilogue-hardware-control-v1/run.log
```

The full launch command is in `gdn-epilogue-hardware-control-v1/launch.json`.
Results will be `gdn-epilogue-hardware-v1/epilogue.json` and `hardware.xml` on the
host. Keep failed attempts and launch a new versioned directory for retries.
To cancel only this follower, stop its exact systemd service; do not reset
hardware while a preceding job is using it.
