# Optional prefill-budget capacity failure and restored queue

Observed Oct 9, 2026. The original `qwen38-gpqa-first-v2-20261009.service`
completed corrected GPQA, the 32K-budget sweep, the 128K/near256K sweep and the
16K-budget sweep. At 20:59 UTC its optional **64K prefill-budget** sweep stopped
at 32K/B32 on a DRAM allocator error. The benchmark recorded
`state=allocation_failed`, an OOM cell and `cleanup_completed=true`.

The allocation requested 570,425,344 bytes over eight banks, or 71,303,168 per
bank. Each bank had 235,596,288 free bytes but its largest contiguous block was
63,394,688 bytes. This is an allocator/capacity limit, not evidence of a dispatch
hang. The remaining cells are explicitly `not_run`; they are not passes.

The old container and optimized-GDN followers correctly stopped before touching
hardware when their predecessor failed. All three original failed receipts and
units remain preserved. The recovery audit independently verified the exact
stopped invocation, ordinary exit code 1, completed preceding stages, the narrow
DRAM-allocation signature, cleanup and availability of the shared lock. It does
not claim hardware health. The following container job performs an authorized
reset under the lock before opening devices.

Recovery validation: **35 tests passed**. The future queue controller now records
this optional clean 64K-budget allocator limit without preventing later work.
Dispatch timeouts, incomplete cleanup, other stages and different exit codes
still stop the queue. No original benchmark, score or failed outcome was rewritten.

## Restored persistent work

- `qwen38-image-hardware-v3-20261009.service`: PID 2628711, invocation
  `93f5759d1e4049508202587de2adf31b`. Actual pinned-image startup, API/tool smoke
  and full OpenBench. Reset succeeded; all eight workers were loading model
  layers at the latest observation. Hardware/API/eval qualification is pending.
- `qwen38-bfp8-gdn-v2-20261009.service`: PID 2628723, invocation
  `c3f1a7cad8e64f04bf68976b09a505fc`. Waiting for successful container cleanup;
  retains the exact tested `bfp8-gdn-source-v1` snapshot and CPU preflight.
  Native/shared/native B16/B32 sweeps, four stage profiles and G0/full GPQA follow.
- Complete launch commands and audit are retained below this directory.
  Both jobs survive session disconnect, not host reboot. Original time/resource
  bounds remain. No firmware, NFS or shared installation changed.

## Prefill-budget observations

At 32K input, B16, reducing the budget from 32K to 16K lowered aggregate prefill
from 5,321 to 4,726 input tok/s. At B32 it fell from 4,692 to 3,714. Steady decode
was effectively unchanged. The 64K budget improved the completed B16 point to
5,870 input tok/s but OOMed at B32. These are fixed offered-batch measurements,
not a mixed-request fairness/latency study. Retain 32K as the working budget.
