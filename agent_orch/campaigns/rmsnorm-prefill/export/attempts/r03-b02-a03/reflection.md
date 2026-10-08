# r03-b02-a03 result: 1.3000 (ok)

## What happened vs expected
The run is valid. PCC is 0.9999985 and max_abs is 0.0204-0.0240, the parent's values. So the alternate command
buffer wrote the right bytes from the right source to the right pages. Restoring cmd buf 1's return coordinate after
the drain also worked: the next call's stick reads and semaphore atomics on that buffer behaved normally (no hang, and
go -> W_DRAIN start is 0.59 µs, the same as the parent's 0.60).

It is slower on every shape. Parent r03-b02-a02 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.59 | 12.85 | +2.1% |
| h4096 | 14.05 | 14.38 | +2.3% |
| h6144 | 17.55 | 18.20 | +3.7% |
| h7168 | 19.32 | 19.73 | +2.1% |

Score is 1.3000 vs 1.3333 (-2.5%), outside the ±1% noise. I expected the per-core drain rate to rise and the
drain to shrink by 0.5-0.8 µs. Instead it got ~8% slower.

## Why (profiler evidence)
`drain2.py` and `drain.py` are in this dir. Outputs: `drain2_out.txt` / `drain2_parent_out.txt`, and
`drain_out.txt` / `drain_parent_out.txt`. Values are medians over measured calls and all 4 chips.

| shape | drain end - first POST unpack, parent -> this (µs) | per-core tiles/µs (W_DRAIN zone), parent -> this |
|---|---|---|
| h3584 | 2.94 -> 3.24 | 7.8-8.2 -> 7.4 (all cores) |
| h4096 | 3.49 -> 3.78 | 6.8-8.8 -> 6.5-7.9 |
| h6144 | 5.14 -> 5.56 | 7.5-9.1 -> 7.1-8.3 |
| h7168 | 5.58 -> 6.05 | 8.8-9.6 -> 8.2-8.7 |

- **Nothing outside the drain changed.** Compute POST (T0 1.71 / 1.97 / 3.00 / 3.52 µs, T2 the same), F_COLLECT,
  go, and the stick read are the parent's values. The whole loss is drain throughput, ~0.3-0.47 µs per shape. That
  matches the kernel-end change.
- **The per-core drain cap is not command-buffer serialization.** With two write requests allowed in the NIU per NoC,
  every core got *uniformly* slower (h3584: every core 7.4 tiles/µs, still no position gradient). If the issue path
  had been the limit (BRISC address math + counter RMWs + NIU setup serialized behind each 2 KB injection),
  overlapping it would have helped. So the cap is downstream of issue: NIU injection / end-to-end flow control or
  response handling for non-posted writes, or the destination side. The noc.h comment ("busy while back-pressured")
  is consistent with that: the buffer was busy *because* the NoC back-pressured it, not because of issue overhead.
- The slowdown is most likely two packets from the same core competing for one NIU injection port and VC.
  Consecutive tiles go to different banks, so the core's packets now interleave at packet granularity toward two
  destinations instead of streaming one at a time. That adds head-of-line blocking on the shared links/DRAM queues.
  The cmd buf 1 setup/restore costs only ~10 register ops per call and can't explain ~0.3 µs.

## Classification
flawed idea. The per-core drain rate (~10.5 tiles/µs = ~21 GB/s per core) is not set by the single command
buffer or BRISC issue overhead, so a second in-flight request per NoC only adds contention. The plumbing is
correct and reusable: switching cmd buf 1 to writes under DM_DYNAMIC_NOC, setting its TARG coordinate, and restoring
its RET coordinate.

## What a child of this node should try next
1. **Revert this drain change** (start from the parent r03-b02-a02 or drop the `two_cmd_bufs` path). Don't add more
   in-flight requests per core on the drain.
2. **Posted output writes** are the remaining cheap test of the per-core cap (r03-b04-a02 #2), since the cap now
   looks like response/flow control, not issue. Keep one cmd buf. Use `noc.async_write<NocOptions::POSTED>` and
   `async_writes_flushed<POSTED>` before each pop, and keep a final full barrier. Caveat: posted writes have no ack,
   so the kernel can finish before the last bytes land. That is fine for the profiled test, but justify it for
   production, e.g. end with one non-posted write per bank.
3. If posted is also flat, the per-core cap is the NoC/DRAM path, and the drain can only shrink by moving bytes
   off the hot links/banks (r02-b02-a01 #1: per-core NoC0 share from `analysis/opt.py`) or by starting it earlier
   (post-AG fixed costs: stick read 0.59 µs, combine 0.38 µs).
