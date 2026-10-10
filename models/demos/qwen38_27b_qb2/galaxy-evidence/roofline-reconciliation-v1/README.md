# Reconciling the plan's roofline with B16/32K

October 10, 2026 UTC. The saved specification's 86 tokens/s/user roofline is
**B16, 8K context, BFP4 weights**. The current 16.55 tokens/s/user measurement
is **B16, 32K context, BFP8 weights**, with accurate attention and HiFi2
projections. Both retain BFP8 KV and FP32 recurrent state. Comparing the two
numbers directly overstates the implementation gap.

The artifact's HTML shell was accessible on October 10, but its content request
returned HTTP 403. This audit uses the user's saved October 6 PDF extraction,
hashed in [spec-snapshot.json](spec-snapshot.json). It does not establish that
the current online artifact has identical contents.

## Matched traffic calculation

Per chip and decode step, using measured padded projection shapes:

| Term | Current B16/32K bytes | Floor at assumed 512 GB/s |
|---|---:|---:|
| BFP8 weights, including padding and vocabulary head | 7.112 GB | 13.890 ms |
| BFP8 KV reads | 4.563 GB | 8.913 ms |
| FP32 recurrent-state read and write | 1.208 GB | 2.359 ms |
| Total useful traffic | 12.883 GB | 25.162 ms |

This is a useful-byte lower bound, not a calibrated compute/memory roofline:
it excludes additional physical transactions, activation traffic, compute,
communication and dependency stalls. The 128-token measurement scans a growing
context; this calculation uses its initial 32K length. No physical DRAM counters
were measured.

The original plan rounded weights to 3.60 GB/chip. BFP8 uses 1088 bytes per
32x32 tile versus 576 for BFP4; today's padded shapes additionally differ from
the plan's parameter-only count. Increasing context from 8K to 32K quadruples
the modeled KV traffic. These are workload changes, not performance regressions.
The retained FP32 state is already charged in both scenarios.

| Comparison | Step time | Tokens/s/user |
|---|---:|---:|
| Original specification's rounded BFP4/8K byte-only floor | 11.6 ms | 86 |
| Current BFP8/32K byte-only floor | 25.16 ms | 39.74 |
| Current bytes with the plan's 85% bandwidth + 4.7 ms P2 overhead assumptions | 34.30 ms | 29.15 |
| Measured direct-preparation/fused-epilogue candidate | 60.42 ms | 16.55 |

The current candidate achieves 41.64% of this useful-byte-only ceiling. Reaching
the plan's P2 scenario at the current precision/context still requires a 1.76x
throughput increase, or 26.12 ms less step time. The plan's 85% and 4.7 ms are
engineering assumptions, not established implementation properties or a promise
of attainable performance. Its 14K Galaxy throughput target is a different
operating point again: B64 per replica at 8K across eight replicas.

## What accounts for the remaining implementation gap

The [full operator inventory](../operator-scope-v1/INVENTORY.md) covers the
earlier 67.24-ms shared-Q/K control across 64 layers, four ranks and three
replays. Profiled family sums are not an additive wall-time decomposition;
profiling adds about 7.46%, and current fusion has already removed some work.

- Layout operations account for about 22.58 ms of profiled kernel sums.
  The source expands one useful decode row per user into separate padded tile
  planes at several operator boundaries, then repacks them for matmuls.
  Tracing retains these operations; it does not fuse them or remove their
  memory traffic. The compact GDN prototype is queued to remove more of them.
- Matmuls account for 18.89 ms. Their encoded-weight estimate is about 74% of
  peak, with gate/up around 83% and output projections around 57%. They have
  uneven tuning headroom, but making all of them ideal weight readers would
  recover only about 5.00 ms against this profiled weight-only bound. That is
  an optimistic bound with omitted required work, not an attainable timing.
- Attention accounts for 12.63 ms versus an 8.91-ms KV-only bound. Accurate
  exponentiation, dot products, reductions, page lookup and delivery also take
  time. The rejected reader prototype performed worse than production once
  consumer delivery was included; read-only bandwidth cannot substitute for
  that complete measurement.
- GDN preparation/recurrence uses SFPU reductions, L1 intermediate buffers and
  synchronization despite reading/writing FP32 state only once in DRAM.
  Communication and norms also require work omitted from the byte-only floor.
  The original LoFi arithmetic assumptions differ from the qualified HiFi2
  policy; the isolated full-model cost of that difference is not established.

The measured compact-preparation/epilogue candidate saves about 6.82 ms and
increases throughput by 11.3%. The final before/after B16/32K controls drifted
only 0.0031%, with matching output hashes. Candidate GPQA qualification remained
live at collection; this report does not promote it or claim its GPQA passed.

## Implication for the native 30-TSU target

Thirty TSU means a 33.33-ms step: another 27.09 ms must be removed. At 85%
useful-byte bandwidth the modeled streaming cost is 29.60 ms, leaving only
3.73 ms for unhidden work. We need both efficient streaming and much less
intermediate materialization/dependency overhead.

The persistent compact-GDN experiment targets 4-6 ms of additional savings;
the broader compact/fused decoder target of 10-15 ms includes that first piece.
Neither establishes 30 TSU. Full-boundary fusion and overlapping compatible
stages remain the larger intervention; projection and SDPA tuning support it.
These estimates must be replaced by matched complete-model measurements.

## Reproduction and evidence

Run `python3 reconcile.py` in this directory. It reads the retained completed
candidate sweep, native control comparison, plan assumptions and previously
verified operator inventory; validates precision/batch/context and three timing
samples; and writes [comparison.json](comparison.json). It does not open hardware.
The JSON includes BFP4/BFP8 and 8K/32K traffic scenarios with identical padded
projection shapes. BFP4 rows are hypothetical byte substitutions, not qualified
precision changes.

`qualification-queue.json` and `live-unit.txt` record the specific live job at
collection time. They are historical snapshots, not current liveness checks.
