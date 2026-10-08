# Optimization follow-up, Oct 8 2026 UTC

User priority is **32K ISL first, 16K second**. Both 128K and 256K remain active
secondary optimization targets. Report throughput, per-user speed, TTFT, memory
and numerical/quality tradeoffs together. Hardware policy remains BFP8 KV,
BF16 activations and FP32 recurrent state; existing BFP4 weights are unchanged.

`collection.json` records 73 original artifact hashes and collection-time
systemd state. Compressed files preserve the exact original bytes. Running
queue receipts are snapshots, not evidence of later completion.

## Full-model recurrence comparison

The 128K/B16 pair completed on one TP4 with three measured repetitions per
arm and clean device close. Source hashes, prompt hashes and precision match
except the intended recurrence selector. Each arm repeats its output hash;
native and candidate outputs are not bit-identical to each other. This is a
performance comparison, not model-eval qualification.

| Metric | Native recurrence | Single-step recurrence |
|---|---:|---:|
| Decode output tok/s, TP4 total | 142.063 | 163.359 |
| Decode output tok/s/user | 8.879 | 10.210 |
| TPOT, ms | 112.626 | 97.944 |
| Prefill input tok/s, TP4 total | 4044.021 | 4043.965 |
| TTFT p50, seconds | 518.719 | 518.750 |

Decode throughput improves **14.990%**. Prefill is unchanged. The 8x projection
is 1306.87 output tok/s/Galaxy; physical eight-replica performance is unmeasured.
Rates exclude weight loading/first-use compilation. Decode throughput excludes
prefill; TTFT and the separate input rate retain the prefill cost. The full
64-layer comparison plot and JSON/CSV are in `capacity-pairs-v1/s131072-b16/`.
The ongoing controller next measures the 32K/B32 pair, then near-256K/B8.

## Placement/chunk diagnostic: completed 78 cases

All six geometries completed and devices closed cleanly at 06:09:53 UTC.
Thirteen variants each include native controls bracketing the measurement.
Failed numerical candidates are preserved and excluded from wins.

| ISL / batch | Best passing placement / chunk | Attention-call gain vs native |
|---|---|---:|
| 32K / 16 | outer 80 cores / 256 | 2.52% |
| 32K / 32 | outer 80 cores / 256 | 1.94% |
| 128K / 8 | outer 80 cores / 256 | 3.01% |
| 128K / 16 | outer 96 cores / 256 | 3.57% |
| 262016 / 4 | native / 256 | no gain |
| 262016 / 8 | outer 80 cores / 512 | 3.60% |

Matched 80-core 512-chunk row/outer placements at 256K/B8 have bit-identical
outputs and a 4.75% location-only gain; changing core count/chunk versus native
changes reduction geometry. The 256-chunk 80/96-core candidates at this geometry
fail the unchanged numerical gate. Chunk 512 passes, consistent with reduction
depth affecting error; placement itself is not a proven numerical root cause.

Useful KV bandwidth is 357.88 GB/s/chip at 32K/B16 and about 385-387 GB/s/chip
at 32K/B32 and the larger 128K/256K batches, approximately 70-76% of assumed
512 GB/s. These are useful bytes divided by call time, not DRAM-counter
utilization or full-model bandwidth. No full-model placement uplift is claimed.

## Accurate partial-query path: simulator pass, hardware pending

The candidate removes the forced approximate-exp condition on partial faces
and restricts FP32 scalar scaling to the same valid faces. Full-tile arithmetic
is unchanged. The native installation is untouched: separate source overlays
and JIT caches select each arm, with generated TRISC includes checked against
the exact override. Q/KV/output precision and numerical thresholds are fixed.

- 285 CPU tests plus 40 subtests passed.
- V1 stopped after its first passing full-tile case because the evidence checker
  looked for compute includes in a dataflow-only generated filename. Device
  cleanup completed; the log, receipt and exact source are retained.
- V2 checks generated unpack/math/pack source wrappers and rejects missing or
  mixed overrides. Both simulator processes completed cleanly, exit 0.
- All 16 candidate cases passed at 1K, 32K, 128K and near-256K with two causal
  bounds and full/partial query tiles. Six native partial-query cases failed.
- All eight candidate partial-query outputs are bit-identical to native
  full-query controls. All full-query controls are unchanged across overlays.
- Candidate relative RMS versus quantized-operand reference is 0.69-1.40%,
  compared with 1.85-2.20% for the original partial-query path.

The simulator uses the previously documented explicit-instruction compiler
fallback for unsupported SFPLOADMACRO, with checks intact. No physical hardware
test, production-instruction parity, speedup or full-model/eval pass follows
from this screen. The candidate does not reduce KV bytes; hardware timing must
determine whether reduced query compute and padding improve throughput.

At 32K/B16, the synthetic full attention call is ~0.810 ms. Sixteen such calls
would be ~13.0 ms of the measured ~61.7-ms full-model decode step. This is a
cross-benchmark estimate, not stage attribution. Hypothetical attention speedups
of 10%/20% imply roughly 1.95%/3.63% full-model throughput gains if the estimate
transfers and other stages are unchanged. These are sensitivity cases, not
measured gains or a confidence interval. KV reads do not shrink with Q tiles.

## Persistent profiling and priority change

At 06:16:39 UTC, queued `qwen38-bounded-layer-profile-v2-20261008.service` after
276 CPU tests plus 40 subtests passed. Order: 32K/B16, 32K/B32, 16K/B16,
16K/B32, 128K/B16, 256K/B8; native and single-step at each geometry. The older
queue was frozen, verified to have only controller/timeout/shell/flock processes,
and stopped before any device worker began. The active 32K model run was not
interrupted. `bounded-layer-profile-replacement-v2.json` preserves that evidence.

The new queue retains the shared lock, 12-hour deadline, 64-GiB memory limit,
eight-CPU cap and bounded exports. The capacity controller has a 14-hour deadline.
Live `loginctl show-user ttuser` confirmed `Linger=yes`; both services were
active independently of SSH. They survive client/session disconnects, subject
to their deadlines, errors and host uptime. They are not reboot-resume services.

## Blaze bandwidth reference

Inspected cached `tt-blaze` revision `9415da978b35a2ea9b28ab736aa485fc4d06d8ab`
(`origin/yugao/reload-pipeline-v2`), `docs/plans/reload_pipeline.md` sections 4.7
and C.1. Its recorded **471 GB/s / 92%** figure comes from routed-expert
streaming matmuls: 24.75 MB / 52.5 us. The document explicitly says applying
that rate to reload is a modeling assumption, not measured reload efficiency.
No new silicon measurement was made for this source inspection.

The local Blaze streaming implementation at `75ae38aabb0c3cbd12750dfc9398f7b0db01026e`
provides applicable mechanisms: bank-adjacent NOC0 workers, contiguous bank-local
weight shards, packet sizes up to 16 KiB on Blackhole, reused NoC address state,
and transaction-ID buffering that waits for the oldest needed transaction rather
than draining every outstanding read. `blaze/ops/dram_streaming_matmul/common.py`
and `kernels/op.hpp` implement these. Cache/model precision differs from Qwen KV.

Qwen's KV remains paged/interleaved; relocating workers alone does not establish
bank locality. Larger barrier thresholds already regressed, so the useful next
comparison is bank-local/transaction-pipelined reading versus interleaved reading
with matched BFP8 bytes, independent of attention math. This is a candidate
experiment, not an implemented or tested Qwen optimization.

## Remaining qualification

Hardware timing for the partial-query candidate, model-boundary validation of
placement, 16K full-model measurements, fused GDN output gating, shared-Q/K
preparation, B64 support and physical Galaxy scaling remain open. Last full GPQA
is still 171/198, below the plan's gate; no new policy is model-eval qualified.
