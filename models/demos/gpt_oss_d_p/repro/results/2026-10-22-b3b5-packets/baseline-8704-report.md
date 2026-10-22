# GPT-OSS-120B P/D: 2,880 vs 4,352 vs 8,704-byte fabric payloads

Snapshot: 2026-10-21T17:59:50.858938+00:00. **29/29 cases complete at 8,704 bytes.**

Current phase: All 29 measurements are complete and audited. See completion records for publication and resource cleanup.

## Measurement contract

- Fresh 8,704-byte sweep explicitly authorized after the completed 4,352-byte sweep. Previous results are frozen.
- Same 29 valid pairs: ISL 1K,2K,4K,5K,8K,16K,32K,55K,64K; chunks 1K,4K,8K,16K,32K,64K, chunk <= ISL; K = 1,024.
- One warmup and three measured requests per case; output cap 32 with EOS enabled. No selective reruns or exclusion of slower results.
- Full36 model; prefill SP4/TP8; four 4x2 reload decoder stages on one Galaxy; two allocated slots, one active request, 131,072-token capacity.
- Prefill topology stays linear / FABRIC_1D. The only model change from the 4,352-byte configuration is FABRIC_PAYLOAD_SIZE = 8704.
- No golden generation/use, PCC or standalone rerun. Completion establishes measurement/accounting checks, not numerical accuracy.

## Matched prefill throughput

Means use the three measured requests; throughput is actual newly prefilled tokens divided by the host-observed prefill service span. Improvement = (8704 mean / reference mean − 1) × 100%. Sample SD is retained in the interactive table and raw evidence.

| ISL | Chunk | 2,880 B (t/s) | 4,352 B (t/s) | 8,704 B (t/s) | 8,704 vs 2,880 | 8,704 vs 4,352 | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1K | 1K | 2,536.8 | 2,550.5 | 2,510.2 | -1.05% | -1.58% | DONE |
| 2K | 1K | 2,650.4 | 2,682.1 | 2,607.8 | -1.61% | -2.77% | DONE |
| 4K | 1K | 2,717.0 | 2,730.1 | 2,637.2 | -2.94% | -3.41% | DONE |
| 4K | 4K | 10,308.1 | 10,654.5 | 10,255.4 | -0.51% | -3.75% | DONE |
| 5K | 1K | 2,702.3 | 2,727.4 | 2,653.0 | -1.83% | -2.73% | DONE |
| 5K | 4K | 6,631.0 | 6,658.6 | 6,583.0 | -0.72% | -1.14% | DONE |
| 8K | 1K | 2,688.3 | 2,684.9 | 2,669.0 | -0.72% | -0.59% | DONE |
| 8K | 4K | 10,550.9 | 10,754.7 | 10,589.0 | 0.36% | -1.54% | DONE |
| 8K | 8K | 19,631.1 | 20,720.4 | 20,642.2 | 5.15% | -0.38% | DONE |
| 16K | 1K | 2,722.2 | 2,675.3 | 2,643.2 | -2.90% | -1.20% | DONE |
| 16K | 4K | 10,880.3 | 10,758.7 | 10,720.8 | -1.47% | -0.35% | DONE |
| 16K | 8K | 20,696.6 | 21,255.9 | 20,500.7 | -0.95% | -3.55% | DONE |
| 16K | 16K | 23,780.0 | 25,551.1 | 27,483.0 | 15.57% | 7.56% | DONE |
| 32K | 1K | 2,713.9 | 2,747.5 | 2,661.4 | -1.94% | -3.14% | DONE |
| 32K | 4K | 10,983.7 | 10,993.9 | 10,621.7 | -3.30% | -3.39% | DONE |
| 32K | 8K | 21,080.7 | 20,849.9 | 21,027.1 | -0.25% | 0.85% | DONE |
| 32K | 16K | 22,994.1 | 24,457.1 | 26,187.4 | 13.89% | 7.07% | DONE |
| 32K | 32K | 23,201.5 | 24,980.1 | 26,784.9 | 15.44% | 7.22% | DONE |
| 55K | 1K | 2,749.1 | 2,751.7 | 2,671.6 | -2.82% | -2.91% | DONE |
| 55K | 4K | 10,512.9 | 10,393.7 | 10,545.6 | 0.31% | 1.46% | DONE |
| 55K | 8K | 19,884.7 | 20,789.9 | 20,716.9 | 4.19% | -0.35% | DONE |
| 55K | 16K | 18,977.2 | 20,105.4 | 21,346.5 | 12.49% | 6.17% | DONE |
| 55K | 32K | 19,350.9 | 20,891.6 | 22,263.2 | 15.05% | 6.57% | DONE |
| 64K | 1K | 2,715.2 | 2,730.7 | 2,692.3 | -0.84% | -1.41% | DONE |
| 64K | 4K | 10,776.0 | 10,895.3 | 10,588.8 | -1.74% | -2.81% | DONE |
| 64K | 8K | 20,058.0 | 20,653.2 | 20,543.1 | 2.42% | -0.53% | DONE |
| 64K | 16K | 21,641.7 | 23,030.6 | 24,540.1 | 13.39% | 6.55% | DONE |
| 64K | 32K | 21,945.4 | 23,428.0 | 25,048.7 | 14.14% | 6.92% | DONE |
| 64K | 64K | 21,114.0 | 22,539.8 | 24,003.3 | 13.68% | 6.49% | DONE |

### Observed differences against 2,880 bytes

13 higher means; 16 lower means. Range -3.295% to +15.572%. Descriptive results, not a significance test.

| Chunk | Cases | Lowest change | Highest change |
|---:|---:|---:|---:|
| 1K | 9 | -2.938% | -0.715% |
| 4K | 7 | -3.295% | +0.361% |
| 8K | 5 | -0.947% | +5.151% |
| 16K | 4 | +12.485% | +15.572% |
| 32K | 3 | +14.141% | +15.444% |
| 64K | 1 | +13.684% | +13.684% |

All slower means, without selective exclusions:

- 1K / 1K: -1.048%; reference 2,536.8 ± 61.4, new 2,510.2 ± 28.1 tokens/s. New samples: 2,478.5, 2,520.2, 2,532.0.
- 2K / 1K: -1.609%; reference 2,650.4 ± 19.3, new 2,607.8 ± 72.1 tokens/s. New samples: 2,626.6, 2,668.5, 2,528.2.
- 4K / 1K: -2.938%; reference 2,717.0 ± 2.1, new 2,637.2 ± 22.3 tokens/s. New samples: 2,618.0, 2,661.7, 2,631.9.
- 4K / 4K: -0.511%; reference 10,308.1 ± 323.5, new 10,255.4 ± 191.6 tokens/s. New samples: 10,476.6, 10,141.2, 10,148.5.
- 5K / 1K: -1.826%; reference 2,702.3 ± 35.6, new 2,653.0 ± 23.8 tokens/s. New samples: 2,625.9, 2,662.7, 2,670.4.
- 5K / 4K: -0.724%; reference 6,631.0 ± 103.6, new 6,583.0 ± 189.3 tokens/s. New samples: 6,742.1, 6,373.6, 6,633.3.
- 8K / 1K: -0.715%; reference 2,688.3 ± 36.9, new 2,669.0 ± 18.6 tokens/s. New samples: 2,678.6, 2,647.6, 2,680.9.
- 16K / 1K: -2.902%; reference 2,722.2 ± 16.8, new 2,643.2 ± 52.8 tokens/s. New samples: 2,696.2, 2,642.7, 2,590.6.
- 16K / 4K: -1.466%; reference 10,880.3 ± 85.5, new 10,720.8 ± 146.8 tokens/s. New samples: 10,554.8, 10,774.1, 10,833.5.
- 16K / 8K: -0.947%; reference 20,696.6 ± 63.2, new 20,500.7 ± 594.4 tokens/s. New samples: 19,837.4, 20,984.9, 20,679.8.
- 32K / 1K: -1.938%; reference 2,713.9 ± 2.2, new 2,661.4 ± 8.5 tokens/s. New samples: 2,670.1, 2,653.1, 2,660.9.
- 32K / 4K: -3.295%; reference 10,983.7 ± 18.5, new 10,621.7 ± 103.6 tokens/s. New samples: 10,712.8, 10,509.0, 10,643.3.
- 32K / 8K: -0.254%; reference 21,080.7 ± 126.1, new 21,027.1 ± 164.6 tokens/s. New samples: 21,167.3, 21,068.1, 20,845.8.
- 55K / 1K: -2.818%; reference 2,749.1 ± 25.8, new 2,671.6 ± 13.2 tokens/s. New samples: 2,686.4, 2,660.9, 2,667.6.
- 64K / 1K: -0.843%; reference 2,715.2 ± 6.8, new 2,692.3 ± 24.4 tokens/s. New samples: 2,669.8, 2,718.2, 2,688.8.
- 64K / 4K: -1.737%; reference 10,776.0 ± 71.5, new 10,588.8 ± 84.0 tokens/s. New samples: 10,684.5, 10,527.6, 10,554.3.

### Observed differences against 4,352 bytes

10 higher means; 19 lower means. Range -3.745% to +7.561%. Descriptive results, not a significance test.

| Chunk | Cases | Lowest change | Highest change |
|---:|---:|---:|---:|
| 1K | 9 | -3.405% | -0.590% |
| 4K | 7 | -3.745% | +1.461% |
| 8K | 5 | -3.553% | +0.850% |
| 16K | 4 | +6.173% | +7.561% |
| 32K | 3 | +6.565% | +7.225% |
| 64K | 1 | +6.493% | +6.493% |

All slower means, without selective exclusions:

- 1K / 1K: -1.580%; reference 2,550.5 ± 113.7, new 2,510.2 ± 28.1 tokens/s. New samples: 2,478.5, 2,520.2, 2,532.0.
- 2K / 1K: -2.772%; reference 2,682.1 ± 40.2, new 2,607.8 ± 72.1 tokens/s. New samples: 2,626.6, 2,668.5, 2,528.2.
- 4K / 1K: -3.405%; reference 2,730.1 ± 44.1, new 2,637.2 ± 22.3 tokens/s. New samples: 2,618.0, 2,661.7, 2,631.9.
- 4K / 4K: -3.745%; reference 10,654.5 ± 74.8, new 10,255.4 ± 191.6 tokens/s. New samples: 10,476.6, 10,141.2, 10,148.5.
- 5K / 1K: -2.729%; reference 2,727.4 ± 16.4, new 2,653.0 ± 23.8 tokens/s. New samples: 2,625.9, 2,662.7, 2,670.4.
- 5K / 4K: -1.135%; reference 6,658.6 ± 91.9, new 6,583.0 ± 189.3 tokens/s. New samples: 6,742.1, 6,373.6, 6,633.3.
- 8K / 1K: -0.590%; reference 2,684.9 ± 9.6, new 2,669.0 ± 18.6 tokens/s. New samples: 2,678.6, 2,647.6, 2,680.9.
- 8K / 4K: -1.541%; reference 10,754.7 ± 140.2, new 10,589.0 ± 53.3 tokens/s. New samples: 10,546.2, 10,648.7, 10,572.1.
- 8K / 8K: -0.377%; reference 20,720.4 ± 262.2, new 20,642.2 ± 183.5 tokens/s. New samples: 20,464.9, 20,630.4, 20,831.4.
- 16K / 1K: -1.199%; reference 2,675.3 ± 24.2, new 2,643.2 ± 52.8 tokens/s. New samples: 2,696.2, 2,642.7, 2,590.6.
- 16K / 4K: -0.352%; reference 10,758.7 ± 73.0, new 10,720.8 ± 146.8 tokens/s. New samples: 10,554.8, 10,774.1, 10,833.5.
- 16K / 8K: -3.553%; reference 21,255.9 ± 260.9, new 20,500.7 ± 594.4 tokens/s. New samples: 19,837.4, 20,984.9, 20,679.8.
- 32K / 1K: -3.136%; reference 2,747.5 ± 11.0, new 2,661.4 ± 8.5 tokens/s. New samples: 2,670.1, 2,653.1, 2,660.9.
- 32K / 4K: -3.386%; reference 10,993.9 ± 51.7, new 10,621.7 ± 103.6 tokens/s. New samples: 10,712.8, 10,509.0, 10,643.3.
- 55K / 1K: -2.910%; reference 2,751.7 ± 12.1, new 2,671.6 ± 13.2 tokens/s. New samples: 2,686.4, 2,660.9, 2,667.6.
- 55K / 8K: -0.351%; reference 20,789.9 ± 258.9, new 20,716.9 ± 403.9 tokens/s. New samples: 20,251.3, 20,972.7, 20,926.6.
- 64K / 1K: -1.407%; reference 2,730.7 ± 14.9, new 2,692.3 ± 24.4 tokens/s. New samples: 2,669.8, 2,718.2, 2,688.8.
- 64K / 4K: -2.813%; reference 10,895.3 ± 123.7, new 10,588.8 ± 84.0 tokens/s. New samples: 10,684.5, 10,527.6, 10,554.3.
- 64K / 8K: -0.533%; reference 20,653.2 ± 582.7, new 20,543.1 ± 293.1 tokens/s. New samples: 20,616.7, 20,220.2, 20,792.4.

## Interpretation limits

The first eight 8,704-byte cases (64K,32K,16K chunks) used b05u02 prefill / b03u02 decode. Slurm preempted that decoder allocation at16:02:23UTC. The remaining21 cases (8K,4K,1K chunks) used b05u02 / b08u02. This decoder change is an additional comparison confound. The initial c05u02 prefill was stopped and released at the user request before any measurements. The 4,352 baseline used c05u02 / b03u02, so prefill host identity changes; neither comparison isolates packet size alone. The harness, D-gen and topology remain unchanged; the model delta is the prefill packet size. Independent startups, cache state, run order and three samples per case remain limitations. The 2,880-byte run used c02u02/c06u14 and historical D-gen repair cohorts, so that comparison also changes host identity. Prompts use the existing synthetic fresh-nonce harness, not the realistic-text prompt from the motivating note.

## Timing definitions

TTFT is the request-to-first-streamed-token interval. Prefill rate uses actual newly prefilled tokens; logical/reused and padded processing counts remain distinct. KV bytes are wire payload. Transfer sum, union-active duration and span are separate. Overlap is the intersection of prefill and transfer active intervals; non-overlap is the respective remainder. Initial prefill and final transfer tail expose start/end serialization. Decode t/s/user uses native host-commit intervals (generated tokens minus one divided by the first-to-last interval). Do not add overlapping durations, transfer sums or concurrent stage totals to wall time.

## Setup, recovery and reload phases

The prior 4,352-byte models and allocations had already been released. Fresh allocations 141892 (c05u02) and 141893 (b03u02) were granted at 14:03:46 UTC, expiring 19:03:46. Decoder site recovery was required before reusing firmware parked by the previous owner-stop; it passed on its first bounded attempt, with firmware/device/link validation retained. This was preparation for reuse, not a failed 8,704-byte decoder startup. After the user reassigned prefill, b05u02 was verified under existing allocation141909 (expires20:25:51UTC); the original process deadlines remain in force even after decoder141893 was preempted. The replacement initially returned PCIe read0xffffffff on device26. Bounded site recovery passed first attempt at14:34:21UTC, and a fresh health check confirmed32devices/32DRAMready. Its build uses separate logs. Native builds reuse node-local build/cache directories; actual recompilation is included in setup time.

The first decoder attempt failed on a cached metadata read (EBADF): rank0 failed14:38:06UTC, overall exit1 followed14:53:03 after35/41 captures. Repeated sidecar reads, the blob checksum and idle32-device health checks passed. An unchanged retry began14:59:48UTC with the ready prefill retained. The original cause remains unresolved; no source/cache mutation or device reset was applied. Failed and retry setup timings remain separate.

Slurm preempted decoder141893 at16:02:23UTC after eight audited cases. No8K request had begun. The ready8K prefill was retained while b03u08 was reserved as a replacement. Its idle PCIe probe failed on device24 and bounded site recovery failed with all eight UBB4 devices unreachable; no model was started there. After descendants/holders were verified absent,142022 was released. A fresh six-hour decoder allocation142028 on b08u02 expires22:13:34UTC and passed32device/DRAM checks. The user requested three extra hours: Slurm denied in-place increases; prefill141909 still expires20:25:51UTC. The fresh decoder reservation incorporates the extra three hours. Existing task deadlines are unchanged. See the handoff for current process identities.

| Phase | Seconds | Status | Finished UTC |
|---|---:|---|---|
| decode on bh-glx-120-b03u02: native compilation | 387.3 | DONE | 2026-10-21T14:16:48.353399+00:00 |
| decode on bh-glx-120-b03u02: imports/environment check | 21.6 | DONE | 2026-10-21T14:17:10.000843+00:00 |
| decode-b08 on bh-glx-120-b08u02: native compilation | 474.7 | DONE | 2026-10-21T16:22:12.150232+00:00 |
| decode-b08 on bh-glx-120-b08u02: imports/environment check | 41.4 | DONE | 2026-10-21T16:22:53.560802+00:00 |
| prefill on bh-glx-120-c05u02: native compilation | 323.2 | DONE | 2026-10-21T14:12:41.532213+00:00 |
| prefill on bh-glx-120-c05u02: imports/environment check | 29.3 | DONE | 2026-10-21T14:13:10.860158+00:00 |
| prefill-b05 on bh-glx-120-b05u02: native compilation | 360.3 | DONE | 2026-10-21T14:42:00.024854+00:00 |
| prefill-b05 on bh-glx-120-b05u02: imports/environment check | 33.4 | DONE | 2026-10-21T14:42:33.460191+00:00 |
| sweep-b05-pd-c1024: launch to ready | 238.9 | DONE | 2026-10-21 17:39:53.841 UTC |
| sweep-b05-pd-c16384: launch to ready | 192.5 | DONE | 2026-10-21 15:50:58.097 UTC |
| sweep-b05-pd-c32768: launch to ready | 276.1 | DONE | 2026-10-21 15:41:10.472 UTC |
| sweep-b05-pd-c4096: launch to ready | 206.5 | DONE | 2026-10-21 17:21:53.394 UTC |
| sweep-b05-pd-c65536: launch to ready | 991.5 | DONE | 2026-10-21 14:59:24.226 UTC |
| sweep-b05-pd-c8192: launch to ready | 212.9 | DONE | 2026-10-21 16:02:57.548 UTC |
| sweep-pd-c65536: launch to ready | 310.4 | DONE | 2026-10-21 14:18:49.919 UTC |
| decode-attempt1-cache-read: launch to ready/exit | 2,132.7 | FAILED | 2026-10-21T14:53:02.288631+00:00 |
| b03u02 retry (subsequently preempted): launch to ready/exit | 1,902.4 | DONE | 2026-10-21T15:31:31.112000+00:00 |
| replacement decoder b08u02: launch to ready/exit | 2,613.7 | DONE | 2026-10-21T17:06:42.218000+00:00 |
| decode-b03u08 site recovery | 62.3 | FAILED | 2026-10-21T16:11:53.035080+00:00 |
| decoder site recovery | 104.8 | DONE | 2026-10-21T14:09:00.086338+00:00 |
| prefill-b05 site recovery | 122.8 | DONE | 2026-10-21T14:34:21.340337+00:00 |

### Completed reload images by concurrent stage

| Attempt | Stage | Images | Load/build s | Staging s | Tap s | Configure s | Readback s |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode-attempt1-cache-read | 0 | 5 | 655.9 | 220.1 | 0.0 | 123.0 | 10.5 |
| decode-attempt1-cache-read | 1 | 10 | 1,304.2 | 312.5 | 0.0 | 389.3 | 22.3 |
| decode-attempt1-cache-read | 2 | 10 | 1,297.7 | 304.9 | 0.0 | 407.4 | 20.3 |
| decode-attempt1-cache-read | 3 | 10 | 1,298.3 | 307.4 | 0.0 | 398.8 | 21.7 |
| b03u02 retry (subsequently preempted) | 0 | 11 | 1,300.0 | 339.9 | 0.0 | 123.3 | 22.4 |
| b03u02 retry (subsequently preempted) | 1 | 10 | 1,272.5 | 304.5 | 0.0 | 38.3 | 20.3 |
| b03u02 retry (subsequently preempted) | 2 | 10 | 1,262.1 | 305.5 | 0.0 | 56.7 | 20.7 |
| b03u02 retry (subsequently preempted) | 3 | 10 | 1,287.1 | 306.0 | 0.0 | 26.5 | 20.5 |
| replacement decoder b08u02 | 0 | 11 | 1,621.5 | 567.5 | 0.0 | 235.7 | 25.0 |
| replacement decoder b08u02 | 1 | 10 | 1,641.2 | 394.3 | 0.0 | 357.4 | 23.8 |
| replacement decoder b08u02 | 2 | 10 | 1,657.8 | 323.8 | 0.0 | 380.2 | 30.5 |
| replacement decoder b08u02 | 3 | 10 | 1,658.6 | 325.6 | 0.0 | 375.7 | 32.5 |

Stage totals overlap; their sum is not startup wall time. Initial setup, per-group setup and request performance are reported separately.

## Source pins and evidence

- Metal branch: `divanovic/mg-gptoss-pd-payload8704-20261021-metal`, from `9f7df82859f1b5e2e7d7f91e6fb3b1e69e9d628f` (completed 4,352 results). Configuration commit: `076d79443138613e9e1a96b948d02db69858274a`.
- Blaze: `d6d0b09f22f60a161d66fe4db90b9abed81ef82d`; nested Metal: `44d66500520fda9f2c7060c0f6b41ec48f7ab37e`.
- D-gen: `5ca79e3076f193c092962d34a3b69731e39aff91`.
- Remote runtime/evidence: `/data/divanovic/gptoss-payload8704-20261021`. Each group records its effective imported payload in `prefill/fabric-config.json`.
- Saved-evidence audit covers token/byte/overlap/TSU/mean/variance accounting and all four requests per case. Completion records below distinguish measurements from publication and resource cleanup.
- [Interactive graphs and comparison](http://127.0.0.1:8768/payload4352-comparison.html)
- [Matched three-way CSV](http://127.0.0.1:8768/payload4352-comparison.csv)
- [Configuration and exact commands](http://127.0.0.1:8768/PAYLOAD8704_MEASUREMENTS.md)

## Completion records

| Check | Recorded state |
|---|---|
| Final saved-evidence audit | 29/29 cases, zero errors |
| Model configuration publication | Pushed; remote hash verified |
| Task services and model owners | Stopped; no remaining device holders on either task host |
| Reservations 141909 / 142028 | Released and verified |
| Source and raw evidence archive | All archive member SHA256 values verified |

## Accepted request accounting

116 requests: 29 warmups and 87 measured requests. First request 2026-10-21T15:34:18.060038+00:00; last response 2026-10-21T17:57:19.535874+00:00.
Elapsed span 8,581.476s; sum of individual request durations 484.153s. Request span includes group switching and the decoder-preemption/replacement gap, including replacement decoder startup. Duration sum excludes setup and gaps. Neither includes decoder startup before the first accepted request.

[All requests](PAYLOAD8704_RUNS.csv) · [Means, sample variance and SD](PAYLOAD8704_STATISTICS.csv) · [Accounting summary](measurement-summary.json)

### Preserved teardown warnings

- Decoder owner-stop completed with MPI launcher exit1 (rank3 improper termination) at18:07:52.569903UTC. All recorded owners, task processes and device holders were independently verified gone. This is not a clean decoder exit0.

### Source and evidence snapshot

Archive: `/data/divanovic/gptoss-payload8704-20261021/archives/gptoss-payload8704-20261021-source-evidence.tar.gz`.
SHA256: `ac9ffd73ddd147c677d2cd9918a0b432d740cbf4499d53ef203a4f2a59ec3bfb`.
3,394 files; 104,306,507 compressed bytes. The archive captures the tested source and measurement/cleanup evidence before the result-publication commit. Weight caches, environments and native build trees are excluded.
