# GPT-OSS-120B: P/D fabric packet-size comparison

Snapshot: 2026-10-21T13:51:38.071633+00:00. **29/29 new cases complete.**

Measurement state: all 29 cases audited; task resources released.

## Experiment

- Historical prefill fabric payload: 2,880 bytes; new payload: **4,352 bytes**.
- PR #60485 at head `b1b537740f5206e47cdbef3ecd5c880bbd197f06` uses 8,704 bytes. That setting is not measured in either table here; the historical pinned source uses 2,880, and the user selected 4,352 for this comparison.
- ISLs: 1K, 2K, 4K, 5K, 8K, 16K, 32K, 55K, 64K. K = 1,024 tokens.
- Chunks: 1K, 4K, 8K, 16K, 32K, 64K; only chunk <= ISL, giving 29 cases.
- Each case: one warmup plus three measured requests, output cap 32 with EOS enabled.
- Full 36-layer model; SP4/TP8 prefill, four 4x2 reload decode stages, 131,072-token capacity, two allocated slots, one active request.
- Prefill topology is linear with `FABRIC_1D` in both sweeps. The physical mesh uses the single-Galaxy XY-torus descriptor; that does not select ring fabric mode.
- No golden generation/use and no PCC checks. Performance completion does not establish numerical accuracy.
- Prefill: bh-glx-120-c05u02; decode: bh-glx-120-b03u02. Allocations 141422 (prefill) and 141554 (decode).

## How long did the previous P/D sweep take?

Accepted P/D requests span 03:25:12.509–05:04:39.511 UTC on October 21: **1h 39m 27s elapsed**. This includes interleaved standalone tests, chunk changes and repairs. The sum of the 116 accepted P/D request durations was **487.158s (8m 07s)**. That sum excludes startup, switching and failed attempts, so it is not the experiment duration. The earlier decoder cold startup was **2,673.897s (44m 34s)**, before the sweep.

## Matched prefill-throughput comparison

Throughput is actual newly prefilled tokens divided by the host-observed prefill service span. Rows match the same ISL and chunk. Improvement = (new mean / old mean - 1) x 100%. Means use the three measured requests; raw runs and sample standard deviations remain in the live page/JSON.

| ISL | Chunk | Old 2,880 B (tokens/s) | New 4,352 B (tokens/s) | Difference (tokens/s) | Improvement | New status |
|---:|---:|---:|---:|---:|---:|---|
| 1K | 1K | 2,536.8 | 2,550.5 | 13.7 | 0.54% | DONE |
| 2K | 1K | 2,650.4 | 2,682.1 | 31.7 | 1.20% | DONE |
| 4K | 1K | 2,717.0 | 2,730.1 | 13.1 | 0.48% | DONE |
| 4K | 4K | 10,308.1 | 10,654.5 | 346.3 | 3.36% | DONE |
| 5K | 1K | 2,702.3 | 2,727.4 | 25.1 | 0.93% | DONE |
| 5K | 4K | 6,631.0 | 6,658.6 | 27.6 | 0.42% | DONE |
| 8K | 1K | 2,688.3 | 2,684.9 | -3.4 | -0.13% | DONE |
| 8K | 4K | 10,550.9 | 10,754.7 | 203.9 | 1.93% | DONE |
| 8K | 8K | 19,631.1 | 20,720.4 | 1,089.3 | 5.55% | DONE |
| 16K | 1K | 2,722.2 | 2,675.3 | -46.9 | -1.72% | DONE |
| 16K | 4K | 10,880.3 | 10,758.7 | -121.6 | -1.12% | DONE |
| 16K | 8K | 20,696.6 | 21,255.9 | 559.3 | 2.70% | DONE |
| 16K | 16K | 23,780.0 | 25,551.1 | 1,771.1 | 7.45% | DONE |
| 32K | 1K | 2,713.9 | 2,747.5 | 33.6 | 1.24% | DONE |
| 32K | 4K | 10,983.7 | 10,993.9 | 10.3 | 0.09% | DONE |
| 32K | 8K | 21,080.7 | 20,849.9 | -230.8 | -1.09% | DONE |
| 32K | 16K | 22,994.1 | 24,457.1 | 1,463.0 | 6.36% | DONE |
| 32K | 32K | 23,201.5 | 24,980.1 | 1,778.5 | 7.67% | DONE |
| 55K | 1K | 2,749.1 | 2,751.7 | 2.6 | 0.09% | DONE |
| 55K | 4K | 10,512.9 | 10,393.7 | -119.2 | -1.13% | DONE |
| 55K | 8K | 19,884.7 | 20,789.9 | 905.2 | 4.55% | DONE |
| 55K | 16K | 18,977.2 | 20,105.4 | 1,128.2 | 5.95% | DONE |
| 55K | 32K | 19,350.9 | 20,891.6 | 1,540.8 | 7.96% | DONE |
| 64K | 1K | 2,715.2 | 2,730.7 | 15.6 | 0.57% | DONE |
| 64K | 4K | 10,776.0 | 10,895.3 | 119.2 | 1.11% | DONE |
| 64K | 8K | 20,058.0 | 20,653.2 | 595.2 | 2.97% | DONE |
| 64K | 16K | 21,641.7 | 23,030.6 | 1,388.9 | 6.42% | DONE |
| 64K | 32K | 21,945.4 | 23,428.0 | 1,482.5 | 6.76% | DONE |
| 64K | 64K | 21,114.0 | 22,539.8 | 1,425.9 | 6.75% | DONE |

### Observed spread and slower cases

Across the 29 completed matched cases, 24 have higher mean prefill throughput and 5 have lower means. The observed changes range from -1.72% to +7.96%. This is a descriptive comparison of three measured requests per case, not a statistical significance or packet-only causal claim. Slower measurements are retained without selective reruns or exclusions.

- 8K input / 1K chunk: -0.13%. Old 2,688.3 ± 36.9, new 2,684.9 ± 9.6 tokens/s (mean ± sample SD). New measured runs: 2,673.8, 2,690.1, 2,690.7 tokens/s. No cause is established by these observations.
- 16K input / 1K chunk: -1.72%. Old 2,722.2 ± 16.8, new 2,675.3 ± 24.2 tokens/s (mean ± sample SD). New measured runs: 2,660.9, 2,703.2, 2,661.6 tokens/s. No cause is established by these observations.
- 16K input / 4K chunk: -1.12%. Old 10,880.3 ± 85.5, new 10,758.7 ± 73.0 tokens/s (mean ± sample SD). New measured runs: 10,788.3, 10,812.3, 10,675.5 tokens/s. No cause is established by these observations.
- 32K input / 8K chunk: -1.09%. Old 21,080.7 ± 126.1, new 20,849.9 ± 490.1 tokens/s (mean ± sample SD). New measured runs: 21,167.0, 20,285.4, 21,097.3 tokens/s. No cause is established by these observations.
- 55K input / 4K chunk: -1.13%. Old 10,512.9 ± 93.7, new 10,393.7 ± 264.1 tokens/s (mean ± sample SD). New measured runs: 10,651.7, 10,405.4, 10,123.9 tokens/s. No cause is established by these observations.

## Time required for the new sweep

The 116 accepted requests (29 warmups and 87 measurements) ran from 2026-10-21T12:27:05.775034+00:00 to 2026-10-21T13:38:47.166399+00:00. Elapsed request span: **71.69 minutes**. Sum of accepted request durations: **480.382 seconds**. The span includes chunk switching and the prefill recovery gap; the sum excludes them. Both exclude the initial decoder bringup. Unlike the old sweep, this run did not interleave standalone tests, so the two elapsed sweep spans are not directly comparable performance measures.

### Observed prefill-throughput change by chunk

| Chunk | Cases | Lowest change | Highest change |
|---:|---:|---:|---:|
| 1K | 9 | -1.72% | +1.24% |
| 4K | 7 | -1.13% | +3.36% |
| 8K | 5 | -1.09% | +5.55% |
| 16K | 4 | +5.95% | +7.45% |
| 32K | 3 | +6.76% | +7.96% |
| 64K | 1 | +6.75% | +6.75% |

## Interpretation limits

The historical hosts were c02u02/c06u14; the new hosts are c05u02/b03u02. The new sweep uses the final validated D-gen source throughout, while historical cases span the recorded repair cohorts. These are observed before/after results across different hosts, not a controlled same-machine estimate of packet-size causality. Prompts follow the original synthetic fresh-nonce harness; these are not the realistic-text measurements quoted in the motivating note.

TTFT, prefill span/rate, transferred bytes, transfer sum/active/span, overlap, non-overlapped prefill/transfer, initial prefill, final transfer tail, decode interval/host-commit tokens/s/user and total request time are retained per run. Overlapping intervals and concurrent stage durations are not added to wall time.

## Bringup and obstacles

Both node-local native builds/imports passed. The first decoder startup failed before weight loading or reload capture: ARC firmware on device 0 did not become ready within 300 seconds. The failed attempt was preserved under `evidence/failed/decode-attempt1-arc`. The bounded site recovery timed out. The idle c08u02 node was released while retaining prefill. Replacement b03u02 passed 32-device/DRAM discovery and the TCP round-trip check; its local build and decoder setup are tracked separately. The first 32K prefill restart then failed during mesh initialization with an Ethernet heartbeat timeout, before model loading. That attempt is preserved under `evidence/failed/prefill-c32768-attempt1-ethernet`. Prefill-only site recovery passed in about a minute, including all 32 devices and healthy links; decode remained loaded. The fresh 32K owner reached readiness in 222.565 seconds. Later chunk transitions use the runner’s existing shutdown sentinel and require an explicit mesh-close log, exit zero and idle devices. The earlier signal-based shutdown lacked that explicit close evidence; it is not established as the cause of the firmware timeout. After all audited measurements, decoder owner-stop caused MPI launcher exit 1 (improper rank-0 termination). All decoder ranks and the driver exited; both hosts had no remaining task processes or device holders. This is a preserved teardown defect, not a clean-shutdown pass. The prefill native-sentinel shutdown exited 0. Both reservations were released after idle verification. The setup timings below are separate from request performance.

## Setup timing breakdown

| Phase | Seconds | Status | Finished (UTC) |
|---|---:|---|---|
| decode on bh-glx-120-c08u02: native compilation | 380.5 | DONE | 2026-10-21T11:12:47.274328+00:00 |
| decode on bh-glx-120-c08u02: imports/environment check | 19.0 | DONE | 2026-10-21T11:13:06.284328+00:00 |
| decode-b03u02 on bh-glx-120-b03u02: native compilation | 283.8 | DONE | 2026-10-21T11:45:25.312762+00:00 |
| decode-b03u02 on bh-glx-120-b03u02: imports/environment check | 20.4 | DONE | 2026-10-21T11:45:45.687149+00:00 |
| prefill on bh-glx-120-c05u02: native compilation | 336.6 | DONE | 2026-10-21T11:11:56.084019+00:00 |
| prefill on bh-glx-120-c05u02: imports/environment check | 20.8 | DONE | 2026-10-21T11:12:16.863548+00:00 |
| sweep-pd-c1024: launch to ready | 189.0 | DONE | 2026-10-21 13:24:32.955 UTC |
| sweep-pd-c16384: launch to ready | 230.7 | DONE | 2026-10-21 12:52:26.763 UTC |
| sweep-pd-c32768: launch to ready | 222.6 | DONE | 2026-10-21 12:39:51.810 UTC |
| sweep-pd-c4096: launch to ready | 191.6 | DONE | 2026-10-21 13:12:32.212 UTC |
| sweep-pd-c65536: launch to ready | 603.0 | DONE | 2026-10-21 11:22:43.928 UTC |
| sweep-pd-c8192: launch to ready | 192.1 | DONE | 2026-10-21 13:02:16.380 UTC |
| decode-attempt1-arc: launch to ready/exit | 301.8 | FAILED | 2026-10-21T11:18:34.103716+00:00 |
| current decoder: launch to ready/exit | 2,340.2 | DONE | 2026-10-21T12:25:04.115000+00:00 |
| prefill-c32768-attempt1-ethernet: launch to failure | 80.5 | FAILED | 2026-10-21T12:30:31.895504+00:00 |
| decoder site recovery | 900.2 | FAILED | 2026-10-21T11:37:25.935822+00:00 |
| prefill-c32768 site recovery | 64.6 | DONE | 2026-10-21T12:34:40.044181+00:00 |

### Completed reload-image phases by stage

These phase totals include only images whose capture has completed. The four stages run concurrently; adding their durations would overstate startup wall time. Load/build includes the combined preparation reported by the reload pipeline, not a separately measured pure compiler duration.

| Attempt | Stage | Captured images | Load/build (s) | Weight staging (s) | Tap (s) | Configure (s) | Readback (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| current decoder | 0 | 11 | 1,623.6 | 369.4 | 0.0 | 219.6 | 22.4 |
| current decoder | 1 | 10 | 1,504.3 | 372.6 | 0.0 | 258.3 | 21.7 |
| current decoder | 2 | 10 | 1,495.5 | 309.2 | 0.0 | 322.5 | 22.8 |
| current decoder | 3 | 10 | 1,487.4 | 305.1 | 0.0 | 326.6 | 25.7 |

Same-host intervals. Concurrent stages and builds overlap; do not sum them into wall time. Failed exit times use exit-marker mtime; image phase values are rounded log measurements.

## Sources and evidence

- Metal base: `aa93f1b795694a2cea1dd8ceb952147dce620463`; experiment branch `divanovic/mg-gptoss-pd-payload4352-20261021-metal`.
- Tested configuration commit: `2898a1b47b7411611e9ab41b2873e16cf5aa9ac7` (pushed and remote verified).
- Model-source delta: `GptOss120BConfig.FABRIC_PAYLOAD_SIZE = 4352`. The external wrapper records the effective imported value for every chunk group.
- Blaze: `d6d0b09f22f60a161d66fe4db90b9abed81ef82d`; nested Metal: `44d66500520fda9f2c7060c0f6b41ec48f7ab37e`.
- D-gen: `5ca79e3076f193c092962d34a3b69731e39aff91`.
- Remote evidence root: `/data/divanovic/gptoss-payload4352-20261021`.
- Final measurement audit: all 29 cases complete, zero errors; no goldens or PCC.
- Publication and cleanup: Tested configuration commit 2898a1b47b7411611e9ab41b2873e16cf5aa9ac7 is pushed and remote verified. The 3,201-file measurement/source archive is verified. Both reservations were released after empty process/device-holder checks. Decoder shutdown returned MPI exit 1, preserved as a teardown warning. Results, commands and complete timing CSVs accompany this report.

## Interactive artifacts

- [Graphs, toggles, progress and per-run details](http://127.0.0.1:8768/payload4352-comparison.html)
- [Experiment configuration, commands and evidence guide](http://127.0.0.1:8768/PAYLOAD4352_MEASUREMENTS.md)
- [All 116 requests, including warmups](PAYLOAD4352_RUNS.csv)
- [Case means, sample variances and standard deviations](PAYLOAD4352_STATISTICS.csv)
- [Matched throughput CSV](http://127.0.0.1:8768/payload4352-comparison.csv)
- [Complete live data](http://127.0.0.1:8768/payload4352-comparison.json)


## Raw source and evidence archive

Path: `/data/divanovic/gptoss-payload4352-20261021/archives/gptoss-payload4352-20261021-source-evidence.tar.gz`

SHA256: `6f6c460a231e6d0da2991981c2753dd15f1d834cc0545908bdd5a3e98c715c56`

3,201 saved files; 67,770,826 compressed bytes. Includes raw prompts, outputs, logs, failed attempts, source patches and completed coordinator scripts. Native build trees and weight caches are excluded. The archive predates the result-publication commit.
