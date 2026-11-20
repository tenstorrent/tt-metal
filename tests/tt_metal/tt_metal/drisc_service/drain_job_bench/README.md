# Drain jobs in the DRISC service: measurements

The streaming profiler's drainers run as drain jobs inside the DRISC service (`impl/drisc_service/kernel/drain_job.hpp`),
16 per chip, one on each seat of each bank. A drain job takes a turn in the service's round-robin after every
`N` copy turns (`TT_METAL_STREAMING_PROFILER_DRAIN_HOLD_TURNS`, default 1; a copy turn is up to 24 KB) and holds the
seat for `M` sweeps (`TT_METAL_STREAMING_PROFILER_DRAIN_HOLD_SWEEPS`, default 1). These scripts measure what that costs
each side. All run from the tt-metal root after `./build_metal.sh --build-tests`.

| Script | What it does |
|---|---|
| `knee.sh` | Profiler knee: full grid, uniform zone rate, each run about 200 ms of producer time |
| `sweep.sh` + `sweep_row.py` | Knee runs under a steady copy load (`DriscService.SteadyLoad`) from a second process; per run, drainer stalls and the copy rate and probe latency before and during the workload |
| `knee_search.sh` | Raises the delay per load and N until every rep has 0 stalls |
| `stress_with_knee.sh` | Byte-verified copies (`DriscService.Stress`) while knee runs come and go |

They need Blackhole, firmware bundle ≥ 19.12.0, and KMD 2.10 or KMD 2.9 with `dma_address_bits=58`. Without that, the
KMD hands out IOVAs at the top of the 64-bit space once a chip's low 4 GiB runs out, and the service refuses them.

## Results on bh-38, 2026-11-20

p150b, PCIe Gen4 x16, KMD 2.9.0 with `dma_address_bits=58`, fast dispatch, 128 MB spool per bank and 128 MB host FIFO
per drainer. Stalls are the profiler's device stall count per rep; stall counts are bimodal between reps. "Copy" is
the copy load's rate during the profiler workload, with its rate before the profiler attached in brackets. No run
hung, and no capture reported missing words.

With nothing else on the seats, `SteadyLoad` (16 seats, 1 MiB commands) moves 220.6 GB/s local with a 12.4 µs probe
median, and 20.4 GB/s d2h with a 62.7 µs probe median. The d2h load alone also runs at about 12 GB/s on some runs.

### Three scenarios, N = 1, M = 1

| Delay | Baseline: main, 8 relays | Drain jobs, no copies | Drain jobs + local copies | Drain jobs + d2h copies |
|---|---|---|---|---|
| 100 | 719 / 1609 / 0 | 0 in 6 of 6 | 272 / 709 / 0; copy 66-79 GB/s (≈224) | 9736 / 1264 / 760; copy 8-13 GB/s (12-21) |
| 50 | 1780 / 1820 / 600 | 0 in 7 of 14, others 571-1730 | 1130 / 585 / 565; copy 60-62 GB/s (≈223) | ~200k; workload 321 ms not 190; copy 7-8 GB/s (12) |
| 25 | 175k-189k; 219 ms | 244k-288k; ~250 ms | 239k-326k; ~248 ms; copy 47-72 GB/s (≈222) | 0.97M-1.18M; 376-531 ms; copy 8-11 GB/s (12-21) |

### Hold settings at delay 50, local copies

| N, M | Stalls | Copy during workload (of ≈223 GB/s) |
|---|---|---|
| 1, 1 | 1130 / 585 | 60-62 GB/s |
| 4, 1 | 1180 / 570 | 141-156 GB/s |
| 16, 1 | 0 / 905 | 181-186 GB/s |
| 1, 4 | 585 / 582 | 23-33 GB/s |
| 4, 4 | 1195 / 1748 | 69-71 GB/s |

With d2h copies a larger N hurts both sides: N = 16 stretched the 190 ms workload to 2.1-3.7 s (955k stalls), since
copies then take more of the PCIe link.

### Findings

- Local copies barely change the drainer's stalls; they lose throughput instead, about 70% at N = 1 while the profiler
  is busy, because each 24 KB copy turn is followed by a drain of the copy pipeline and a drainer hold.
- d2h copies and drain jobs share the PCIe link, so at N = 1 and delay 50 the drainer goes from ~1k to ~200k stalls.
- The 64 B probe's median stays 9-15 µs, but single holds reach 9-37 ms. A hold waits for its bounces to reach the
  host before it hands the seat back, so a slow host holds the seat.
- Every copy completes. `SteadyLoad` checks completion, not bytes; `stress_with_knee.sh` checks bytes.

### Knee with copies, and byte-verified copies

The knee is the first delay at which all 3 reps show 0 stalls (`knee_search.sh`, M = 1, one delay above it also run).

| Scenario | Knee | Copy rate at the knee |
|---|---|---|
| Baseline, main | above 100 (stalls in 2 of 3 reps at 100) | |
| Drain jobs, no copies | 100 or below | |
| + local copies, N = 1 | 150 | 85-88 GB/s of ≈223 |
| + local copies, N = 16 | 150 | 194-220 GB/s of ≈223 |
| + d2h copies, N = 1 | 300 | about the copy load's own rate |
| + d2h copies, N = 16 | none up to 1200 (~2.9k stalls there) | about the copy load's own rate |

`stress_with_knee.sh` (180 s of `DriscService.Stress`, about 27k byte-verified commands, while knee runs at delays
100, 50 and 25 come and go) passes at N = 1 and N = 16 with no mismatches. Before the drain job bumped the service
heartbeat during its holds, 1-2 stress clients per run refused to attach: a client reads the heartbeat twice 20 ms
apart and takes a still one for a hung kernel, and a hold that waits on the host can outlast that.
