# Drain jobs in the DRISC service: measurements

The streaming profiler's drainers run as drain jobs inside the DRISC service (`impl/drisc_service/kernel/drain_job.hpp`),
16 per chip, one on each seat of each bank. A drain job takes a turn in the service's round-robin after every
`N` copy turns (`TT_METAL_STREAMING_PROFILER_DRAIN_HOLD_TURNS`, default 4; a copy turn is up to 24 KB) and holds the
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

N defaulted to 16 after these results, since it keeps local copies at 194-220 GB/s with the same knee as N = 1; the
bh-17 results below moved it to 4.

### Copies with the profiler attached but idle

Delay 100000 over 3 s (`TARGET_MS=3000 sweep.sh <out> 100000 2 "1:1 16:1" "local d2h"`), about 12 MB/s of profiler
data in all, so the drain jobs' sweeps find almost nothing. Copy rate before the profiler attached, then during:

| N, M | Local copies | d2h copies |
|---|---|---|
| 16, 1 | 221-225 then 231 GB/s | 12.2 then 12.1, 20.4 then 20.2 GB/s |
| 1, 1 | 220-222 then 140 GB/s | 20.7 then 20.4, 11.8 then 11.7 GB/s |

At N = 16 an idle profiler costs the copies nothing measurable. At N = 1 it costs local copies 37%, the price of a
handover (drain the copy pipeline, an empty sweep, re-arm) after every 24 KB turn. d2h copies are bound by PCIe, not
by the seat, at either N.

### Where the 64 B probe's ~9 ms worst case comes from

In idle runs the probe's worst case reached 8.6-10.4 ms, against about 0.3 ms before the profiler attaches. Three
checks place it at the profiler's close, not in normal operation:

- `SteadyLoad`'s own loop never stalled more than 0.3 ms, so the probing thread was not off the CPU.
- The drain jobs' device counters (longest hold, longest handover, holds over 1 ms) stayed at 6-160 µs on bh-38 and
  14-18 µs on bh-17 for every hold but the last, which they leave out: the finishing hold that drains the whole spool
  to the host when the profiler closes.
- Without the profiler, even with two more idle clients on the seat, no probe took over 2 ms. With it, each run has
  exactly one slow probe, its client's control page silent for the whole of it, and on bh-17 it falls in the 50 ms
  window ending 30-43 ms after "workload done", where the close begins.

So the finishing hold kept the seat for about 9 ms while the host took the spool's tail. The finish now runs over the
job's turns, at most 100 µs of draining per turn, and the counters cover every hold. On bh-17 the idle probe's worst
case went from about 9.5 ms to 0.24-0.25 ms, no hold exceeded 22 µs, and full-grid runs at delays 100 and 50 stayed
lossless. Closing (from "workload done" to the capture summary) took about 30 ms idle, 341 ms at delay 50 and about
525 ms at delay 100; there is no figure from the single-hold finish on bh-17 to compare.

## Results on bh-17 (Gen5), 2026-11-23

One p150-class card at PCIe Gen5 x16 with 7 DRAM banks (so 14 drain jobs), KMD 2.9.0 with `dma_address_bits=58`, a
6-core Ryzen 5 7600X host, fast dispatch, 128 MB spool per bank and 128 MB host FIFO per drainer. bh-17's AICLK runs
slower than bh-38's (a 200 ms target takes about 250 ms), so delays here mean less load per second than on bh-38.
Copies alone run at 195.7 GB/s local (probe median 8.7 µs) and 54.5 GB/s d2h (probe median 21.4 µs).

### Profiler only

| Delay | Baseline: main at 121306e3d78, 8 relays | Drain jobs (N = 16) |
|---|---|---|
| 10 | 2 / 13 / 5 | 383 / 251 / 227 |
| 15 | 0 / 1 / 0 | 0 / 0 / 0 |
| 20 to 100 | 0 in every rep | 0 in every rep |
| Knee | 20 | 15 |
| Workload at delay 100 | 248 ms | 268-281 ms |

With no stalls at all, the workload still runs about 10% longer with the drain jobs, at every delay; the cause is not
known yet. bh-38 showed no such difference (198 ms in both).

### With copies

| Copies | Knee, N = 1 | Copy rate there | Knee, N = 16 | Copy rate there |
|---|---|---|---|---|
| local | 25 or lower | 70-84 GB/s | 35 | 181 GB/s |
| d2h | 50 | 40-42 GB/s | 300 | about 52 GB/s |

At delay 50:

| N, M | local: stalls, copy rate | d2h: stalls, copy rate, workload |
|---|---|---|
| 1, 1 | 0, 79-86 GB/s | 0, 40-43 GB/s |
| 4, 1 | 0, 149-154 GB/s | about 220, 45-46 GB/s |
| 16, 1 | 0, 185 GB/s | 661k, 41 GB/s, 590 ms |
| 64, 1 | 850k, 190 GB/s | 934k, 45 GB/s, 1.78 s |
| 1, 4 | 0, 28-36 GB/s | 0, 24-27 GB/s |
| 16, 4 | 0, 164 GB/s | 331k, 46 GB/s |

An idle profiler costs d2h copies nothing at N = 1 or 16 (54 GB/s), and local copies nothing at N = 16 (42% at N = 1).
No hold exceeded 1 ms in any run (longest 613 µs, d2h at N = 64).

N = 16 suits local copies, but with d2h copies it gives them enough of the PCIe link that the drain jobs fall behind
until delay 300; N = 4 is close to balanced for both. The best N depends on the kind of copy.

### Defaults: N = 4, M = 1

N = 16 suits local copies but, under d2h copies, gives them enough of the PCIe link that the drain jobs fall behind
until delay 300 on bh-17. At delay 50, N = 4 kept the drain jobs at 0 stalls under local copies and about 220 under d2h
copies, with local copies at 149-154 of 198 GB/s and d2h copies at 45-46 of 54 GB/s, so it is the default.
`TT_METAL_STREAMING_PROFILER_DRAIN_HOLD_TURNS=16` favours local copies, `=1` the profiler. M stays 1: more sweeps per
hold only cost copies.


### Models, 2026-11-24

ResNet-50 (batch 16, trace, 2 CQs) and Llama-3.1-8B decode (batch 1, 40 tokens), main against the drain jobs at N = 4,
M = 1. Copies are `DriscService.SteadyLoad` d2h running alongside the model, about 52 GB/s inside the model's window;
the copy rows are 3 alternating runs each.

| | ResNet-50 FPS | Llama-3.1-8B tok/s/user |
|---|---|---|
| main, profiler off | 10,492 | 36.18 |
| main, profiler on | 9,961 (-5.1%) | 35.60 (-1.6%) |
| drain jobs, profiler off | 10,483 | 36.13 |
| drain jobs, profiler on | 9,935 (-5.2%) | 35.60 (-1.5%) |
| drain jobs, profiler on, with copies | 2,501-2,570 | 28.0-28.2 |
| copies only, profiler off | 176-177 | 14.24-14.28 |

The profiler costs the same with the drain jobs as with main's relays. Every profiled run had 0 stalls and 0 missing
zones; ResNet captured 1,589,504 zone rows in each, and Llama 18,889,336 with or without copies (main's Llama run has
441 more kernel launches on every RISC, a difference in the workload, with every zone start matched by an end).

The copies are what slow the models, and they slow them more with the profiler off, although the copy rate barely
changes between the two (52.5-54.7 GB/s alone, 50.6-52.3 GB/s with the profiler). The likely cause, not yet tested, is
that the copies saturate the PCIe path that fast dispatch also uses, and the drain jobs' holds open gaps in it. Sharing
that path is a question for the service, not the profiler.

### Why copies slow the models, 2026-11-25

The copies here are synthetic: ResNet does not use the service; a separate SteadyLoad process keeps every loaded seat
busy with 1 MiB copies (from a DRAM region ResNet never touches, into a 64 MiB host buffer) for the whole run. It is the
worst case for one client's copies against another process's fast dispatch, continuous and uncapped, not a model doing
its own transfers.

ResNet-50 on bh-17 against SteadyLoad copies, 2 runs each (`DRISC_SERVICE_LOAD_BANKS` limits the copies to the first
N banks; 2 seats per bank). Copy rates are inside the model's window.

| Copies | Profiler off FPS | Profiler on FPS | Copy rate |
|---|---|---|---|
| none | 10,496-10,509 | 9,923-9,937 | - |
| local, 7 banks | 6,763-7,216 | 8,833-8,842 | 200-202 GB/s |
| d2h, 1 bank | 9,166-9,186 | | 49.4-49.6 GB/s |
| d2h, 2 banks | 3,589-3,620 | | 50.6-51.0 GB/s |
| d2h, 4 banks | 800-802 | | 52.0 GB/s |
| d2h, 7 banks | 176-177 | 2,712-2,722 | 52.6 GB/s |
| d2h, 7 banks, 1 or 6 spinning host threads | 176-177 | | 52.6-53.7 GB/s |

Host memcpy (numpy, 1 / 6 processes): 39.2 / 57.7 GB/s idle, 16.1 / 49.4 GB/s under d2h copies, 38.5 / 70.1 GB/s under
local copies.

- PCIe bandwidth is not it. One bank already moves 49.6 GB/s, within 6% of seven, yet costs the model 13% where seven
  cost it 98%. The model's loss follows the number of seats writing to the host, not the bytes.
- Host memory is not it. Its load is the same at 1 bank and at 7, and the model differs 52x between them.
- Host CPU is not it. Spinning threads change nothing, so the profiler's own threads are not what helps.
- Local copies at 200 GB/s cost the model a third, far less than 1/7 of that rate crossing PCIe.

What fits is queueing at the PCIe tile. Fast dispatch's prefetcher reads its commands from host memory, and a device
read request cannot pass the posted writes queued ahead of it toward the host, so each read waits behind every seat's
outstanding writes; more writers make that queue deeper at the same bandwidth. The drain jobs' holds take each seat's
copies off the queue for a while, which is why the profiler helps (and helps under local copies too, by the same
pause). Not yet measured directly; a 64 B host-to-device probe timed against the number of writing banks would confirm
it.
