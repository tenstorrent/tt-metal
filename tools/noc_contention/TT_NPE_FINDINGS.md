# tt-npe: findings, gaps and improvements from a trace-driven analysis

Context: tt-npe (main `42cf88f8`, built 2026-10-29 with `./build-npe.sh`) replaying NoC traces that tt-metal (main
`7aad8aacd5b`) captured on a Blackhole p150 with `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` for the fused
`chunk_gated_delta_rule` prefill (one call, 108 cores, 152 µs, 120 588 events / 27 MB per trace). The op streams
seven tensors (92 KB per chunk) from 7 producer cores to each head's 2 receiver cores; the analysis is in
`README.md` and the scripts next to this file. Everything below was seen on that workload; the numbers are tt-npe's
own outputs.

Items 1–6 are tt-npe; 7–8 are in the trace tt-metal's profiler hands it; 9 is a calibration question.

## 1. `link_demand` is offered bytes per cycle, not percent — and this is undocumented

The timeline's `timestep_data[i].link_demand` rows are `[chip, row, col, kind, value]`. `value` is the *offered*
rate in bytes per cycle: every active transfer contributes its injection rate, and a transfer that is being
stretched by contention keeps contributing while it waits, so two full-rate transfers on one link show
`2 × link_bandwidth` (121.8 on Blackhole, whose model has `link_bandwidth: 60.9`). Nothing in the file says so;
the `common_info` block next to it reports `link_demand`, `max_link_demand`, `link_util` as percentages, and the
natural reading of the per-timestep value is "percent" too. Read that way, a link at 136 % offered demand looks
like 83 % and a 57 %-contended run looks 44 %-contended (that is what happened here, and the first version of our
tool and write-up carried the wrong numbers).

Improvement: document the timeline schema with units; emit per link both the offered demand and the delivered
utilization `min(demand, bandwidth)`, or normalize `link_demand` to the link bandwidth as `common_info` already does.

## 2. The summary averages the problem away

```
      avg Link util:   3.9%      max Link util:   8.8%
    avg Link demand:   6.7%    max Link demand:  19.5%
  congestion impact:   0.0%
```
for a run whose per-timestep rows show the hand-off links into every receiver pair at 123–137 % offered demand,
47–61 % delivered, contended in 47–57 % of the 6420 timesteps with peaks of 6–9×. The averages run over every link
of a 17 × 12 chip, most of which are idle, and over the whole run including the fill and drain; "max link util"
is the maximum of a per-timestep *average*, not of any link.

Improvement: a per-link report — for the N busiest links: mean offered demand, mean delivered utilization,
fraction of timesteps contended, peak, and the window in which they are contended — printed with the summary and
writable as CSV. `noc_link_demand.py` next to this file is a working sketch (`--top`, `--csv`, per-row and
per-column tables).

## 3. Golden cycles are wrong for current tt-metal traces, which voids two headline numbers

```
   estimated cycles: 205404
      golden cycles: 18446743773018584033
   cycle pred error:   0.0%
  congestion impact:   0.0%
```
`golden cycles` ≈ 2^64 − t0: `computeGoldenCyclesAndT0` (`npeWorkloadIngest.cpp`) found no per-core kernel
timestamps for the device, so `min_kernel_cycles` stayed at `numeric_limits<Cycle>::max()`, `max_kernel_cycles` at
0, and both wrapped when `t0_timestamp` was subtracted. The trace does carry the kernel zones (events with
`zone: "BRISC-KERNEL"` / `zone_phase: "ZONE_START"|"ZONE_END"`, absolute timestamps ≈ 3.0e11, `src_device_id: 0`),
so either the zone naming or the device-id keying has drifted between tt-metal's writer and tt-npe's reader.
With golden empty, `cycle pred error` and `congestion impact` (which compares the estimate against the golden span)
print 0.0 % — the two numbers a user reads first.

Improvement: fix the kernel-zone pickup for the current trace format, and refuse to print `cycle pred error` /
`congestion impact` (or print `n/a`) when the golden span is empty or wrapped.

## 4. Trace-driven replay cannot show the cost of congestion end to end; a per-transfer statistic can

Replaying recorded start times keeps the real run's stalls in the schedule: a transfer that was late because its
predecessor queued starts late in the replay too, so the end-to-end delta between the contended and uncontended
simulations is ≈ 0 even when every transfer of a class queues. What a user wants to know is how much each transfer
of a class was stretched: here the kernels' own zones measured 2.5–3.5 µs to issue a 92 KB write that needs 1.7 µs
at 82 GB/s, and the replay's link demand says why, but tt-npe reports no per-transfer or per-class delay.

Improvement: per transfer, `modelled_end − uncontended_end` (the simulator has both), aggregated by source /
destination / size class as mean, p50, p99 and total; that is the "congestion impact" that survives trace-driven
start times.

## 5. `build-npe.sh` does not install the Python module

After `./build-npe.sh`, `install/lib/` holds only `libfmt.so`; `tt_npe_pybind.cpython-310-x86_64-linux-gnu.so` and
`libtt_npe.so` stay in `build/tt_npe/`. `ENV_SETUP` puts `install/lib` on `PYTHONPATH`, so
`npe_analyze_noc_trace_dir.py` fails with `ModuleNotFoundError: No module named 'tt_npe_pybind'` and `tt_npe.py`
only works with `PYTHONPATH=build/tt_npe`. (`npe_replay.sh` next to this file does exactly that.)

Improvement: install the module and the shared library, or have `ENV_SETUP` add `build/tt_npe`.

## 6. Timeline size

One 152 µs op produced a 202 MB timeline JSON (6420 timesteps, every link with traffic listed per timestep, plus
29 964 transfers with full routes). Loading it in Python takes tens of seconds; a sweep over variants is slow.

Improvement: a compact output mode (per-link CSV of demand / utilization per timestep or per window, transfers
without routes), `--compress-timeline-output-file` by default, or streaming the per-timestep section.

## 7. (tt-metal) rectangle multicasts arrive as one unicast per destination

`noc_async_write_multicast` to a 1 × 2 receiver rectangle appears in the trace as two `WRITE_` events with `dx, dy`
set to each receiver, 0.1 µs apart, and no `mcast_start_* / mcast_end_*` fields — on this workload 0 multicast
events in 15 552 writes. tt-npe then routes two unicasts: the hop into the first receiver carries the bytes twice,
`avg Mcast link util` is 0.0 %, and the NIU injection count doubles. The trace format supports multicasts; the
kernel-side event capture (`PROFILE_NOC_EVENTS` instrumentation of the multicast API) seems to expand them.

Improvement (tt-metal profiler): log a multicast once with its rectangle; (tt-npe) warn when a trace contains
writes that look like expanded multicasts (same source, same bytes, consecutive destinations within a few cycles).

## 8. (tt-metal) semaphore traffic is not traced

No `SEMAPHORE_*` / atomic events in the trace, although the enum exists in `profiler.cpp`'s event switch. For a
credit / flag hand-off the semaphore increments and multicast sets are the control path whose latency the
receivers wait on; their bytes are negligible, their timing is not.

Improvement: trace `noc_semaphore_inc` / `noc_semaphore_set_*` (and raw atomic writes) as events with their
destination; tt-npe can then model them as small, latency-bound transfers and report their flight time.

## 9. Calibration of the link model (open question, not a bug)

Blackhole's `link_bandwidth: 60.9` B/cycle (≈ 82 GB/s at 1.35 GHz) is consistent with what the kernels measured
here (a 92 KB write needs 1.7 µs uncontended; measured 2.5–3.5 µs on links the replay shows 47–61 % delivered,
contended half the time), but nothing in this analysis checked a single uncontended link against silicon, nor the
DRAM-port rows (whose +x links also carry a bank's read responses) against the model's port handling.

Improvement: ship a calibration workload (N writers → one core in one row; a DRAM reader in a port row and in a
plain row) and the measured numbers it was validated against, per architecture.

## Reproducing

```bash
# tt-metal, any op (this one: the fused GDN prefill at BH 12, one call)
TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 python my_op_once.py
# replay
TT_NPE_ROOT=~/tt-npe tools/noc_contention/npe_replay.sh generated/profiler/.logs out/ P150 -- --bin 8 --top 20
# items 1-3: compare tt_npe.py's summary with out/links_*.txt; item 7: noc_trace_summary.py shows 0 multicast events;
# item 8: grep -c SEMAPHORE generated/profiler/.logs/noc_trace_dev0_ID*.json
```
