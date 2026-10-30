# NoC contention analysis

A method, with three small scripts, for finding out whether a kernel is bound by the NoC, on which links, in which
time windows, and by which traffic — and for pricing a change before building it. It is built on two things
tt-metal and tt-npe already provide: the device profiler's NoC-event tracing, and tt-npe's trace replay.

```
tools/noc_contention/
  noc_trace_summary.py      what the trace contains: event types, bytes, destinations, per-core GB/s per window
  npe_replay.sh             replay every trace of a .logs dir in tt-npe, write the timelines and the link reports
  noc_link_demand.py        per-link demand over time from a tt-npe timeline: the oversubscribed links, rows, windows
  noc_trace_whatif.py       edit a trace (scale / drop / move a traffic class) for a what-if replay
  examples/gdn_handoff_timeline.py   layer 1 for one op: the per-chunk hand-off timeline of the fused GDN prefill
```

## The three layers

**1. Where does the time go, per unit of work?** Zones in the dataflow kernels around every wait (for a credit,
for a flag, for a NoC barrier, for CB space), read back from the device profiler with their timestamps, and paired
across the cores that talk to each other (producer item n ↔ receiver chunk c). Their *tails* name the rows and the
time windows where the NoC is slow; their means say whether the compute is starved at all. This layer is
op-specific — `examples/gdn_handoff_timeline.py` is the one written for the fused GDN prefill (it pairs the
producer's ready / credit-seen / issued / VALID-sent timestamps with the receivers' credit-sent / VALID-seen / step,
per head and chunk, through the op's own core map).

**2. Which links, and how hard?** Capture every NoC transfer of one run with the profiler's NoC-event tracing,
replay it in tt-npe, and read tt-npe's *per-timestep* link demand (not its summary): mean demand and
oversubscribed timesteps per link, per physical row and time window, per link kind.

**3. What if?** Edit the trace — halve a class's bytes, drop it, move a source or a destination — and replay. The
difference in link demand is the price of the change, available before any kernel is touched.

## Step by step

```bash
# 2a. capture: one call of the op (keep it to one call; the device holds a limited number of events)
TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 \
    python my_op_once.py
ls generated/profiler/.logs/      # noc_trace_dev0_ID<program>.json, soc_descriptor.yaml, topology.json, profile_log_device.csv

# 2b. what is in it (no simulator needed): types, bytes, DRAM ports hit, per-core inbound / outbound GB/s
python3 -I tools/noc_contention/noc_trace_summary.py generated/profiler/.logs/noc_trace_dev0_ID1024.json --core 3,2

# 2c. replay in tt-npe and report the links (tt-npe: git clone https://github.com/tenstorrent/tt-npe && ./build-npe.sh)
TT_NPE_ROOT=~/tt-npe tools/noc_contention/npe_replay.sh generated/profiler/.logs out/ P150 -- --bin 8 --rows 2,3

# 2d. look closer at one row: per-column demand of each link kind per window
python3 -I tools/noc_contention/noc_link_demand.py out/npe_timeline_noc_trace_dev0_ID1024.json --bin 8 --rows 2 --top 20

# 3. what if the hand-off carried half the bytes
python3 -I tools/noc_contention/noc_trace_whatif.py generated/profiler/.logs/noc_trace_dev0_ID1024.json whatif/noc_trace_dev0_ID1024.json \
    --type WRITE_ --dst 3,2 --dst 4,2 --bytes 4096,8192,16384 --scale-bytes 0.5
cp generated/profiler/.logs/{soc_descriptor.yaml,topology.json} whatif/ && TT_NPE_ROOT=~/tt-npe tools/noc_contention/npe_replay.sh whatif out_whatif/
```

`noc_link_demand.py` prints, in this order: the busiest links (oversubscribed timesteps, mean demand, peak, and the
window they are oversubscribed in), the oversubscribed link-timesteps per physical row and window, the same per
row by link kind, and for `--rows` the per-column heatmap of each `--kinds` kind. `--csv` writes the per-link,
per-window table for plotting. Link kinds: `NOC0_EAST` / `NOC0_SOUTH` are the +x / +y hops out of a core on
NOC_0 (which routes +x then +y, wrapping the torus), `NOC1_WEST` / `NOC1_NORTH` the −x / −y hops on NOC_1 (−x then
−y), `*_IN` is the core's NIU injecting into the NoC, `*_OUT` the NoC ejecting into the core. Coordinates are the
trace's physical NoC coordinates (Blackhole: workers x = 1–7 and 10–16, y = 2–11; DRAM ports at x = 0 and x = 9).

## What to know about the inputs

- **tt-npe's summary hides the problem.** Its "congestion impact" and "max link util" average over the run and every
  link; a hand-off link above 100 % demand for half the run showed as "congestion impact 0 %, max link util 9 %". Use
  the per-timestep `link_demand`, which is what `noc_link_demand.py` reads.
- **The link model is 32 B per cycle**, ≈ 43 GB/s at 1.35 GHz on Blackhole. That number is the one to compare a
  traffic class against (7 producers × 92 KB per 12.3 µs ≈ 52 GB/s does not fit one link).
- **A rectangle multicast is logged as one unicast per destination core**, so its link load is counted once per hop
  per destination (an overestimate on the shared hops) and the multicast link-utilization statistic stays 0.
- **Semaphore increments and sets are not traced**: credits and flags are invisible. Their bytes are negligible,
  their latency is not; that is what layer 1 measures.
- **DRAM reads carry the port's coordinates**, so the "DRAM-port row" effect (a port's responses run along its row
  on NOC_0, +x from x = 0) is explicit in the replay. Which of a bank's three ports a core uses is firmware-dependent
  (`soc_descriptors/*.yaml`, `worker_endpoint` / `relocated_worker_endpoint`); the trace settles it.
- **`READ_WITH_STATE` events carry 0 bytes**; the bytes sit on the preceding `READ_SET_STATE`.
- **One call per capture.** The device buffers a bounded number of events; profile one op instance (set
  `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000` so the DRAM buffer holds the event stream).
- The trace costs a few percent of kernel time (one profiler write per NoC call); timings taken with it on are
  not the op's timings.

## How it was used: the fused GDN prefill on Blackhole (2026-10-29)

The op streams seven fp32 tensors (76 KB per 32-token chunk) from 7 producer cores to each head's 2 receiver cores
over a credit / VALID hand-off; the receivers' scan chain is the critical path. Three of twelve heads at BH 12 ended
10–14 µs after the rest and the op is their maximum.

- Layer 1 (`examples/gdn_handoff_timeline.py`) showed the receivers' compute waiting 0.15–0.5 µs of every 1.75 µs
  step for its next chunk, more in specific *physical* rows (rotating the heads over the rows moved nothing); the
  hand-off's issue time per item was 3.3–3.9 µs for ~92 KB and grew with the hand-off depth (a deeper ring was
  slower), while the receiver's reader was not the bottleneck (a non-blocking reader changed nothing). So: the
  hand-off was link-bound, and some rows worse.
- The routing rules plus the SoC descriptor said which rows: NOC_0 routes +x first, and a DRAM port's read
  responses run east along the port's row; the rows late were exactly the four rows with an x = 0 NOC_0 port.
- The fix (PR #63364): put four of the seven producers west of the receivers writing on NOC_0 and three east on
  NOC_1 — two links into each receiver pair instead of one — and only two west in the port rows. BH 12
  151.2 → 143.6 µs, bit-identical.
- Layer 2 then quantified what remained: with the solve, the NOC_1 link into a 2-west row's receivers at 83 % mean
  demand and above 100 % in 2700 of 6420 timesteps (peaks 4–5×), the NOC_0 link into a 4-west row's receivers at
  70–82 %; with the faster FPU-chain producers the same picture plus the kickoff read burst in the top rows. That
  is why a 25 % shorter producer item bought nothing at this shape: the chain was eating link time, not item time.
- Layer 3 priced the remaining lever: halving the hand-off bytes in the trace took every hand-off link out of
  oversubscription (steady-state oversubscribed link-timesteps per row and 8 µs window 1500–3400 → 100–300); what
  remained was the kickoff read burst through the DRAM-port columns and rows. fp16 then turned out to be refused at
  kernel build (one exponent family per core) and bf16 is a numerics change, so the decision is the model owner's —
  but the price was known before either was tried.

## Using it on another kernel

1. Instrument the waits. Put a `DeviceZoneScopedN` around every blocking NoC-adjacent wait in the dataflow kernels
   (semaphore waits, `noc_async_*_barrier`, `cb_wait_front` / `cb_reserve_back` that depend on another core), and
   around the issue of each large transfer. Pair the zones across cores by the unit of work they belong to (a
   chunk, a tile row, a block) — that pairing is the only op-specific code, and it is where the time goes.
2. Capture one call with NoC events on, run `noc_trace_summary.py`: name the traffic classes (which cores write to
   whom, how many bytes per window, which DRAM ports), and check the classes you expect are all there (the gaps
   above).
3. Replay and read `noc_link_demand.py`: links above 100 % for a large share of timesteps, in rows that match
   layer 1's slow rows, are the bound. Links at 70–85 % mean with large peaks are the ones a small change tips.
4. For each candidate fix, edit the trace and replay before coding: fewer bytes (`--scale-bytes`), a different
   placement (`--remap-src` / `--remap-dst`), a dropped class (`--drop`), a different port row. Keep the one whose
   link demand drops where layer 1 says the time is lost.
5. Build the winner, re-capture, and check layer 1 again: the wait that was the bound should be gone, and the next
   one is visible.

Routing facts to reason with on Blackhole: NOC_0 goes +x then +y, NOC_1 −x then −y, both wrap the torus; a
transfer's whole route lies in the source row then the destination column; DRAM ports sit at x = 0 and x = 9 in
every row 0–11, and which port a bank answers on depends on the NoC and the firmware's endpoint table.
