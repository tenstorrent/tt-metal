# r05-b03-a01: two-wave row pipeline on 40 half-row workers: wave B's DRAM read overlaps wave A's all-gather, wave A's drain overlaps wave B's all-gather

## Motivation
Both DRAM phases of the op are already close to the chip's DRAM bandwidth, and nothing else uses DRAM in between:
- Input read: aggregate-bound at ~440-480 GB/s with 20 cores (r04-b04-a03: more read depth or bank order does not
  help; r04-b03-a03: bank de-phasing does not help).
- Output drain: ~93 ns per 2 KB tile per core, so 20 cores give ~430 GB/s. Every per-core knob is ruled out (issue,
  cmd buf, VC, ack, bank phase). r04-b02-a02/-a03 found that synchronized drains contend with each other.
- Between them sits a fixed latency chain with the DRAM idle: PRE tail (~0.8 µs at HiFi4), stick push (0.25),
  all-gather (F_COLLECT end -> F_FABRIC end, median 2.35-2.6 µs), go fan-out, stick read and combine
  (F_FABRIC end -> C_POST start, 1.6 µs). That is ~5 µs of the 12-18 µs kernel (r04-b04-a02 `tl.py`, h3584: read
  ends 3.2, POST starts 8.5, drain ends 11.6).

So the kernel is roughly `read(all rows) + fixed AG chain + drain(all rows)`. The only way to shrink it further is to
**overlap DRAM traffic with the AG chain**: pipeline the rows in two waves, so that one wave's read or drain runs
while the other wave is in its all-gather.

r03-b04-a02 tried two waves of 10 full-row workers (even/odd). It lost 5%, because 10 cores cannot run a wave at
full rate. PRE is per-core compute-bound (~110 ns/tile at HiFi4), the drain is per-core-capped, and 10 cores only
read at ~273 GB/s. Its reflection (#4) says waves are only worth a retry "with more cores than rows: e.g. a k=2
column split (40 workers, as two waves of 20)". This node does that.

## Mechanism
On the AG path, when the shape allows (BH, RMS, one link, even row count, row count ≤ one packet's sticks, even
width, resident POST with broadcast gamma, no rope/bias), the factory switches to **40 half-row workers in two
waves of 20**:
- Worker (wave w, j in 0..19) owns tile row `w*R/2 + j/2` and column half `j%2`, so each core has num_tile_cols/2
  columns. Wave A = workers 0-19 (grid rows 0-1), wave B = workers 20-39.
- **PRE:** each half-row core pushes its partial sum(x²) stick to slot j of packet buffer w. The forked forwarder
  sends wave w's 20 sticks (2560 B) as **its own fabric packet to its own scratch page (round = wave)**. The page
  geometry comes from `compute_sizing` (2 pages per forwarder), so create_stats_buffer and validate agree.
- **Combine:** after go, a worker reads both halves' sticks of its row from all 4 devices, giving 8 gathered tiles.
  Compute is unchanged: it gets `stats_tiles_cols = 2*ring` and `num_tile_cols = W/2`. Its pairwise ELWADD then sums
  8 partials, and `1/(num_tile_cols*32*stats_tiles_cols)` is still 1/H_full.
- **Wave gate:** a wave-B reader waits on a start semaphore before its input read. Its wave-A partner (same j) ups it
  when its own read has 2 blocks left in flight. B's reads then queue right behind A's tail, A's read keeps the
  full DRAM rate, and the DRAM stays busy.
- **Forwarder:** a fork of r03-b04-a02's validated two-wave forwarder with per-wave 16-bit fields in the arrival
  and out_ready semaphores. One poll loop sends wave w once its 20 arrivals are in, and releases wave w's go once
  all peers' wave-w packets have landed. Waves are independent, so wave B's packet never waits behind wave A's
  release.
- Reader, writer and gamma addressing take a column offset (input `row*W + off + c`, gamma page `off + j`, output
  column `off + c`). The writer reads `col_split` sticks per device.

Files (all in the op dir):
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`: sizing, decomposition, RT/CT args.
- `device/dit_fused_distributed_rmsnorm_device_operation_types.hpp`: sizing fields.
- `kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: column offset and the wave gate.
- `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: column offset, per-wave page/packet/arrival field, and
  2 sticks per device.
- New `kernels/dataflow/dit_rmsnorm_wave_forwarder.cpp`.

The compute kernel is untouched: HiFi4 everywhere, no approx mode. The non-wave path keeps the exact current
behaviour (same kernels, default RT values).

## Why this is not a repeat
- **r03-b04-a02 (waves of 10 full-row cores, -5.3%):** each wave had half the cores and therefore half the per-core
  throughput. Here each wave has 20 cores, the same number that runs the whole op today, with half the columns
  each. Per-core PRE, POST and drain work halves, and the per-wave read and drain stay at the full aggregate rate.
  That reflection named this exact configuration as the condition for retrying waves.
- **r01-b03-a01/a02/a03 (column split k=3/4 without waves):** the split there only shortened per-core phases that
  turned out to be aggregate-bound. It also paid a leader-combine hop (~1-1.5 µs) on the AG-start path. Here the
  split exists so that each wave runs at full width. There is no on-chip hop: the half-row partials travel as
  separate sticks and are summed in the existing post-AG combine (8 tiles instead of 4).
- **r04-b02-* (forwarder multicast):** different idea (AG release).
- No drain-, read- or PRE-micro-optimisation, and no fidelity change.

## Expected effect and risk
Model for the critical chip, using measured rates (read ~440 GB/s + 0.6 µs start, AG ~1-2.4 µs, post-AG fixed
1.6 µs, drain ~93 ns/tile/core):
- h7168: today ~read 6.4 + chain 5.6 + drain 5.2. Two waves: A ends at ~3.2 (read) + 5.6 + 2.6 ≈ 11.4, and B's
  read ends ~6, so B ends at ≈ 6 + 5.6 + 2.6 ≈ 14.2. That is about -3 µs.
- h3584: about -1.5..-2 µs.
- Expected score 1.55-1.7 if the model holds. Even half of that is well outside noise.

Risks:
1. **Hang**, from an arrival/out_ready field mismatch, a wrong go-release list, or the wave gate. The arithmetic is
   checked against the r03-b04-a02 forwarder, which ran clean on HW.
2. **Accuracy**: a wrong column offset or a missing half-stick would give PCC ≪ 0.99999 at once. A wrong page or
   slot would give stale or zero sums.
3. **Wave B's read contending with wave A's drain** on the narrow shapes. Then the wave B tail grows, which shows up
   in per-wave R_INPUT/W_DRAIN zones.
4. **16 stick reads instead of 8** after go (+~0.1 µs).
5. More dispatch work (41 cores, one kernel group).

How to tell from the eval:
- Per-wave R_INPUT end, W_PUSH end, W_AGWAIT end and W_DRAIN end from the zones (wave = worker index ≥ 20). The
  timelines should be staggered by about one wave's read.
- The forwarder's F_SEND/F_GO zones per wave.
