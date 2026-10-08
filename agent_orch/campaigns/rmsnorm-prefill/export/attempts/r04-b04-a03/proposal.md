# r04-b04-a03: position-aware input-read depth: workers left of the x=9 DRAM column keep 6 trid blocks (24 tiles) of input reads in flight instead of 4, so the slow side's read and stick push stop gating F_COLLECT

## Motivation
On the parent (r04-b04-a02), the AG start (F_COLLECT end) is gated by the **slowest worker's stick push**. That worker is
always on the left half of the grid (NoC0 x < 9). The cause is the input read, not compute or the push.

`analysis/lr.py` on the parent's report (medians over measured calls, per device, µs from the chip's first kernel start).
Columns are group max over left (x<9) vs right (x>9) workers:

| shape | rd_end max L / R (dev 0..3) | push_e max L / R (dev 0..3) | F_COLLECT end |
|---|---|---|---|
| h3584 | 3.86/3.80, 3.59/3.37, 3.39/3.26, 3.35/3.17 | 4.85/4.83, 4.54/4.25, 4.40/4.10, 4.33/4.00 | = L push + 0.1 |
| h4096 | 4.30/4.01, 4.38/3.70, 4.22/3.85, 4.17/3.89 | 5.66/5.01, 5.70/4.72, 5.41/4.85, 5.36/4.81 | = L push + 0.1 |
| h6144 | 5.82/5.19, 6.06/5.36, 5.83/5.29, 5.76/5.19 | 7.22/6.28, 7.44/6.46, 7.17/6.36, 7.18/6.24 | = L push + 0.1 |
| h7168 | 6.65/6.57, 6.66/6.28, 6.30/5.87, 6.31/5.74 | 8.04/7.63, 7.90/7.33, 7.67/6.88, 7.78/6.81 | = L push + 0.1 |

- On every shape and chip the left group finishes its read later: 0.1-0.7 µs in rd_end max, and 0.3-1.0 µs in push.
  F_COLLECT ends 0.1 µs after the left group's last push, so the right group's 0.3-1.0 µs of slack is wasted.
- Two parts, from `percore.py` / `pushgate.py`:
  - The **read itself is slower on the left**. At h7168 on dev 2/3 the cores start within 0.28 µs of each other. Even
    so, the row-2 left cores end at 6.1-6.3 µs and the right cores at 5.3-5.9. Median read duration is L 5.83-6.23 vs
    R 5.35-5.79 at h7168.
  - On h4096/h6144 the left cores also **launch** ~1 µs later. They drain last in the previous call and so restart
    last: a cross-call loop of the kind r03-b04-a03 found. A faster left read shortens that loop as well.
- The same pattern holds on the HiFi2 stack r04-b01-a03 (`analysis/lr_r04-b01-a03_out.txt`): L push max is
  0.2-0.9 µs behind R, and F_COLLECT follows L.
- The trid-pipelined reader (`read_input_pass_pipelined`) keeps `kInputLookahead = 4` blocks of `block_size` (4)
  tiles in flight, i.e. 16 x 2 KB, the same on every core. With equal outstanding reads, a core's read rate is
  outstanding / round-trip latency. The left cores' round trip is longer: their x=9 bank responses take the east wrap
  on NoC0. So they get a smaller share of the bandwidth.
- Per-core read rate does respond to depth: r03-b04-a02 measured 13.7 vs 8.9 tiles/µs per core with lookahead 8 vs 4
  (in a 10-core wave).

## Mechanism
Reader only (`dit_rmsnorm_fused_reader.cpp`, `read_input_pass_pipelined`). The lookahead becomes a per-core runtime
value instead of the constexpr 4:
- `my_x[0] < 9` (left of the x=9 DRAM column; same test as the writer's dual-NoC drain rule): lookahead **6** blocks
  (24 tiles, 48 KB in flight);
- right side: lookahead **4** (unchanged).

Trids stay 1..14 (lookahead 6 < 14). The input CB already holds the whole row, so no CB change is needed. No host or
factory change, so no rebuild. Compute, writer and forwarder are untouched.

## Why this is not a repeat
- r04-b03-a03 (bank de-phasing per core) changed *which banks* each core hits in lockstep: neutral. This node leaves
  the access order alone and changes *how much* each side keeps in flight, based on the measured per-side lag.
- r03-b04-a02 used lookahead 8 inside a 2-wave schedule (10 cores per wave). That failure came from the waves, not the
  depth. No node has changed the read depth for the 20-core single-wave read, and none has made it depend on position.
- r02-b02-a01 #4 / dual-NoC reads: not done here. The read stays on NoC0, so there are no new link paths.

## Expected effect and risk
- If the left read is outstanding/latency-limited, the left read finishes 0.3-0.6 µs earlier at h6144/h7168 and
  0.1-0.2 at h3584. The gate then moves toward the right group's push (R push max is 0.3-1.0 µs earlier today).
- If the read is aggregate-bound, the right side slows a little, and the two sides meet near the mean, which is still
  earlier than today's left max.
- Expected: F_COLLECT end -0.15..-0.5 µs. At HiFi4, part of a faster read is eaten by the PRE compute backlog (left
  cores' push - read end is 1.1 µs today), so kernel end -0.1..-0.4 µs and score ~1.41-1.42.
- Risk: a deeper queue on the left only adds latency (link-bound): neutral. If the right group gets starved past the
  left, it becomes the gate. `lr.py` shows which side gates, so it will be visible.
- Accuracy is unchanged: same bytes in the same CB slots. Hang risk: none new. The trid ring is unchanged and the
  lookahead stays below the trid count.
